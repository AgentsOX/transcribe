"""
transcribe — local audio → clean text, tuned for Hebrew & English.

Engine: faster-whisper (CTranslate2) running ivrit.ai's Hebrew-tuned Whisper
models. 100% local / offline once the model is cached.

Extras:
  --enhance        light ffmpeg speech chain (highpass + denoise + loudnorm)
  --denoise        heavy neural denoise via DeepFilterNet (needs `denoise` extra)
  --ab             A/B test raw vs enhanced and report which is better
  --reference FILE compute real WER vs a human transcript (definitive)
  --speakers       who-said-what diarization (needs `speakers` extra; local, no token)
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import tempfile
import warnings

# Keep first-run output clean. The HF Hub logs "you are sending unauthenticated
# requests… set a HF_TOKEN" while downloading public models — which wrongly
# implies a token is needed. It isn't: everything here uses public models.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
for _noisy in ("huggingface_hub", "speechbrain", "pyannote"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)
# Benign NumPy NaN/overflow chatter from Whisper's mel feature extraction.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=".*encountered in matmul.*")
from dataclasses import dataclass, field
from pathlib import Path

from . import audio
from . import diarize
from . import metrics as M
from . import report_html

# ── model aliases ────────────────────────────────────────────────────────────
MODELS = {
    "turbo": "ivrit-ai/whisper-large-v3-turbo-ct2",   # fast, ~1.6GB — default
    "large-v3": "ivrit-ai/whisper-large-v3-ct2",       # max accuracy, ~3GB
    "accurate": "ivrit-ai/whisper-large-v3-ct2",
    "openai-large-v3": "large-v3",
}

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".mov", ".aac", ".flac", ".ogg",
              ".opus", ".webm", ".m4v", ".mkv"}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    speaker: str | None = None
    words: list = field(default_factory=list)


# ── transcription ────────────────────────────────────────────────────────────
def transcribe_wav(wav: Path, model_repo: str, language: str | None,
                   compute_type: str) -> tuple[list[Segment], str]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_repo, device="cpu", compute_type=compute_type)
    segs, info = model.transcribe(
        str(wav), language=language, vad_filter=True,
        word_timestamps=True, beam_size=5,
    )
    out: list[Segment] = []
    for s in segs:
        words = [{"start": w.start, "end": w.end, "word": w.word}
                 for w in (s.words or [])]
        out.append(Segment(
            s.start, s.end, s.text.strip(),
            avg_logprob=s.avg_logprob,
            compression_ratio=s.compression_ratio,
            no_speech_prob=s.no_speech_prob,
            words=words,
        ))
    return out, info.language


def plain_text(segs: list[Segment]) -> str:
    return " ".join(s.text for s in segs).strip()


# Speaker diarization lives in diarize.py (classic embeddings + clustering,
# token-free, fully local).


# ── writers ──────────────────────────────────────────────────────────────────
def _ts(t: float, sep: str = ",") -> str:
    ms = int(round((t - int(t)) * 1000))
    whole = int(t)
    h, whole = divmod(whole, 3600)
    m, s = divmod(whole, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_txt(segs: list[Segment], path: Path, with_speakers: bool) -> None:
    if with_speakers:
        lines, cur, buf = [], None, []
        for s in segs:
            if s.speaker != cur:
                if buf:
                    lines.append(f"{cur}: {' '.join(buf)}")
                cur, buf = s.speaker, [s.text]
            else:
                buf.append(s.text)
        if buf:
            lines.append(f"{cur}: {' '.join(buf)}")
        body = "\n\n".join(lines)
    else:
        body = plain_text(segs)
    path.write_text(body.strip() + "\n", encoding="utf-8")


def write_srt(segs: list[Segment], path: Path, with_speakers: bool) -> None:
    out = []
    for i, s in enumerate(segs, 1):
        text = f"{s.speaker}: {s.text}" if with_speakers and s.speaker else s.text
        out.append(f"{i}\n{_ts(s.start)} --> {_ts(s.end)}\n{text}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def write_vtt(segs: list[Segment], path: Path, with_speakers: bool) -> None:
    out = ["WEBVTT\n"]
    for s in segs:
        text = f"{s.speaker}: {s.text}" if with_speakers and s.speaker else s.text
        out.append(f"{_ts(s.start, '.')} --> {_ts(s.end, '.')}\n{text}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def write_json(segs: list[Segment], path: Path, language: str, model: str) -> None:
    payload = {
        "language": language, "model": model,
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3),
             "text": s.text, "speaker": s.speaker,
             "avg_logprob": round(s.avg_logprob, 4),
             "compression_ratio": round(s.compression_ratio, 3),
             "words": s.words}
            for s in segs
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")


WRITERS = {"txt": write_txt, "srt": write_srt, "vtt": write_vtt}


def write_outputs(segs, dest: Path, stem: str, formats, language, model,
                  with_speakers) -> None:
    for fmt in formats:
        outfile = dest / f"{stem}.{fmt}"
        if fmt == "json":
            write_json(segs, outfile, language, model)
        else:
            WRITERS[fmt](segs, outfile, with_speakers)
        print(f"  ✓ {outfile}")


# ── A/B harness ──────────────────────────────────────────────────────────────
def variant_specs(denoise: bool) -> list[dict]:
    specs = [
        {"name": "raw", "enhance": False, "denoise": False},
        {"name": "enhanced", "enhance": True, "denoise": False},
    ]
    if denoise:
        specs.append({"name": "enhanced+denoised", "enhance": True,
                      "denoise": True})
    return specs


def render_report(stem: str, variants: list[dict], winner: str,
                  has_ref: bool) -> str:
    lines = [f"A/B QUALITY REPORT — {stem}", "=" * 60, ""]
    if has_ref:
        lines.append(f"{'variant':<20}{'WER↓':>8}{'wlogprob↑':>12}"
                     f"{'low-conf%':>11}{'halluc':>8}{'words':>8}")
    else:
        lines.append(f"{'variant':<20}{'wlogprob↑':>12}{'low-conf%':>11}"
                     f"{'halluc':>8}{'comprss':>9}{'words':>8}")
    lines.append("-" * 60)
    for v in variants:
        r = v["report"]
        if has_ref:
            lines.append(f"{v['name']:<20}{v['wer']:>8.3f}"
                         f"{r['weighted_logprob']:>12.3f}"
                         f"{r['low_conf_pct']:>11}{r['hallucination_segs']:>8}"
                         f"{r['words']:>8}")
        else:
            lines.append(f"{v['name']:<20}{r['weighted_logprob']:>12.3f}"
                         f"{r['low_conf_pct']:>11}{r['hallucination_segs']:>8}"
                         f"{r['mean_compression']:>9}{r['words']:>8}")
    lines += ["-" * 60, "", f"WINNER: {winner}"]
    if has_ref:
        lines.append("(by lowest WER vs your reference — definitive.)")
    else:
        lines.append("(by highest duration-weighted confidence — a PROXY, not "
                     "ground truth.\n Add --reference <transcript.txt> for a "
                     "definitive WER verdict.)")
    return "\n".join(lines) + "\n"


def run_ab(src: Path, model_repo: str, language: str | None, compute_type: str,
           denoise: bool, reference: str | None, dest: Path) -> None:
    ref_text = Path(reference).expanduser().read_text(encoding="utf-8") \
        if reference else None
    specs = variant_specs(denoise)

    with tempfile.TemporaryDirectory() as tmp:
        for spec in specs:
            print(f"  · variant '{spec['name']}'…")
            wav = audio.to_wav(src, tmp, enhance=spec["enhance"],
                               denoise=spec["denoise"], tag=spec["name"])
            segs, lang = transcribe_wav(wav, model_repo, language, compute_type)
            spec["segs"], spec["lang"] = segs, lang
            spec["text"] = plain_text(segs)
            spec["report"] = M.confidence_report(segs)
            if ref_text is not None:
                spec["wer"] = M.wer_against(ref_text, spec["text"])

    winner_name = M.pick_winner(specs, has_reference=ref_text is not None)
    report = render_report(src.stem, specs, winner_name,
                           has_ref=ref_text is not None)
    print("\n" + report)

    # persist: per-variant transcripts, a raw-vs-winner diff, the report
    for spec in specs:
        (dest / f"{src.stem}.{spec['name']}.txt").write_text(
            spec["text"] + "\n", encoding="utf-8")
    (dest / f"{src.stem}.ab-report.txt").write_text(report, encoding="utf-8")

    raw = next(s for s in specs if s["name"] == "raw")
    win = next(s for s in specs if s["name"] == winner_name)
    if win is not raw:
        diff = difflib.unified_diff(
            raw["text"].split(), win["text"].split(),
            fromfile="raw", tofile=winner_name, lineterm="")
        (dest / f"{src.stem}.ab-diff.txt").write_text(
            "\n".join(diff), encoding="utf-8")

    # visual side-by-side viewer with embedded audio (before vs the best
    # optimized variant; falls back to 'enhanced' when raw itself won)
    after = win if win is not raw else next(
        (s for s in specs if s["name"] != "raw"), raw)
    sub = (f"model {model_repo} · "
           + ("WER verdict" if ref_text is not None else "confidence proxy"))
    html_path = dest / f"{src.stem}.ab.html"
    report_html.build_html(
        src, raw["text"], after["text"], html_path,
        left_label="BEFORE (raw)", right_label=f"AFTER ({after['name']})",
        winner=winner_name, subtitle=sub)
    print(f"  ✓ wrote per-variant transcripts + report + {html_path.name}")


# ── orchestration ────────────────────────────────────────────────────────────
def gather_inputs(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p).expanduser()
        if path.is_dir():
            files += [f for f in sorted(path.iterdir())
                      if f.suffix.lower() in AUDIO_EXTS]
        elif path.exists():
            files.append(path)
        else:
            print(f"  ! not found: {p}", file=sys.stderr)
    return files


def run(args: argparse.Namespace) -> int:
    import shutil
    if shutil.which("ffmpeg") is None:
        print("✗ ffmpeg not found — it's required to read audio.\n"
              "  macOS:  brew install ffmpeg\n"
              "  Linux:  sudo apt install ffmpeg", file=sys.stderr)
        return 3

    model_repo = MODELS.get(args.model, args.model)
    formats = ["txt", "srt", "vtt", "json"] if "all" in args.format else args.format
    language = None if args.lang == "auto" else args.lang
    out_dir = Path(args.output_dir).expanduser() if args.output_dir else None

    files = gather_inputs(args.input)
    if not files:
        print("✗ no audio files to process.", file=sys.stderr)
        return 1

    mode = "A/B" if args.ab else "transcribe"
    print(f"▶ {mode} | model: {model_repo} | device: cpu/{args.compute_type} | "
          f"lang: {args.lang} | enhance: {args.enhance} | denoise: {args.denoise}")

    for src in files:
        print(f"\n▶ {src.name}")
        dest = out_dir or src.parent
        dest.mkdir(parents=True, exist_ok=True)

        if args.ab:
            run_ab(src, model_repo, language, args.compute_type,
                   args.denoise, args.reference, dest)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            wav = audio.to_wav(src, tmp, enhance=args.enhance,
                               denoise=args.denoise)
            print("  · transcribing…")
            segs, lang = transcribe_wav(wav, model_repo, language,
                                        args.compute_type)
            print(f"  · detected language: {lang}  ({len(segs)} segments)")
            if args.speakers:
                print("  · diarizing (speaker labels)…")
                n = diarize.assign_speakers(
                    wav, segs, num_speakers=args.num_speakers)
                if args.num_speakers:
                    print(f"  · labeled {n} speaker(s)")
                else:
                    print(f"  · auto-detected {n} speaker(s) — if that's off, "
                          f"re-run with --num-speakers N")
            write_outputs(segs, dest, src.stem, formats, lang, model_repo,
                          args.speakers)

    print("\n✓ done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description="Local audio → clean text (Hebrew & English, ivrit.ai + "
                    "faster-whisper).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  transcribe call.m4a\n"
               "  transcribe call.m4a --enhance --format srt\n"
               "  transcribe call.m4a --ab                 # raw vs enhanced\n"
               "  transcribe call.m4a --ab --reference truth.txt   # real WER\n"
               "  transcribe ./recordings --format all --model large-v3\n"
               "  transcribe call.m4a --speakers --num-speakers 2   # local, no token\n",
    )
    p.add_argument("input", nargs="+", help="audio/video file(s) or a folder")
    p.add_argument("--lang", default="auto", choices=["auto", "he", "en"],
                   help="language (default: auto-detect)")
    p.add_argument("--model", default="turbo",
                   help="turbo (default) | large-v3 | any HF repo id")
    p.add_argument("--format", default=["txt"], nargs="+",
                   choices=["txt", "srt", "vtt", "json", "all"],
                   help="output format(s) (default: txt)")
    p.add_argument("--enhance", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="light ffmpeg speech chain (ON by default; "
                        "use --no-enhance for raw audio)")
    p.add_argument("--denoise", action="store_true",
                   help="heavy neural denoise (DeepFilterNet; needs `denoise` extra)")
    p.add_argument("--ab", action="store_true",
                   help="A/B test raw vs enhanced and report which wins")
    p.add_argument("--reference", default=None,
                   help="human transcript (.txt) → compute real WER in --ab mode")
    p.add_argument("--speakers", action="store_true",
                   help="label who-said-what (needs `speakers` extra; local, "
                        "no token)")
    p.add_argument("--num-speakers", type=int, default=0,
                   help="force the speaker count (0 = auto-detect); set it when "
                        "you know it, e.g. 2 for a call")
    p.add_argument("--output-dir", default=None,
                   help="where to write outputs (default: next to each input)")
    p.add_argument("--compute-type", default="int8",
                   choices=["int8", "int8_float32", "float32"],
                   help="CTranslate2 compute type (default: int8)")
    args = p.parse_args(argv)

    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n✗ interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
