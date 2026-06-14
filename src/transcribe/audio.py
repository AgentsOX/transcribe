"""Audio decode + optional enhancement, all feeding 16kHz mono wav to Whisper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Conservative ffmpeg speech chain (the "safe default" — no neural models):
#   highpass=80   → remove low rumble / AC hum / desk thumps
#   afftdn        → light FFT denoise (nr=10dB is gentle on purpose)
#   loudnorm      → EBU R128 loudness normalization (biggest safe win for
#                   quiet / far-from-mic / uneven recordings)
# Heavy denoising is intentionally NOT here — it can hurt Whisper accuracy.
ENHANCE_CHAIN = "highpass=f=80,afftdn=nr=10:nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def deepfilter_denoise(src: Path, tmpdir: str) -> Path:
    """Heavy neural denoise via DeepFilterNet (optional extra). Returns a wav.

    Uses the `deepFilter` CLI installed by the `denoise` extra. Use only on
    genuinely noisy audio (phone/street/crowd) — and always A/B it.
    """
    if shutil.which("deepFilter") is None:
        raise RuntimeError(
            "--denoise needs DeepFilterNet. Install it with:\n"
            "    uv sync --extra denoise")
    outdir = Path(tmpdir) / "df"
    outdir.mkdir(exist_ok=True)
    _run(["deepFilter", str(src), "-o", str(outdir)])
    produced = next(outdir.glob("*.wav"), None)
    if produced is None:
        raise RuntimeError("DeepFilterNet produced no output.")
    return produced


def to_wav(src: Path, tmpdir: str, *, enhance: bool = False,
           denoise: bool = False, tag: str = "") -> Path:
    """Decode `src` → 16kHz mono PCM wav, optionally enhanced/denoised.

    Pipeline order: [DeepFilterNet denoise] → ffmpeg [enhance chain] → 16k mono.
    `tag` distinguishes variants when A/B testing (e.g. "raw", "enhanced").
    """
    source = src
    if denoise:
        source = deepfilter_denoise(src, tmpdir)

    out = Path(tmpdir) / f"{src.stem}{('_' + tag) if tag else ''}.wav"
    af = f"{ENHANCE_CHAIN}," if enhance else ""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source)]
    if af:
        cmd += ["-af", af.rstrip(",")]
    cmd += ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    _run(cmd)
    return out
