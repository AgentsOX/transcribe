# apps/transcribe — local audio → text (Hebrew & English)

A small CLI that turns audio/video into clean text. Built for AgentsOX internal
use (transcribing discovery calls, voice notes, client recordings). **Runs 100%
locally** — recordings never leave the machine.

## Stack (and why)

- **faster-whisper** (CTranslate2) as the runtime — no PyTorch needed for plain
  transcription, so the base install is light and fast.
- **ivrit.ai** Hebrew-tuned Whisper models (fine-tuned on 22k+ hrs of Hebrew).
  They beat stock Whisper on Hebrew and handle English fine.
  - `turbo` → `ivrit-ai/whisper-large-v3-turbo-ct2` (~1.6GB, default)
  - `large-v3` → `ivrit-ai/whisper-large-v3-ct2` (~3GB, max accuracy)
- **pyannote.audio** for optional speaker labels (`--speakers`). This is the only
  feature that pulls in PyTorch, so it lives behind the `speakers` extra.

> We deliberately did **not** use WhisperX: its word-level alignment step has no
> Hebrew model (adds nothing for us) and it drags in heavier deps. We use the same
> underlying pieces (faster-whisper + pyannote) directly.

## Setup

```bash
cd apps/transcribe
uv sync                     # base: transcription + txt/srt/vtt/json + --enhance + --ab
uv sync --extra eval        # adds jiwer for real WER (--ab --reference)
uv sync --extra speakers    # adds pyannote + torch for --speakers
uv sync --extra denoise     # adds DeepFilterNet for --denoise
```

First run downloads the model (cached in ~/.cache/huggingface afterwards).

## Usage

```bash
uv run transcribe call.m4a                          # → call.txt (auto HE/EN)
uv run transcribe call.m4a --lang he --format srt    # Hebrew subtitles
uv run transcribe ./recordings --format all          # batch a folder, all formats
uv run transcribe call.m4a --model large-v3          # max accuracy
uv run transcribe call.m4a --speakers                # who-said-what (needs HF_TOKEN)
```

Outputs are written next to each input file (or use `--output-dir`).

## Audio enhancement + A/B testing

The light ffmpeg speech chain (`--enhance`) is **on by default** — verified to
read better on our Hebrew call recordings. Heavy neural denoising (`--denoise`)
stays off, since it can *hurt* Whisper. When in doubt on a new kind of audio,
don't guess — measure with `--ab`.

```bash
transcribe call.m4a --no-enhance             # raw audio, skip the chain
transcribe call.m4a --denoise                # heavy neural denoise (DeepFilterNet)
transcribe call.m4a --denoise                # heavy neural denoise (DeepFilterNet)
transcribe call.m4a --ab                     # transcribe raw vs enhanced, report the winner
transcribe call.m4a --ab --reference truth.txt   # definitive: real WER vs your transcript
```

**Two ways `--ab` decides the winner:**
- **No reference** → highest duration-weighted Whisper confidence (`avg_logprob`),
  plus low-confidence-% and hallucination flags. A strong *proxy*, not proof.
- **`--reference <human transcript.txt>`** → real **WER**. The only 100%-objective
  measure — use it before trusting `--enhance` on a class of recordings.

`--ab` writes per-variant transcripts (`<name>.raw.txt`, `<name>.enhanced.txt`),
an `<name>.ab-report.txt` scorecard, and an `<name>.ab-diff.txt` (raw vs winner).

The `ENHANCE_CHAIN` (in `audio.py`) is deliberately conservative:
`highpass=80 → afftdn(light) → loudnorm` (EBU R128). Loudness normalization is the
biggest safe win for quiet / far-from-mic recordings.

### Speaker labels (one-time setup)

`--speakers` needs a free HuggingFace token and accepting pyannote's terms:
1. Accept terms at <https://huggingface.co/pyannote/speaker-diarization-3.1>
2. `export HF_TOKEN=hf_...` (or pass `--hf-token`)

## Flags

| Flag | Default | Notes |
|------|---------|-------|
| `--lang` | `auto` | `auto` \| `he` \| `en` |
| `--model` | `turbo` | `turbo` \| `large-v3` \| any HF repo id |
| `--format` | `txt` | `txt` `srt` `vtt` `json` `all` (multiple allowed) |
| `--enhance` / `--no-enhance` | **on** | light ffmpeg speech chain (disable with `--no-enhance`) |
| `--denoise` | off | heavy neural denoise (needs `denoise` extra) |
| `--ab` | off | A/B raw vs enhanced + winner report |
| `--reference` | — | human transcript for real WER in `--ab` |
| `--speakers` | off | diarization (needs `speakers` extra + HF_TOKEN) |
| `--compute-type` | `int8` | `int8` is fast & accurate on this Mac |
| `--output-dir` | input's folder | override output location |

## Notes

- ffmpeg is required (already installed via Homebrew) — used to decode any input
  to 16kHz mono before transcription.
- CTranslate2 runs on CPU on Apple Silicon (no Metal); the M4 Pro still does
  several× realtime with the turbo model.
