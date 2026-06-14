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
- **SpeechBrain ECAPA-TDNN** voice embeddings + agglomerative clustering for
  optional speaker labels (`--speakers`). Classic diarization: embed each
  segment, cluster the voices. Uses a **public** model — no HuggingFace token, no
  gated terms. Lives behind the `speakers` extra (pulls in PyTorch).

> We deliberately did **not** use WhisperX (its word-level alignment has no Hebrew
> model) or pyannote (gated models + token = bad onboarding). The diarizer is a
> plain embed-then-cluster pipeline anyone can run with one `uv sync`.

## Setup

```bash
cd apps/transcribe
uv sync                     # base: transcription + txt/srt/vtt/json + --enhance + --ab
uv sync --extra eval        # adds jiwer for real WER (--ab --reference)
uv sync --extra speakers    # adds SpeechBrain + torch for --speakers (no token)
uv sync --extra denoise     # adds DeepFilterNet for --denoise
```

First run downloads the model (cached in ~/.cache/huggingface afterwards).

## Usage

```bash
uv run transcribe call.m4a                          # → call.txt (auto HE/EN)
uv run transcribe call.m4a --lang he --format srt    # Hebrew subtitles
uv run transcribe ./recordings --format all          # batch a folder, all formats
uv run transcribe call.m4a --model large-v3          # max accuracy
uv run transcribe call.m4a --speakers --num-speakers 2   # who-said-what (local, no token)
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

### Speaker labels (`--speakers`)

Classic, fully local, **no token / no signup**:

```bash
uv sync --extra speakers
transcribe call.m4a --speakers --num-speakers 2   # pass the count when you know it
transcribe call.m4a --speakers                     # or let it auto-detect
```

Each segment gets a `SPEAKER_xx` tag (shown in txt/srt/vtt and the json). The
ECAPA model is downloaded once (~80MB, cached in `~/.cache/transcribe/ecapa`),
then it runs offline. Passing `--num-speakers` is the most reliable option —
auto-detection guesses the count from voice similarity.

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
| `--speakers` | off | diarization, local & token-free (needs `speakers` extra) |
| `--num-speakers` | `0` (auto) | force the speaker count, e.g. `2` for a call |
| `--compute-type` | `int8` | `int8` is fast & accurate on this Mac |
| `--output-dir` | input's folder | override output location |

## Notes

- ffmpeg is required (already installed via Homebrew) — used to decode any input
  to 16kHz mono before transcription.
- CTranslate2 runs on CPU on Apple Silicon (no Metal); the M4 Pro still does
  several× realtime with the turbo model.
