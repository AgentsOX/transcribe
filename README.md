# transcribe

Turn audio/video into clean text — **locally**, tuned for **Hebrew & English**.

Powered by [ivrit.ai](https://huggingface.co/ivrit-ai)'s Hebrew-tuned Whisper
models on [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Your
recordings never leave the machine.

## Quick start

You need [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` installed.

```bash
git clone https://github.com/AgentsOX/transcribe.git
cd transcribe
uv sync                       # one-time setup
uv run transcribe call.m4a    # → call.txt next to your audio
```

That's it. The first run downloads the model (~1.6GB, cached after). No API keys,
no signup, no cloud.

## Common uses

```bash
uv run transcribe call.m4a --format srt          # subtitles
uv run transcribe call.m4a --format all          # txt + srt + vtt + json
uv run transcribe ./recordings                   # batch a whole folder
uv run transcribe call.m4a --model large-v3      # max accuracy (slower)
uv run transcribe call.m4a --lang he             # force Hebrew (default: auto)
```

### Speaker labels — who said what

```bash
uv sync --extra speakers
uv run transcribe call.m4a --speakers --num-speakers 2
```

Classic, fully local, **no token / no signup**. Each line gets a `SPEAKER_xx`
tag. Pass `--num-speakers` when you know the count, or omit it to auto-detect.

### Not sure if cleanup helps? A/B it.

`--enhance` (light audio cleanup) is **on by default**. To prove it helps on your
kind of audio:

```bash
uv run transcribe call.m4a --ab        # transcribes raw vs enhanced, picks a winner
```

This also writes an **HTML viewer** (`call.ab.html`) showing the two side by side
with word-level diff highlighting and a button to play the audio while you read.

## Output

Files are written next to each input (override with `--output-dir`):

- `.txt` — clean readable transcript
- `.srt` / `.vtt` — timestamped subtitles
- `.json` — segments with timestamps, confidence, and speaker

## More

See [`AGENTS.md`](AGENTS.md) for the full flag reference, the optional extras
(`speakers`, `denoise`, `eval`), and how the pieces fit together.
