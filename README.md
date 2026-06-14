# transcribe

Turn audio/video into clean text — **locally**, tuned for **Hebrew & English**.

Powered by [ivrit.ai](https://huggingface.co/ivrit-ai)'s Hebrew-tuned Whisper
models on [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Your
recordings never leave the machine.

## Install (global CLI)

Install once, then run `transcribe` in **any folder / any repo**. Needs
[`uv`](https://docs.astral.sh/uv/) and `ffmpeg` (`brew install ffmpeg`).

```bash
uv tool install "git+https://github.com/AgentsOX/transcribe.git"
```

That drops a `transcribe` command on your PATH. To include speaker labels:

```bash
uv tool install "transcribe[speakers] @ git+https://github.com/AgentsOX/transcribe.git"
```

- **Upgrade:** `uv tool upgrade transcribe`
- **Uninstall:** `uv tool uninstall transcribe`
- No `uv`? Get it with `curl -LsSf https://astral.sh/uv/install.sh | sh`
  (or use `pipx install "git+https://github.com/AgentsOX/transcribe.git"`).

## Quick start

```bash
cd ~/any-repo
transcribe meeting.m4a        # → meeting.txt right next to your audio
```

That's it. The first run downloads the model (~1.6GB, cached after). No API keys,
no signup, no cloud.

## Common uses

```bash
transcribe call.m4a --format srt          # subtitles
transcribe call.m4a --format all          # txt + srt + vtt + json
transcribe ./recordings                   # batch a whole folder
transcribe call.m4a --model large-v3      # max accuracy (slower)
transcribe call.m4a --lang he             # force Hebrew (default: auto)
```

### Speaker labels — who said what

```bash
transcribe call.m4a --speakers --num-speakers 2   # needs the [speakers] install
```

Classic, fully local, **no token / no signup**. Each line gets a `SPEAKER_xx`
tag. Pass `--num-speakers` when you know the count, or omit it to auto-detect.

### Not sure if cleanup helps? A/B it.

`--enhance` (light audio cleanup) is **on by default**. To prove it helps on your
kind of audio:

```bash
transcribe call.m4a --ab        # transcribes raw vs enhanced, picks a winner
```

This also writes an **HTML viewer** (`call.ab.html`) showing the two side by side
with word-level diff highlighting and a button to play the audio while you read.

## Output

Files are written next to each input (override with `--output-dir`):

- `.txt` — clean readable transcript
- `.srt` / `.vtt` — timestamped subtitles
- `.json` — segments with timestamps, confidence, and speaker

## Develop from source

```bash
git clone https://github.com/AgentsOX/transcribe.git
cd transcribe
uv sync                       # base env
uv run transcribe call.m4a    # run without a global install
```

## More

See [`AGENTS.md`](AGENTS.md) for the full flag reference, the optional extras
(`speakers`, `denoise`, `eval`), and how the pieces fit together.
