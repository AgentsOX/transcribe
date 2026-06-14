"""Self-contained HTML A/B viewer: before vs after, word-diff highlighted,
with the full audio embedded so you can listen while you read."""

from __future__ import annotations

import base64
import difflib
import html
import mimetypes
from pathlib import Path


def _diff_spans(raw_words: list[str], enh_words: list[str]):
    """Word-level diff → (left_spans, right_spans, counts).

    Tags: eq (same), del (only on left/raw), ins (only on right/enhanced),
    rep (changed — shown highlighted on both sides).
    """
    sm = difflib.SequenceMatcher(a=raw_words, b=enh_words, autojunk=False)
    left, right = [], []
    counts = {"del": 0, "ins": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            left += [("eq", w) for w in raw_words[i1:i2]]
            right += [("eq", w) for w in enh_words[j1:j2]]
        else:  # delete / insert / replace -> red on left, green on right
            left += [("del", w) for w in raw_words[i1:i2]]
            right += [("ins", w) for w in enh_words[j1:j2]]
            counts["del"] += i2 - i1
            counts["ins"] += j2 - j1
    return left, right, counts


def _render(spans) -> str:
    out = []
    for tag, w in spans:
        esc = html.escape(w)
        cls = "w" if tag == "eq" else f"w {tag}"
        out.append(f'<span class="{cls}">{esc}</span>')
    return " ".join(out)


# Force browser-playable MIME types. Python guesses .m4a as "audio/mp4a-latm",
# which browsers refuse — map AAC/M4A containers to audio/mp4 explicitly.
_AUDIO_MIME = {
    ".m4a": "audio/mp4", ".aac": "audio/mp4", ".mp4": "audio/mp4",
    ".m4v": "audio/mp4", ".mov": "audio/mp4",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac",
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".webm": "audio/webm",
}


def _audio_data_uri(audio: Path) -> str:
    mime = _AUDIO_MIME.get(audio.suffix.lower()) \
        or mimetypes.guess_type(str(audio))[0] or "audio/mpeg"
    b64 = base64.b64encode(audio.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_html(audio: Path, raw_text: str, enh_text: str, out_path: Path,
               *, left_label: str = "BEFORE (raw)",
               right_label: str = "AFTER (enhanced)",
               winner: str = "", subtitle: str = "") -> Path:
    left, right, counts = _diff_spans(raw_text.split(), enh_text.split())
    audio_uri = _audio_data_uri(audio)
    changed = counts["del"] + counts["ins"]

    doc = f"""<!doctype html>
<html lang="he">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B transcript — {html.escape(audio.name)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font:16px/1.7 -apple-system,Segoe UI,Roboto,Arial,sans-serif;
         background:#0f1115; color:#e7e9ee; }}
  header {{ position:sticky; top:0; z-index:10; background:#171a21;
           border-bottom:1px solid #2a2f3a; padding:14px 20px;
           box-shadow:0 2px 12px rgba(0,0,0,.35); }}
  h1 {{ margin:0 0 4px; font-size:18px; }}
  .sub {{ color:#9aa3b2; font-size:13px; }}
  .winner {{ color:#7ee787; font-weight:600; }}
  .controls {{ display:flex; align-items:center; gap:14px; margin-top:10px;
              flex-wrap:wrap; }}
  button.play {{ background:#2f81f7; color:#fff; border:0; border-radius:8px;
                padding:10px 18px; font-size:15px; font-weight:600;
                cursor:pointer; }}
  button.play:hover {{ background:#1f6fe5; }}
  audio {{ height:34px; vertical-align:middle; }}
  .legend {{ font-size:13px; color:#9aa3b2; display:flex; gap:16px;
            flex-wrap:wrap; }}
  .legend b {{ font-weight:600; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:0; }}
  .pane {{ padding:22px 26px; min-height:60vh; }}
  .pane:first-child {{ border-inline-end:1px solid #2a2f3a; }}
  .pane h2 {{ position:sticky; top:0; margin:0 0 14px; font-size:14px;
             letter-spacing:.04em; text-transform:uppercase; color:#9aa3b2; }}
  .text {{ }}
  .w {{ padding:1px 2px; border-radius:3px; }}
  .del {{ background:rgba(248,81,73,.22); color:#ffb3ae;
         text-decoration:line-through; }}
  .ins {{ background:rgba(63,185,80,.22); color:#9ff0a8; }}
  .rep {{ background:rgba(210,153,34,.28); color:#ffd479; }}
  @media (max-width:760px) {{ .cols {{ grid-template-columns:1fr; }}
    .pane:first-child {{ border-inline-end:0;
      border-bottom:1px solid #2a2f3a; }} }}
</style>
</head>
<body>
<header>
  <h1>A/B transcript — {html.escape(audio.name)}</h1>
  <div class="sub">{html.escape(subtitle)}
    {f'· <span class="winner">winner: {html.escape(winner)}</span>' if winner else ''}
    · {changed} words differ</div>
  <div class="controls">
    <button class="play" id="playBtn">▶ Play full audio</button>
    <audio id="aud" controls preload="metadata" src="{audio_uri}"></audio>
    <div class="legend">
      <span><b style="color:#ffb3ae">■</b> only in BEFORE (raw)</span>
      <span><b style="color:#9ff0a8">■</b> only in AFTER (enhanced)</span>
    </div>
  </div>
</header>
<div class="cols">
  <div class="pane">
    <h2>{html.escape(left_label)}</h2>
    <div class="text" dir="auto">{_render(left)}</div>
  </div>
  <div class="pane">
    <h2>{html.escape(right_label)}</h2>
    <div class="text" dir="auto">{_render(right)}</div>
  </div>
</div>
<script>
  const btn = document.getElementById('playBtn');
  const aud = document.getElementById('aud');
  btn.addEventListener('click', () => {{
    if (aud.paused) {{ aud.play(); btn.textContent = '⏸ Pause'; }}
    else {{ aud.pause(); btn.textContent = '▶ Play full audio'; }}
  }});
  aud.addEventListener('ended', () => btn.textContent = '▶ Play full audio');
</script>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
