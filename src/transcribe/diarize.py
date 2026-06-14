"""Classic, fully-local speaker diarization — no token, no gated terms.

How it works (the textbook approach):
  1. take each transcribed segment's audio,
  2. turn it into a speaker embedding with a public ECAPA-TDNN model
     (SpeechBrain's spkrec-ecapa-voxceleb — open, no login),
  3. cluster the embeddings (agglomerative, cosine) so segments from the same
     voice land in the same group → SPEAKER_00, SPEAKER_01, …

The model file is downloaded once on first use and cached under
~/.cache/transcribe/ecapa; after that it runs 100% offline. Speaker embeddings
are acoustic, so this is language-agnostic (Hebrew & English alike).
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 16000
_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
_CACHE = Path.home() / ".cache" / "transcribe" / "ecapa"

_encoder = None  # lazily loaded singleton


def _load_wav(path: Path) -> np.ndarray:
    """Read our 16kHz mono PCM wav into a float32 [-1, 1] array."""
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _get_encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        _CACHE.mkdir(parents=True, exist_ok=True)
        _encoder = EncoderClassifier.from_hparams(
            source=_MODEL, savedir=str(_CACHE), run_opts={"device": "cpu"})
    return _encoder


def _embed(encoder, clip: np.ndarray) -> np.ndarray:
    import torch
    sig = torch.from_numpy(clip).unsqueeze(0)          # [1, time]
    with torch.no_grad():
        emb = encoder.encode_batch(sig).squeeze().cpu().numpy()
    return emb                                          # 192-d


def assign_speakers(wav_path: Path, segments, *, num_speakers: int = 0,
                    threshold: float = 0.55) -> int:
    """Tag each segment with SPEAKER_xx (in place). Returns #speakers found.

    num_speakers > 0 forces that many speakers (most reliable when you know it);
    otherwise the count is inferred via a cosine-distance threshold.
    """
    from sklearn.cluster import AgglomerativeClustering

    audio = _load_wav(wav_path)
    encoder = _get_encoder()

    embs, idx = [], []
    for i, seg in enumerate(segments):
        a, b = int(seg.start * SR), int(seg.end * SR)
        if b - a < SR:                      # widen to ~1s for a stable embedding
            mid = (a + b) // 2
            a, b = max(0, mid - SR // 2), min(len(audio), mid + SR // 2)
        clip = audio[a:b]
        if len(clip) < int(0.3 * SR):
            continue
        try:
            embs.append(_embed(encoder, clip))
            idx.append(i)
        except Exception:
            continue

    if not embs:
        for seg in segments:
            seg.speaker = "SPEAKER_00"
        return 1

    X = np.vstack(embs)
    if len(embs) == 1:
        labels = np.zeros(1, dtype=int)
    elif num_speakers and num_speakers > 0:
        labels = AgglomerativeClustering(
            n_clusters=num_speakers, metric="cosine",
            linkage="average").fit_predict(X)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average").fit_predict(X)

    # stable SPEAKER_xx names by order of first appearance
    remap: dict[int, str] = {}
    for lab in labels:
        remap.setdefault(int(lab), f"SPEAKER_{len(remap):02d}")
    seg_label = {idx[k]: remap[int(labels[k])] for k in range(len(idx))}

    last = seg_label.get(idx[0], "SPEAKER_00")
    for i, seg in enumerate(segments):
        if i in seg_label:
            last = seg_label[i]
        seg.speaker = last          # carry label across short/skipped segments
    return len(remap)
