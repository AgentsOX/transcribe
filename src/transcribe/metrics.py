"""Quality metrics for A/B testing transcription variants.

Two ways to judge which variant is better:

1. Confidence proxy (no reference needed) — derived from Whisper's own
   per-segment signals. Strong indicator, but NOT ground truth.
2. WER against a human reference (`--reference`) — the only 100%-objective
   measure. Needs the `eval` extra (jiwer).
"""

from __future__ import annotations

import re

# avg_logprob below this = the model was unsure on that segment.
LOW_CONF = -0.8
# compression_ratio above this often means repetition / hallucination.
HALLUCINATION_CR = 2.4


def confidence_report(segs) -> dict:
    """Aggregate Whisper confidence signals into one comparable scorecard."""
    if not segs:
        return {"weighted_logprob": -99.0, "low_conf_pct": 100.0,
                "hallucination_segs": 0, "segments": 0, "words": 0,
                "mean_compression": 0.0}

    total_dur = sum(s.end - s.start for s in segs) or 1.0
    # duration-weighted mean log-prob — higher (closer to 0) is better.
    wlogprob = sum(s.avg_logprob * (s.end - s.start) for s in segs) / total_dur
    low = sum(1 for s in segs if s.avg_logprob < LOW_CONF)
    halluc = sum(1 for s in segs
                 if s.compression_ratio > HALLUCINATION_CR or s.avg_logprob < -1.0)
    words = sum(len(s.text.split()) for s in segs)
    mean_cr = sum(s.compression_ratio for s in segs) / len(segs)

    return {
        "weighted_logprob": round(wlogprob, 4),
        "low_conf_pct": round(100 * low / len(segs), 1),
        "hallucination_segs": halluc,
        "segments": len(segs),
        "words": words,
        "mean_compression": round(mean_cr, 3),
    }


def _normalize(text: str) -> str:
    """Light normalization so WER measures words, not punctuation/spacing.

    Works for Hebrew and English: strip punctuation, collapse whitespace.
    """
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def wer_against(reference_text: str, hypothesis_text: str) -> float:
    """Real Word Error Rate vs a human reference (lower is better). Needs jiwer."""
    import jiwer

    ref, hyp = _normalize(reference_text), _normalize(hypothesis_text)
    return round(jiwer.wer(ref, hyp), 4)


def pick_winner(variants: list[dict], has_reference: bool) -> str:
    """Return the name of the best variant.

    With a reference → lowest WER (definitive). Otherwise → highest
    duration-weighted confidence (proxy).
    """
    if has_reference:
        return min(variants, key=lambda v: v["wer"])["name"]
    return max(variants, key=lambda v: v["report"]["weighted_logprob"])["name"]
