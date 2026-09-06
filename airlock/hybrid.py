"""The detector Airlock actually ships: the model, unioned with the validators.

Neither half is sufficient. The model finds the identifiers that have no fixed
shape and the validators find the ones that carry a checksum, and each covers a
class the other measurably misses. Overlaps are merged into one span over the
union of their extents so nothing is left half-masked, and a proven identifier
keeps its own label.
"""
from __future__ import annotations

from .detect import Detector, Span
from .validators import ChecksumValidators


def merge_spans(spans: list, priority: set | None = None) -> list:
    """Union overlapping spans. `priority` labels win the type of the merged
    span; where neither is priority the longer contributor wins."""
    if not spans:
        return []
    priority = priority or set()
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out = [ordered[0]]
    for s in ordered[1:]:
        last = out[-1]
        if s.start < last.end:  # overlap
            take = s if (s.label in priority and last.label not in priority) or \
                (s.label in priority) == (last.label in priority) and \
                (s.end - s.start) > (last.end - last.start) else last
            out[-1] = Span(start=min(last.start, s.start), end=max(last.end, s.end),
                           label=take.label, score=max(last.score, s.score))
        else:
            out.append(s)
    return out


class HybridDetector:
    """Same surface as `Detector`, so the gate and the harness are unchanged."""

    def __init__(self, path: str = "models/detector", **kw):
        self.model = Detector(path, **kw)
        self.validators = ChecksumValidators()
        self.device = self.model.device
        self.path = path

    def _token_scores(self, texts):
        return self.model._token_scores(texts)

    def _snap(self, text, a, b):
        return self.model._snap(text, a, b)

    def _decode(self, text: str, toks, threshold: float, snap: bool = True):
        model_spans = self.model._decode(text, toks, threshold, snap)
        proven = self.validators.find(text)
        merged = merge_spans(model_spans + proven, priority={s.label for s in proven})
        for s in merged:
            s.text = text[s.start:s.end]
        return merged

    def find(self, text: str, threshold: float = 0.2):
        return self._decode(text, self._token_scores([text])[0], threshold)

    def find_batch(self, texts, threshold: float = 0.2, batch_size: int = 8):
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            for t, toks in zip(chunk, self._token_scores(chunk)):
                out.append(self._decode(t, toks, threshold))
        return out
