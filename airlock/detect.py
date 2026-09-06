"""Stage 1: the identifier detector.

Turns raw text into character spans with a per-span confidence. Long documents
are processed as overlapping windows and the windows are merged in character
space, so a span that straddles a window boundary is not cut in half.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .labels import ID2LABEL, IDENTIFYING, LABELS, type_of

# column indices of the identifying labels, so a per-token probability of
# "this is an identifier we are contractually obliged to remove" can be read
# off directly -- company/date/time carry probability mass that is not a leak.
IDENT_COLS = [i for i, l in enumerate(LABELS) if l != "O" and type_of(l) in set(IDENTIFYING)]


@dataclass
class Span:
    start: int
    end: int
    label: str
    score: float
    text: str = ""

    def as_dict(self):
        return asdict(self)


class Detector:
    """Token-classification detector over char spans."""

    def __init__(self, path: str = "models/detector", device: str | None = None,
                 max_length: int = 384, stride: int = 96):
        self.path = path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device).eval()
        self.max_length = max_length
        self.stride = stride

    # -- internals -------------------------------------------------------
    def _token_scores(self, texts: list[str]):
        """-> list (per text) of [(start, end, p_pii, label_str, p_identifying)].

        Windows are merged in character space keeping, for each token, the
        window that scored it highest."""
        enc = self.tok(
            texts, return_offsets_mapping=True, return_overflowing_tokens=True,
            truncation=True, max_length=self.max_length, stride=self.stride,
            padding=True, return_tensors="pt",
        )
        mapping = enc["overflow_to_sample_mapping"].tolist()
        offsets = enc["offset_mapping"]
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            ).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu()
        p_pii = 1.0 - probs[:, :, 0]
        p_ident = probs[:, :, IDENT_COLS].sum(dim=-1)
        # best non-O label
        best = probs[:, :, 1:].argmax(dim=-1) + 1

        per_text: list[dict] = [dict() for _ in texts]
        for w, sample in enumerate(mapping):
            acc = per_text[sample]
            for t in range(offsets.shape[1]):
                a, b = int(offsets[w, t, 0]), int(offsets[w, t, 1])
                if a == b:
                    continue
                score = float(p_pii[w, t])
                key = (a, b)
                if key not in acc or score > acc[key][0]:
                    acc[key] = (score, ID2LABEL[int(best[w, t])], float(p_ident[w, t]))
        out = []
        for acc in per_text:
            toks = [(a, b, s, l, pi) for (a, b), (s, l, pi) in acc.items()]
            toks.sort(key=lambda x: (x[0], x[1]))
            out.append(toks)
        return out

    # characters a word can contain. A mask that stops mid-word leaves the
    # rest of the word on screen -- `[STREET_ADDRESS_1]npur` was a real output
    # of this decoder before snapping was added -- and by this project's own
    # definition a partially masked identifier is a leaked one. Spans are
    # therefore only ever widened to word boundaries, never narrowed.
    WORD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") | set(".@_-+/#'&")

    @classmethod
    def _snap(cls, text: str, a: int, b: int) -> tuple:
        while a > 0 and text[a - 1] in cls.WORD_CHARS:
            a -= 1
        while b < len(text) and text[b] in cls.WORD_CHARS:
            b += 1
        # a widened span must not end on trailing sentence punctuation
        while b > a and text[b - 1] in ".'-":
            b -= 1
        return a, b

    @staticmethod
    def _decode(text: str, toks, threshold: float, snap: bool = True) -> list[Span]:
        spans: list[Span] = []
        cur = None  # [start, end, type, [scores]]
        for (a, b, score, label, _p_ident) in toks:
            typ = type_of(label)
            is_pii = score >= threshold
            if not is_pii:
                cur = None if cur is None else spans_flush(spans, cur, text)
                continue
            starts_new = label.startswith("B-")
            if cur is not None and cur[2] == typ and not starts_new and \
                    text[cur[1]:a].strip() == "":
                cur[1] = b
                cur[3].append(score)
            else:
                if cur is not None:
                    spans_flush(spans, cur, text)
                cur = [a, b, typ, [score]]
        if cur is not None:
            spans_flush(spans, cur, text)
        if snap:
            for sp in spans:
                sp.start, sp.end = Detector._snap(text, sp.start, sp.end)
                sp.text = text[sp.start:sp.end]
        return spans

    # -- public ----------------------------------------------------------
    def find(self, text: str, threshold: float = 0.5) -> list[Span]:
        return self.find_batch([text], threshold)[0]

    def find_batch(self, texts: list[str], threshold: float = 0.5, batch_size: int = 8):
        results: list[list[Span]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            for text, toks in zip(chunk, self._token_scores(chunk)):
                results.append(self._decode(text, toks, threshold))
        return results


def spans_flush(spans: list, cur, text: str):
    spans.append(Span(start=cur[0], end=cur[1], label=cur[2],
                      score=round(min(cur[3]), 4), text=text[cur[0]:cur[1]]))
    return None
