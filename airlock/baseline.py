"""Regex-and-heuristics detector -- the control the model has to beat.

This is what a careful team writes in an afternoon without any ML: patterns for
the identifier shapes that have one, plus a capitalised-bigram guess at names.
It exposes the same `find`/`find_batch` interface as the model detector so the
evaluation harness scores them through identical code.
"""
from __future__ import annotations

import re

from .detect import Span

PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("ipv6", re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("swift_bic_code", re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("phone_number", re.compile(r"(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3,4}[ .-]\d{3,4}(?:[ .-]\d{2,4})?")),
    ("date_of_birth", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("local_latlng", re.compile(r"-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}")),
    ("api_key", re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")),
    ("user_name", re.compile(r"(?<![\w@])@[A-Za-z0-9_]{3,}\b")),
    # a capitalised bigram is the classic no-ML guess at a person's name
    ("name", re.compile(r"\b[A-Z][a-z]{1,15}(?:[ -][A-Z][a-z]{1,15}){1,2}\b")),
]


class RegexBaseline:
    name = "regex_baseline"

    def find(self, text: str, threshold: float = 0.5):
        hits = []
        for label, rx in PATTERNS:
            for m in rx.finditer(text):
                if m.end() > m.start():
                    hits.append((m.start(), m.end(), label))
        hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        out, last_end = [], -1
        for a, b, lab in hits:
            if a >= last_end:
                out.append(Span(start=a, end=b, label=lab, score=1.0, text=text[a:b]))
                last_end = b
        return out

    def find_batch(self, texts, threshold: float = 0.5, batch_size: int = 8):
        return [self.find(t, threshold) for t in texts]
