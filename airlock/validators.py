"""Deterministic detectors for identifiers that carry their own proof.

The learned detector is good at the fuzzy identifiers -- names, addresses,
employee ids that follow no fixed shape. It is not reliable on the ones that do
have a fixed shape: at threshold 0.2 it left `4111 1111 1111 1111`, the most
widely published test card number in the world, in the outgoing text. That was
found by a control in `tests/run_checks.py`, not by reading the model's metrics.

An identifier with a checksum does not need a model. A Luhn-valid 16-digit run
IS a card number; an IBAN whose mod-97 is 1 IS an account. These validators
fire on that proof, so they are high precision by construction, and Airlock
takes the UNION of the two detectors: the model covers what has no shape, the
validators cover what does.
"""
from __future__ import annotations

import re

from .detect import Span

CARD_RE = re.compile(r"(?<![\d\-])(?:\d[ \-]?){12,18}\d(?![\d\-])")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
IPV6_RE = re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b")
SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")


def luhn_ok(digits: str) -> bool:
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    v = value.replace(" ", "").upper()
    if not (15 <= len(v) <= 34) or not v[:2].isalpha() or not v[2:4].isdigit():
        return False
    rearranged = v[4:] + v[:4]
    total = 0
    for ch in rearranged:
        if ch.isdigit():
            total = (total * 10 + int(ch)) % 97
        elif ch.isalpha():
            total = (total * 100 + (ord(ch) - 55)) % 97
        else:
            return False
    return total == 1


class ChecksumValidators:
    """Same interface as the model detector, so the harness scores them alike."""

    name = "checksum_validators"

    def find(self, text: str, threshold: float = 0.0):
        out = []
        for m in CARD_RE.finditer(text):
            if luhn_ok(re.sub(r"[ \-]", "", m.group())):
                out.append(("credit_card_number", m.start(), m.end()))
        for m in IBAN_RE.finditer(text):
            if iban_ok(m.group()):
                out.append(("iban", m.start(), m.end()))
        for rx, label in ((EMAIL_RE, "email"), (IPV4_RE, "ipv4"),
                          (IPV6_RE, "ipv6"), (SSN_RE, "ssn")):
            for m in rx.finditer(text):
                out.append((label, m.start(), m.end()))
        out.sort(key=lambda x: (x[1], -(x[2] - x[1])))
        spans, last = [], -1
        for label, a, b in out:
            if a >= last:
                spans.append(Span(start=a, end=b, label=label, score=1.0, text=text[a:b]))
                last = b
        return spans

    def find_batch(self, texts, threshold: float = 0.0, batch_size: int = 8):
        return [self.find(t) for t in texts]
