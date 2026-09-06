"""Stage 2: the gate. Redaction is best effort; the gate is the decision.

The measured fact this exists for: at its most aggressive setting the stage-1
detector still leaves an annotated identifier in ~10% of documents. A system
that redacts and forwards regardless is a system that leaks one document in ten
and says nothing about it. So Airlock scores what it may have missed and
refuses to forward a document whose residual risk exceeds the operator's
budget; those documents are answered by the local model instead.

Risk statistic
--------------
After redaction, every token that SURVIVED into the outgoing text carries the
detector's own probability that it is an identifier, `p_ident`. Their sum is
the expected number of surviving identifier tokens; under independence
`1 - exp(-sum)` is the probability that at least one survived. Independence is
false -- the tokens of one name are correlated -- so that number is used only
as a monotone statistic and is mapped onto observed leak frequency by isotonic
regression fitted on documents held out of the detector's training.

Every calibrated bin reports a Jeffreys posterior mean and a one-sided 95%
upper bound, and the gate decides on the UPPER bound. A bin with no observed
leak returns a small positive risk, never zero: "we saw none" is not "there
are none".
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------- statistic
def surviving_tokens(text: str, toks, spans):
    """Tokens whose characters are not entirely inside a redacted span."""
    n = len(text)
    mask = bytearray(n)
    for s in spans:
        d = s if isinstance(s, dict) else s.as_dict()
        for i in range(max(0, d["start"]), min(n, d["end"])):
            mask[i] = 1
    out = []
    for (a, b, p_pii, label, p_ident) in toks:
        if not all(mask[a:b]):
            out.append((a, b, p_ident, label))
    return out


def raw_risk(text: str, toks, spans):
    """-> (expected_surviving_identifier_tokens, naive_P_at_least_one)."""
    total = sum(p for (_a, _b, p, _l) in surviving_tokens(text, toks, spans))
    return total, 1.0 - math.exp(-total)


# -------------------------------------------------------------- calibration
def _wilson_upper(successes: float, n: int, z: float = 1.645) -> float:
    """One-sided upper bound, so an unobserved leak is not read as impossible."""
    if n <= 0:
        return 1.0
    p = successes / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n)))
    return min(1.0, (centre + half) / d)


def pava(pairs):
    """Pool-adjacent-violators. pairs = [(x, y in {0,1})] sorted by x ascending."""
    blocks = []  # [x_max, sum_y, count]
    for x, y in pairs:
        blocks.append([x, float(y), 1])
        while len(blocks) > 1 and (blocks[-2][1] / blocks[-2][2]) >= (blocks[-1][1] / blocks[-1][2]):
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([b[0], a[1] + b[1], a[2] + b[2]])
    return blocks


@dataclass
class Calibrator:
    """Isotonic map from the raw statistic to a leak probability."""

    x: list            # right edge of each block
    p_hat: list        # Jeffreys posterior mean
    p_upper: list      # one-sided 95% upper bound
    n: list            # block sizes
    fitted_on: str = ""
    fitted_docs: int = 0
    observed_leak_rate: float = 0.0

    @classmethod
    def fit(cls, raw_scores, leaked, fitted_on: str = "", min_block: int = 25):
        pairs = sorted(zip(raw_scores, [1 if l else 0 for l in leaked]))
        blocks = pava(pairs)
        # merge forward until every block has enough observations to bound
        merged = []
        for b in blocks:
            if merged and merged[-1][2] < min_block:
                prev = merged.pop()
                merged.append([b[0], prev[1] + b[1], prev[2] + b[2]])
            else:
                merged.append(list(b))
        while len(merged) > 1 and merged[-1][2] < min_block:
            b = merged.pop()
            prev = merged.pop()
            merged.append([b[0], prev[1] + b[1], prev[2] + b[2]])
        xs = [b[0] for b in merged]
        hat = [(b[1] + 0.5) / (b[2] + 1.0) for b in merged]          # Jeffreys
        up = [_wilson_upper(b[1], int(b[2])) for b in merged]
        # isotonic is destroyed by the merge only if a merge inverted an order;
        # enforce monotonicity explicitly so the gate's threshold is a threshold
        for i in range(1, len(hat)):
            hat[i] = max(hat[i], hat[i - 1])
            up[i] = max(up[i], up[i - 1])
        return cls(x=xs, p_hat=hat, p_upper=up, n=[int(b[2]) for b in merged],
                   fitted_on=fitted_on, fitted_docs=len(pairs),
                   observed_leak_rate=sum(y for _, y in pairs) / max(1, len(pairs)))

    def predict(self, raw: float, conservative: bool = True) -> float:
        table = self.p_upper if conservative else self.p_hat
        i = bisect.bisect_left(self.x, raw)
        if i >= len(table):
            i = len(table) - 1
        return table[i]

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


# --------------------------------------------------------------- the gate
@dataclass
class Decision:
    forward: bool
    risk: float
    risk_naive: float
    expected_surviving_identifier_tokens: float
    budget: float
    route: str
    reason: str


class Gate:
    def __init__(self, detector, calibrator: Calibrator | None = None,
                 threshold: float = 0.2, budget: float = 0.12,
                 conservative: bool = True):
        self.detector = detector
        self.calibrator = calibrator
        self.threshold = threshold
        self.budget = budget
        self.conservative = conservative

    def process(self, text: str):
        from .redact import redact

        toks = self.detector._token_scores([text])[0]
        spans = self.detector._decode(text, toks, self.threshold)
        redacted, vault = redact(text, spans)
        expected, naive = raw_risk(text, toks, spans)
        risk = self.calibrator.predict(expected, self.conservative) if self.calibrator else naive
        forward = risk <= self.budget
        return {
            "spans": [s.as_dict() for s in spans],
            "redacted": redacted,
            "vault": vault,
            "decision": Decision(
                forward=forward,
                risk=round(risk, 5),
                risk_naive=round(naive, 5),
                expected_surviving_identifier_tokens=round(expected, 4),
                budget=self.budget,
                route="hosted_model" if forward else "local_model",
                reason=("residual leak risk %.3f is within the %.3f budget" % (risk, self.budget))
                if forward else
                ("residual leak risk %.3f exceeds the %.3f budget -- answering locally"
                 % (risk, self.budget)),
            ),
        }
