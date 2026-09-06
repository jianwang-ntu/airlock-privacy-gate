"""Consistent pseudonymisation, the vault, and re-hydration.

Every distinct identifier surface becomes a stable typed placeholder --
`[NAME_1]`, `[IBAN_2]` -- so the text that leaves the machine still reads as a
document and a hosted model can still reason over it ("send NAME_1 the invoice
for IBAN_2"). The mapping stays on the local machine; when the answer comes
back, `rehydrate` puts the real values in. The hosted model never sees them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .labels import IDENTIFYING


@dataclass
class Vault:
    """placeholder -> original surface. Never leaves the local process."""

    mapping: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.mapping, ensure_ascii=False)

    @property
    def size(self) -> int:
        return len(self.mapping)


def _placeholder(label: str, n: int) -> str:
    return f"[{label.upper()}_{n}]"


def redact(text: str, spans, redact_types=None):
    """-> (redacted_text, Vault). `spans` is any iterable of objects or dicts
    carrying start/end/label. Overlapping spans are resolved by taking the
    longest; nothing is left half-masked."""
    redact_types = set(IDENTIFYING if redact_types is None else redact_types)
    items = []
    for s in spans:
        d = s if isinstance(s, dict) else s.as_dict()
        if d["label"] in redact_types:
            items.append((d["start"], d["end"], d["label"]))
    items.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    chosen, last_end = [], -1
    for a, b, lab in items:
        if a >= last_end:
            chosen.append((a, b, lab))
            last_end = b
        elif b > last_end:  # overlap: extend the mask rather than leave a tail
            a0, _, lab0 = chosen[-1]
            chosen[-1] = (a0, b, lab0)
            last_end = b

    vault, seen, counters, out, cursor = Vault(), {}, {}, [], 0
    for a, b, lab in chosen:
        surface = text[a:b]
        key = (lab, surface.strip().casefold())
        if key not in seen:
            counters[lab] = counters.get(lab, 0) + 1
            ph = _placeholder(lab, counters[lab])
            seen[key] = ph
            vault.mapping[ph] = surface
        out.append(text[cursor:a])
        out.append(seen[key])
        cursor = b
    out.append(text[cursor:])
    return "".join(out), vault


def rehydrate(text: str, vault: Vault) -> str:
    """Put the real values back into a hosted model's answer, locally."""
    for ph, original in sorted(vault.mapping.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(ph, original)
    return text


PLACEHOLDER_RE = re.compile(r"\[([A-Z_]+)_(\d+)\]")


def placeholders_in(text: str):
    return PLACEHOLDER_RE.findall(text)


def rehydrate_checked(text: str, vault: Vault):
    """Re-hydrate, and say plainly when the answer cannot be trusted.

    A hosted model does not have to give the placeholders back. Measured here
    on Qwen2.5-0.5B-Instruct: asked to preserve `[IBAN_1]` it answered with an
    invented account number instead. Silently returning that text would hand a
    reader a fabricated identifier wearing a real one's clothes, so the failure
    is reported rather than absorbed: `intact` lists the placeholders that came
    back, `missing` the ones that did not, and `unknown` any placeholder-shaped
    token the vault has never heard of.
    """
    found = {f"[{a}_{b}]" for a, b in placeholders_in(text)}
    expected = set(vault.mapping)
    missing = sorted(expected - found)
    unknown = sorted(found - expected)
    report = {
        "expected": len(expected),
        "intact": sorted(expected & found),
        "missing": missing,
        "unknown_placeholders": unknown,
        "trustworthy": not missing and not unknown,
    }
    return rehydrate(text, vault), report
