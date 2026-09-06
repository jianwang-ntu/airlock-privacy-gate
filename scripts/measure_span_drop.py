#!/usr/bin/env python3
"""How many annotated identifiers the loader discards, and whether that hides a leak.

`airlock.data.load_split` keeps one span per overlapping group -- BIO tagging
cannot put two labels on one token -- so the gold set every number in this
repository is scored against is smaller than the set the corpus ships. That
makes the reported denominator ours, not the corpus's, and a denominator you
chose yourself is the kind of thing a reader should be handed rather than left
to find.

This script measures the gap against the SHIPPED loader: it calls `load_split`
for the kept set instead of re-implementing the rule, so if the rule changes
the number here changes with it. It then asks the question that decides how
much the gap matters -- of the discarded identifiers, how many name a stretch
of text that no kept span covers? A discarded span always overlaps a kept one,
so "missing from the denominator" and "able to leak unseen" are different
claims, and coverage is checked against the union of kept spans rather than
against the rule that produced them.

Writes evidence/loader_span_drop.json. Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airlock.data import DATA_DIR, load_split  # noqa: E402
from airlock.labels import IDENTIFYING, NON_IDENTIFYING  # noqa: E402

IDENT = set(IDENTIFYING)
SHIPPED_THRESHOLD = "0.2"

checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def uncovered_chars(span, kept):
    """Chars of `span` that no span in `kept` covers. 0 => wholly inside the kept set."""
    todo = [(span["start"], span["end"])]
    for k in kept:
        nxt = []
        for a, b in todo:
            if k["end"] <= a or k["start"] >= b:
                nxt.append((a, b))
                continue
            if a < k["start"]:
                nxt.append((a, k["start"]))
            if k["end"] < b:
                nxt.append((k["end"], b))
        todo = nxt
    return sum(b - a for a, b in todo)


def main():
    path = os.path.join(DATA_DIR, "english_test.jsonl")
    if not os.path.exists(path):
        sys.stderr.write(f"{path} missing -- run: python3 scripts/prepare_data.py\n")
        return 2

    # The kept set comes from the shipped loader, not from a copy of its rule.
    kept_docs = load_split("test")
    raw_rows = [json.loads(l) for l in open(path)]
    check("loader and corpus agree on the document count",
          len(kept_docs) == len(raw_rows), f"{len(kept_docs)} vs {len(raw_rows)}")

    raw_ident = kept_ident = dropped = dropped_covered = 0
    dropped_partial = residual = 0
    by_label = {}

    for row, doc in zip(raw_rows, kept_docs):
        kept = [s for s in doc.spans]
        kept_i = [s for s in kept if s["label"] in IDENT]
        kept_ident += len(kept_i)

        raw = [s for s in row["spans"] if s["end"] > s["start"]]
        raw_i = [s for s in raw if s["label"] in IDENT]
        raw_ident += len(raw_i)

        kept_keys = {(s["start"], s["end"], s["label"]) for s in kept}
        for s in raw_i:
            if (s["start"], s["end"], s["label"]) in kept_keys:
                continue
            dropped += 1
            by_label[s["label"]] = by_label.get(s["label"], 0) + 1
            u = uncovered_chars(s, kept)
            if u == 0:
                dropped_covered += 1
            else:
                dropped_partial += 1
                residual += u

    det = json.load(open(os.path.join(ROOT, "evidence", "detector_eval.json")))
    h = det["by_threshold_hybrid"][SHIPPED_THRESHOLD]
    caught, gold = h["identifiers_caught"], h["gold_identifiers"]

    r_reported = caught / gold
    r_worst = caught / raw_ident
    overstatement_pp = 100 * (r_reported - r_worst)

    print(f"\nidentifying annotations, corpus   {raw_ident}")
    print(f"identifying annotations, scored   {kept_ident}")
    print(f"discarded by the overlap rule     {dropped}")
    print(f"  covered by the kept spans       {dropped_covered}")
    print(f"  not covered                     {dropped_partial}  ({residual} chars)")
    print(f"by label                          {json.dumps(by_label, sort_keys=True)}")
    print(f"\nrecall, reported ({gold})        {r_reported:.6f}")
    print(f"recall, over all {raw_ident} as misses  {r_worst:.6f}")
    print(f"overstatement                     {overstatement_pp:.4f} pp")

    check("the scored set is the one detector_eval.json reports",
          kept_ident == gold, f"{kept_ident} vs {gold}")
    check("the reported recall is the README headline 93.96%",
          round(100 * r_reported, 2) == 93.96, f"{100 * r_reported:.4f}%")
    check("the corpus set is larger than the scored set",
          raw_ident > kept_ident, f"{raw_ident} > {kept_ident}")
    check("the difference is exactly the discarded spans",
          raw_ident - kept_ident == dropped, f"{raw_ident - kept_ident} vs {dropped}")
    check("every discarded span is accounted for as covered or not",
          dropped_covered + dropped_partial == dropped, "")
    check("re-scoring over the corpus set lowers recall",
          r_worst < r_reported, f"{r_worst:.6f} < {r_reported:.6f}")

    # --- controls: uncovered_chars must be able to say both things ------------
    k = [{"start": 0, "end": 10, "label": "name"}]
    check("CONTROL a span inside the kept set reports 0 uncovered chars",
          uncovered_chars({"start": 2, "end": 6}, k) == 0, "")
    check("CONTROL a span extending past it reports exactly the overhang",
          uncovered_chars({"start": 5, "end": 14}, k) == 4, "")
    check("CONTROL a span straddling a gap between two kept spans counts the gap",
          uncovered_chars({"start": 0, "end": 20},
                          [{"start": 0, "end": 5}, {"start": 15, "end": 20}]) == 10, "")
    check("CONTROL a disjoint span is wholly uncovered",
          uncovered_chars({"start": 30, "end": 33}, k) == 3, "")
    check("CONTROL non-identifying labels are outside the count",
          not (NON_IDENTIFYING & IDENT), f"non-identifying: {sorted(NON_IDENTIFYING)}")

    n_ok = sum(1 for c in checks if c["ok"])
    print(f"\n{n_ok}/{len(checks)} checks pass")

    out = {
        "documents": len(raw_rows),
        "identifying_annotations_in_corpus": raw_ident,
        "identifying_annotations_scored": kept_ident,
        "discarded_by_overlap_rule": dropped,
        "discarded_covered_by_kept_spans": dropped_covered,
        "discarded_not_covered": dropped_partial,
        "uncovered_characters": residual,
        "discarded_by_label": by_label,
        "identifiers_caught": caught,
        "recall_reported": r_reported,
        "recall_if_all_discarded_are_misses": r_worst,
        "overstatement_percentage_points": overstatement_pp,
        "shipped_threshold": float(SHIPPED_THRESHOLD),
        "checks_passed": n_ok,
        "checks_total": len(checks),
        "note": (
            "The overlap rule is in airlock/data.py load_split(): BIO tags cannot "
            "express two labels on one token, so of a group of overlapping spans "
            "only the first is kept. This file measures what that costs. Coverage "
            "is computed against the union of kept spans, not against the rule, so "
            "it answers whether the text is still scored rather than whether the "
            "rule fired."
        ),
    }
    with open(os.path.join(ROOT, "evidence", "loader_span_drop.json"), "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
