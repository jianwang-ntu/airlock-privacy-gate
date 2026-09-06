"""Measure whether Airlock protects some people's identifiers less than others.

Rule 8 of the UPAI-Hackdays rules asks teams to "consider privacy, security,
fairness, and responsible use of AI". Fairness is the bullet with an operand
this project can actually measure, so it is measured here rather than asserted.

The question
------------
A redaction gate that catches 94% of identifiers is not thereby fair. If the
6% it misses is concentrated on one kind of person, then the people in that
group are the ones who get exposed, and the headline number hides it. So:

    conditional on a gold identifier being present, does the probability that
    Airlock redacts it depend on properties of the person it belongs to?

Pre-registered contrasts, fixed before any number below was read:

    C1  person-name types            vs  every other identifying type
    C2  last_name                    vs  name                (within person names)
    C3  surface unseen in training   vs  seen                (person names)
    C4  contains a non-ASCII char    vs  pure ASCII          (person names)
    C5  C3 restricted to single-token person names   (length-confound control)
    C6  C4 restricted to single-token person names   (length-confound control)

One further contrast is POST-HOC and labelled as such wherever it appears:

    C7  last_name vs first_name, single-token only

C7 was added after C2 and the token-count table had been read, because C2's
arms differ in length as well as in type and C2 alone cannot separate the two.
It is carried in the same Holm family as the other six, so adding it costs all
seven contrasts power rather than buying C7 a free pass, and its p-value should
be read as the weakest of the seven.

C5 and C6 exist because span length is the obvious confound: `last_name` spans
are one token and `name` spans are two or three, so any raw gap between them
could be a gap in span length wearing a fairness costume. The token-count table
is reported in full for the same reason.

Holm-Bonferroni is applied across the six contrasts. Every rate carries a
Wilson 95% interval, because the smallest cell here has 78 spans in it and a
point estimate at that n is not a finding.

What this script does not export
--------------------------------
No identifier surface string is written to the evidence file, printed, or
returned. Buckets are computed from the strings inside this process and only
counts leave it. The corpus is synthetic (gretelai/synthetic_pii_finance_-
multilingual, Apache-2.0) so no real person is involved either way, but a
fairness audit of a privacy tool should not itself be the leak, and
tests/check_fairness_measurement.py asserts the evidence file is free of them.

The instrument is checked before the finding is read: this script re-scores the
held-out split through the shipped hybrid detector at the shipped threshold and
requires its aggregate and all 25 per-type cells to reproduce
evidence/detector_eval.json exactly. A fairness split of a population this
script scored differently from the shipped evaluation would not be about the
shipped system.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlock.data import load_split
from airlock.evaluate import score_docs, _mask_array
from airlock.labels import IDENTIFYING

PERSON_NAME_TYPES = ("name", "first_name", "last_name")
SHIPPED_THRESHOLD = 0.2  # airlock/cli.py and airlock/gate.py default


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion. Never returns [1,1]."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]], by the sum-of-smaller-tables
    definition: total probability of every table at least as extreme as the one
    observed, conditional on both margins.

    Implemented here rather than imported so the repository does not depend on
    a library's choice of definition; measure_fairness cross-checks it against
    scipy.stats.fisher_exact when scipy is importable.
    """
    n = a + b + c + d
    r1, c1 = a + b, a + c
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    logden = _log_binom(n, c1)

    def logp(x):
        return _log_binom(r1, x) + _log_binom(n - r1, c1 - x) - logden

    obs = logp(a)
    tol = 1e-7
    total = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= obs + tol:
            total += math.exp(lp)
    return min(1.0, total)


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values, order-preserving and monotone."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


# ------------------------------------------------------------------ bucketing

def normalise_surface(s: str) -> str:
    return " ".join(s.split()).casefold()


def is_ascii(s: str) -> bool:
    return s.isascii()


def token_bucket(s: str) -> str:
    n = len(s.split())
    return "1" if n <= 1 else ("2" if n == 2 else "3+")


# -------------------------------------------------------------------- scoring

def leaked_flags(docs, preds, redact_types):
    """Per gold identifying span: (doc_index, label, surface, leaked?).

    Coverage rule is character-identical to airlock.evaluate.score_docs: a gold
    span leaks when at least one of its characters survives the mask.
    """
    redact_types = set(redact_types)
    rows = []
    for di, (doc, pred) in enumerate(zip(docs, preds)):
        n = len(doc.text)
        mask = _mask_array(n, pred, redact_types)
        for g in doc.spans:
            if g["label"] not in redact_types:
                continue
            covered = sum(mask[g["start"]:g["end"]])
            width = max(1, g["end"] - g["start"])
            rows.append((di, g["label"], doc.text[g["start"]:g["end"]], covered < width))
    return rows


def rate_block(rows, label):
    n = len(rows)
    leaked = sum(1 for r in rows if r[3])
    caught = n - leaked
    lo, hi = wilson(caught, n)
    return {
        "bucket": label,
        "spans": n,
        "leaked": leaked,
        "recall": round(caught / n, 4) if n else None,
        "recall_ci95": [round(lo, 4), round(hi, 4)] if n else None,
    }


def contrast(name, question, rows_a, label_a, rows_b, label_b):
    A, B = rate_block(rows_a, label_a), rate_block(rows_b, label_b)
    a_c, a_l = A["spans"] - A["leaked"], A["leaked"]
    b_c, b_l = B["spans"] - B["leaked"], B["leaked"]
    p = fisher_exact_two_sided(a_c, a_l, b_c, b_l)
    gap = (None if A["recall"] is None or B["recall"] is None
           else round(A["recall"] - B["recall"], 4))
    return {
        "id": name,
        "question": question,
        "arm_a": A,
        "arm_b": B,
        "recall_gap_a_minus_b": gap,
        "fisher_exact_two_sided_p": p,
        "table": [[a_c, a_l], [b_c, b_l]],
    }


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--split", default="test")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--threshold", type=float, default=SHIPPED_THRESHOLD)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reference", default="evidence/detector_eval.json")
    ap.add_argument("--out", default="evidence/fairness.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-instrument-mismatch", action="store_true",
                    help="controls only: score without requiring reproduction")
    args = ap.parse_args()

    t0 = time.time()
    docs = load_split(args.split, limit=args.limit)

    from airlock.hybrid import HybridDetector
    hyb = HybridDetector(args.model)
    det = hyb.model
    toks = []
    for i in range(0, len(docs), args.batch):
        toks.extend(det._token_scores([d.text for d in docs[i:i + args.batch]]))
    preds = [hyb._decode(d.text, tk, args.threshold) for d, tk in zip(docs, toks)]

    # ---- instrument check, before any fairness number is read
    scored = score_docs(docs, preds)
    instrument = {"reference": args.reference, "threshold": args.threshold}
    if os.path.exists(args.reference) and args.limit is None:
        ref = json.load(open(args.reference))["by_threshold_hybrid"][f"{args.threshold:g}"]
        mismatches = []
        for k in ("gold_identifiers", "identifiers_caught", "identifiers_leaked",
                  "identifier_recall", "typed_recall", "doc_leak_rate"):
            if scored[k] != ref[k]:
                mismatches.append({"field": k, "ours": scored[k], "shipped": ref[k]})
        for t in sorted(set(ref["per_type"]) | set(scored["per_type"])):
            o, s = scored["per_type"].get(t), ref["per_type"].get(t)
            if o != s:
                mismatches.append({"field": f"per_type.{t}", "ours": o, "shipped": s})
        instrument.update({
            "reproduces_shipped_evaluation": not mismatches,
            "per_type_cells_compared": len(set(ref["per_type"]) | set(scored["per_type"])),
            "aggregate_fields_compared": 6,
            "mismatches": mismatches,
            "identifier_recall": scored["identifier_recall"],
            "identifiers_leaked": scored["identifiers_leaked"],
            "gold_identifiers": scored["gold_identifiers"],
        })
        if mismatches and not args.allow_instrument_mismatch:
            print(json.dumps({"instrument_check": "FAILED", "mismatches": mismatches[:8]},
                             indent=2))
            raise SystemExit("instrument does not reproduce the shipped evaluation; "
                             "refusing to report a fairness result from it")
    else:
        instrument["reproduces_shipped_evaluation"] = None
        instrument["note"] = "reference absent or --limit set; not a shipped-figure run"

    # ---- population
    rows = leaked_flags(docs, preds, IDENTIFYING)
    person = [r for r in rows if r[1] in PERSON_NAME_TYPES]
    other = [r for r in rows if r[1] not in PERSON_NAME_TYPES]

    # ---- seen-in-training lookup, built from the detector's ACTUAL gradient set.
    # airlock/train.py holds out docs[-2000:] as dev and trains on docs[:n-2000],
    # so a surface that occurs only in the dev tail was never seen by the weights.
    # Counting those as "seen" would dilute C3 toward the null. The all-docs
    # definition is carried alongside as a robustness check.
    train_all = load_split(args.train_split)
    meta_path = os.path.join(args.model, "train_meta.json")
    n_grad = json.load(open(meta_path))["train_docs"] if os.path.exists(meta_path) else len(train_all)
    train_grad = train_all[:n_grad]

    def surfaces_of(dset):
        out = set()
        for d in dset:
            for sp in d.spans:
                if sp["label"] in PERSON_NAME_TYPES:
                    out.add(normalise_surface(d.text[sp["start"]:sp["end"]]))
        return out

    seen = surfaces_of(train_grad)
    seen_all = surfaces_of(train_all)
    train_surface_forms = len(seen)

    def is_seen(r):
        return normalise_surface(r[2]) in seen

    p_seen = [r for r in person if is_seen(r)]
    p_unseen = [r for r in person if not is_seen(r)]
    p_ascii = [r for r in person if is_ascii(r[2])]
    p_nonascii = [r for r in person if not is_ascii(r[2])]
    p_1tok = [r for r in person if token_bucket(r[2]) == "1"]

    contrasts = [
        contrast("C1", "Are person-name identifiers redacted less often than "
                       "every other identifying type?",
                 person, "person_name_types", other, "all_other_identifying_types"),
        contrast("C2", "Within person names, is a bare surname redacted less "
                       "often than a full name?",
                 [r for r in person if r[1] == "last_name"], "last_name",
                 [r for r in person if r[1] == "name"], "name"),
        contrast("C3", "Does protection depend on the model having seen that "
                       "exact name during training?",
                 p_unseen, "surface_unseen_in_training",
                 p_seen, "surface_seen_in_training"),
        contrast("C4", "Are names containing a non-ASCII character redacted "
                       "less often than pure-ASCII names?",
                 p_nonascii, "name_contains_non_ascii", p_ascii, "name_pure_ascii"),
        contrast("C5", "C3 restricted to single-token names (length control).",
                 [r for r in p_1tok if not is_seen(r)], "unseen_single_token",
                 [r for r in p_1tok if is_seen(r)], "seen_single_token"),
        contrast("C6", "C4 restricted to single-token names (length control).",
                 [r for r in p_1tok if not is_ascii(r[2])], "non_ascii_single_token",
                 [r for r in p_1tok if is_ascii(r[2])], "ascii_single_token"),
        contrast("C7", "POST-HOC: is a bare surname redacted less often than a bare "
                       "given name, with span length held at one token?",
                 [r for r in p_1tok if r[1] == "last_name"], "last_name_single_token",
                 [r for r in p_1tok if r[1] == "first_name"], "first_name_single_token"),
    ]
    contrasts[-1]["pre_registered"] = False
    contrasts[-1]["post_hoc_reason"] = (
        "added after C2 and the token-count table were read; C2's arms differ in "
        "length as well as in type, so C2 alone cannot separate the two")
    for c in contrasts[:-1]:
        c["pre_registered"] = True
    adj = holm({c["id"]: c["fisher_exact_two_sided_p"] for c in contrasts})
    for c in contrasts:
        c["holm_adjusted_p"] = adj[c["id"]]
        c["significant_at_05_after_holm"] = adj[c["id"]] < 0.05

    # C3 under the looser seen-set definition, as a robustness check
    r_unseen_all = [r for r in person if normalise_surface(r[2]) not in seen_all]
    r_seen_all = [r for r in person if normalise_surface(r[2]) in seen_all]
    c3_robust = contrast("C3_all_train_docs",
                         "C3 with the seen-set built from all train docs including "
                         "the 2000-doc dev tail the weights never saw.",
                         r_unseen_all, "surface_unseen_in_any_train_doc",
                         r_seen_all, "surface_seen_in_any_train_doc")

    # ---- descriptive tables (no contrast, no p-value)
    by_token = {}
    for b in ("1", "2", "3+"):
        by_token[b] = rate_block([r for r in person if token_bucket(r[2]) == b],
                                 f"person_name_tokens_{b}")
    by_type = {}
    for t in PERSON_NAME_TYPES:
        by_type[t] = rate_block([r for r in person if r[1] == t], t)
    by_type_single_token = {}
    for t in PERSON_NAME_TYPES:
        by_type_single_token[t] = rate_block(
            [r for r in p_1tok if r[1] == t], f"{t}_single_token")

    # If every person-name span were an unseen surface -- which is what deployment
    # on real names looks like -- what would the headline identifier recall be?
    unseen_rate = rate_block(p_unseen, "u")["recall"]
    proj_person_leaked = round(len(person) * (1.0 - unseen_rate)) if unseen_rate is not None else None
    other_leaked = sum(1 for r in other if r[3])
    projection = {
        "measured_identifier_recall": scored["identifier_recall"],
        "person_name_spans_sharing_a_surface_with_training": len(p_seen),
        "share_of_person_name_spans_seen_in_training": round(len(p_seen) / len(person), 4) if person else None,
        "projected_identifier_recall_if_no_name_were_seen_in_training": (
            round((len(rows) - other_leaked - proj_person_leaked) / len(rows), 4)
            if proj_person_leaked is not None and rows else None),
        "basis": ("holds the non-name types at their measured leak count and reprices "
                  "every person-name span at the measured unseen-surface leak rate"),
        "caveat": ("a projection, not a measurement: it assumes unseen names in "
                   "deployment behave like unseen names in this corpus, and this "
                   "corpus's names are synthetic. It is reported because the "
                   "measured headline is flattered by a train/test surface overlap "
                   "that a real deployment would not enjoy."),
    }

    # distinct people affected: distinct normalised surfaces that leaked at least once
    leaked_surfaces = {normalise_surface(r[2]) for r in person if r[3]}
    all_surfaces = {normalise_surface(r[2]) for r in person}

    # cross-check our Fisher against scipy, if importable
    scipy_check = {"available": False}
    try:
        from scipy.stats import fisher_exact as _sf
        worst = 0.0
        for c in contrasts:
            (a, b), (cc, d) = c["table"]
            if min(a + b, cc + d) == 0:
                continue
            worst = max(worst, abs(float(_sf([[a, b], [cc, d]])[1])
                                   - c["fisher_exact_two_sided_p"]))
        scipy_check = {"available": True, "max_abs_difference": float(worst),
                       "agrees_within_1e_9": bool(worst < 1e-9)}
    except Exception as e:  # pragma: no cover - environment dependent
        scipy_check = {"available": False, "error": type(e).__name__}

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": ("Conditional on a gold identifier being present, does the "
                     "probability that Airlock redacts it depend on properties "
                     "of the person it belongs to?"),
        "split": args.split,
        "documents": len(docs),
        "threshold": args.threshold,
        "threshold_basis": "airlock/cli.py and airlock/gate.py default; the shipped value",
        "detector": "hybrid (model UNION checksum validators), as the CLI and gate use",
        "corpus": "gretelai/synthetic_pii_finance_multilingual (Apache-2.0), English split",
        "instrument_check": instrument,
        "population": {
            "gold_identifying_spans": len(rows),
            "person_name_spans": len(person),
            "other_identifying_spans": len(other),
            "person_name_types": list(PERSON_NAME_TYPES),
            "distinct_person_name_surfaces": len(all_surfaces),
            "distinct_person_name_surfaces_leaked_at_least_once": len(leaked_surfaces),
            "train_person_name_surface_forms": train_surface_forms,
            "train_documents_total": len(train_all),
            "train_documents_used_for_gradient": n_grad,
            "train_gradient_set_basis": ("airlock/train.py holds out docs[-2000:] as dev; "
                                         "train_docs from models/detector/train_meta.json"),
            "train_person_name_surface_forms_including_dev_tail": len(seen_all),
        },
        "buckets": {
            "seen_in_training": rate_block(p_seen, "surface_seen_in_training"),
            "unseen_in_training": rate_block(p_unseen, "surface_unseen_in_training"),
            "pure_ascii": rate_block(p_ascii, "name_pure_ascii"),
            "non_ascii": rate_block(p_nonascii, "name_contains_non_ascii"),
            "by_token_count": by_token,
            "by_person_name_type": by_type,
            "by_person_name_type_single_token": by_type_single_token,
        },
        "contrasts": contrasts,
        "robustness": {"C3_alternative_seen_set": c3_robust},
        "deployment_projection": projection,
        "multiple_comparisons": "Holm-Bonferroni across the six pre-registered contrasts",
        "fisher_implementation": "local sum-of-smaller-tables; cross-checked against scipy",
        "scipy_cross_check": scipy_check,
        "label_caveat": ("The corpus's spans are incomplete, so every recall here is an "
                         "UPPER bound. That bias applies to both arms of every contrast, "
                         "but not necessarily equally: if the annotation misses hard names "
                         "more often than easy ones, the true gaps are wider than measured, "
                         "not narrower."),
        "no_surfaces_exported": True,
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "instrument_reproduces_shipped": instrument.get("reproduces_shipped_evaluation"),
        "contrasts": [{"id": c["id"], "a": c["arm_a"]["bucket"], "b": c["arm_b"]["bucket"],
                       "recall_a": c["arm_a"]["recall"], "recall_b": c["arm_b"]["recall"],
                       "n_a": c["arm_a"]["spans"], "n_b": c["arm_b"]["spans"],
                       "gap": c["recall_gap_a_minus_b"],
                       "p_holm": round(c["holm_adjusted_p"], 6),
                       "sig": c["significant_at_05_after_holm"]} for c in contrasts],
        "by_token_count": {k: (v["recall"], v["spans"]) for k, v in by_token.items()},
        "scipy_cross_check": scipy_check,
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
