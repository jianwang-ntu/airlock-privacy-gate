"""Score the gate on the untouched test split and write evidence/gate_eval.json.

Reports, for a grid of leak budgets: what fraction of documents Airlock is
willing to forward (coverage) and how often a forwarded document actually
leaked. The control that carries the weight is the random gate -- the same
coverage, documents chosen at random. If the risk score carried no information
the two would agree.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlock.data import load_split
from airlock.hybrid import HybridDetector
from airlock.gate import Calibrator, raw_risk
from airlock.labels import IDENTIFYING
from scripts.fit_gate import leaked


def auc(scores, labels):
    """Rank-based AUC; ties get average rank."""
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, l in pairs if l)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    s = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (s - pos * (pos + 1) / 2) / (pos * neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--calibration", default="models/gate_calibration.json")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--budgets", default="0.02,0.034,0.05,0.07,0.1,0.15,0.2,0.3,1.0")
    ap.add_argument("--out", default="evidence/gate_eval.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    docs = load_split("test", limit=args.limit)
    det = HybridDetector(args.model)
    cal = Calibrator.load(args.calibration)
    redact_types = set(IDENTIFYING)

    risks, risks_hat, raws, labels = [], [], [], []
    for i in range(0, len(docs), 16):
        chunk = docs[i:i + 16]
        for doc, toks in zip(chunk, det._token_scores([d.text for d in chunk])):
            spans = det._decode(doc.text, toks, args.threshold)
            expected, _ = raw_risk(doc.text, toks, spans)
            raws.append(expected)
            risks.append(cal.predict(expected, conservative=True))
            risks_hat.append(cal.predict(expected, conservative=False))
            labels.append(leaked(doc, spans, redact_types))

    n = len(docs)
    base_rate = sum(labels) / n
    rows = []
    rng = random.Random(0)
    for b in [float(x) for x in args.budgets.split(",")]:
        fwd = [i for i in range(n) if risks[i] <= b]
        held = [i for i in range(n) if risks[i] > b]
        fwd_leaks = sum(1 for i in fwd if labels[i])
        ctrl = []
        for seed in range(20):
            r = random.Random(seed)
            pick = r.sample(range(n), len(fwd)) if fwd else []
            ctrl.append(sum(1 for i in pick if labels[i]) / len(pick) if pick else 0.0)
        rows.append({
            "budget": b,
            "coverage": round(len(fwd) / n, 4),
            "documents_forwarded": len(fwd),
            "documents_held_local": len(held),
            "leaks_among_forwarded": fwd_leaks,
            "observed_leak_rate_forwarded": round(fwd_leaks / len(fwd), 4) if fwd else None,
            "observed_leak_rate_held": round(sum(1 for i in held if labels[i]) / len(held), 4) if held else None,
            "budget_respected": (fwd_leaks / len(fwd) <= b) if fwd else None,
            "random_gate_same_coverage_leak_rate_mean": round(statistics.mean(ctrl), 4) if fwd else None,
            "random_gate_same_coverage_leak_rate_sd": round(statistics.pstdev(ctrl), 4) if fwd else None,
        })

    # reliability of the conservative risk against observed frequency
    order = sorted(range(n), key=lambda i: risks[i])
    bins, size = [], max(1, n // 10)
    for s in range(0, n, size):
        idx = order[s:s + size]
        if not idx:
            continue
        bins.append({
            "n": len(idx),
            "mean_predicted_upper": round(sum(risks[i] for i in idx) / len(idx), 4),
            "mean_predicted_hat": round(sum(risks_hat[i] for i in idx) / len(idx), 4),
            "observed": round(sum(1 for i in idx if labels[i]) / len(idx), 4),
        })
    ece_hat = sum(b["n"] * abs(b["mean_predicted_hat"] - b["observed"]) for b in bins) / n
    ece_up = sum(b["n"] * abs(b["mean_predicted_upper"] - b["observed"]) for b in bins) / n
    conservative_bins = sum(1 for b in bins if b["mean_predicted_upper"] >= b["observed"])

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "test (never used for training or calibration)",
        "documents": n,
        "redaction_threshold": args.threshold,
        "forward_everything_leak_rate": round(base_rate, 4),
        "calibration": {"fitted_on": cal.fitted_on, "fitted_docs": cal.fitted_docs,
                        "dev_leak_rate": round(cal.observed_leak_rate, 4),
                        "blocks": len(cal.x), "min_certifiable_upper_bound": min(cal.p_upper)},
        "ranking_auc_raw_statistic": round(auc(raws, labels), 4),
        "risk_coverage": rows,
        "reliability_deciles": bins,
        "ece_posterior_mean": round(ece_hat, 4),
        "ece_upper_bound": round(ece_up, 4),
        "reliability_bins_where_upper_bound_is_conservative": f"{conservative_bins}/{len(bins)}",
        "seconds": round(time.time() - t0, 1),
        "caveat": ("'Leak' means an identifier the CORPUS annotates survived redaction. The "
                   "corpus's annotation is incomplete, so the true leak rate is higher than "
                   "every number here and the gate is calibrated against annotated leaks only."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("documents", "forward_everything_leak_rate", "ranking_auc_raw_statistic",
                       "ece_posterior_mean", "ece_upper_bound",
                       "reliability_bins_where_upper_bound_is_conservative")}, indent=2))
    for r in rows:
        print(f"  budget {r['budget']:<6} coverage {r['coverage']:<7} "
              f"leak(fwd) {r['observed_leak_rate_forwarded']} "
              f"random-gate {r['random_gate_same_coverage_leak_rate_mean']} "
              f"respected={r['budget_respected']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
