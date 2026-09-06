"""Is the model's POSTERIOR load-bearing in the gate, or only its decision?

The claim this exists to test is the one rule 3 of this hackathon turns on:
that the learned model is integrated, not bolted on. Airlock's risk statistic
is the SUM of the detector's per-token probability `p_ident` over the tokens
that survived redaction. Two cheaper things could have been done instead with
exactly the same detector, and if either ranks documents as well then the
posterior is decoration:

  posterior  sum of p_ident over surviving tokens          <- shipped
  count      number of surviving tokens                    <- uncertainty discarded
  flag@tau   number of surviving tokens with p_ident >= tau <- uncertainty thresholded

`count` is the honest control: this project already reports that its statistic
grows with document length, so "you have merely built a length detector" is the
live objection and it is tested here rather than argued with.

All four statistics are computed from the SAME forward pass and the SAME
redaction spans, so the only thing that differs between them is how the
detector's output is read. Comparison is at FIXED COVERAGE -- forward the k
lowest-scoring documents under each statistic, same k -- because only the
shipped statistic has a calibrator and comparing budgets would compare
calibrations instead of rankings.

Writes evidence/posterior_ablation.json. Reads no document text into any
output: only counts, rates and AUCs leave this script.
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
from airlock.gate import surviving_tokens
from airlock.labels import IDENTIFYING
from scripts.eval_gate import auc
from scripts.fit_gate import leaked


def leak_rate_at_coverage(scores, labels, k):
    """Leak rate among the k lowest-scoring documents. Ties broken by index,
    which is the arrival order -- deterministic and independent of the score."""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))[:k]
    return (sum(1 for i in order if labels[i]) / k) if k else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--taus", default="0.01,0.05,0.2")
    ap.add_argument("--coverages", default="0.6365,0.0948")
    ap.add_argument("--out", default="evidence/posterior_ablation.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    taus = [float(x) for x in args.taus.split(",")]
    docs = load_split("test", limit=args.limit)
    det = HybridDetector(args.model)
    redact_types = set(IDENTIFYING)

    stats = {"posterior": [], "count": []}
    for tau in taus:
        stats["flag@%g" % tau] = []
    labels, surviving_counts = [], []

    for i in range(0, len(docs), 16):
        chunk = docs[i:i + 16]
        for doc, toks in zip(chunk, det._token_scores([d.text for d in chunk])):
            spans = det._decode(doc.text, toks, args.threshold)
            surv = surviving_tokens(doc.text, toks, spans)
            ps = [p for (_a, _b, p, _l) in surv]
            stats["posterior"].append(sum(ps))
            stats["count"].append(float(len(ps)))
            for tau in taus:
                stats["flag@%g" % tau].append(float(sum(1 for p in ps if p >= tau)))
            surviving_counts.append(len(ps))
            labels.append(leaked(doc, spans, redact_types))

    n = len(docs)
    base_rate = sum(labels) / n
    covs = [float(x) for x in args.coverages.split(",")]

    rows = []
    for name, s in stats.items():
        row = {"statistic": name, "auc": round(auc(s, labels), 4),
               "distinct_values": len(set(s))}
        for c in covs:
            k = int(round(c * n))
            row["leak_rate_at_coverage_%.4f" % c] = round(leak_rate_at_coverage(s, labels, k), 4)
        rows.append(row)

    # random ranking at the same coverages, 20 seeds -- the floor every row
    # must beat before any of them means anything.
    rand = {"statistic": "random ranking (20 seeds)", "auc": None, "distinct_values": None}
    for c in covs:
        k = int(round(c * n))
        vals = []
        for seed in range(20):
            r = random.Random(seed)
            pick = r.sample(range(n), k)
            vals.append(sum(1 for i in pick if labels[i]) / k)
        rand["leak_rate_at_coverage_%.4f" % c] = round(statistics.mean(vals), 4)
        rand["sd_at_coverage_%.4f" % c] = round(statistics.pstdev(vals), 4)
    rows.append(rand)

    shipped = next(r for r in rows if r["statistic"] == "posterior")
    countrow = next(r for r in rows if r["statistic"] == "count")
    flags = [r for r in rows if r["statistic"].startswith("flag@")]
    best_flag = max(flags, key=lambda r: r["auc"])
    worst_flag = min(flags, key=lambda r: r["auc"])

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": ("Does the gate need the detector's per-token POSTERIOR, or would its "
                     "binary decision (or the token count alone) rank documents just as well?"),
        "split": "test (never used for training or calibration)",
        "documents": n,
        "redaction_threshold": args.threshold,
        "base_leak_rate": round(base_rate, 4),
        "surviving_tokens_per_document_mean": round(sum(surviving_counts) / n, 2),
        "statistics": rows,
        "auc_delta_posterior_minus_count": round(shipped["auc"] - countrow["auc"], 4),
        "best_flag_statistic": best_flag["statistic"],
        "auc_delta_posterior_minus_best_flag": round(shipped["auc"] - best_flag["auc"], 4),
        "worst_flag_statistic": worst_flag["statistic"],
        "auc_delta_posterior_minus_worst_flag": round(shipped["auc"] - worst_flag["auc"], 4),
        "coverages_compared": covs,
        "method": ("One forward pass, one set of redaction spans. Each statistic is a "
                   "different reading of the SAME detector output. Coverage is fixed across "
                   "statistics: forward the k lowest-scoring documents, same k, and measure "
                   "the leak rate among them."),
        "caveat": ("'Leak' means an identifier the CORPUS annotates survived redaction. The "
                   "corpus's annotation is incomplete, so every rate here is a lower bound on "
                   "the true rate. The comparison between statistics is unaffected: all of "
                   "them are scored against the same labels."),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
