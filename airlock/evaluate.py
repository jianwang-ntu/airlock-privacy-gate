"""Measure the gate on the held-out split, and write the evidence file.

Definitions, stated once so no number is read the wrong way:

leak            a gold IDENTIFYING span at least one character of which
                survives redaction. Partial masking counts as a leak: half an
                account number still identifies an account.
doc leak rate   fraction of documents with at least one leaked identifier.
over-redaction  a predicted span overlapping no gold span at all. The corpus's
                own labels are known to be incomplete (documents contain names
                the annotation misses), so over-redaction is an UPPER bound on
                false positives and recall is an UPPER bound on true recall.
                Both are reported in that direction on purpose.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .data import load_split
from .labels import IDENTIFYING


def _mask_array(n: int, spans, keep_types) -> bytearray:
    m = bytearray(n)
    for s in spans:
        d = s if isinstance(s, dict) else s.as_dict()
        if d["label"] in keep_types:
            for i in range(max(0, d["start"]), min(n, d["end"])):
                m[i] = 1
    return m


def score_docs(docs, preds, redact_types=None):
    redact_types = set(IDENTIFYING if redact_types is None else redact_types)
    gold_total = gold_leaked = gold_partial = 0
    typed_correct = 0
    docs_with_leak = docs_with_gold = 0
    over_spans = pred_spans = 0
    masked_chars = total_chars = 0
    per_type = {}

    for doc, pred in zip(docs, preds):
        n = len(doc.text)
        total_chars += n
        mask = _mask_array(n, pred, redact_types)
        masked_chars += sum(mask)
        pred_list = [p if isinstance(p, dict) else p.as_dict() for p in pred]
        pred_spans += len(pred_list)
        gold_ident = [s for s in doc.spans if s["label"] in redact_types]
        if gold_ident:
            docs_with_gold += 1
        leaked_here = 0
        for g in gold_ident:
            gold_total += 1
            covered = sum(mask[g["start"]:g["end"]])
            width = max(1, g["end"] - g["start"])
            t = per_type.setdefault(g["label"], {"gold": 0, "leaked": 0, "typed_correct": 0})
            t["gold"] += 1
            if covered < width:
                gold_leaked += 1
                leaked_here += 1
                t["leaked"] += 1
                if covered > 0:
                    gold_partial += 1
            else:
                hit = next((p for p in pred_list
                            if p["start"] < g["end"] and p["end"] > g["start"]), None)
                if hit and hit["label"] == g["label"]:
                    typed_correct += 1
                    t["typed_correct"] += 1
        if gold_ident and leaked_here:
            docs_with_leak += 1
        gold_mask = _mask_array(n, doc.spans, {s["label"] for s in doc.spans})
        for p in pred_list:
            if not any(gold_mask[p["start"]:p["end"]]):
                over_spans += 1

    caught = gold_total - gold_leaked
    return {
        "gold_identifiers": gold_total,
        "identifiers_caught": caught,
        "identifiers_leaked": gold_leaked,
        "identifiers_partially_masked_still_leaked": gold_partial,
        "identifier_recall": round(caught / gold_total, 4) if gold_total else None,
        "typed_recall": round(typed_correct / gold_total, 4) if gold_total else None,
        "docs_with_identifiers": docs_with_gold,
        "docs_with_leak": docs_with_leak,
        "doc_leak_rate": round(docs_with_leak / docs_with_gold, 4) if docs_with_gold else None,
        "predicted_spans": pred_spans,
        "over_redacted_spans_upper_bound": over_spans,
        "over_redaction_rate_upper_bound": round(over_spans / pred_spans, 4) if pred_spans else None,
        "chars_masked_fraction": round(masked_chars / total_chars, 4) if total_chars else None,
        "per_type": {k: {**v, "recall": round((v["gold"] - v["leaked"]) / v["gold"], 4)}
                     for k, v in sorted(per_type.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--thresholds", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="evidence/detector_eval.json")
    args = ap.parse_args()

    docs = load_split(args.split, limit=args.limit)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    from .baseline import RegexBaseline
    from .detect import Detector
    from .hybrid import HybridDetector
    from .validators import ChecksumValidators

    hyb = HybridDetector(args.model)
    det = hyb.model
    t0 = time.time()
    toks = []
    for i in range(0, len(docs), args.batch):
        toks.extend(det._token_scores([d.text for d in docs[i:i + args.batch]]))
    infer_s = time.time() - t0

    results, results_hybrid = {}, {}
    for th in thresholds:
        results[f"{th:g}"] = score_docs(docs, [det._decode(d.text, tk, th)
                                               for d, tk in zip(docs, toks)])
        results_hybrid[f"{th:g}"] = score_docs(docs, [hyb._decode(d.text, tk, th)
                                                     for d, tk in zip(docs, toks)])

    b = RegexBaseline()
    t1 = time.time()
    bpred = b.find_batch([d.text for d in docs])
    base_s = time.time() - t1
    baseline = score_docs(docs, bpred)
    validators_only = score_docs(docs, ChecksumValidators().find_batch([d.text for d in docs]))

    meta = {}
    mpath = os.path.join(args.model, "train_meta.json")
    if os.path.exists(mpath):
        meta = json.load(open(mpath))

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "documents": len(docs),
        "corpus": "gretelai/synthetic_pii_finance_multilingual (Apache-2.0), English split",
        "label_caveat": ("The corpus's spans are known to be incomplete -- documents contain "
                         "identifiers the annotation misses. Recall is therefore an UPPER bound "
                         "on true recall and over-redaction an UPPER bound on false positives. "
                         "The planted-identifier stress set measures recall against ground truth "
                         "that is exact by construction."),
        "redact_policy_types": sorted(IDENTIFYING),
        "model": {**meta, "path": args.model},
        "throughput": {
            "model_docs_per_second": round(len(docs) / infer_s, 2),
            "model_seconds": round(infer_s, 2),
            "baseline_docs_per_second": round(len(docs) / base_s, 2),
            "baseline_seconds": round(base_s, 2),
            "device": det.device,
        },
        "by_threshold_model_only": results,
        "by_threshold_hybrid": results_hybrid,
        "regex_baseline": baseline,
        "checksum_validators_only": validators_only,
        "ablation_note": ("hybrid = model UNION checksum validators, which is what the CLI and "
                          "the gate use. model_only and validators_only are each scored through "
                          "the identical harness so the union's contribution is visible."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    keys = ("identifier_recall", "doc_leak_rate", "over_redaction_rate_upper_bound")
    print(json.dumps({
        "model_only": {t: {k: r[k] for k in keys} for t, r in results.items()},
        "hybrid": {t: {k: r[k] for k in keys} for t, r in results_hybrid.items()},
        "validators_only": {k: validators_only[k] for k in keys},
        "regex_baseline": {k: baseline[k] for k in keys},
    }, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
