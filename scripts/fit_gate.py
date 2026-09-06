"""Fit the gate's calibration on documents held out of the detector's training.

The 2000 documents used here are exactly the tail of the training split that
`airlock/train.py` withholds (DEV_DOCS). The test split is never touched.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlock.data import load_split
from airlock.hybrid import HybridDetector
from airlock.gate import Calibrator, raw_risk
from airlock.labels import IDENTIFYING
from airlock.train import DEV_DOCS


def leaked(doc, spans, redact_types):
    n = len(doc.text)
    mask = bytearray(n)
    for s in spans:
        if s.label in redact_types:
            for i in range(max(0, s.start), min(n, s.end)):
                mask[i] = 1
    for g in doc.spans:
        if g["label"] in redact_types and sum(mask[g["start"]:g["end"]]) < (g["end"] - g["start"]):
            return True
    return False


def collect(det, docs, threshold, batch=16):
    raws, labels = [], []
    redact_types = set(IDENTIFYING)
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        for doc, toks in zip(chunk, det._token_scores([d.text for d in chunk])):
            spans = det._decode(doc.text, toks, threshold)
            expected, _ = raw_risk(doc.text, toks, spans)
            raws.append(expected)
            labels.append(leaked(doc, spans, redact_types))
    return raws, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out", default="models/gate_calibration.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    all_train = load_split("train")
    dev = all_train[-DEV_DOCS:]
    if args.limit:
        dev = dev[: args.limit]
    det = HybridDetector(args.model)
    raws, labels = collect(det, dev, args.threshold)
    cal = Calibrator.fit(
        raws, labels,
        fitted_on=f"train split tail ({len(dev)} docs held out of detector training), "
                  f"redaction threshold {args.threshold}",
    )
    cal.save(args.out)
    print(json.dumps({
        "dev_docs": len(dev),
        "observed_leak_rate": round(cal.observed_leak_rate, 4),
        "blocks": len(cal.x),
        "block_sizes": cal.n,
        "p_upper": [round(v, 4) for v in cal.p_upper],
        "x_edges": [round(v, 3) for v in cal.x],
        "seconds": round(time.time() - t0, 1),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
