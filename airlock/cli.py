"""Airlock command line: one document through the whole path.

    python3 -m airlock.cli --file note.txt
    python3 -m airlock.cli --text "..." --ask "What is being requested?"
    python3 -m airlock.cli --file note.txt --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .hybrid import HybridDetector
from .gate import Calibrator, Gate
from .redact import rehydrate_checked


def build_gate(args) -> Gate:
    det = HybridDetector(args.model)
    cal = Calibrator.load(args.calibration) if os.path.exists(args.calibration) else None
    return Gate(det, cal, threshold=args.threshold, budget=args.budget)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="airlock")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file")
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--calibration", default="models/gate_calibration.json")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="detector probability above which a token is redacted")
    ap.add_argument("--budget", type=float, default=0.12,
                    help="largest residual leak probability the operator will forward")
    ap.add_argument("--ask", help="question to answer about the document; runs a model")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    text = args.text if args.text is not None else open(args.file, encoding="utf-8").read()
    gate = build_gate(args)
    r = gate.process(text)
    d = r["decision"]

    payload = {
        "characters": len(text),
        "identifiers_found": len(r["spans"]),
        "spans": r["spans"],
        "redacted": r["redacted"],
        "vault_size": r["vault"].size,
        "decision": d.__dict__,
    }

    if args.ask:
        from .route import HostedStandIn, LocalLM, PRESERVE_SYSTEM

        engine = HostedStandIn() if d.forward else LocalLM()
        prompt = f"{args.ask}\n\n---\n{r['redacted']}\n---"
        ans = engine.answer(prompt, system=PRESERVE_SYSTEM)
        restored, report = rehydrate_checked(ans.text, r["vault"])
        payload["answer"] = {
            "route": ans.route, "model": ans.model, "seconds": ans.seconds,
            "text_as_returned": ans.text,
            "text_rehydrated": restored,
            "integrity": report,
        }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"input: {len(text)} characters, {len(r['spans'])} identifiers detected")
    for s in r["spans"]:
        print(f"  {s['label']:<26} p={s['score']:.3f}  {s['text']!r}")
    print("\n--- text that would leave this machine ---")
    print(r["redacted"])
    print("\n--- gate ---")
    print(f"  expected surviving identifier tokens : {d.expected_surviving_identifier_tokens}")
    print(f"  calibrated residual leak risk        : {d.risk}   (budget {d.budget})")
    print(f"  route                                : {d.route}")
    print(f"  {d.reason}")
    if args.ask:
        a = payload["answer"]
        print(f"\n--- answer via {a['route']} ({a['model']}, {a['seconds']}s) ---")
        print(a["text_as_returned"])
        print("\n--- re-hydrated locally ---")
        print(a["text_rehydrated"])
        i = a["integrity"]
        print(f"\n  placeholders expected {i['expected']}, intact {len(i['intact'])}, "
              f"missing {i['missing']}, trustworthy={i['trustworthy']}")
        if not i["trustworthy"]:
            print("  WARNING: the answer did not return every placeholder. Do not treat the "
                  "re-hydrated text as faithful -- a value in it may be invented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
