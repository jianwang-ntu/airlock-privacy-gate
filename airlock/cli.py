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


def _die(msg: str, remedy: str | None = None):
    """Stop the way argparse already stops: one line naming what is wrong, on
    stderr, exit 2. The two mistakes argparse handles (`no source`, `two
    sources`) were the only ones that behaved like this; the paths below used to
    raise instead, and a traceback is not a message to an operator."""
    print(f"airlock: error: {msg}", file=sys.stderr)
    if remedy:
        print(f"airlock: {remedy}", file=sys.stderr)
    raise SystemExit(2)


def read_source(args) -> str:
    if args.text is not None:
        return args.text
    if not os.path.isfile(args.file):
        _die(f"--file {args.file}: no such file")
    try:
        return open(args.file, encoding="utf-8").read()
    except UnicodeDecodeError:
        _die(f"--file {args.file}: not UTF-8 text")
    except OSError as e:
        _die(f"--file {args.file}: {e.strerror}")


def build_gate(args, calibration_named: bool) -> "tuple[Gate, bool]":
    """Returns the gate and whether it is CALIBRATED. The second value is not
    bookkeeping: `Gate` falls back to the naive risk when there is no
    calibrator, and the naive risk is a different number on the same document.
    Whoever prints the result has to be able to say which one it is."""
    try:
        det = HybridDetector(args.model)
    except Exception:
        if os.path.isdir(args.model):
            raise      # the directory is there; whatever is wrong is inside it
        _die(f"--model {args.model}: not a directory, and not a model this "
             f"machine has already cached",
             "run `bash scripts/fetch_model.sh` for the exact weights every "
             "number in the README was measured with, or `bash "
             "scripts/train_all.sh` to rebuild them from the corpus.")

    if os.path.exists(args.calibration):
        cal = Calibrator.load(args.calibration)
        return Gate(det, cal, threshold=args.threshold, budget=args.budget), True

    if calibration_named:
        _die(f"--calibration {args.calibration}: no such file",
             "refusing to substitute the uncalibrated risk for a calibration "
             "file you asked for by name.")

    print(f"airlock: WARNING: {args.calibration} not found -- the gate is "
          f"UNCALIBRATED and the risk below is the naive estimate, not the "
          f"calibrated one. Run `bash scripts/fetch_model.sh`.", file=sys.stderr)
    return Gate(det, None, threshold=args.threshold, budget=args.budget), False


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

    raw = sys.argv[1:] if argv is None else list(argv)
    calibration_named = any(a == "--calibration" or a.startswith("--calibration=")
                            for a in raw)

    text = read_source(args)
    gate, calibrated = build_gate(args, calibration_named)
    r = gate.process(text)
    d = r["decision"]

    payload = {
        "characters": len(text),
        "identifiers_found": len(r["spans"]),
        "spans": r["spans"],
        "redacted": r["redacted"],
        "vault_size": r["vault"].size,
        "calibrated": calibrated,
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
    risk_label = ("calibrated residual leak risk" if calibrated
                  else "UNCALIBRATED naive leak risk")
    print(f"  {risk_label:<37}: {d.risk}   (budget {d.budget})")
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
