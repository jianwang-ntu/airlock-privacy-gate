"""Exercise the whole path on two documents and record what actually happened.

One document the gate forwards, one it refuses. Both go through detection,
redaction, the routing decision, an answer from a model, and re-hydration with
the integrity check. Everything printed here is produced by the run; nothing is
transcribed by hand.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlock.gate import Calibrator, Gate
from airlock.hybrid import HybridDetector
from airlock.redact import rehydrate_checked
from airlock.route import HostedStandIn, LocalLM, PRESERVE_SYSTEM

DOCS = [
    ("support_ticket",
     "Ticket 88231 (billing). The caller says the direct debit failed twice this month. "
     "Reference INV-2024-0912. Please confirm whether the mandate is still active and what "
     "the retry schedule is. No action taken yet.",
     "What is the customer asking for, and what should the agent check first?"),
    ("payment_instruction",
     "From: Meera Subramanian <meera.s@example.co.uk>\n"
     "Please transfer GBP 12,400 to IBAN DE89370400440532013000 held by Hartley Freight Ltd, "
     "reference HF-2291. Card on file 4111 1111 1111 1111 should not be charged. "
     "Contact me on +44 7700 900123 if the payment fails. Employee id EMP-40221.",
     "Summarise the instruction and list what must be verified before paying."),
]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "evidence/demo_run.json"
    det = HybridDetector("models/detector")
    cal = Calibrator.load("models/gate_calibration.json")
    gate = Gate(det, cal, threshold=0.2, budget=0.12)

    engines = {}
    records = []
    for name, text, question in DOCS:
        t0 = time.time()
        r = gate.process(text)
        d = r["decision"]
        route = "hosted_model_standin" if d.forward else "local_model"
        if route not in engines:
            engines[route] = HostedStandIn() if d.forward else LocalLM()
        ans = engines[route].answer(f"{question}\n\n---\n{r['redacted']}\n---",
                                    system=PRESERVE_SYSTEM)
        restored, integrity = rehydrate_checked(ans.text, r["vault"])
        records.append({
            "document": name,
            "characters": len(text),
            "question": question,
            "identifiers_detected": [{k: s[k] for k in ("label", "score", "text")}
                                     for s in r["spans"]],
            "text_leaving_the_machine": r["redacted"],
            "vault_entries": r["vault"].size,
            "decision": d.__dict__,
            "answer_model": ans.model,
            "answer_route": ans.route,
            "answer_seconds": ans.seconds,
            "answer_as_returned": ans.text,
            "answer_rehydrated_locally": restored,
            "rehydration_integrity": integrity,
            "wall_seconds": round(time.time() - t0, 2),
        })
        print(f"[{name}] route={d.route} risk={d.risk} forward={d.forward} "
              f"identifiers={len(r['spans'])} integrity_ok={integrity['trustworthy']}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector": "models/detector (hybrid: fine-tuned distilbert-base-cased + checksum validators)",
        "redaction_threshold": 0.2,
        "leak_budget": 0.12,
        "note": ("No third-party API is called anywhere in this run. The document the gate "
                 "forwards is answered by a larger model on this same machine standing in for "
                 "the hosted one, so the loop is exercised without anything leaving the host."),
        "runs": records,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
