"""Measure what the corpus's own annotation misses, and record it.

Two numbers the README depends on, and neither flatters this project:
  * how many of the corpus's card numbers are Luhn-valid -- which bounds how
    much the checksum validators can possibly contribute here;
  * how often a validator fires with proof on a string the annotation does not
    mark -- which is why "over-redaction" in this repository is reported as an
    upper bound rather than as a false-positive rate.
"""
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlock.data import load_split
from airlock.validators import ChecksumValidators, iban_ok, luhn_ok

SAMPLE = 800


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "evidence/corpus_gaps.json"
    docs = load_split("test")
    cards = [d.text[s["start"]:s["end"]] for d in docs for s in d.spans
             if s["label"] == "credit_card_number"]
    ibans = [d.text[s["start"]:s["end"]] for d in docs for s in d.spans if s["label"] == "iban"]
    v = ChecksumValidators()
    extra = collections.Counter()
    for d in docs[:SAMPLE]:
        gold = bytearray(len(d.text))
        for s in d.spans:
            for i in range(s["start"], s["end"]):
                gold[i] = 1
        for sp in v.find(d.text):
            if not any(gold[sp.start:sp.end]):
                extra[sp.label] += 1
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": "test",
        "documents": len(docs),
        "credit_card_spans": len(cards),
        "credit_card_spans_luhn_valid": sum(luhn_ok(re.sub(r"[ \-]", "", c)) for c in cards),
        "iban_spans": len(ibans),
        "iban_spans_mod97_valid": sum(iban_ok(i) for i in ibans),
        "sampled_documents": SAMPLE,
        "validator_hits_outside_any_gold_span": dict(sorted(extra.items())),
        "reading": ("A checksum validator fires only on proof. Where it fires on a string the "
                    "annotation does not mark, the likelier explanation is a gap in the "
                    "annotation than an error in arithmetic -- so recall here is an upper bound "
                    "on true recall and over-redaction an upper bound on false positives."),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
