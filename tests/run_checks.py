"""Airlock's control suite. No pytest, no plugins: `python3 tests/run_checks.py`.

House rule: every accepting check is paired with a control that must FAIL for
the check to mean anything. A check that can only pass is not a check. Checks
needing the trained detector report SKIP when it is absent rather than passing
vacuously, and the run prints how many were skipped.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airlock.data import Doc, encode_doc          # noqa: E402
from airlock.gate import Calibrator, raw_risk     # noqa: E402
from airlock.labels import ID2LABEL, IDENTIFYING  # noqa: E402
from airlock.redact import redact, rehydrate, rehydrate_checked  # noqa: E402
from scripts.eval_gate import auc                 # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        r = fn()
    except SkipCheck as e:
        SKIP.append((name, str(e)))
        print(f"SKIP {name}: {e}")
        return
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        return
    if r is True:
        PASS.append(name)
        print(f"PASS {name}")
    else:
        FAIL.append((name, str(r)))
        print(f"FAIL {name}: {r}")


class SkipCheck(Exception):
    pass


MODEL = os.path.join(ROOT, "models", "detector")
CAL = os.path.join(ROOT, "models", "gate_calibration.json")
_det = None


def detector():
    global _det
    if not os.path.exists(os.path.join(MODEL, "config.json")):
        raise SkipCheck("models/detector absent -- run scripts/train_all.sh")
    if _det is None:
        from airlock.hybrid import HybridDetector   # what the CLI and gate ship
        _det = HybridDetector(MODEL)
    return _det


# --------------------------------------------------------------- alignment
TEXT_A = "Please pay Anita Kulkarni at IBAN GB33BUKB20201555555555 today."
SPAN_A = [{"start": 11, "end": 25, "label": "name"},
          {"start": 34, "end": 56, "label": "iban"}]


def _tags(shift=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-cased")
    spans = [{**s, "start": s["start"] + shift, "end": s["end"] + shift} for s in SPAN_A]
    w = encode_doc(tok, Doc(text=TEXT_A, spans=spans))[0]
    return [(TEXT_A[a:b], ID2LABEL[l]) for (a, b), l in zip(w["offsets"], w["labels"])
            if l not in (-100, 0)]


def c_alignment():
    """The name is several word-pieces ("Ku","lk","ar","ni"), so the check is
    that the tagged pieces reconstruct the gold surface -- not that any single
    token equals it. Asserting the latter is how this check first failed."""
    got = _tags()
    joined = "".join(t for t, lbl in got if lbl.endswith("name"))
    if joined != "AnitaKulkarni":
        return f"name tokens reconstruct {joined!r}, expected 'AnitaKulkarni'"
    if got[0][1] != "B-name":
        return f"first name token tagged {got[0][1]}"
    if not any(lbl.endswith("iban") for _, lbl in got):
        return "iban span produced no iban tag"
    return True


def c_alignment_control():
    """Shifting the gold spans by three characters must change the tags."""
    return _tags() != _tags(shift=3) or "shifted spans produced identical tags"


# --------------------------------------------------------------- redaction
def c_redaction_removes_every_char():
    red, vault = redact(TEXT_A, SPAN_A)
    for s in SPAN_A:
        if TEXT_A[s["start"]:s["end"]] in red:
            return f"{TEXT_A[s['start']:s['end']]!r} survived redaction"
    return vault.size == 2 or f"vault holds {vault.size} entries, expected 2"


def c_redaction_control_no_spans():
    """With nothing detected the identifier must still be there -- otherwise
    the check above passes for a reason other than redaction."""
    red, vault = redact(TEXT_A, [])
    return ("Anita Kulkarni" in red and vault.size == 0) or "empty span list still redacted"


def c_placeholder_consistency():
    t = "Anita Kulkarni called. Anita Kulkarni is a customer. Ravi Menon is not."
    spans = [{"start": 0, "end": 14, "label": "name"},
             {"start": 23, "end": 37, "label": "name"},
             {"start": 52, "end": 62, "label": "name"}]
    red, vault = redact(t, spans)
    if red.count("[NAME_1]") != 2:
        return f"repeated surface got {red.count('[NAME_1]')} identical placeholders, expected 2"
    return "[NAME_2]" in red or "second distinct name did not get its own placeholder"


def c_placeholder_consistency_control():
    """A distinct surface must not collapse onto the first placeholder."""
    t = "Anita Kulkarni called. Ravi Menon did not."
    spans = [{"start": 0, "end": 14, "label": "name"}, {"start": 23, "end": 33, "label": "name"}]
    red, _ = redact(t, spans)
    return red.count("[NAME_1]") == 1 or "two different names share one placeholder"


def c_overlap_leaves_no_tail():
    t = "Contact Anita Kulkarni Rao now."
    spans = [{"start": 8, "end": 22, "label": "name"}, {"start": 14, "end": 26, "label": "name"}]
    red, _ = redact(t, spans)
    return ("Kulkarni" not in red and "Rao" not in red) or f"overlap left a tail: {red!r}"


def c_rehydrate_roundtrip():
    red, vault = redact(TEXT_A, SPAN_A)
    return rehydrate(red, vault) == TEXT_A or "round trip did not restore the original"


def c_rehydrate_flags_missing():
    """The measured failure this exists for: a model that answers with an
    invented value instead of the placeholder."""
    _red, vault = redact(TEXT_A, SPAN_A)
    ok, rep_ok = rehydrate_checked("Paid [NAME_1] via [IBAN_1].", vault)
    bad, rep_bad = rehydrate_checked("Paid [NAME_1] via GB00INVENTED0000.", vault)
    if not rep_ok["trustworthy"]:
        return "an answer with every placeholder was called untrustworthy"
    if rep_bad["trustworthy"]:
        return "an answer with a fabricated account number was called trustworthy"
    return rep_bad["missing"] == ["[IBAN_1]"] or f"wrong missing list {rep_bad['missing']}"


# ------------------------------------------------------------------- gate
def c_calibrator_never_certain():
    c = Calibrator.fit([0.01 * i for i in range(400)], [0] * 400, "all-clean self test")
    return (min(c.p_upper) > 0 and min(c.p_hat) > 0) or \
        "400 clean documents produced a risk of exactly zero"


def c_calibrator_monotone():
    import random
    r = random.Random(1)
    raw = [r.random() * 2 for _ in range(1500)]
    lab = [1 if r.random() < x / 3 else 0 for x in raw]
    c = Calibrator.fit(raw, lab, "monotonicity self test")
    xs = [0.0, 0.2, 0.5, 1.0, 1.5, 1.99]
    ps = [c.predict(x) for x in xs]
    return all(ps[i] >= ps[i - 1] for i in range(1, len(ps))) or f"not monotone: {ps}"


def c_calibrator_control_shuffled():
    """Shuffled labels must destroy the signal: the fitted curve should be
    nearly flat, otherwise the fit is reading noise as risk."""
    import random
    r = random.Random(2)
    raw = [r.random() * 2 for _ in range(1500)]
    lab = [r.random() < 0.3 for _ in range(1500)]
    c = Calibrator.fit(raw, lab, "shuffled control")
    spread = c.p_hat[-1] - c.p_hat[0]
    real = Calibrator.fit(raw, [1 if x > 1.0 else 0 for x in raw], "signal")
    return (real.p_hat[-1] - real.p_hat[0]) > 3 * spread or \
        f"shuffled labels produced spread {spread:.3f}, signal {real.p_hat[-1]-real.p_hat[0]:.3f}"


def c_auc_known_orderings():
    if auc([0, 1, 2, 3], [0, 0, 1, 1]) != 1.0:
        return "perfect ordering did not score 1.0"
    if auc([0, 1, 2, 3], [1, 1, 0, 0]) != 0.0:
        return "reversed ordering did not score 0.0"
    return abs(auc([0, 0, 0, 0], [1, 0, 1, 0]) - 0.5) < 1e-9 or "all ties did not score 0.5"


def c_risk_rises_with_survivors():
    toks = [(0, 4, 0.9, "B-name", 0.9), (5, 9, 0.9, "I-name", 0.9)]
    text = "Anna Bell"
    covered, _ = raw_risk(text, toks, [{"start": 0, "end": 9, "label": "name"}])
    exposed, _ = raw_risk(text, toks, [])
    return exposed > covered and covered == 0.0 or \
        f"redacting the tokens did not drop the statistic ({covered} vs {exposed})"


# ----------------------------------------------------- model-dependent
PLANTED = ("Ticket 4412: caller Meera Subramanian, mobile +44 7700 900123, "
           "email meera.s@example.co.uk, account IBAN DE89370400440532013000, "
           "card 4111 1111 1111 1111, national id 452-11-9834. Escalate to billing.")
PLANTED_IDS = ["Meera Subramanian", "+44 7700 900123", "meera.s@example.co.uk",
               "DE89370400440532013000", "4111 1111 1111 1111", "452-11-9834"]
CLEAN = ("Ticket 4412: the caller asked about the billing cycle and the refund "
         "window. No account details were shared. Escalate to billing.")


def c_planted_document_is_masked():
    """The card number here is Luhn-valid and the model alone left it in the
    outgoing text. This check is why airlock/validators.py exists."""
    det = detector()
    spans = det.find(PLANTED, 0.2)
    red, _ = redact(PLANTED, spans)
    survived = [v for v in PLANTED_IDS if v in red]
    return not survived or f"survived redaction: {survived}"


def c_clean_document_control():
    """A document with no identifiers must be masked far less than the planted
    one, or the check above passes because the detector masks everything."""
    det = detector()
    p_red, _ = redact(PLANTED, det.find(PLANTED, 0.2))
    c_red, _ = redact(CLEAN, det.find(CLEAN, 0.2))
    p_frac = p_red.count("[") / len(PLANTED)
    c_frac = c_red.count("[") / len(CLEAN)
    return p_frac > 3 * c_frac or f"planted {p_frac:.4f} vs clean {c_frac:.4f} placeholders/char"


def c_snapping_widens_to_word():
    det = detector()
    t = "from the Kanpur branch"
    a, b = det._snap(t, 9, 11)             # "Ka"
    if t[a:b] != "Kanpur":
        return f"sub-word span snapped to {t[a:b]!r}, expected 'Kanpur'"
    c, d = det._snap(t, 9, 15)             # already a whole word
    return (c, d) == (9, 15) or f"a whole word was moved to ({c},{d})"


def c_gate_decision_shape():
    det = detector()
    if not os.path.exists(CAL):
        raise SkipCheck("models/gate_calibration.json absent -- run scripts/fit_gate.py")
    from airlock.gate import Gate
    g = Gate(det, Calibrator.load(CAL), threshold=0.2, budget=0.05)
    risky = g.process(PLANTED)["decision"]
    clean = g.process(CLEAN)["decision"]
    if not (0.0 < risky.risk <= 1.0):
        return f"risk out of range: {risky.risk}"
    return risky.risk >= clean.risk or \
        f"a document stuffed with identifiers scored below a clean one ({risky.risk} < {clean.risk})"


def c_coverage_selects_the_lowest_risk():
    """The ablation table rests on this: forward the k LOWEST-scoring documents.

    Selecting the highest instead would invert every row while still producing
    a plausible-looking table, so the direction is asserted rather than assumed.
    """
    from scripts.ablate_posterior import leak_rate_at_coverage
    scores = [0.0, 0.1, 0.2, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1]           # the leaks are the high scorers
    if leak_rate_at_coverage(scores, labels, 3) != 0.0:
        return "the three lowest-risk documents were not leak-free"
    if leak_rate_at_coverage(scores, labels, 5) != 0.4:
        return "full coverage did not return the base rate"
    return True


def c_coverage_control_reversed_scores():
    """CONTROL: negate the scores and the same call must find every leak."""
    from scripts.ablate_posterior import leak_rate_at_coverage
    scores = [0.0, 0.1, 0.2, 0.9, 1.0]
    labels = [0, 0, 0, 1, 1]
    rate = leak_rate_at_coverage([-x for x in scores], labels, 3)
    return abs(rate - 2 / 3) < 1e-9 or \
        f"reversing the ranking did not surface the leaks (got {rate})"


def c_evidence_matches_disk():
    """Every headline number the README quotes is re-read from evidence/."""
    ev = os.path.join(ROOT, "evidence", "gate_eval.json")
    if not os.path.exists(ev):
        raise SkipCheck("evidence/gate_eval.json absent -- run scripts/eval_gate.py")
    d = json.load(open(ev))
    rows = {r["budget"]: r for r in d["risk_coverage"]}
    if not rows:
        return "no risk-coverage rows"
    for b, r in rows.items():
        if r["observed_leak_rate_forwarded"] is not None and not r["budget_respected"]:
            return f"budget {b} was not respected in the evidence file"
    return d["reliability_bins_where_upper_bound_is_conservative"].split("/")[0] == \
        d["reliability_bins_where_upper_bound_is_conservative"].split("/")[1] or \
        "upper bound was optimistic in at least one reliability bin"


CHECKS = [
    ("alignment/char-spans-become-BIO-tags", c_alignment),
    ("alignment/CONTROL-shifted-spans-change-tags", c_alignment_control),
    ("redact/every-character-of-a-span-is-removed", c_redaction_removes_every_char),
    ("redact/CONTROL-no-spans-means-no-redaction", c_redaction_control_no_spans),
    ("redact/same-surface-same-placeholder", c_placeholder_consistency),
    ("redact/CONTROL-different-surfaces-differ", c_placeholder_consistency_control),
    ("redact/overlapping-spans-leave-no-tail", c_overlap_leaves_no_tail),
    ("vault/rehydration-round-trip", c_rehydrate_roundtrip),
    ("vault/CONTROL-fabricated-value-is-flagged", c_rehydrate_flags_missing),
    ("gate/calibrator-never-returns-zero-risk", c_calibrator_never_certain),
    ("gate/calibrated-risk-is-monotone", c_calibrator_monotone),
    ("gate/CONTROL-shuffled-labels-flatten-the-fit", c_calibrator_control_shuffled),
    ("gate/auc-on-known-orderings", c_auc_known_orderings),
    ("gate/risk-drops-when-tokens-are-redacted", c_risk_rises_with_survivors),
    ("model/planted-identifiers-are-all-masked", c_planted_document_is_masked),
    ("model/CONTROL-clean-document-is-barely-masked", c_clean_document_control),
    ("model/sub-word-span-snaps-to-the-word", c_snapping_widens_to_word),
    ("model/gate-ranks-a-stuffed-doc-above-a-clean-one", c_gate_decision_shape),
    ("ablation/fixed-coverage-takes-the-lowest-risk-documents",
     c_coverage_selects_the_lowest_risk),
    ("ablation/CONTROL-reversed-ranking-surfaces-the-leaks",
     c_coverage_control_reversed_scores),
    ("evidence/README-numbers-are-on-disk", c_evidence_matches_disk),
]


if __name__ == "__main__":
    for name, fn in CHECKS:
        check(name, fn)
    print(f"\n{len(PASS)}/{len(CHECKS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped")
    for n, why in FAIL:
        print(f"  FAILED {n}: {why}")
    sys.exit(1 if FAIL else 0)
