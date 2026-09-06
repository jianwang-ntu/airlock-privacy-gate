"""Delete the learned model. Does the product still work?

Rule 3 of UPAI-Hackdays: *"AI/ML must play a meaningful role in the submitted
solution ... Simply adding an AI chatbot or API without meaningful integration
is discouraged."* That is a counterfactual claim, so it is answered with a
counterfactual rather than with prose.

This repository already reports that the no-ML *detector* is far worse than the
learned one (38.45% identifier recall against 93.96%). That is not the same
claim, and using it to answer rule 3 was a gap. **The product is not the
detector, it is the gate**, and the fair objection is that a no-ML pipeline
would simply *withhold* the documents it cannot handle -- the gate is supposed
to absorb a weak detector. Nobody had ever run that pipeline. This script runs
it end to end.

Three arms, the same 2,891 held-out test documents, the same leak definition:

  S  shipped           model union validators redaction; risk = sum of p_ident
                       over the tokens that survived redaction
  B  no model anywhere regex union validators redaction; risk = best no-ML
                       statistic. This is the whole product with the learning
                       deleted.
  R  redaction-API     model union validators redaction; risk = best no-ML
                       statistic computed from the redacted OUTPUT alone. The
                       model is used as a black box -- exactly what a commercial
                       redaction API hands you -- and the gate is rebuilt on top
                       of it without any posterior.

Arm S is scored on the one statistic it ships. Arms B and R are allowed to pick
their risk statistic from TWELVE candidates -- six no-ML statistics and both
signs of each -- **by looking at the test labels**. That is an oracle they would
not have in production, and it is granted deliberately: the finding has to
survive the competitor being given more than it could really have.

Every arm carries its own leak labels, because a leak is defined against the
redaction that arm performed. Base rate, ranked leak rate and whole-corpus
exposure are all reported per arm so the redaction effect and the ranking effect
can be read apart instead of being conflated into one headline.

Writes evidence/learned_model_ablation.json. No document text, no span text and
no identifier value is read into any output: only counts, rates and AUCs leave
this script.
"""
import argparse
import json
import os
import random
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airlock.baseline import RegexBaseline
from airlock.data import load_split
from airlock.gate import surviving_tokens
from airlock.hybrid import HybridDetector, merge_spans
from airlock.labels import IDENTIFYING
from airlock.validators import ChecksumValidators
from scripts.eval_gate import auc
from scripts.fit_gate import leaked


# ------------------------------------------------------------------ no-ML statistics
WORD_RE = re.compile(r"\S+")
CAP_RE = re.compile(r"^[A-Z][a-z]{1,20}$")
DIGITS_RE = re.compile(r"\d{2,}")


def _mask(text, spans):
    """1 where a character is inside a redacted span."""
    n = len(text)
    m = bytearray(n)
    for s in spans:
        d = s if isinstance(s, dict) else s.as_dict()
        for i in range(max(0, d["start"]), min(n, d["end"])):
            m[i] = 1
    return m


def no_ml_statistics(text, spans):
    """Six risk statistics a team with no model could compute from the redacted
    document. Higher is meant to be riskier. Nothing here touches the detector's
    probabilities -- only the text and the spans that were masked.

    These are the honest no-ML options: how much document survived, how many
    capitalised words survived (the classic 'a name we missed' heuristic), how
    many digit runs survived, and how much PII was found (a document full of
    identifiers probably holds more).
    """
    m = _mask(text, spans)
    surviving_chars = len(m) - sum(m)
    words = 0
    caps = 0
    digits = 0
    for w in WORD_RE.finditer(text):
        a, b = w.start(), w.end()
        if all(m[a:b]):          # entirely redacted
            continue
        words += 1
        tok = w.group().strip(".,;:!?()[]\"'")
        if CAP_RE.match(tok):
            caps += 1
        if DIGITS_RE.search(tok):
            digits += 1
    return {
        "surviving_chars": float(surviving_chars),
        "surviving_words": float(words),
        "surviving_capitalised_words": float(caps),
        "surviving_digit_runs": float(digits),
        "spans_redacted": float(len(spans)),
        "surviving_capitalised_plus_digits": float(caps + digits),
    }


NO_ML_NAMES = list(no_ml_statistics("", []).keys())


# ------------------------------------------------------------------ scoring helpers
def leak_rate_at_coverage(scores, labels, k):
    """Leak rate among the k lowest-scoring documents. Ties broken by index --
    arrival order, deterministic and independent of the score."""
    if not k:
        return None
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))[:k]
    return sum(1 for i in order if labels[i]) / k


def leaks_at_coverage(scores, labels, k):
    if not k:
        return 0
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))[:k]
    return sum(1 for i in order if labels[i])


def max_coverage_within_exposure(scores, labels, exposure_budget_count):
    """Largest k whose forwarded set contains no more than `exposure_budget_count`
    leaking documents. The practical question: how much can this arm forward and
    still be as safe, in absolute terms, as the shipped arm?"""
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    seen = 0
    for k, i in enumerate(order):
        if labels[i]:
            seen += 1
            if seen > exposure_budget_count:
                return k
    return len(order)


def random_gate(labels, k, seeds=20):
    n = len(labels)
    vals = []
    for seed in range(seeds):
        r = random.Random(seed)
        pick = r.sample(range(n), k)
        vals.append(sum(1 for i in pick if labels[i]) / k)
    return statistics.mean(vals), statistics.pstdev(vals)


def pick_best_no_ml(stats, labels, coverages, n):
    """Oracle choice: every no-ML statistic, both signs, scored against the test
    labels; the one with the highest AUC wins. Reported in full so the size of
    the advantage granted is visible, not just its result."""
    rows = []
    for name in NO_ML_NAMES:
        for sign, tag in ((1.0, ""), (-1.0, " (negated)")):
            s = [sign * v for v in stats[name]]
            a = auc(s, labels)
            row = {"statistic": name + tag, "auc": round(a, 4) if a is not None else None,
                   "distinct_values": len(set(s))}
            for c in coverages:
                k = int(round(c * n))
                row["leak_rate_at_coverage_%.4f" % c] = round(leak_rate_at_coverage(s, labels, k), 4)
            rows.append(row)
    best = max(rows, key=lambda r: (r["auc"] if r["auc"] is not None else -1))
    return rows, best


def arm_report(name, description, scores, labels, coverages, n, shipped_exposure_counts=None):
    base_leaks = sum(labels)
    a = auc(scores, labels)
    out = {
        "arm": name,
        "description": description,
        "documents": n,
        "base_leak_rate_all_documents": round(base_leaks / n, 4),
        "base_leaking_documents": base_leaks,
        "ranking_auc": round(a, 4) if a is not None else None,
        "at_coverage": {},
    }
    for c in coverages:
        k = int(round(c * n))
        lk = leaks_at_coverage(scores, labels, k)
        rmean, rsd = random_gate(labels, k)
        out["at_coverage"]["%.4f" % c] = {
            "documents_forwarded": k,
            "leaking_documents_forwarded": lk,
            "leak_rate_among_forwarded": round(lk / k, 4),
            "whole_corpus_exposure": round(lk / n, 4),
            "random_gate_same_coverage_leak_rate": round(rmean, 4),
            "random_gate_sd": round(rsd, 4),
        }
    if shipped_exposure_counts:
        out["coverage_needed_to_match_shipped_exposure"] = {}
        for c, budget in shipped_exposure_counts.items():
            k = max_coverage_within_exposure(scores, labels, budget)
            out["coverage_needed_to_match_shipped_exposure"][c] = {
                "shipped_leaking_documents_forwarded": budget,
                "max_documents_this_arm_can_forward": k,
                "max_coverage": round(k / n, 4),
            }
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--coverages", default="0.6365,0.0948")
    ap.add_argument("--out", default="evidence/learned_model_ablation.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    covs = [float(x) for x in args.coverages.split(",")]
    docs = load_split("test", limit=args.limit)
    n = len(docs)
    redact_types = set(IDENTIFYING)

    det = HybridDetector(args.model)
    base = RegexBaseline()
    val = ChecksumValidators()

    shipped_scores, shipped_labels = [], []
    api_stats = {k: [] for k in NO_ML_NAMES}
    nom_stats = {k: [] for k in NO_ML_NAMES}
    nom_labels = []

    for i in range(0, n, 16):
        chunk = docs[i:i + 16]
        for doc, toks in zip(chunk, det._token_scores([d.text for d in chunk])):
            # ---- arms S and R share one forward pass and one set of spans
            spans = det._decode(doc.text, toks, args.threshold)
            surv = surviving_tokens(doc.text, toks, spans)
            shipped_scores.append(sum(p for (_a, _b, p, _l) in surv))
            shipped_labels.append(leaked(doc, spans, redact_types))
            for k, v in no_ml_statistics(doc.text, spans).items():
                api_stats[k].append(v)

            # ---- arm B: the model never runs
            b_spans = base.find(doc.text)
            proven = val.find(doc.text)
            b_merged = merge_spans(list(b_spans) + list(proven),
                                   priority={s.label for s in proven})
            nom_labels.append(leaked(doc, b_merged, redact_types))
            for k, v in no_ml_statistics(doc.text, b_merged).items():
                nom_stats[k].append(v)

    # ---- arm S -------------------------------------------------------------
    shipped = arm_report(
        "S_shipped",
        "Model union validators redaction; risk = sum of the detector's per-token "
        "p_ident over the tokens that survived redaction. One statistic, no choice.",
        shipped_scores, shipped_labels, covs, n)
    shipped_exposure_counts = {
        "%.4f" % c: shipped["at_coverage"]["%.4f" % c]["leaking_documents_forwarded"]
        for c in covs}

    # ---- arm B -------------------------------------------------------------
    b_rows, b_best = pick_best_no_ml(nom_stats, nom_labels, covs, n)
    b_sign = -1.0 if b_best["statistic"].endswith("(negated)") else 1.0
    b_name = b_best["statistic"].replace(" (negated)", "")
    b_scores = [b_sign * v for v in nom_stats[b_name]]
    no_model = arm_report(
        "B_no_learned_model",
        "The learned model never runs. Regex union checksum validators redaction, and "
        "the best of twelve no-ML risk statistics chosen against the test labels.",
        b_scores, nom_labels, covs, n, shipped_exposure_counts)
    no_model["oracle_statistic_chosen"] = b_best["statistic"]
    no_model["candidates"] = b_rows

    # ---- arm R -------------------------------------------------------------
    r_rows, r_best = pick_best_no_ml(api_stats, shipped_labels, covs, n)
    r_sign = -1.0 if r_best["statistic"].endswith("(negated)") else 1.0
    r_name = r_best["statistic"].replace(" (negated)", "")
    r_scores = [r_sign * v for v in api_stats[r_name]]
    api = arm_report(
        "R_redaction_api",
        "The model is used as a black-box redactor -- what a commercial redaction API "
        "returns -- and the gate is rebuilt from the redacted output with the best of "
        "twelve no-ML risk statistics chosen against the test labels. Identical "
        "redaction to arm S, so the leak labels are identical too, and the ONLY "
        "difference from S is that the posterior is unavailable.",
        r_scores, shipped_labels, covs, n, shipped_exposure_counts)
    api["oracle_statistic_chosen"] = r_best["statistic"]
    api["candidates"] = r_rows

    head = "%.4f" % covs[0]
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": ("Rule 3 asks that AI/ML play a meaningful role. Delete the learned "
                     "model from the PRODUCT -- not from the detector benchmark -- and "
                     "measure what the gate can still do."),
        "split": "test (never used for training or calibration)",
        "documents": n,
        "redaction_threshold": args.threshold,
        "coverages_compared": covs,
        "leak_definition": ("An identifier the corpus annotates as one of the 25 IDENTIFYING "
                            "types survived into the outgoing text. Each arm is scored against "
                            "the redaction that arm performed, which is why the base rates differ."),
        "arms": [shipped, no_model, api],
        "headline": {
            "coverage": covs[0],
            "shipped_leak_rate_among_forwarded": shipped["at_coverage"][head]["leak_rate_among_forwarded"],
            "no_learned_model_leak_rate_among_forwarded": no_model["at_coverage"][head]["leak_rate_among_forwarded"],
            "redaction_api_leak_rate_among_forwarded": api["at_coverage"][head]["leak_rate_among_forwarded"],
            "shipped_whole_corpus_exposure": shipped["at_coverage"][head]["whole_corpus_exposure"],
            "no_learned_model_whole_corpus_exposure": no_model["at_coverage"][head]["whole_corpus_exposure"],
            "redaction_api_whole_corpus_exposure": api["at_coverage"][head]["whole_corpus_exposure"],
            "no_learned_model_max_coverage_at_shipped_exposure":
                no_model["coverage_needed_to_match_shipped_exposure"][head]["max_coverage"],
            "redaction_api_max_coverage_at_shipped_exposure":
                api["coverage_needed_to_match_shipped_exposure"][head]["max_coverage"],
        },
        "advantage_granted_to_the_no_ml_arms": (
            "Arms B and R choose their risk statistic from twelve candidates -- six "
            "statistics and both signs of each -- by looking at the test labels. Arm S is "
            "scored on the single statistic it ships, with no choice and no sign flip. The "
            "comparison is therefore biased in favour of the no-ML arms."),
        "what_this_does_not_show": (
            "It does not show that no no-ML gate could ever work: only the six statistics "
            "named here were tried, and a team with more time might find a better one. It "
            "does not vary the redaction threshold. Every leak rate is scored against the "
            "corpus's own annotation, which is incomplete -- the checksum validators find "
            "email addresses it does not mark -- so all of them are lower bounds on the true "
            "rate, in every arm equally. Arm B's base rate is high because its redaction is "
            "bad; the ranking effect is read from the base-rate-to-forwarded-rate drop within "
            "each arm, not from comparing arms' forwarded rates alone. Single run, one corpus, "
            "one language, no confidence intervals."),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    for a in out["arms"]:
        print(json.dumps({k: v for k, v in a.items() if k != "candidates"}, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
