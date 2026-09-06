"""Controls for scripts/ablate_learned_model.py -- the rule 3 counterfactual.

The finding this apparatus produces is "delete the learned model and the gate
cannot recover". That finding is worthless unless the harness can be shown to
recognise a GOOD ranking when it is handed one -- otherwise "the no-ML gate
scored badly" is indistinguishable from "the scorer scores everything badly".
So the first two controls hand it a perfect ranking and a perfectly wrong one.

  A1 ORACLE       rank by the true leak label -> AUC 1.0 and ZERO leaks among
                  the forwarded set. The harness can express a perfect gate.
  A2 ANTI-ORACLE  rank by the negated label -> AUC 0.0 and the forwarded set
                  saturated with leaks. Fixes the direction convention: the k
                  LOWEST-scoring documents are the ones forwarded.
  A3 REPRODUCTION arm S must reproduce evidence/gate_eval.json and
                  evidence/posterior_ablation.json to the digit, by a second
                  code path. Paired with a perturbation that must NOT match.
  A4 MASK         no_ml_statistics with no spans must count the whole document;
                  with the document fully spanned it must count nothing. Both
                  sides, because a mask stuck at 0 passes one of them.
  A5 LEAK LABEL   leaked() must fire on an annotated identifier left in the
                  clear and must not fire when it is covered.
  A6 COVERAGE     leak_rate_at_coverage forwards exactly k;
                  max_coverage_within_exposure stops at the document that
                  exceeds the budget, not after it.
  A7 ORACLE PICK  pick_best_no_ml must really choose the best of twelve,
                  including a NEGATED candidate. A picker that quietly failed
                  to consider sign would understate the competitor and flatter
                  this project, so it is tested in the direction that hurts us.
  A8 LABELS MOVE  arm B's leak labels must differ from arm S's. A leak is
                  defined against the redaction that arm performed; if the two
                  label sets were identical the arms would be scoring the wrong
                  thing.
  A9 ADVANTAGE    the chosen no-ML statistic must be at least as good as every
                  other candidate reported, so the "best of twelve" claim is
                  checked against the table it is drawn from.

Each control states what it expects before it runs. No document text is read.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from airlock.data import Doc                                   # noqa: E402
from airlock.detect import Span                                # noqa: E402
from airlock.labels import IDENTIFYING                         # noqa: E402
from scripts.eval_gate import auc                              # noqa: E402
from scripts.fit_gate import leaked                            # noqa: E402
from scripts.ablate_learned_model import (                     # noqa: E402
    NO_ML_NAMES, leak_rate_at_coverage, leaks_at_coverage,
    max_coverage_within_exposure, no_ml_statistics, pick_best_no_ml,
)

SCRIPT = os.path.join(REPO, "scripts", "ablate_learned_model.py")
ABLATION = os.path.join(REPO, "evidence", "learned_model_ablation.json")
GATE_EVAL = os.path.join(REPO, "evidence", "gate_eval.json")
POSTERIOR = os.path.join(REPO, "evidence", "posterior_ablation.json")
EVIDENCE = os.path.join(REPO, "evidence", "learned_model_ablation_controls.json")

results = []


def check(name, expect, ok, detail=""):
    results.append((name, expect, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {expect}" + (f"  [{detail}]" if detail else ""))


# --------------------------------------------------------------- A1 / A2
# A synthetic corpus with the same shape as the real one: 2891 documents,
# 10.03% of them leaking. The labels are the only thing these two controls need.
N = 2891
N_LEAK = 290
labels = [i < N_LEAK for i in range(N)]
K = int(round(0.6365 * N))

oracle = [1.0 if l else 0.0 for l in labels]          # leakers score highest
check("A1 oracle auc", "a perfect ranking scores AUC 1.0",
      auc(oracle, labels) == 1.0, f"auc={auc(oracle, labels)}")
check("A1 oracle gate", f"a perfect ranking forwards {K} documents with 0 leaks",
      leaks_at_coverage(oracle, labels, K) == 0 and leak_rate_at_coverage(oracle, labels, K) == 0.0,
      f"leaks={leaks_at_coverage(oracle, labels, K)}")

anti = [0.0 if l else 1.0 for l in labels]            # leakers score lowest
check("A2 anti-oracle auc", "a perfectly wrong ranking scores AUC 0.0",
      auc(anti, labels) == 0.0, f"auc={auc(anti, labels)}")
check("A2 anti-oracle gate", "a perfectly wrong ranking forwards every leaker first",
      leaks_at_coverage(anti, labels, K) == N_LEAK,
      f"leaks={leaks_at_coverage(anti, labels, K)} of {N_LEAK}")

# --------------------------------------------------------------- A3
abl = json.load(open(ABLATION, encoding="utf-8"))
ge = json.load(open(GATE_EVAL, encoding="utf-8"))
po = json.load(open(POSTERIOR, encoding="utf-8"))
armS = next(a for a in abl["arms"] if a["arm"] == "S_shipped")
cov0 = "%.4f" % abl["coverages_compared"][0]

ge_base = ge["forward_everything_leak_rate"]
check("A3 base rate", f"arm S base leak rate reproduces gate_eval.json ({ge_base})",
      armS["base_leak_rate_all_documents"] == ge_base,
      f"ablation={armS['base_leak_rate_all_documents']} gate_eval={ge_base}")
check("A3 base rate control", "a perturbed base rate must NOT match",
      (armS["base_leak_rate_all_documents"] + 0.0001) != ge_base)

po_post = next(r for r in po["statistics"] if r["statistic"] == "posterior")
po_key = "leak_rate_at_coverage_" + cov0
check("A3 auc", f"arm S ranking AUC reproduces posterior_ablation.json AND\n      gate_eval.json ({po_post['auc']})",
      armS["ranking_auc"] == po_post["auc"] == ge["ranking_auc_raw_statistic"],
      f"ablation={armS['ranking_auc']} posterior={po_post['auc']} "
      f"gate_eval={ge['ranking_auc_raw_statistic']}")
check("A3 leak rate", f"arm S leak rate at coverage {cov0} reproduces "
      f"posterior_ablation.json ({po_post[po_key]})",
      armS["at_coverage"][cov0]["leak_rate_among_forwarded"] == po_post[po_key],
      f"ablation={armS['at_coverage'][cov0]['leak_rate_among_forwarded']} "
      f"posterior={po_post[po_key]}")
check("A3 auc control", "a perturbed AUC must NOT match",
      round(armS["ranking_auc"] + 0.01, 4) != po_post["auc"])
check("A3 documents", f"all three arms score the same {abl['documents']} documents",
      all(a["documents"] == abl["documents"] for a in abl["arms"]) and
      abl["documents"] == po["documents"],
      f"arms={[a['documents'] for a in abl['arms']]} posterior={po['documents']}")

# --------------------------------------------------------------- A4
text = "Meera Subramanian paid 4111 1111 1111 1111 on 2026-01-02 at Acme Ltd today."
open_stats = no_ml_statistics(text, [])
full = [Span(start=0, end=len(text), label="name", score=1.0)]
closed_stats = no_ml_statistics(text, full)
check("A4 unmasked", "with no spans, every surviving count is positive",
      open_stats["surviving_chars"] == len(text) and open_stats["surviving_words"] > 0
      and open_stats["surviving_capitalised_words"] > 0
      and open_stats["surviving_digit_runs"] > 0,
      json.dumps({k: v for k, v in open_stats.items() if k != "spans_redacted"}))
check("A4 fully masked", "with the whole document spanned, every surviving count is 0",
      closed_stats["surviving_chars"] == 0 and closed_stats["surviving_words"] == 0
      and closed_stats["surviving_capitalised_words"] == 0
      and closed_stats["surviving_digit_runs"] == 0,
      json.dumps({k: v for k, v in closed_stats.items() if k != "spans_redacted"}))
check("A4 span count", "spans_redacted counts the spans it was given",
      open_stats["spans_redacted"] == 0.0 and closed_stats["spans_redacted"] == 1.0)
check("A4 statistic set", f"exactly {len(NO_ML_NAMES)} no-ML statistics are computed",
      len(NO_ML_NAMES) == 6 and set(open_stats) == set(NO_ML_NAMES),
      ",".join(NO_ML_NAMES))

# --------------------------------------------------------------- A5
ident = sorted(IDENTIFYING)[0]
doc = Doc(text=text, spans=[{"start": 0, "end": 17, "label": ident}])
uncovered = leaked(doc, [], set(IDENTIFYING))
covered = leaked(doc, [Span(start=0, end=17, label=ident, score=1.0)], set(IDENTIFYING))
check("A5 leak fires", "an annotated identifier left in the clear is a leak", uncovered is True)
check("A5 leak clears", "the same identifier, covered, is not a leak", covered is False)
partial = leaked(doc, [Span(start=0, end=9, label=ident, score=1.0)], set(IDENTIFYING))
check("A5 partial", "half-covering an identifier is still a leak", partial is True)

# --------------------------------------------------------------- A6
# scores 0..9, documents 3 and 7 leak. Budget of 1 leaking document must stop
# at k=7 -- the moment the SECOND leaker would be forwarded, not after it.
s6 = [float(i) for i in range(10)]
l6 = [i in (3, 7) for i in range(10)]
check("A6 exact k", "leak_rate_at_coverage scores exactly k documents",
      leaks_at_coverage(s6, l6, 5) == 1 and leak_rate_at_coverage(s6, l6, 5) == 0.2,
      f"leaks={leaks_at_coverage(s6, l6, 5)}")
check("A6 budget stop", "max_coverage_within_exposure stops before exceeding the budget",
      max_coverage_within_exposure(s6, l6, 1) == 7, f"k={max_coverage_within_exposure(s6, l6, 1)}")
check("A6 budget zero", "a budget of 0 leaking documents stops at the first leaker",
      max_coverage_within_exposure(s6, l6, 0) == 3, f"k={max_coverage_within_exposure(s6, l6, 0)}")
check("A6 budget all", "a budget larger than the corpus forwards everything",
      max_coverage_within_exposure(s6, l6, 99) == 10)

# --------------------------------------------------------------- A7
# A statistic that is perfectly ANTI-correlated with leaking. The picker must
# choose its negated form -- if it cannot flip sign it hands the no-ML arms a
# worse statistic than they really have, which flatters this project.
n7 = 200
l7 = [i < 40 for i in range(n7)]
planted = {k: [0.0] * n7 for k in NO_ML_NAMES}
planted[NO_ML_NAMES[0]] = [0.0 if l else 1.0 for l in l7]      # anti-correlated
rows7, best7 = pick_best_no_ml(planted, l7, [0.6365], n7)
check("A7 sign flip", "the picker chooses the NEGATED form of an anti-correlated statistic",
      best7["statistic"] == NO_ML_NAMES[0] + " (negated)" and best7["auc"] == 1.0,
      f"chose={best7['statistic']} auc={best7['auc']}")
check("A7 twelve candidates", "twelve candidates are scored -- six statistics, both signs",
      len(rows7) == 12, f"rows={len(rows7)}")
check("A7 picks the max", "the chosen candidate has the highest AUC of the twelve",
      all((r["auc"] is None) or r["auc"] <= best7["auc"] for r in rows7))

# --------------------------------------------------------------- A8 / A9
armB = next(a for a in abl["arms"] if a["arm"] == "B_no_learned_model")
armR = next(a for a in abl["arms"] if a["arm"] == "R_redaction_api")
check("A8 labels move", "arm B's base leak rate differs from arm S's -- a leak is "
      "defined against the redaction that arm performed",
      armB["base_leak_rate_all_documents"] != armS["base_leak_rate_all_documents"],
      f"B={armB['base_leak_rate_all_documents']} S={armS['base_leak_rate_all_documents']}")
check("A8 labels held", "arm R redacts identically to arm S, so its base rate is identical",
      armR["base_leak_rate_all_documents"] == armS["base_leak_rate_all_documents"],
      f"R={armR['base_leak_rate_all_documents']}")

for arm in (armB, armR):
    aucs = [r["auc"] for r in arm["candidates"] if r["auc"] is not None]
    chosen = next(r for r in arm["candidates"] if r["statistic"] == arm["oracle_statistic_chosen"])
    check(f"A9 best of twelve ({arm['arm']})",
          "the reported statistic is the highest-AUC candidate in its own table",
          len(arm["candidates"]) == 12 and chosen["auc"] == max(aucs),
          f"chose={arm['oracle_statistic_chosen']} auc={chosen['auc']} max={max(aucs)}")
    check(f"A9 arm auc matches ({arm['arm']})",
          "the arm's headline AUC is the chosen candidate's AUC",
          arm["ranking_auc"] == chosen["auc"],
          f"arm={arm['ranking_auc']} candidate={chosen['auc']}")

n_pass = sum(1 for _, _, ok, _ in results if ok)
print(f"\n{n_pass}/{len(results)} controls passed")

with open(SCRIPT, "rb") as fh:
    script_sha = hashlib.sha256(fh.read()).hexdigest()
with open(EVIDENCE, "w", encoding="utf-8") as fh:
    json.dump({
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": "Controls for scripts/ablate_learned_model.py -- the rule 3 "
                "counterfactual. A1 and A2 hand the harness a perfect ranking and a "
                "perfectly wrong one, so 'the no-ML gate scored badly' can be told "
                "apart from 'the scorer scores everything badly'. A7 is run in the "
                "direction that hurts this project.",
        "script": "scripts/ablate_learned_model.py",
        "script_sha256": script_sha,
        "controls": [{"name": n, "expects": e, "passed": ok, "detail": d}
                     for n, e, ok, d in results],
        "passed": n_pass,
        "total": len(results),
    }, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"wrote {EVIDENCE}")
sys.exit(0 if n_pass == len(results) else 1)
