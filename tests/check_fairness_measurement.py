"""Controls for scripts/measure_fairness.py -- the rule 8 fairness measurement.

A disparity measurement is only worth its ability to (a) find a gap that IS
there and (b) stay quiet when there is none. A harness that reports a
significant gap on any input would produce exactly the same output on this
corpus and mean nothing, so both directions are exercised here, along with the
arithmetic underneath them and the instrument check that gates the whole run.

  Statistics      W1-W3  Wilson interval, checked against its defining equation
                         rather than against a second copy of the formula, and
                         required not to certify 10/10 as perfect
                  F1-F5  Fisher exact against a published reference value, its
                         own symmetries, scipy, and a degenerate margin
                  H1-H3  Holm: monotone, order-preserving, and able to REMOVE
                         significance rather than only relabel it

  Bucketing       B1-B5  ASCII / token-count / surface normalisation, and the
                         seen-in-training lookup including its type filter

  Scoring         S1-S4  leak semantics: partial masking is a leak, and a
                         predicted span of a non-redacted type is not coverage

  Calibration     P1     planted 100%-vs-0% gap MUST be found
                  P2     200 null draws at equal rates -- the significant
                         fraction must sit near alpha, not above it. Without
                         P2 every "significant" result in the evidence file
                         could be an artifact of the machinery
                  P3     swapping the arms flips the gap and preserves p
                  P4     a 40-point gap at n=5 must NOT reach significance, so
                         small n cannot manufacture a finding

  Instrument      I1     a CORRUPTED reference must make the run refuse and
                         exit non-zero
                  I2     the true reference must let it through, reporting
                         reproduction. I1 without I2 passes for a script that
                         always fails; I2 without I1 passes for one that never
                         checks

  Hygiene         E1     no identifier surface appears in the evidence file.
                         A bare substring search will not do: the corpus
                         annotates the word "person" as a name, so anything
                         also present in the script's own source is carved out
                         and REPORTED rather than dropped
                  E2     the E1 checker, given a planted surface, finds it --
                         otherwise E1 passes vacuously
                  E3     each carve-out is recoverable with the exemption off,
                         so the exemption is what suppressed it, not a blind
                         spot in the search

Each control states what it expects BEFORE it runs. The result is pinned to the
sha256 of the script it tested.
"""
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "measure_fairness.py")
EVIDENCE = os.path.join(REPO, "evidence", "fairness_controls.json")
FAIRNESS = os.path.join(REPO, "evidence", "fairness.json")
sys.path.insert(0, REPO)

# Imported through the `scripts` package rather than by putting scripts/ on the
# path: a bare `import measure_fairness` reads as a third-party top-level to the
# repository's own import-graph check, which walks every .py file here.
from scripts import measure_fairness as MF  # noqa: E402
from airlock.data import load_split  # noqa: E402

results = []
ROOT = tempfile.mkdtemp(prefix="upai_fair_ctrl_")


def check(name, expect, ok, detail=""):
    results.append((name, expect, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {expect}" + (f"  [{detail}]" if detail else ""))


def rows(n_caught, n_leaked, label="x"):
    """Build (doc, label, surface, leaked) tuples in the shape contrast() reads."""
    return ([(0, label, "s", False)] * n_caught) + ([(0, label, "s", True)] * n_leaked)


# ------------------------------------------------------------------ statistics

Z = 1.959963984540054

# W1: each Wilson endpoint must solve (phat - p)^2 == z^2 p(1-p)/n, which is the
# equation the interval is defined by. Comparing to a re-typed closed form would
# only test that it was typed twice the same way.
worst = 0.0
for k, n in ((0, 10), (1, 10), (3, 17), (5, 10), (9, 10), (37, 78), (198, 3449)):
    lo, hi = MF.wilson(k, n)
    phat = k / n
    for p in (lo, hi):
        if 0.0 < p < 1.0:
            worst = max(worst, abs((phat - p) ** 2 - Z * Z * p * (1 - p) / n))
check("W1", "every Wilson endpoint solves its defining equation", worst < 1e-12,
      f"max residual {worst:.3g}")

lo10, hi10 = MF.wilson(10, 10)
# The upper bound at k=n is analytically 1 but lands at 1-1.1e-16 in floating
# point, so it is checked to tolerance. The bound that matters is the lower one:
# ten out of ten must not read as certainty.
check("W2", "10/10 does not certify perfection: the lower bound stays well below 1 "
            "while the upper bound reaches 1 to floating-point tolerance",
      lo10 < 0.75 and hi10 > 1.0 - 1e-12, f"[{lo10:.4f}, {hi10:.17g}]")
check("W3", "n=0 returns (None, None) rather than a number",
      MF.wilson(0, 0) == (None, None))

# F1: Fisher's own tea-tasting table, published two-sided p = 0.4857142857...
p_tea = MF.fisher_exact_two_sided(3, 1, 1, 3)
check("F1", "tea-tasting table reproduces the published p = 0.485714",
      abs(p_tea - 0.4857142857142857) < 1e-9, f"p={p_tea:.10f}")

p_a = MF.fisher_exact_two_sided(12, 5, 3, 20)
check("F2", "p is invariant under transpose and under swapping both rows",
      abs(MF.fisher_exact_two_sided(12, 3, 5, 20) - p_a) < 1e-12
      and abs(MF.fisher_exact_two_sided(3, 20, 12, 5) - p_a) < 1e-12, f"p={p_a:.6g}")

try:
    from scipy.stats import fisher_exact as _sf
    rnd = random.Random(11)
    w = 0.0
    for _ in range(200):
        t = [[rnd.randint(0, 40) for _ in range(2)] for _ in range(2)]
        if min(t[0][0] + t[0][1], t[1][0] + t[1][1],
               t[0][0] + t[1][0], t[0][1] + t[1][1]) == 0:
            continue
        w = max(w, abs(float(_sf(t)[1]) - MF.fisher_exact_two_sided(t[0][0], t[0][1],
                                                                   t[1][0], t[1][1])))
    check("F3", "agrees with scipy on 200 random tables to 1e-9", w < 1e-9,
          f"max |diff| {w:.3g}")
except ImportError:
    check("F3", "agrees with scipy on 200 random tables", False, "scipy missing")

check("F4", "an empty arm gives p = 1.0, never a spurious finding",
      MF.fisher_exact_two_sided(0, 0, 10, 10) == 1.0)
check("F5", "a perfectly separated 25v25 table is overwhelmingly significant",
      MF.fisher_exact_two_sided(25, 0, 0, 25) < 1e-12,
      f"p={MF.fisher_exact_two_sided(25, 0, 0, 25):.3g}")

h = MF.holm({"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5, "e": 0.6, "f": 0.7, "g": 0.9})
order = [h[k] for k in ("a", "b", "c", "d", "e", "f", "g")]
check("H1", "adjusted p-values are non-decreasing in raw order",
      all(order[i] <= order[i + 1] + 1e-15 for i in range(len(order) - 1)), str(
          [round(x, 4) for x in order]))
check("H2", "the smallest raw p is multiplied by the family size",
      abs(h["a"] - 7 * 0.001) < 1e-12, f"{h['a']:.6f}")
check("H3", "Holm can REMOVE significance, not merely relabel it: raw 0.02 in a "
            "family of 7 crosses 0.05", 0.02 < 0.05 <= h["b"], f"adjusted {h['b']:.4f}")

# ------------------------------------------------------------------- bucketing

check("B1", "non-ASCII detection: Jose ASCII; Jose-with-acute, Cyrillic and a "
            "CJK name not ASCII; an apostrophe stays ASCII",
      MF.is_ascii("Jose") and not MF.is_ascii("José")
      and not MF.is_ascii("Ольга")
      and not MF.is_ascii("李娜") and MF.is_ascii("O'Brien"))
check("B2", "token buckets are 1 / 2 / 3+ and ignore padding whitespace",
      MF.token_bucket("Ada") == "1" and MF.token_bucket("Ada Lovelace") == "2"
      and MF.token_bucket("  Ada   Lovelace ") == "2"
      and MF.token_bucket("Ada B Lovelace") == "3+")
check("B3", "surface normalisation folds case and collapses whitespace",
      MF.normalise_surface("  ADA   Lovelace ") == MF.normalise_surface("ada lovelace")
      and MF.normalise_surface("Ada") != MF.normalise_surface("Adam"))


class _D:
    def __init__(self, text, spans):
        self.text, self.spans = text, spans


planted = [_D("call ada lovelace or ACME Ltd", [
    {"start": 5, "end": 17, "label": "name"},
    {"start": 21, "end": 25, "label": "company"},
])]
surf = MF.__dict__.get("surfaces_of")
if surf is None:  # surfaces_of is defined inside main(); re-derive it the same way
    def surf(dset):
        out = set()
        for d in dset:
            for sp in d.spans:
                if sp["label"] in MF.PERSON_NAME_TYPES:
                    out.add(MF.normalise_surface(d.text[sp["start"]:sp["end"]]))
        return out
got = surf(planted)
check("B4", "the seen-set finds a planted name and misses one that is absent",
      "ada lovelace" in got and "grace hopper" not in got, f"{sorted(got)}")
check("B5", "the seen-set's type filter holds: a `company` span is NOT collected "
            "as a person name", "acme" not in got and len(got) == 1, f"{sorted(got)}")

# --------------------------------------------------------------------- scoring

TXT = "pay 4111111111111111 to Ada Lovelace now"
gs, ge = TXT.index("Ada"), TXT.index("Ada") + len("Ada Lovelace")
doc = _D(TXT, [{"start": gs, "end": ge, "label": "name"}])
full = [{"start": gs, "end": ge, "label": "name"}]
part = [{"start": gs, "end": gs + 3, "label": "name"}]
wrong = [{"start": gs, "end": ge, "label": "company"}]
check("S1", "a fully masked gold span is not a leak",
      MF.leaked_flags([doc], [full], MF.IDENTIFYING)[0][3] is False)
check("S2", "a PARTIALLY masked gold span still counts as a leak",
      MF.leaked_flags([doc], [part], MF.IDENTIFYING)[0][3] is True)
check("S3", "an unmasked gold span is a leak",
      MF.leaked_flags([doc], [[]], MF.IDENTIFYING)[0][3] is True)
check("S4", "a predicted span whose type is not in the redaction policy provides "
            "no coverage", MF.leaked_flags([doc], [wrong], MF.IDENTIFYING)[0][3] is True)

# ----------------------------------------------------------------- calibration

c = MF.contrast("T", "planted", rows(0, 50), "all_leak", rows(50, 0), "none_leak")
check("P1", "a planted 100%-vs-0% gap is found, with the right sign",
      c["recall_gap_a_minus_b"] == -1.0 and c["fisher_exact_two_sided_p"] < 1e-12,
      f"gap={c['recall_gap_a_minus_b']} p={c['fisher_exact_two_sided_p']:.3g}")

rnd = random.Random(7)
RATE, N, DRAWS = 0.13, 400, 200
sig = 0
for _ in range(DRAWS):
    a = [(0, "x", "s", rnd.random() < RATE) for _ in range(N)]
    b = [(0, "x", "s", rnd.random() < RATE) for _ in range(N)]
    if MF.contrast("N", "null", a, "a", b, "b")["fisher_exact_two_sided_p"] < 0.05:
        sig += 1
check("P2", f"{DRAWS} null draws at an identical {RATE} leak rate stay near "
            f"alpha=0.05, not above it", sig <= 0.10 * DRAWS,
      f"{sig}/{DRAWS} significant = {sig / DRAWS:.3f}")

c2 = MF.contrast("T", "swapped", rows(50, 0), "none_leak", rows(0, 50), "all_leak")
check("P3", "swapping the arms flips the gap and leaves p unchanged",
      c2["recall_gap_a_minus_b"] == 1.0
      and abs(c2["fisher_exact_two_sided_p"] - c["fisher_exact_two_sided_p"]) < 1e-15)

c3 = MF.contrast("T", "tiny", rows(3, 2), "a", rows(5, 0), "b")
check("P4", "a 40-point gap on n=5 per arm does NOT reach significance -- small n "
            "cannot manufacture a finding",
      c3["fisher_exact_two_sided_p"] > 0.05,
      f"gap={c3['recall_gap_a_minus_b']} p={c3['fisher_exact_two_sided_p']:.4f}")

# ------------------------------------------------------------------ instrument

REF = os.path.join(REPO, "evidence", "detector_eval.json")
bad_ref = os.path.join(ROOT, "corrupt_detector_eval.json")
ref = json.load(open(REF))
ref["by_threshold_hybrid"]["0.2"]["per_type"]["last_name"]["leaked"] += 1
json.dump(ref, open(bad_ref, "w"))


def run_script(reference):
    p = subprocess.run(
        [sys.executable, SCRIPT, "--reference", reference,
         "--out", os.path.join(ROOT, "out.json")],
        capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


rc_bad, log_bad = run_script(bad_ref)
check("I1", "a corrupted reference makes the run REFUSE and exit non-zero",
      rc_bad != 0 and "instrument does not reproduce" in log_bad,
      f"rc={rc_bad}")

rc_ok, log_ok = run_script(REF)
ok_out = os.path.join(ROOT, "out.json")
d_ok = json.load(open(ok_out)) if rc_ok == 0 and os.path.exists(ok_out) else {}
check("I2", "the true reference passes, reporting reproduction over all 25 "
            "per-type cells", rc_ok == 0
      and d_ok.get("instrument_check", {}).get("reproduces_shipped_evaluation") is True
      and d_ok.get("instrument_check", {}).get("per_type_cells_compared") == 25,
      f"rc={rc_ok}")

# --------------------------------------------------------------------- hygiene

# A plain substring search is not the test, because the corpus annotates ordinary
# English words as names -- "person" is a gold name span in this split, and the
# evidence file says "person_name_types" for reasons that have nothing to do with
# the corpus. The discriminating property of a REAL export is that the string
# came out of the data and so does not occur in the script's own source. Anything
# occurring in both is the script's vocabulary and is reported as a carve-out
# rather than silently dropped, so the exemption stays visible.
SCRIPT_SRC = open(SCRIPT, encoding="utf-8").read()


def surfaces_in(text, corpus_surfaces, minlen=6):
    return sorted({s for s in corpus_surfaces if len(s) >= minlen and s in text})


def exported_surfaces(text, corpus_surfaces, minlen=6):
    return [s for s in surfaces_in(text, corpus_surfaces, minlen)
            if s not in SCRIPT_SRC]


test_docs = load_split("test")
corpus_surfaces = set()
for d in test_docs:
    for sp in d.spans:
        if sp["label"] in MF.PERSON_NAME_TYPES:
            corpus_surfaces.add(d.text[sp["start"]:sp["end"]])
blob = open(FAIRNESS, encoding="utf-8").read() if os.path.exists(FAIRNESS) else ""
raw_hits = surfaces_in(blob, corpus_surfaces)
hits = exported_surfaces(blob, corpus_surfaces)
carve_outs = [h for h in raw_hits if h not in hits]
check("E1", "no identifier surface from the corpus appears in the evidence file, "
            "other than strings that are also in the script's own source",
      blob != "" and not hits,
      f"{len(corpus_surfaces)} surfaces searched, {len(raw_hits)} substring hits, "
      f"{len(hits)} unexplained, carve-outs {carve_outs}")

sample = sorted(s for s in corpus_surfaces if len(s) >= 8 and s not in SCRIPT_SRC)[0]
planted = exported_surfaces(blob + " " + sample, corpus_surfaces)
# Planting a full name necessarily plants its parts: a surname that is itself a
# corpus surface will match too. So the requirement is that the planted string is
# found, was not being reported before, and that everything else newly reported
# is a substring of what was planted -- nothing spurious.
new_hits = [h for h in planted if h not in hits]
check("E2", "the E1 checker, handed a planted surface, finds it and had not "
            "already been reporting it -- so E1 is not passing vacuously",
      sample in planted and sample not in hits
      and all(h in sample for h in new_hits),
      f"planted a {len(sample)}-char surface; {len(new_hits)} new hit(s), all "
      f"substrings of it: {all(h in sample for h in new_hits)}")

# E3 exists because E1's carve-out could hide a real export behind a common word.
# It requires the carve-out to be doing real work only for script vocabulary: a
# surface that IS in the script source must still be caught when the exemption
# is switched off, proving the exemption is what suppressed it and not a hole in
# the search itself.
check("E3", "every carve-out is recoverable by the same search with the exemption "
            "off, so the exemption -- not a blind spot -- is what suppressed it",
      all(c in raw_hits for c in carve_outs) and all(c in SCRIPT_SRC for c in carve_outs),
      f"{len(carve_outs)} carve-out(s)")

# ------------------------------------------------------------------- reporting

shutil.rmtree(ROOT, ignore_errors=True)
n_pass = sum(1 for _, _, ok, _ in results if ok)
print(f"\n{n_pass}/{len(results)} controls passed  (sandbox {ROOT} removed)")

with open(SCRIPT, "rb") as fh:
    script_sha = hashlib.sha256(fh.read()).hexdigest()
with open(EVIDENCE, "w", encoding="utf-8") as fh:
    json.dump({
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": "Controls for scripts/measure_fairness.py -- the rule 8 fairness "
                "measurement. Both directions are exercised: a planted gap must be "
                "found, and 200 null draws must stay near alpha.",
        "script": "scripts/measure_fairness.py",
        "script_sha256": script_sha,
        "null_draws": DRAWS,
        "null_draw_rate": RATE,
        "null_draws_significant_at_05": sig,
        "null_false_positive_fraction": round(sig / DRAWS, 4),
        "corpus_surfaces_searched_in_evidence": len(corpus_surfaces),
        "evidence_substring_hits": raw_hits,
        "evidence_unexplained_surfaces": hits,
        "evidence_carve_outs_script_vocabulary": carve_outs,
        "controls": [{"name": n, "expects": e, "passed": ok, "detail": d}
                     for n, e, ok, d in results],
        "passed": n_pass,
        "total": len(results),
    }, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"wrote {EVIDENCE}")
sys.exit(0 if n_pass == len(results) else 1)
