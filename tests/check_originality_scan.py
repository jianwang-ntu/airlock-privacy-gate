"""Controls for scripts/measure_originality.py -- the rule 2 copy search.

A search that reports "no copy found" is worth exactly as much as its ability
to find a copy that IS there. This exercises all three of the scanner's
outcomes, against trees that did not exist before this run:

  C1 ACCEPT     a root holding only unrelated material -> NO_COPY_FOUND
  C2 EXACT      a byte-identical copy of a tracked file planted on the root
                -> REVIEW_REQUIRED, naming that file
  C3 NEAR       a copy of a tracked file with ~1 line in 6 rewritten, planted
                where EXACT cannot see it -> REVIEW_REQUIRED, flagged
  C4 DEGRADED   roots that do not exist -> NOT_RUN, never a pass
  C5 LICENCE    the MIT licence text alone must NOT flip the verdict, because
                the licence is not our authorship -- but it must still appear
                in `matches`, so the carve-out is visible rather than silent
  C6 SHORTLINE  a file of only short lines cannot reach the near threshold by
                boilerplate alone
  C7 SELF       a copy sitting inside THIS job's own workspace is reported but
                does not flip the verdict -- it is not a prior project
  C8 SELF-PAIR  the SAME copy, with the self-workspace pointed elsewhere, DOES
                flip it. Without C8, C7 would pass for a scanner that had
                simply stopped looking

Each control states what it expects BEFORE running, and a control that passes
for the wrong reason is reported as a defect rather than counted.
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCANNER = os.path.join(REPO, "scripts", "measure_originality.py")
EVIDENCE = os.path.join(REPO, "evidence", "originality_controls.json")

ROOT = tempfile.mkdtemp(prefix="upai_orig_ctrl_")
OUT = os.path.join(ROOT, "out.json")

results = []


def run(roots, focus, selfdir=None):
    cmd = [sys.executable, SCANNER, "--roots", *roots, "--focus", *focus, "--out", OUT]
    if selfdir:
        cmd += ["--self", selfdir]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if p.returncode != 0:
        return None, p.returncode, p.stdout + p.stderr
    return json.load(open(OUT, encoding="utf-8")), p.returncode, p.stdout


def check(name, expect, ok, detail=""):
    results.append((name, expect, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {expect}" + (f"  [{detail}]" if detail else ""))


# --- C1: a root with only unrelated material must ACCEPT -------------------
clean = os.path.join(ROOT, "clean")
os.makedirs(clean)
with open(os.path.join(clean, "unrelated.py"), "w") as fh:
    fh.write("def totally_unrelated_helper(alpha, beta):\n"
             "    accumulator = alpha * beta + 17\n"
             "    return accumulator / (alpha + beta + 1)\n" * 40)
d, rc, out = run([clean], [clean])
check("C1 accept", "unrelated root -> NO_COPY_FOUND",
      d and d["status"] == "NO_COPY_FOUND" and d["search_exact"]["files_walked"] > 0
      and d["search_near_duplicate"]["candidates_walked"] > 0,
      f"status={d and d['status']} walked={d and d['search_exact']['files_walked']}")

# --- C2: a verbatim copy must be found -------------------------------------
exact_root = os.path.join(ROOT, "prior_project_exact")
os.makedirs(exact_root)
shutil.copyfile(os.path.join(REPO, "airlock", "gate.py"),
                os.path.join(exact_root, "risk_gate.py"))
d, rc, out = run([exact_root], [clean])
hit = d and any(m["tracked_file"] == "airlock/gate.py" for m in d["search_exact"]["matches"])
check("C2 exact", "verbatim copy -> REVIEW_REQUIRED naming airlock/gate.py",
      d and d["status"] == "REVIEW_REQUIRED" and hit,
      f"status={d and d['status']} matches={d and d['search_exact']['match_count']}")

# --- C3: an edited copy must be found by NEAR, and not by EXACT ------------
near_root = os.path.join(ROOT, "prior_project_near")
os.makedirs(near_root)
src = open(os.path.join(REPO, "airlock", "detect.py"), encoding="utf-8").read().splitlines()
random.seed(11)
edited = []
rewritten = 0
for i, line in enumerate(src):
    if len(line.strip()) >= 8 and i % 6 == 0:
        edited.append("    # rewritten line %d by the control" % i)
        rewritten += 1
    else:
        edited.append(line)
open(os.path.join(near_root, "pii_detect.py"), "w", encoding="utf-8").write("\n".join(edited))
d, rc, out = run([near_root], [near_root])
near = d["search_near_duplicate"] if d else {}
flagged = [h for h in near.get("hits", [])
           if h["tracked_file"] == "airlock/detect.py" and h["flagged_probable_copy"]]
check("C3 near", f"{rewritten} lines rewritten -> flagged as probable copy",
      d and d["status"] == "REVIEW_REQUIRED" and flagged,
      f"status={d and d['status']} jaccard={flagged[0]['jaccard'] if flagged else None}")
check("C3b near-only", "the edited copy is NOT byte-identical, so EXACT must miss it",
      d and d["search_exact"]["match_count"] == 0,
      f"exact_matches={d and d['search_exact']['match_count']}")

# --- C4: absent roots must degrade to NOT_RUN, never to a pass -------------
absent = os.path.join(ROOT, "does_not_exist")
d, rc, out = run([absent], [absent])
check("C4 degraded", "absent roots -> NOT_RUN (not a pass)",
      d and d["status"] == "NOT_RUN" and d["search_exact"]["files_walked"] == 0,
      f"status={d and d['status']}")

# --- C5: the MIT licence must be visible but must not flip the verdict -----
lic_root = os.path.join(ROOT, "someone_elses_mit_project")
os.makedirs(lic_root)
shutil.copyfile(os.path.join(REPO, "LICENSE"), os.path.join(lic_root, "LICENSE"))
d, rc, out = run([lic_root], [clean])
lic_match = d and [m for m in d["search_exact"]["matches"] if m["tracked_file"] == "LICENSE"]
check("C5 licence carve-out", "MIT licence text appears in matches but keeps NO_COPY_FOUND",
      d and d["status"] == "NO_COPY_FOUND" and lic_match,
      f"status={d and d['status']} licence_matches={len(lic_match or [])}")

# --- C6: boilerplate short lines must not reach the near threshold ---------
short_root = os.path.join(ROOT, "short_lines")
os.makedirs(short_root)
open(os.path.join(short_root, "boiler.py"), "w").write(
    "\n".join(["import os", "import sys", "import json", "return", "pass",
               "if x:", "else:", "try:", ")", "}", "]"] * 60))
d, rc, out = run([short_root], [short_root])
check("C6 short-line floor", "a file of only short lines cannot be flagged",
      d and d["status"] == "NO_COPY_FOUND"
      and d["search_near_duplicate"]["flagged_count"] == 0,
      f"status={d and d['status']}")

# --- C7/C8: the self-workspace carve-out, both ways ------------------------
fake_ws = os.path.join(ROOT, "fake_workspace")
os.makedirs(os.path.join(fake_ws, "sub"))
shutil.copyfile(os.path.join(REPO, "airlock", "redact.py"),
                os.path.join(fake_ws, "sub", "redact.py"))

d, rc, out = run([fake_ws], [clean], selfdir=fake_ws)
ex = d["search_exact"] if d else {}
check("C7 self carve-out", "copy inside this job's workspace: listed, verdict unchanged",
      d and d["status"] == "NO_COPY_FOUND" and ex.get("match_count") == 1
      and ex.get("match_count_outside_this_job") == 0
      and ex["matches"][0]["in_this_jobs_own_workspace"] is True,
      f"status={d and d['status']} matches={ex.get('match_count')} "
      f"outside={ex.get('match_count_outside_this_job')}")

d, rc, out = run([fake_ws], [clean], selfdir=os.path.join(ROOT, "somewhere_else"))
ex = d["search_exact"] if d else {}
check("C8 self pair", "the SAME copy, self pointed elsewhere -> REVIEW_REQUIRED",
      d and d["status"] == "REVIEW_REQUIRED" and ex.get("match_count") == 1
      and ex.get("match_count_outside_this_job") == 1,
      f"status={d and d['status']} outside={ex.get('match_count_outside_this_job')}")

shutil.rmtree(ROOT, ignore_errors=True)

n_pass = sum(1 for _, _, ok, _ in results if ok)
print(f"\n{n_pass}/{len(results)} controls passed  (sandbox {ROOT} removed)")

# The result is pinned to the sha256 of the scanner it was run against, so a
# later edit to the scanner cannot inherit a green run it was never subjected to.
with open(SCANNER, "rb") as fh:
    scanner_sha = hashlib.sha256(fh.read()).hexdigest()
with open(EVIDENCE, "w", encoding="utf-8") as fh:
    json.dump({
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": "Controls for scripts/measure_originality.py -- the rule 2 copy "
                "search. Each exercises one outcome of the scanner against a tree "
                "built for the run.",
        "scanner": "scripts/measure_originality.py",
        "scanner_sha256": scanner_sha,
        "near_control_lines_rewritten": rewritten,
        "near_control_jaccard": flagged[0]["jaccard"] if flagged else None,
        "controls": [{"name": n, "expects": e, "passed": ok, "detail": d}
                     for n, e, ok, d in results],
        "passed": n_pass,
        "total": len(results),
    }, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print(f"wrote {EVIDENCE}")
sys.exit(0 if n_pass == len(results) else 1)
