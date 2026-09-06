#!/usr/bin/env python3
"""What an operator actually experiences driving Airlock, measured.

Rules section 9 names six judging criteria. Five of them have a section in the
write-up. **User Experience** had none, and nothing in this repository had ever
measured one -- so this script exists to produce the numbers rather than to
illustrate a claim already made.

It measures three things, all by running the real command line in a real
subprocess:

  1. the operator path       -- how many commands from a clean clone to a first
                                decision, and how many flags the first one needs
  2. what it feels like      -- cold start over repeated fresh processes, and
                                throughput once a process is warm
  3. what happens when the   -- four wrong invocations, run against the CURRENT
     operator gets it wrong     tree and against the shipped baseline commit, so
                                a change to an error path is visible as a change
  4. what happens when it    -- the calibration file absent, and the whole model
     goes wrong QUIETLY         directory absent (an operator who skipped step 2
                                of three). A wrong answer delivered confidently
                                is worse than an error, so these are measured
                                separately from the four above.

The fourth measurement is the one that matters and it is deliberately hostile to
us: an error path is scored on whether it names the thing that is wrong and
stops, not on whether it says something. A Python traceback is recorded as a
traceback.

Writes evidence/ux.json. Re-runnable; every number in the write-up comes from
here and tests/check_ux_numbers.py re-derives them.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "evidence")

# The README's own worked example, verbatim, so the measurement is of the thing
# the operator is actually told to type.
EXAMPLE = ("Please wire GBP 12,400 to IBAN DE89370400440532013000 for "
           "Meera Subramanian; card 4111 1111 1111 1111 must not be charged.")

# The four ways to get it wrong that cost nothing to try and that an operator
# will hit in their first ten minutes.
WRONG = {
    "no_source":     ["--json"],
    "two_sources":   ["--text", "x", "--file", "y"],
    "missing_file":  ["--file", "/tmp/airlock_no_such_file_9e1c.txt"],
    "bad_model_dir": ["--text", "hello", "--model", "models/not_a_model"],
}


def run(argv, cwd, timeout=300):
    """One CLI invocation. stdout and stderr are kept apart on purpose -- the
    question of which stream carries what is itself part of the measurement."""
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "airlock.cli", *argv],
                       cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return {
        "argv": argv,
        "returncode": p.returncode,
        "seconds": round(time.perf_counter() - t0, 3),
        "stdout": p.stdout,
        "stderr": p.stderr,
    }


def _is_traceback(stderr: str) -> bool:
    return "Traceback (most recent call last)" in stderr


def _last_line(s: str) -> str:
    lines = [l for l in s.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def score_error_path(name: str, r: dict) -> dict:
    """An error path is graded on four mechanical properties, none of which is a
    matter of taste:

      exits_nonzero        -- a script that wraps this can tell it failed
      no_traceback         -- the operator is not shown our call stack
      names_the_input      -- the offending value appears in the message
      no_unrelated_remedy  -- the message does not send the operator to fix
                              something that is not the problem (the measured
                              case: a bad local --model path produces Hugging
                              Face's "log in with `hf auth login`", which is
                              advice about credentials for a problem that is a
                              wrong path)
    """
    err = r["stderr"]
    offending = {
        "no_source": None,
        "two_sources": None,
        "missing_file": "/tmp/airlock_no_such_file_9e1c.txt",
        "bad_model_dir": "models/not_a_model",
    }[name]
    names = True if offending is None else (offending in err)
    unrelated = any(k in err for k in ("hf auth login", "huggingface.co",
                                       "private repository", "token having permission"))
    props = {
        "exits_nonzero": r["returncode"] != 0,
        "no_traceback": not _is_traceback(err),
        "names_the_input": bool(names),
        "no_unrelated_remedy": not unrelated,
    }
    return {
        "returncode": r["returncode"],
        "stderr_lines": len([l for l in err.strip().split("\n") if l.strip()]),
        "last_stderr_line": _last_line(err)[:200],
        "properties": props,
        "properties_held": sum(1 for v in props.values() if v),
        "properties_total": len(props),
    }


def measure_error_paths(cwd):
    out = {}
    for name, argv in WRONG.items():
        out[name] = score_error_path(name, run(argv, cwd))
    return out


ABSENT_CAL = "models/airlock_absent_calibration_9e1c.json"


def materialise(ref, dest):
    """Put a copy of the repository's tracked files in `dest`.

    `ref=None` means the WORKING TREE. That distinction is the whole reason this
    function exists: the first version of this script used `git archive HEAD`
    for the "after" column, which reads the COMMITTED bytes, and duly reported
    two defects as unfixed while the fix was sitting uncommitted on disk. A
    before/after instrument that cannot see the change it is measuring is worse
    than no instrument, because it reports a number."""
    if ref is not None:
        subprocess.run(f"git archive {ref} | tar -x -C {dest}", cwd=ROOT,
                       shell=True, check=True)
        return
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split("\n")
    for rel in (t for t in tracked if t):
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _tree_without_calibration(ref: str):
    """The repo at `ref` with the detector weights present and the gate
    calibration file ABSENT at its default path -- the state a clone lands in if
    the calibration is deleted, moved, or not produced by a partial fetch. The
    detector is symlinked so this costs no disk and no download."""
    d = tempfile.mkdtemp(prefix="airlock_ux_nocal_")
    materialise(ref, d)
    os.makedirs(os.path.join(d, "models"), exist_ok=True)
    os.symlink(os.path.join(ROOT, "models", "detector"),
               os.path.join(d, "models", "detector"))
    assert not os.path.exists(os.path.join(d, "models", "gate_calibration.json"))
    return d


def measure_silent_fallback(ref: str, cwd: str):
    """The gate's headline number is printed as "calibrated residual leak risk".
    If the calibration file is not there, `Gate` falls back to the naive risk.
    What this measures is whether the operator is TOLD -- because an
    uncalibrated number under a label that says calibrated is not cosmetic on a
    privacy gate: it is the number the forward/refuse decision is taken on.

    Three arms, all on the SAME document, so any difference in risk or route is
    the calibration and nothing else:

      with_calibration      the normal case, for the reference numbers
      default_path_absent   no --calibration flag, file missing where it is
                            looked for -- the realistic failure
      explicit_missing_path --calibration pointed at a file that is not there --
                            the operator NAMED a file and it was not used
    """
    def arm(where, extra):
        plain = run(["--text", EXAMPLE, *extra], where)
        js = run(["--text", EXAMPLE, "--json", *extra], where)
        try:
            payload = json.loads(js["stdout"])
            d = payload["decision"]
            risk, route = d["risk"], d["route"]
            json_flag = payload.get("calibrated")
        except Exception:
            risk, route, json_flag = None, None, None
        return {
            "returncode": plain["returncode"],
            "risk": risk,
            "route": route,
            "json_calibrated_flag": json_flag,
            "label_says_calibrated": "calibrated residual leak risk" in plain["stdout"],
            "warns_on_stderr": any(w in plain["stderr"] for w in
                                   ("UNCALIBRATED", "uncalibrated", "not calibrated")),
        }

    with_cal = arm(cwd, [])
    nocal_dir = _tree_without_calibration(ref)
    try:
        default_absent = arm(nocal_dir, [])
    finally:
        os.unlink(os.path.join(nocal_dir, "models", "detector"))
        shutil.rmtree(nocal_dir, ignore_errors=True)
    explicit = arm(cwd, ["--calibration", ABSENT_CAL])

    silent = bool(default_absent["returncode"] == 0
                  and default_absent["label_says_calibrated"]
                  and not default_absent["warns_on_stderr"])
    return {
        "document_characters": len(EXAMPLE),
        "with_calibration": with_cal,
        "default_path_absent": default_absent,
        "explicit_missing_path": explicit,
        "risk_differs": (with_cal["risk"] is not None
                         and default_absent["risk"] is not None
                         and with_cal["risk"] != default_absent["risk"]),
        "route_differs": (with_cal["route"] is not None
                          and default_absent["route"] is not None
                          and with_cal["route"] != default_absent["route"]),
        "silently_uncalibrated": silent,
        "explicit_path_silently_ignored": bool(
            explicit["returncode"] == 0 and not explicit["warns_on_stderr"]),
    }


def measure_skipped_fetch(ref: str):
    """A clean clone with `pip install` done and `bash scripts/fetch_model.sh`
    NOT done -- the single most likely first mistake, because models/ is
    gitignored so every clone starts without it. Measured on the real tree at
    `ref` with no models/ directory at all."""
    d = tempfile.mkdtemp(prefix="airlock_ux_nomodel_")
    try:
        materialise(ref, d)
        r = run(["--text", EXAMPLE], d)
        return {
            "returncode": r["returncode"],
            "stderr_lines": len([l for l in r["stderr"].strip().split("\n") if l.strip()]),
            "traceback": _is_traceback(r["stderr"]),
            "last_stderr_line": _last_line(r["stderr"])[:200],
            "names_fetch_model_sh": "fetch_model.sh" in r["stderr"],
            "sends_operator_to_credentials": any(
                k in r["stderr"] for k in ("hf auth login", "private repository",
                                           "token having permission")),
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)


def baseline_tree(ref: str):
    """The shipped bytes at `ref`, extracted to a temp dir with the (gitignored)
    model directory linked in, so the same four wrong invocations can be run
    against the version a judge could have downloaded before this change."""
    d = tempfile.mkdtemp(prefix="airlock_ux_base_")
    materialise(ref, d)
    os.symlink(os.path.join(ROOT, "models"), os.path.join(d, "models"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ref", default="bad73af",
                    help="commit whose error paths are the 'before' column")
    ap.add_argument("--cold-runs", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(EV, "ux.json"))
    args = ap.parse_args()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()

    # ---- 1. the operator path, read out of the README rather than asserted
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    block = readme.split("## Run it", 1)[1].split("```", 2)[1]
    cmds = [l.strip() for l in block.strip().split("\n")
            if l.strip() and not l.strip().startswith("#")]
    # the shortest real path: install, fetch the weights, run one document
    shortest = [c for c in cmds if c.startswith(("pip install", "bash scripts/fetch_model.sh"))]
    shortest.append("python3 -m airlock.cli --text \"...\"")

    # ---- 2. cold start: whole fresh processes, model load included
    cold = [run(["--text", EXAMPLE], ROOT)["seconds"] for _ in range(args.cold_runs)]

    # ---- 3. streams and machine-readability
    plain = run(["--text", EXAMPLE], ROOT)
    asjson = run(["--text", EXAMPLE, "--json"], ROOT)
    try:
        parsed = json.loads(asjson["stdout"])
        json_ok, json_keys = True, sorted(parsed.keys())
    except Exception:
        json_ok, json_keys = False, []

    # ---- 4. throughput once warm, in one process
    sizes = [len(EXAMPLE), len(EXAMPLE) * 8, len(EXAMPLE) * 40]
    sys.path.insert(0, ROOT)
    from airlock.hybrid import HybridDetector      # noqa: E402
    from airlock.gate import Calibrator, Gate      # noqa: E402
    det = HybridDetector(os.path.join(ROOT, "models/detector"))
    cal = Calibrator.load(os.path.join(ROOT, "models/gate_calibration.json"))
    gate = Gate(det, cal, threshold=0.2, budget=0.12)
    gate.process(EXAMPLE)                          # warm the graph, then measure
    thr = []
    for n in sizes:
        doc = (EXAMPLE + " ") * (n // len(EXAMPLE) or 1)
        t0 = time.perf_counter()
        gate.process(doc)
        s = time.perf_counter() - t0
        thr.append({"characters": len(doc), "seconds": round(s, 4),
                    "docs_per_second": round(1.0 / s, 2)})

    # ---- 5. the wrong invocations, now and at the shipped baseline
    after = measure_error_paths(ROOT)
    fallback_after = measure_silent_fallback(None, ROOT)
    base_dir = baseline_tree(args.baseline_ref)
    try:
        before = measure_error_paths(base_dir)
        fallback_before = measure_silent_fallback(args.baseline_ref, base_dir)
    finally:
        os.unlink(os.path.join(base_dir, "models"))
        shutil.rmtree(base_dir, ignore_errors=True)

    doc = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_head": head,
        "after_column_is": "the WORKING TREE, not HEAD",
        "baseline_ref": subprocess.run(["git", "rev-parse", args.baseline_ref], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip(),
        "python": sys.version.split()[0],
        "operator_path": {
            "commands_in_readme_run_it": len(cmds),
            "commands_clean_clone_to_first_decision": len(shortest),
            "commands": shortest,
            "flags_required_for_first_decision": 1,
            "options_with_defaults": ["--model", "--calibration", "--threshold", "--budget"],
        },
        "cold_start": {
            "runs": args.cold_runs,
            "seconds": [round(s, 3) for s in cold],
            "min": round(min(cold), 3),
            "median": round(statistics.median(cold), 3),
            "max": round(max(cold), 3),
        },
        "streams": {
            "decision_lines_on_stdout": len([l for l in plain["stdout"].strip().split("\n") if l]),
            "stdout_has_progress_bar": "Loading weights" in plain["stdout"],
            "stderr_has_progress_bar": "Loading weights" in plain["stderr"],
            "json_stdout_parses": json_ok,
            "json_keys": json_keys,
        },
        "throughput_warm": thr,
        "silent_fallback_calibration": {
            "before": fallback_before,
            "after": fallback_after,
        },
        "skipped_fetch_model": {
            "before": measure_skipped_fetch(args.baseline_ref),
            "after": measure_skipped_fetch(None),
        },
        "error_paths": {
            "scenarios": sorted(WRONG),
            "before": before,
            "after": after,
            "properties_held_before": sum(v["properties_held"] for v in before.values()),
            "properties_held_after": sum(v["properties_held"] for v in after.values()),
            "properties_total": sum(v["properties_total"] for v in after.values()),
        },
    }
    os.makedirs(EV, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {args.out}")
    print(f"  cold start median {doc['cold_start']['median']} s over {args.cold_runs} runs")
    print(f"  silently uncalibrated: before "
          f"{doc['silent_fallback_calibration']['before']['silently_uncalibrated']}, "
          f"after {doc['silent_fallback_calibration']['after']['silently_uncalibrated']}")
    print(f"  skipped fetch_model.sh sends operator to credentials: before "
          f"{doc['skipped_fetch_model']['before']['sends_operator_to_credentials']}, "
          f"after {doc['skipped_fetch_model']['after']['sends_operator_to_credentials']}")
    print(f"  error-path properties: before {doc['error_paths']['properties_held_before']}"
          f"/{doc['error_paths']['properties_total']}, "
          f"after {doc['error_paths']['properties_held_after']}"
          f"/{doc['error_paths']['properties_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
