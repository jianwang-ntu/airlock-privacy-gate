"""The environment the numbers came from, asserted instead of assumed.

`tests/check_readme_numbers.py` compares the prose to `evidence/`. Both can agree
perfectly while the reader's own run disagrees with both, because a version range
in `requirements.txt` does not name an environment. That is not hypothetical here:
`scripts/measure_env_drift.py` measured it, and `evidence/env_drift.json` records
what an unpinned install costs.

This check closes that gap from the environment side. It fails when:

  * `requirements.txt` does not pin transformers to the version the shipped
    evidence was measured with, or
  * the interpreter running it is not in that environment, or
  * the evidence file says the reference arm did NOT reproduce the shipped
    numbers -- in which case the pin points at the wrong environment, or
  * the evidence file says the unpinned arm DID reproduce them -- in which case
    the pin has no measured justification and everything above is vacuous.

Exit 0 is a pass, 1 is a failure, and 2 means NOT CHECKED. Exit 2 does not mean
the environment is right; it means this check could not tell.
"""
import copy
import importlib.metadata
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(ROOT, "evidence", "env_drift.json")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")

# Pinned in requirements.txt. tokenizers is deliberately not a requirements.txt
# line -- the transformers pin already constrains it (see the coexistence probe
# in env_drift.json) -- but it is asserted here, because "inside the window the
# pin allows" is not "the version the numbers were measured with".
PINNED_IN_REQUIREMENTS = ["transformers"]
ASSERTED_AT_RUNTIME = ["transformers", "tokenizers"]


def declared_spec(text: str) -> dict:
    out = {}
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def installed_versions(names) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def evaluate(drift: dict, declared: dict, installed: dict) -> list:
    """Return a list of (check, detail) failures. Pure: controls feed it copies."""
    bad = []
    arms = drift.get("arms", {})
    ref = arms.get("reference", {})
    ref_v = ref.get("installed", {})
    unpinned = arms.get("latest_unpinned", {})

    # 1. The recorded reference must be the environment evidence/ came from.
    if not ref.get("reproduces_shipped_evidence", {}).get("all"):
        bad.append(("reference-arm-reproduces-shipped-evidence",
                    "env_drift.json's reference arm does not reproduce evidence/. "
                    "The pin below would then name the wrong environment."))

    # 2. The unpinned arm must actually differ, or there is nothing to pin against.
    if unpinned.get("reproduces_shipped_evidence", {}).get("all"):
        bad.append(("unpinned-arm-is-a-live-control",
                    "env_drift.json's unpinned arm reproduces the shipped evidence, "
                    "so the pin has no measured justification and this check is "
                    "asserting a difference that was not observed."))

    # 3. requirements.txt must pin, exactly, to the reference arm's version.
    for name in PINNED_IN_REQUIREMENTS:
        want = ref_v.get(name)
        spec = declared.get(name)
        if want is None:
            bad.append((f"reference-records-{name}",
                        "env_drift.json's reference arm records no version for "
                        f"{name}, so there is nothing to pin to."))
        elif spec != f"=={want}":
            bad.append((f"requirements-pins-{name}",
                        f"requirements.txt declares {name}{spec!r}; the shipped "
                        f"numbers were measured with {want}. A range is not an "
                        f"environment."))

    # 4. This interpreter must BE that environment.
    cost = drift.get("drift_cost", {})
    for name in ASSERTED_AT_RUNTIME:
        want, have = ref_v.get(name), installed.get(name)
        if want is None:
            continue
        if have != want:
            r = cost.get("identifier_recall") or [None, None]
            leak = cost.get("doc_leak_rate") or [None, None]
            viol = (cost.get("budgets_violated") or [[], []])[1]
            bad.append((f"runtime-{name}-is-the-measured-version",
                        f"{name}=={have} here, {want} in evidence/. Measured cost of "
                        f"running the unpinned stack: identifier recall "
                        f"{_pct(r[0])} -> {_pct(r[1])}, document leak rate "
                        f"{_pct(leak[0])} -> {_pct(leak[1])}, leak budget violated at "
                        f"{viol}. Numbers in README.md and evidence/ describe the "
                        f"other environment, not this one."))
    return bad


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.2f}%"


def main() -> int:
    if not os.path.exists(EVIDENCE):
        print(f"NOT CHECKED: {os.path.relpath(EVIDENCE, ROOT)} is absent. Run "
              "scripts/measure_env_drift.py. Exit 2 does NOT mean the environment "
              "is the measured one.")
        return 2
    try:
        drift = json.load(open(EVIDENCE, encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"NOT CHECKED: {os.path.relpath(EVIDENCE, ROOT)} is unreadable ({e}). "
              "Exit 2 does NOT mean the environment is the measured one.")
        return 2
    declared = declared_spec(open(REQUIREMENTS, encoding="utf-8").read())
    installed = installed_versions(ASSERTED_AT_RUNTIME)

    bad = evaluate(drift, declared, installed)
    ref_v = drift.get("arms", {}).get("reference", {}).get("installed", {})
    for name in ASSERTED_AT_RUNTIME:
        print(f"{name}: here {installed.get(name)}, measured with {ref_v.get(name)}")
    for name, detail in bad:
        print(f"FAILED {name}: {detail}")
    print(f"{len(bad)} environment check(s) failed")

    # ---- controls. Each mutates the inputs in the direction that would let a
    # wrong environment through, and must be caught. A checker that cannot fail
    # is not a check, and every accepting run above is worth exactly what these
    # are worth.
    good_declared = {"transformers": f"=={ref_v.get('transformers')}"}
    good_installed = {n: ref_v.get(n) for n in ASSERTED_AT_RUNTIME}
    controls = [
        ("unmutated-positive-control", drift, good_declared, good_installed, 0),
        ("requirements-range-instead-of-pin", drift, {"transformers": ">=4.40"},
         good_installed, 1),
        ("requirements-pins-the-wrong-version", drift,
         {"transformers": "==5.16.1"}, good_installed, 1),
        ("runtime-is-a-different-transformers", drift, good_declared,
         dict(good_installed, transformers="5.16.1"), 1),
        ("runtime-is-a-different-tokenizers", drift, good_declared,
         dict(good_installed, tokenizers="0.23.2"), 1),
        ("transformers-not-installed-at-all", drift, good_declared,
         dict(good_installed, transformers=None), 1),
    ]
    ref_broken = copy.deepcopy(drift)
    ref_broken["arms"]["reference"]["reproduces_shipped_evidence"]["all"] = False
    controls.append(("reference-arm-stopped-reproducing", ref_broken, good_declared,
                     good_installed, 1))
    vacuous = copy.deepcopy(drift)
    vacuous["arms"]["latest_unpinned"]["reproduces_shipped_evidence"]["all"] = True
    controls.append(("unpinned-arm-made-identical", vacuous, good_declared,
                     good_installed, 1))

    failed_controls = []
    for name, d, dec, ins, want_min in controls:
        n = len(evaluate(d, dec, ins))
        ok = (n == 0) if want_min == 0 else (n >= want_min)
        print(f"control {name}: {n} failure(s), {'as expected' if ok else 'UNEXPECTED'}")
        if not ok:
            failed_controls.append(name)
    if failed_controls:
        print("CONTROL FAILED: " + ", ".join(failed_controls) +
              " -- this checker cannot be trusted to have checked anything")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
