"""What a fresh `pip install -r requirements.txt` installs today, and what it costs.

Every number in this repository was measured in one environment. `requirements.txt`
is the only thing that tells a reader how to rebuild it, and a range like
`transformers>=4.40` does not name an environment -- it names whatever PyPI is
serving on the day. This script measures the difference instead of assuming it is
zero: it re-runs the two evidence-producing evaluations under the interpreter that
produced the shipped numbers and again under the versions an unpinned install
resolves to now, and writes both, plus the deltas, to `evidence/env_drift.json`.

Arms:

  reference                 the interpreter running this script; the environment
                            `requirements.txt` is pinned to.
  latest_unpinned           transformers/tokenizers/safetensors at whatever PyPI
                            serves today, installed over the reference by version
                            only -- torch, the corpus, the weights, the seed and
                            the code are held fixed, so any delta is the library.
  reference_new_safetensors reference transformers with today's safetensors, to
                            say whether safetensors is implicated or not.

and one probe that runs no evaluation:

  coexistence               can the pinned transformers import at all with today's
                            tokenizers? The answer decides whether `tokenizers`
                            needs its own line in requirements.txt or is already
                            constrained by the transformers pin.

Needs the network (pip) and the trained detector. It writes nothing outside
`evidence/env_drift.json` and its own temporary directory, which it removes.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(ROOT, "evidence", "env_drift.json")

# The budget grid scripts/train_all.sh uses, so gate rows line up with the
# shipped evidence/gate_eval.json row for row.
BUDGETS = "0.045,0.06,0.09,0.12,0.16,0.2,0.3,0.5,1.0"
THRESHOLD = "0.2"

# The libraries whose version an unpinned install is free to move. torch is in
# requirements.txt too but is deliberately not moved here: holding it fixed is
# what makes a delta attributable to this set.
TRACKED = ["transformers", "tokenizers", "safetensors", "torch"]

# Volatile keys: wall-clock, not measurement.
VOLATILE = {"generated_at", "seconds", "wall_seconds", "throughput"}


def versions(python: str) -> dict:
    """Read each tracked library's version in the given interpreter."""
    code = (
        "import json,importlib.metadata as m\n"
        f"out={{}}\n"
        f"for n in {TRACKED!r}:\n"
        "    try: out[n]=m.version(n)\n"
        "    except Exception as e: out[n]=None\n"
        "import sys; out['python']=sys.version.split()[0]\n"
        "print(json.dumps(out))"
    )
    p = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if p.returncode != 0:
        return {"error": (p.stderr or "").strip()[-400:]}
    return json.loads(p.stdout)


def make_venv(work: str, name: str, installs: list) -> str:
    """A venv that inherits the reference environment and shadows `installs`.

    --system-site-packages keeps torch, the corpus reader and everything else
    identical to the reference arm; --ignore-installed forces the named
    distributions into the venv rather than leaving the inherited copy in place,
    and --no-deps stops pip resolving anything the arm did not ask for.
    """
    vd = os.path.join(work, name)
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", vd],
                   check=True, capture_output=True, text=True)
    py = os.path.join(vd, "bin", "python")
    if installs:
        p = subprocess.run([os.path.join(vd, "bin", "pip"), "install", "--no-deps",
                            "--ignore-installed", "--quiet"] + installs,
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"pip failed for {name}: {(p.stderr or '')[-600:]}")
    return py


def evaluate(python: str, work: str, arm: str) -> dict:
    """Run both evidence-producing evaluations and return what they wrote."""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    det_out = os.path.join(work, f"detector_{arm}.json")
    gate_out = os.path.join(work, f"gate_{arm}.json")
    for cmd in (
        [python, "-m", "airlock.evaluate", "--thresholds", THRESHOLD, "--out", det_out],
        [python, os.path.join("scripts", "eval_gate.py"), "--budgets", BUDGETS,
         "--out", gate_out],
    ):
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{arm}: {' '.join(cmd[1:3])} exited {p.returncode}: "
                               f"{(p.stderr or '')[-600:]}")
    return {"detector": json.load(open(det_out)), "gate": json.load(open(gate_out))}


def shipped() -> dict:
    ev = os.path.join(ROOT, "evidence")
    return {"detector": json.load(open(os.path.join(ev, "detector_eval.json"))),
            "gate": json.load(open(os.path.join(ev, "gate_eval.json")))}


def reproduces(run: dict, ship: dict) -> dict:
    """Does this arm reproduce the shipped evidence, field for field?

    Compared as whole objects rather than on a chosen headline, so a delta this
    script's author did not think to look at still shows up. Timing keys are
    excluded because they are not measurements of the model.
    """
    det_keys = [("by_threshold_hybrid", THRESHOLD), ("by_threshold_model_only", THRESHOLD),
                ("regex_baseline", None), ("checksum_validators_only", None)]
    det = {}
    for top, sub in det_keys:
        a, b = run["detector"].get(top), ship["detector"].get(top)
        if sub is not None:
            a, b = (a or {}).get(sub), (b or {}).get(sub)
        det[top if sub is None else f"{top}[{sub}]"] = (a == b)
    gate = {k: (run["gate"].get(k) == v)
            for k, v in ship["gate"].items() if k not in VOLATILE}
    return {
        "detector_blocks": det,
        "gate_blocks": gate,
        "all": all(det.values()) and all(gate.values()),
    }


def headline(run: dict) -> dict:
    h = run["detector"]["by_threshold_hybrid"][THRESHOLD]
    g = run["gate"]
    violated = [r["budget"] for r in g["risk_coverage"] if not r["budget_respected"]]
    return {
        "identifier_recall": h["identifier_recall"],
        "typed_recall": h["typed_recall"],
        "doc_leak_rate": h["doc_leak_rate"],
        "identifiers_leaked": h["identifiers_leaked"],
        "docs_with_leak": h["docs_with_leak"],
        "over_redaction_rate_upper_bound": h["over_redaction_rate_upper_bound"],
        "gate_ranking_auc": g["ranking_auc_raw_statistic"],
        "ece_posterior_mean": g["ece_posterior_mean"],
        "reliability_bins_where_upper_bound_is_conservative":
            g["reliability_bins_where_upper_bound_is_conservative"],
        "budgets_tested": [r["budget"] for r in g["risk_coverage"]],
        "budgets_violated": violated,
    }


def pypi_latest(name: str) -> dict:
    """What PyPI serves as `name`'s latest release right now."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/json", timeout=30) as r:
            d = json.load(r)
        return {"latest": d["info"]["version"],
                "requires_python": d["info"].get("requires_python")}
    except Exception as e:
        return {"latest": None, "error": f"{type(e).__name__}: {e}"}


def declared_spec() -> dict:
    """The version specifiers requirements.txt actually declares."""
    import re
    out = {}
    for raw in open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def declared_dependency(python: str, dist: str, dep: str) -> list:
    """What `dist` declares about `dep` in its own metadata, read from `python`.

    The pin in requirements.txt says it already constrains tokenizers. This is
    where that sentence gets its evidence, rather than being asserted.
    """
    code = ("import json,importlib.metadata as m\n"
            f"rs=m.requires({dist!r}) or []\n"
            f"print(json.dumps([r for r in rs if r.split(chr(59))[0].strip()"
            f".lower().startswith({dep!r})]))")
    p = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if p.returncode != 0:
        return []
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return []


def coexistence_probe(work: str, tokenizers_version: str) -> dict:
    """Can the pinned transformers import with today's tokenizers?

    If it refuses, the transformers pin already constrains tokenizers and a
    second requirements.txt line would be decoration. If it imports, the pin
    does not cover tokenizers and something else has to.
    """
    out = {"transformers_requires_tokenizers":
           declared_dependency(sys.executable, "transformers", "tokenizers")}
    if not tokenizers_version:
        return dict(out, ran=False, why="no tokenizers version to probe")
    try:
        py = make_venv(work, "coexist", [f"tokenizers=={tokenizers_version}"])
    except RuntimeError as e:
        return dict(out, ran=False, why=str(e)[-300:])
    p = subprocess.run([py, "-c", "import transformers; print(transformers.__version__)"],
                       cwd=ROOT, capture_output=True, text=True)
    lines = [l.strip() for l in (p.stderr or p.stdout).splitlines() if l.strip()]
    # The last line of a traceback is pip's suggestion, not the diagnosis. Take
    # the raised exception line when there is one.
    raised = [l for l in lines if "Error:" in l] or lines[-1:]
    return dict(out,
                ran=True,
                tokenizers_forced=tokenizers_version,
                import_transformers_exit_code=p.returncode,
                imports=p.returncode == 0,
                error_line=(raised[-1][:400] if raised else ""),
                stderr_tail=[l[:200] for l in lines[-5:]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None,
                    help="scratch directory; a fresh temp dir is made and removed "
                         "if not given")
    ap.add_argument("--out", default=EVIDENCE)
    args = ap.parse_args()

    work = args.work or tempfile.mkdtemp(prefix="airlock_env_drift_")
    made_work = args.work is None
    os.makedirs(work, exist_ok=True)
    ship = shipped()
    try:
        ref_versions = versions(sys.executable)
        latest = {n: pypi_latest(n) for n in ("transformers", "tokenizers", "safetensors")}
        wanted = [f"{n}=={latest[n]['latest']}" for n in latest if latest[n]["latest"]]
        if len(wanted) != 3:
            print("cannot reach PyPI; not writing evidence", file=sys.stderr)
            return 2

        arms = {}
        specs = {
            "reference": [],
            "latest_unpinned": wanted,
            "reference_new_safetensors": [f"safetensors=={latest['safetensors']['latest']}"],
        }
        for arm, installs in specs.items():
            py = sys.executable if arm == "reference" else make_venv(work, arm, installs)
            v = versions(py)
            run = evaluate(py, work, arm)
            arms[arm] = {
                "installed": v,
                "pip_install_args": installs,
                "reproduces_shipped_evidence": reproduces(run, ship),
                "headline": headline(run),
            }
            print(f"{arm}: transformers {v.get('transformers')} tokenizers "
                  f"{v.get('tokenizers')} safetensors {v.get('safetensors')} -> "
                  f"reproduces={arms[arm]['reproduces_shipped_evidence']['all']}")

        probe = coexistence_probe(work, latest["tokenizers"]["latest"])
        ref, drift = arms["reference"]["headline"], arms["latest_unpinned"]["headline"]
        out = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "what": "What an unpinned install resolves to today and what it does to "
                    "the numbers. Every arm holds the code, the weights, the corpus, "
                    "the seed and torch fixed and moves only library versions.",
            "shipped_evidence_measured_with": ref_versions,
            "pypi_latest_at_measurement": latest,
            "requirements_txt_declared": declared_spec(),
            "arms": arms,
            "coexistence_probe": probe,
            "drift_cost": {
                "arm": "latest_unpinned",
                "identifier_recall": [ref["identifier_recall"], drift["identifier_recall"]],
                "doc_leak_rate": [ref["doc_leak_rate"], drift["doc_leak_rate"]],
                "typed_recall": [ref["typed_recall"], drift["typed_recall"]],
                "identifiers_leaked": [ref["identifiers_leaked"], drift["identifiers_leaked"]],
                "docs_with_leak": [ref["docs_with_leak"], drift["docs_with_leak"]],
                "gate_ranking_auc": [ref["gate_ranking_auc"], drift["gate_ranking_auc"]],
                "ece_posterior_mean": [ref["ece_posterior_mean"], drift["ece_posterior_mean"]],
                "reliability_bins_where_upper_bound_is_conservative": [
                    ref["reliability_bins_where_upper_bound_is_conservative"],
                    drift["reliability_bins_where_upper_bound_is_conservative"]],
                "budgets_violated": [ref["budgets_violated"], drift["budgets_violated"]],
            },
            "caveat": "One machine, one Python, one torch build. It measures that the "
                      "unpinned range is not the measured environment and what that "
                      "costs here; it does not enumerate every version in the range.",
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print("wrote", args.out)
        return 0
    finally:
        if made_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
