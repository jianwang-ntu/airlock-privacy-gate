"""What this project is actually built out of, measured rather than remembered,
and written to evidence/environment.json.

Requirements 5 of the UPAI-Hackdays brief asks us to "Mention the AI models,
APIs, frameworks, libraries, and other technologies used." A list typed from
memory is exactly the kind of claim this repository does not ship. So the list
is *derived*:

  * the third-party libraries are the top-level modules our own `.py` files
    import, found by parsing every file with `ast` -- not by reading
    requirements.txt, which is the thing being checked;
  * each version is read in its **own subprocess**, importing that module
    first. Import order is not cosmetic here: on this host pyarrow's bundled
    libarrow needs a newer libstdc++ than the system one torch loads, so
    `import torch; import pyarrow` fails while the reverse succeeds. A single
    process probing ten modules would report whichever ones lost that race as
    absent. That mistake was made once while writing this script;
  * the external binaries are the argv[0] literals passed to subprocess;
  * the model ids are parsed out of `airlock/route.py` and the checkpoint's own
    `train_meta.json` / `config.json`;
  * the corpus and its licence come from the data manifest written by
    `scripts/prepare_data.py` at conversion time.

It then compares the derived set against requirements.txt in both directions,
because a dependency we use and do not declare breaks a judge's clean install,
and one we declare and do not import is noise.

Stdlib only in the parent process. Run: python3 scripts/measure_environment.py
"""
from __future__ import annotations

import ast
import datetime
import json
import os
import platform
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(ROOT, "evidence", "environment.json")
sys.path.insert(0, ROOT)

from scripts.pathredact import emit_json  # noqa: E402

FIRST_PARTY = {"airlock", "scripts", "tests"}
SKIP_DIRS = {".git", ".data_cache", "__pycache__"}


def source_files() -> list[str]:
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            if f.endswith(".py"):
                out.append(os.path.relpath(os.path.join(base, f), ROOT))
    return sorted(out)


def imported_top_levels(files: list[str]) -> dict[str, list[str]]:
    """{top-level module: [files importing it]} for THIRD-PARTY modules only."""
    std = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for rel in files:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import -- first-party by construction
                    continue
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in std or top in FIRST_PARTY or top == "__future__":
                    continue
                found.setdefault(top, set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def probe_version(module: str) -> dict:
    """Import `module` FIRST in a fresh interpreter and report its version."""
    code = (
        "import importlib,json,sys\n"
        f"m={module!r}\n"
        "try:\n"
        "    mod=importlib.import_module(m)\n"
        "    print(json.dumps({'import':'OK','version':getattr(mod,'__version__',None),"
        "'file':getattr(mod,'__file__',None)}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'import':'FAIL','error':type(e).__name__,'message':str(e)[:200]}))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    try:
        rec = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        rec = {"import": "FAIL", "error": "no-parseable-output", "message": p.stderr[-200:]}
    rec["probed_alone_in_subprocess"] = True
    return rec


def probe_import_pair(first: str, second: str) -> dict:
    code = (
        "import json\n"
        f"import {first}\n"
        "try:\n"
        f"    import {second}\n"
        "    print(json.dumps({'result':'OK'}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'result':'FAIL','error':type(e).__name__,'message':str(e)[:200]}))\n"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    try:
        rec = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        rec = {"result": "FAIL", "error": "no-parseable-output", "message": p.stderr[-200:]}
    return {"order": f"import {first} then {second}", **rec}


SUBPROCESS_RUNNERS = {"run", "Popen", "call", "check_call", "check_output"}


def _subprocess_local_names(tree) -> dict[str, str]:
    """Local name -> subprocess name, for `from subprocess import run [as r]`.

    Only these may be called unqualified and still mean subprocess. Any other
    bare `run(...)` is somebody's own helper -- `scripts/measure_ux.py`,
    `tests/check_originality_scan.py` and `tests/check_readme_numbers.py` each
    define one -- and its first argument is not an argv.
    """
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess":
            for a in node.names:
                names[a.asname or a.name] = a.name
    return names


def external_binaries(files: list[str]) -> dict[str, list[str]]:
    """argv[0] string literals handed to subprocess.run / Popen / call.

    A call counts only if it is qualified (`subprocess.run([...])`) or its bare
    name was imported from subprocess in that file. Accepting every unqualified
    `run([...])` reported `--text` as an external binary -- the first element of
    the argv `scripts/measure_ux.py` hands to its OWN `run()` helper -- and the
    version probe then recorded a binary that does not exist.
    """
    out: dict[str, set[str]] = {}
    for rel in files:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), rel)
        local = _subprocess_local_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            mod = getattr(getattr(fn, "value", None), "id", None)
            if mod == "subprocess":
                if name not in SUBPROCESS_RUNNERS:
                    continue
            elif mod is None and local.get(name) in SUBPROCESS_RUNNERS:
                pass
            else:
                continue
            for arg in node.args[:1]:
                argv = arg
                if isinstance(argv, ast.Name):  # a list built earlier in the function
                    argv = _resolve_list(tree, argv.id)
                if isinstance(argv, ast.List) and argv.elts:
                    head = argv.elts[0]
                    if isinstance(head, ast.Constant) and isinstance(head.value, str):
                        out.setdefault(head.value, set()).add(rel)
    # sys.executable is this interpreter, not an external tool
    out.pop("sys.executable", None)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _resolve_list(tree, varname):
    last = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == varname and isinstance(node.value, ast.List):
                last = node.value
    return last


def binary_version(cmd: str) -> dict:
    for flag in ("-version", "--version"):
        try:
            p = subprocess.run([cmd, flag], capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            return {"present": False, "error": "FileNotFoundError"}
        except Exception as e:  # pragma: no cover - environment dependent
            return {"present": False, "error": type(e).__name__}
        line = (p.stdout or p.stderr).strip().splitlines()
        if line:
            return {"present": True, "first_line": line[0][:160], "flag": flag}
    return {"present": True, "first_line": None}


def declared_requirements() -> dict:
    path = os.path.join(ROOT, "requirements.txt")
    decl = {}
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#")[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        if m:
            decl[m.group(1)] = m.group(2).strip()
    return decl


# PyPI distribution name -> the top-level module it installs, where they differ.
DIST_TO_MODULE = {"Pillow": "PIL", "pillow": "PIL", "scikit-learn": "sklearn"}


def model_ids() -> dict:
    route = open(os.path.join(ROOT, "airlock", "route.py"), encoding="utf-8").read()
    tree = ast.parse(route)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                consts[t.id] = node.value.value
    meta = json.load(open(os.path.join(ROOT, "models", "detector", "train_meta.json")))
    cfg = json.load(open(os.path.join(ROOT, "models", "detector", "config.json")))
    return {
        "detector_base_checkpoint": meta["base_model"],
        "detector_architecture": cfg.get("architectures", [None])[0],
        "detector_model_type": cfg.get("model_type"),
        "detector_labels": len(cfg.get("id2label", {})),
        "detector_parameters": meta.get("parameters"),
        "local_fallback_model": consts.get("LOCAL_DEFAULT"),
        "hosted_standin_model": consts.get("HOSTED_STANDIN_DEFAULT"),
        "source": {
            "detector_base_checkpoint": "models/detector/train_meta.json:base_model",
            "detector_architecture": "models/detector/config.json:architectures[0]",
            "local_fallback_model": "airlock/route.py:LOCAL_DEFAULT",
            "hosted_standin_model": "airlock/route.py:HOSTED_STANDIN_DEFAULT",
        },
    }


def corpus() -> dict:
    cache = os.environ.get(
        "AIRLOCK_DATA_DIR", os.path.join(ROOT, ".data_cache")
    )
    mpath = os.path.join(cache, "data_manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        return {
            "id": man.get("source"),
            "license": man.get("license"),
            "from": "data_manifest.json written by scripts/prepare_data.py",
            "files": {k: {"rows": v.get("rows"), "sha256": v.get("sha256")}
                      for k, v in sorted(man.get("files", {}).items())},
        }
    src = open(os.path.join(ROOT, "scripts", "prepare_data.py"), encoding="utf-8").read()
    return {
        "id": (re.search(r'"source":\s*"([^"]+)"', src) or [None, None])[1],
        "license": (re.search(r'"license":\s*"([^"]+)"', src) or [None, None])[1],
        "from": "scripts/prepare_data.py source (data manifest not present)",
        "files": {},
    }


# The libraries this project is allowed to import, as an ALLOW-list. A deny-list
# of provider SDKs cannot be complete -- it only ever excludes the vendors
# someone thought to name -- so Requirements 5's "APIs used" answer is derived
# the other way: the third-party set is enumerated from the source, and it is
# this and nothing else. Adding any client library, of any vendor, breaks it.
# scipy is analysis-only: it is imported solely to cross-check this
# repository's own Fisher exact test in scripts/measure_fairness.py against a
# second implementation. It is on the list because it is genuinely imported,
# not because the gate needs it -- nothing in airlock/ imports it.
ALLOWED_THIRD_PARTY = ("PIL", "pandas", "pyarrow", "scipy", "torch", "transformers")

# Stdlib modules through which bytes could leave the machine. These are stdlib
# names, so the check does not depend on knowing any vendor's package name.
STDLIB_NETWORK = (
    "asyncio", "ftplib", "http", "imaplib", "poplib", "smtplib", "socket",
    "socketserver", "ssl", "telnetlib", "urllib", "webbrowser", "xmlrpc",
)


def stdlib_top_levels(files: list[str]) -> dict[str, list[str]]:
    """{stdlib top-level module: [files importing it]} from our own source."""
    std = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for rel in files:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in std and top != "__future__":
                    found.setdefault(top, set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def network_capability(files: list[str], imports: dict) -> dict:
    """Requirements 5 asks which APIs are used. The answer is none, and it is
    derived from the import graph rather than from a search for vendor names."""
    stdlib = stdlib_top_levels(files)
    net = {m: stdlib[m] for m in STDLIB_NETWORK if m in stdlib}
    unexpected = sorted(set(imports) - set(ALLOWED_THIRD_PARTY))
    missing = sorted(set(ALLOWED_THIRD_PARTY) - set(imports))
    return {
        "method": "AST import graph over every .py in the repository, allow-list "
                  "over third-party top-levels plus a stdlib network-module sweep.",
        "allowed_third_party": list(ALLOWED_THIRD_PARTY),
        "third_party_imported": sorted(imports),
        "third_party_outside_allow_list": unexpected,
        "allow_list_entries_not_imported": missing,
        "stdlib_network_list": list(STDLIB_NETWORK),
        "stdlib_network_modules_imported": net,
        "stdlib_modules_imported": sorted(stdlib),
        "files_scanned": len(files),
        "verdict": "NO_NETWORK_CLIENT" if not unexpected and not net else "REVIEW",
    }


def main() -> int:
    files = source_files()
    imports = imported_top_levels(files)
    versions = {m: probe_version(m) for m in imports}
    binaries = external_binaries(files)
    bin_versions = {b: binary_version(b) for b in binaries}
    decl = declared_requirements()
    decl_modules = {DIST_TO_MODULE.get(d, d.lower()): d for d in decl}

    imported_not_declared = sorted(m for m in imports if m.lower() not in decl_modules
                                   and m not in decl_modules)
    declared_not_imported = sorted(d for m, d in decl_modules.items()
                                   if m not in imports and m not in {i.lower() for i in imports})

    out = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": "Requirements 5 (Technology Stack), derived from the source rather than "
                "transcribed. Libraries are the third-party top-level modules our own .py "
                "files import; each version is read in its own subprocess with that module "
                "imported first.",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_is_this_interpreter": True,
        },
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "release": platform.release()},
        "source_files_parsed": len(files),
        "source_files": files,
        "libraries": {
            m: {"files": imports[m], **versions[m]} for m in sorted(imports)
        },
        "external_binaries": {
            b: {"files": binaries[b], **bin_versions[b]} for b in sorted(binaries)
        },
        "requirements_txt": {
            "declared": decl,
            "imported_but_not_declared": imported_not_declared,
            "declared_but_not_imported": declared_not_imported,
        },
        "import_order": [
            probe_import_pair("torch", "pyarrow"),
            probe_import_pair("pyarrow", "torch"),
            probe_import_pair("airlock", "pyarrow"),
        ],
        "models": model_ids(),
        "corpus": corpus(),
        "network_capability": network_capability(files, imports),
    }
    # Committed to a public repository, and produced by reading this host's
    # installed libraries, so absolute paths are rewritten before the file is
    # written and the run refuses rather than publishing one it missed.
    # See scripts/pathredact.py.
    if emit_json(out, EVIDENCE, indent=2) != 0:
        return 2
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
