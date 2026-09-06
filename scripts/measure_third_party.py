"""Every third-party thing this project stands on, measured rather than recalled,
and written to evidence/third_party.json.

Rule 6 of the UPAI-Hackdays rules is two obligations, not one:

    "Teams are responsible for ensuring they have the right to use all code,
     datasets, APIs, images, models, and other third-party resources included
     in their project."
    "Proper attribution must be provided where required."

A hand-typed credits list satisfies neither, because nothing stops it drifting
from what the code actually loads. So the inventory is *derived*:

  * libraries    the third-party top-level modules our own `.py` files import
                 (`ast`), each resolved to the installed distribution that
                 provides it and to that distribution's own METADATA licence
                 fields plus the licence text file shipped inside its
                 `.dist-info`. The licence is read off the artefact on this
                 disk, never from memory;
  * binaries     the argv[0] literals handed to subprocess in our `.py` files,
                 plus the first word of every command line in our `.sh` files
                 that resolves on PATH. Each is asked for its version, its
                 owning distro package, and its packaged copyright file;
  * fonts        the `.ttf` path literals in our source, their owning package
                 and its copyright file. They matter because the demo video
                 ships rendered glyphs;
  * models       the detector's base checkpoint out of the checkpoint's own
                 train_meta.json, and the answering model ids out of
                 airlock/route.py -- then each id matched to the publisher's
                 own licence;
  * datasets     the corpus id and the licence string recorded at conversion
                 time by scripts/prepare_data.py -- then that string CHECKED
                 against the publisher's, because the one in prepare_data.py is
                 a literal somebody typed;
  * egress       every way a byte can leave: the vendor client packages looked
                 for (and not found), the stdlib network modules imported, the
                 URL literals, and the from_pretrained arguments that reach a
                 hub rather than a local path.

What this script deliberately does NOT do is fetch. This repository contains no
network client and the entry claims so; adding one here to look up a licence
would have traded a measured property of a privacy gate for the convenience of
an HTTP call. The publishers' own licence strings are read from a pinned record,
`evidence/publisher_licenses.json`, and every row carries the one-line command
that reproduces it. An id derived from the source with no pinned record is
reported UNVERIFIED -- never assumed permissive.

Stdlib only, and no network. Run: python3 scripts/measure_third_party.py
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE = os.path.join(ROOT, "evidence", "third_party.json")
FIRST_PARTY = {"airlock", "scripts", "tests"}
SKIP_DIRS = {".git", ".data_cache", "__pycache__", "models", "logs"}

# Shell words that are not external programs even when a same-named binary
# exists on PATH (`/usr/bin/test`, `/usr/bin/[`, ...).
SH_NOT_A_BINARY = {
    "if", "then", "else", "elif", "fi", "for", "while", "do", "done", "case",
    "esac", "in", "function", "return", "exit", "set", "cd", "echo", "export",
    "local", "read", "shift", "trap", "source", ".", "[", "test", "true",
    "false", "eval", "exec", "printf", "unset", "wait", "time",
}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def source_files(exts=(".py",)) -> list[str]:
    out = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            if f.endswith(exts):
                out.append(os.path.relpath(os.path.join(base, f), ROOT))
    return sorted(out)


# ------------------------------------------------------------------ libraries
def imported_top_levels(files: list[str]) -> dict[str, list[str]]:
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
                if top in std or top in FIRST_PARTY or top == "__future__":
                    continue
                found.setdefault(top, set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def _files_star_license(body: str) -> str | None:
    """The licence of the `Files: *` stanza in a Debian copyright file.

    A copyright file lists every licence appearing anywhere in the source
    package -- fonts-dejavu-core names both `bitstream-vera` and `GPL-2+`, and
    ffmpeg names eight. Reporting that set alone would leave a reader to pick,
    so the stanza that covers the shipped files is pulled out by itself.
    """
    for stanza in re.split(r"\n\s*\n", body):
        if re.search(r"^Files:\s*\*\s*$", stanza, re.M):
            m = re.search(r"^License:\s*(.+)$", stanza, re.M)
            return m.group(1).strip() if m else None
    return None


def _first_line(text: str | None) -> str | None:
    if not text:
        return None
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:200]
    return None


def distribution_licence(module: str) -> dict:
    """Licence facts for the installed distribution providing `module`.

    Read in a subprocess for the same reason measure_environment.py probes
    versions in one: importing this project's modules in a fixed order is not
    safe on this host, and importlib.metadata is cheap enough to isolate.
    """
    code = r"""
import importlib.metadata as md, json, os, hashlib, sys
mod = sys.argv[1]
pkgs = md.packages_distributions()
names = pkgs.get(mod) or []
out = {"module": mod, "distributions": names}
if not names:
    out["error"] = "no installed distribution provides this module"
    print(json.dumps(out)); raise SystemExit(0)
d = md.distribution(names[0])
m = d.metadata
lf = []
base = None
try:
    base = str(d._path)
except Exception:
    pass
for rel in (m.get_all("License-File") or []):
    # Wheels disagree about where the text actually lands: PEP 639 wheels put it
    # in <dist-info>/licenses/, older ones alongside METADATA, and pyarrow
    # declares "../LICENSE.txt" for a file that is IN its dist-info. Try each,
    # in order, and record which one answered -- resolving only the declared
    # path reported every one of these five as absent.
    cands = []
    if base:
        bn = os.path.basename(rel)
        cands = [os.path.normpath(os.path.join(base, rel)),
                 os.path.normpath(os.path.join(base, "licenses", rel)),
                 os.path.normpath(os.path.join(base, bn)),
                 os.path.normpath(os.path.join(base, "licenses", bn))]
    p = next((c for c in cands if os.path.exists(c)), (cands[0] if cands else None))
    rec = {"declared": rel, "path": p, "present": bool(p and os.path.exists(p)),
           "resolved_by": ("declared" if cands and p == cands[0] else
                           "licenses/ subdirectory" if p and "licenses" in p else
                           "dist-info basename" if p else None),
           "candidates_tried": cands}
    if rec["present"]:
        b = open(p, "rb").read()
        rec["sha256"] = hashlib.sha256(b).hexdigest()
        rec["bytes"] = len(b)
        for line in b.decode("utf-8", "replace").splitlines():
            if line.strip():
                rec["first_line"] = line.strip()[:200]
                break
    lf.append(rec)
out.update({
    "name": m["Name"],
    "version": m["Version"],
    "license_expression": m.get("License-Expression"),
    "license_field": m.get("License"),
    "license_classifiers": [c for c in (m.get_all("Classifier") or [])
                            if c.startswith("License")],
    "license_files": lf,
    "dist_info": base,
})
print(json.dumps(out))
"""
    try:
        p = subprocess.run([sys.executable, "-c", code, module],
                           capture_output=True, text=True, timeout=300)
        rec = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"module": module, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    # The declared licence, and WHICH field it came from. Never normalised into
    # an SPDX id we were not given -- a guess here is exactly the failure mode.
    # pandas ships no License-File and states BSD only inside the full licence
    # TEXT in its License field. The first line of that text is the one place
    # the name appears, so it is recorded rather than inferred from the
    # classifier alone.
    rec["license_field_first_line"] = _first_line(rec.get("license_field"))
    if rec.get("license_expression"):
        rec["license_declared"] = rec["license_expression"]
        rec["license_declared_from"] = "METADATA License-Expression"
    elif rec.get("license_field") and len(rec["license_field"]) < 80:
        rec["license_declared"] = rec["license_field"].strip()
        rec["license_declared_from"] = "METADATA License"
    elif rec.get("license_classifiers"):
        rec["license_declared"] = rec["license_classifiers"][0]
        rec["license_declared_from"] = "METADATA Classifier"
    elif rec.get("license_field"):
        rec["license_declared"] = _first_line(rec["license_field"])
        rec["license_declared_from"] = "METADATA License (first line of full text)"
    else:
        rec["license_declared"] = None
        rec["license_declared_from"] = None
    return rec


# ------------------------------------------------------------------- binaries
def _resolve_list(tree, varname):
    last = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == varname and isinstance(node.value, ast.List):
                last = node.value
    return last


def binaries_from_py(files: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for rel in files:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name not in {"run", "Popen", "call", "check_call", "check_output"}:
                continue
            mod = getattr(getattr(fn, "value", None), "id", None)
            if mod not in (None, "subprocess"):
                continue
            for arg in node.args[:1]:
                argv = arg
                if isinstance(argv, ast.Name):
                    argv = _resolve_list(tree, argv.id)
                if isinstance(argv, ast.List) and argv.elts:
                    head = argv.elts[0]
                    if isinstance(head, ast.Constant) and isinstance(head.value, str):
                        out.setdefault(head.value, set()).add(rel)
    out.pop("sys.executable", None)
    return out


def binaries_from_sh(files: list[str]) -> dict[str, set[str]]:
    """First word of every command line in our shell scripts that resolves on
    PATH and is not a shell keyword. `git`, `curl`, `tar`, `sha256sum` reach the
    inventory this way -- they are third-party programs the project depends on
    exactly as much as ffmpeg does."""
    out: dict[str, set[str]] = {}
    for rel in files:
        for raw in open(os.path.join(ROOT, rel), encoding="utf-8"):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for part in re.split(r"\|\||&&|\||;", line):
                part = part.strip()
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_.\-]*)\b", part)
                if not m:
                    continue
                word = m.group(1)
                if word in SH_NOT_A_BINARY or "=" in part.split(" ")[0]:
                    continue
                if shutil.which(word):
                    out.setdefault(word, set()).add(rel)
    return out


def binary_facts(cmd: str) -> dict:
    path = shutil.which(cmd)
    rec: dict = {"command": cmd, "path": path, "present": bool(path)}
    if not path:
        return rec
    for flag in ("--version", "-version"):
        try:
            p = subprocess.run([cmd, flag], capture_output=True, text=True, timeout=60)
        except Exception:  # noqa: BLE001
            continue
        text = (p.stdout or "") + "\n" + (p.stderr or "")
        lines = [x for x in text.splitlines() if x.strip()]
        if lines:
            rec["version_flag"] = flag
            rec["version_first_line"] = lines[0][:200]
            # ffmpeg prints its build configuration; --enable-gpl / --enable-nonfree
            # decide which licence the BINARY carries, and no other source does.
            cfg = next((x for x in lines if x.strip().startswith("configuration:")), None)
            if cfg:
                rec["build_configuration_flags"] = {
                    "enable_gpl": "--enable-gpl" in cfg,
                    "enable_version3": "--enable-version3" in cfg,
                    "enable_nonfree": "--enable-nonfree" in cfg,
                }
            break
    try:
        p = subprocess.run(["dpkg", "-S", os.path.realpath(path)],
                           capture_output=True, text=True, timeout=60)
        if p.returncode == 0 and ":" in p.stdout:
            pkg = p.stdout.split(":")[0].strip()
            rec["distro_package"] = pkg
            cp = f"/usr/share/doc/{pkg}/copyright"
            rec["copyright_file"] = cp if os.path.exists(cp) else None
            if rec["copyright_file"]:
                body = open(cp, encoding="utf-8", errors="replace").read()
                rec["copyright_file_licenses"] = sorted(
                    {m.group(1).strip() for m in re.finditer(r"^License:\s*(.+)$", body, re.M)})[:20]
                rec["copyright_files_star_license"] = _files_star_license(body)
    except Exception:  # noqa: BLE001
        rec["distro_package"] = None
    return rec


# ---------------------------------------------------------------------- fonts
def font_paths(files: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for rel in files:
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        for m in re.finditer(r"[\"'](/[^\"']*\.(?:ttf|otf|ttc))[\"']", text):
            out.setdefault(m.group(1), set()).add(rel)
    return out


def font_facts(path: str) -> dict:
    rec = {"path": path, "present": os.path.exists(path)}
    if not rec["present"]:
        return rec
    rec["sha256"] = sha256_file(path)
    try:
        p = subprocess.run(["dpkg", "-S", path], capture_output=True, text=True, timeout=60)
        if p.returncode == 0 and ":" in p.stdout:
            pkg = p.stdout.split(":")[0].strip()
            rec["distro_package"] = pkg
            cp = f"/usr/share/doc/{pkg}/copyright"
            rec["copyright_file"] = cp if os.path.exists(cp) else None
            if rec["copyright_file"]:
                body = open(cp, encoding="utf-8", errors="replace").read()
                rec["copyright_file_licenses"] = sorted(
                    {m.group(1).strip() for m in re.finditer(r"^License:\s*(.+)$", body, re.M)})
                rec["copyright_files_star_license"] = _files_star_license(body)
                rec["upstream_source"] = next(
                    (m.group(1).strip() for m in re.finditer(r"^Source:\s*(.+)$", body, re.M)), None)
    except Exception:  # noqa: BLE001
        rec["distro_package"] = None
    return rec


# ------------------------------------------------------- publisher licences
# This repository contains no network client, and that property is measured and
# claimed elsewhere in the entry -- so this script does NOT fetch. The
# publisher's own licence for each model and dataset is read from a pinned
# record, and every row carries the exact one-line command that reproduces it.
# An id we derived that has no pinned record is reported UNVERIFIED; it is never
# assumed permissive.
PINNED = os.path.join(ROOT, "evidence", "publisher_licenses.json")


def curl_line(kind: str, ident: str) -> str:
    return (f"curl -sSL https://huggingface.co/api/{kind}/{ident} "
            f"| python3 -c \"import json,sys; print(json.load(sys.stdin)['cardData']['license'])\"")


def load_pinned() -> dict:
    if not os.path.exists(PINNED):
        return {}
    try:
        return json.load(open(PINNED, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def hf_record(kind: str, ident: str, pinned: dict) -> dict:
    rec = {"id": ident,
           "api_url": f"https://huggingface.co/api/{kind}/{ident}",
           "reproduce": curl_line(kind, ident)}
    row = (pinned.get(kind) or {}).get(ident)
    if not row:
        rec.update({"verification": "UNVERIFIED",
                    "reason": f"no pinned record for {kind}/{ident} in "
                              f"{os.path.relpath(PINNED, ROOT)}",
                    "license": None})
        return rec
    rec.update({
        "verification": "PINNED",
        "canonical_id": row.get("canonical_id"),
        "license": row.get("license"),
        "license_name": row.get("license_name"),
        "license_link": row.get("license_link"),
        "license_tags": row.get("license_tags"),
        "gated": row.get("gated"),
        "private": row.get("private"),
        "last_modified": row.get("last_modified"),
        "fetched_at": row.get("fetched_at"),
        "fetched_by": row.get("fetched_by"),
    })
    return rec


# ------------------------------------------------------------- network egress
# Named because a claim of the form "we call no third-party API" is only worth
# something if the list of things looked for is written down. An empty result
# against an unstated candidate set reads as "no constraint".
VENDOR_CLIENTS = [
    "openai", "anthropic", "google", "google_generativeai", "generativeai",
    "vertexai", "cohere", "mistralai", "together", "replicate", "groq",
    "boto3", "botocore", "azure", "langchain", "llama_index", "litellm",
    "ollama", "huggingface_hub", "datasets", "requests", "httpx", "aiohttp",
    "urllib3", "websockets",
]
# The SAME thirteen names scripts/measure_environment.py sweeps. Two different
# network lists in one entry would be two different claims.
STDLIB_NET = ["asyncio", "ftplib", "http", "imaplib", "poplib", "smtplib", "socket",
              "socketserver", "ssl", "telnetlib", "urllib", "webbrowser", "xmlrpc"]


def network_egress(py: list[str], sh: list[str], imports: dict[str, list[str]]) -> dict:
    """Every way a byte can leave this repository, derived from the source."""
    urls: dict[str, set[str]] = {}
    for rel in py + sh:
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        for m in re.finditer(r"https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+", text):
            urls.setdefault(m.group(0).rstrip(".,)\"'"), set()).add(rel)

    stdlib_net: dict[str, set[str]] = {}
    hub_ids: dict[str, set[str]] = {}
    for rel in py:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding="utf-8").read(), rel)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else ([node.module] if node.module and not node.level else []))
                for n in names:
                    if n.split(".")[0] in STDLIB_NET:
                        stdlib_net.setdefault(n, set()).add(rel)
            # AutoTokenizer.from_pretrained("<hub id>") reaches huggingface.co
            # unless the argument is a local path. Both kinds are recorded.
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "from_pretrained":
                for a in node.args[:1]:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        hub_ids.setdefault(a.value, set()).add(rel)

    present = sorted(set(VENDOR_CLIENTS) & set(imports))
    return {
        "vendor_client_candidates_checked": VENDOR_CLIENTS,
        "vendor_clients_imported": present,
        "stdlib_network_modules_imported": {k: sorted(v) for k, v in sorted(stdlib_net.items())},
        "url_literals": {k: sorted(v) for k, v in sorted(urls.items())},
        "from_pretrained_literal_arguments": {k: sorted(v) for k, v in sorted(hub_ids.items())},
        "reading": "A hub id passed to from_pretrained downloads from huggingface.co on "
                   "first use; a path under models/ does not. Neither is a vendor API "
                   "call, and neither is nothing.",
    }


# ---------------------------------------------------------------------- build
def main(*argv: str) -> int:
    pinned = load_pinned()
    py = source_files((".py",))
    sh = source_files((".sh",))

    import_graph = imported_top_levels(py)
    libs = {}
    for mod, users in import_graph.items():
        rec = distribution_licence(mod)
        rec["imported_by"] = users
        libs[mod] = rec

    bins_seen: dict[str, set[str]] = {}
    for src in (binaries_from_py(py), binaries_from_sh(sh)):
        for k, v in src.items():
            bins_seen.setdefault(k, set()).update(v)
    bins = {}
    for cmd in sorted(bins_seen):
        rec = binary_facts(cmd)
        rec["used_by"] = sorted(bins_seen[cmd])
        bins[cmd] = rec

    fonts = {}
    for path, users in sorted(font_paths(py).items()):
        rec = font_facts(path)
        rec["used_by"] = sorted(users)
        fonts[path] = rec

    # --- models: ids derived from the artefacts, licences fetched per id
    route = ast.parse(open(os.path.join(ROOT, "airlock", "route.py"), encoding="utf-8").read())
    consts = {t.id: n.value.value
              for n in route.body if isinstance(n, ast.Assign)
              for t in n.targets
              if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    meta_path = os.path.join(ROOT, "models", "detector", "train_meta.json")
    base_model = json.load(open(meta_path))["base_model"] if os.path.exists(meta_path) else None
    model_rows = []
    for ident, role, src in (
        (base_model, "detector base checkpoint, fine-tuned and redistributed as a release asset",
         "models/detector/train_meta.json:base_model"),
        (consts.get("LOCAL_DEFAULT"), "local answering model behind the redacted prompt",
         "airlock/route.py:LOCAL_DEFAULT"),
        (consts.get("HOSTED_STANDIN_DEFAULT"),
         "larger local stand-in for the model Airlock is protecting the operator from",
         "airlock/route.py:HOSTED_STANDIN_DEFAULT"),
    ):
        if not ident:
            model_rows.append({"id": None, "role": role, "id_source": src,
                               "error": "id not derivable from the artefact"})
            continue
        row = hf_record("models", ident, pinned)
        row["role"] = role
        row["id_source"] = src
        model_rows.append(row)

    # --- dataset: the licence literal, and the licence the publisher states
    cache = os.environ.get("AIRLOCK_DATA_DIR", os.path.join(ROOT, ".data_cache"))
    mpath = os.path.join(cache, "data_manifest.json")
    declared_id = declared_lic = None
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        declared_id, declared_lic = man.get("source"), man.get("license")
        decl_from = "data_manifest.json written by scripts/prepare_data.py"
    else:
        src = open(os.path.join(ROOT, "scripts", "prepare_data.py"), encoding="utf-8").read()
        m = re.search(r'"source":\s*"([^"]+)".*?"license":\s*"([^"]+)"', src, re.S)
        if m:
            declared_id, declared_lic = m.group(1), m.group(2)
        decl_from = "scripts/prepare_data.py literal (data_manifest.json absent)"
    ds = hf_record("datasets", declared_id, pinned) if declared_id else {"id": None}
    ds.update({
        "role": "training and evaluation corpus for the detector",
        "declared_license_in_repo": declared_lic,
        "declared_license_source": decl_from,
        "publisher_license_agrees": (
            None if ds.get("verification") != "PINNED"
            else (declared_lic == ds.get("license"))),
    })

    lic_path = os.path.join(ROOT, "LICENSE")
    first_party = {
        "our_license_file": {
            "path": "LICENSE",
            "first_line": _first_line(open(lic_path, encoding="utf-8").read())
            if os.path.exists(lic_path) else None,
            "sha256": sha256_file(lic_path),
        },
        "demo_documents": {
            rel: {"sha256": sha256_file(os.path.join(ROOT, rel)),
                  "bytes": os.path.getsize(os.path.join(ROOT, rel))}
            for rel in ("demo/docs/payment_instruction.txt", "demo/docs/support_ticket.txt")
            if os.path.exists(os.path.join(ROOT, rel))
        },
        "demo_documents_provenance": "demo/docs/README.md",
    }

    out = {
        "generated_at": now(),
        "what": "Third-party inventory for UPAI-Hackdays rule 6 (Intellectual Property), "
                "derived from this working tree and this host, not transcribed.",
        "publisher_licenses_pinned_at": os.path.relpath(PINNED, ROOT)
        if os.path.exists(PINNED) else None,
        "source_files_parsed": {"py": len(py), "sh": len(sh)},
        "python_libraries": libs,
        "external_binaries": bins,
        "fonts": fonts,
        "models": model_rows,
        "dataset": ds,
        "first_party": first_party,
        "network_egress": network_egress(py, sh, import_graph),
    }
    unknown = [k for k, v in libs.items() if not v.get("license_declared")]
    out["summary"] = {
        "libraries": len(libs),
        "libraries_with_no_declared_license": unknown,
        "binaries": len(bins),
        "binaries_absent": [k for k, v in bins.items() if not v.get("present")],
        "fonts": len(fonts),
        "fonts_absent": [k for k, v in fonts.items() if not v.get("present")],
        "models": len([m for m in model_rows if m.get("id")]),
        "models_license_unresolved": [m["id"] for m in model_rows
                                      if m.get("id") and not m.get("license")],
        "dataset_license_agrees_with_publisher": ds.get("publisher_license_agrees"),
        "vendor_clients_imported": out["network_egress"]["vendor_clients_imported"],
        "vendor_client_candidates_checked": len(VENDOR_CLIENTS),
    }
    os.makedirs(os.path.dirname(EVIDENCE), exist_ok=True)
    with open(EVIDENCE, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(json.dumps(out["summary"], indent=2))
    print(f"wrote {os.path.relpath(EVIDENCE, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
