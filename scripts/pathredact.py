"""Publish *where* a file was found without publishing the machine it was on.

Three of this repository's evidence files are produced by scanning the host:
`originality.json` walks every source tree on the machine looking for a prior
project this entry could have been copied from, and `third_party.json` /
`environment.json` walk the installed libraries to record their versions and
licences. Those scans are the point -- a copy search that only looks inside its
own repository proves nothing -- but their raw output names directories that
have nothing to do with this hackathon, and that output is committed to a
public repository.

So the paths are rewritten at emit time:

  the repository itself            -> `<repo>`
  the workspace holding it         -> `<workspace>`
  the Python installation          -> `<python-prefix>`
  ordinary system locations        -> unchanged (/usr, /etc, /tmp, ...)
  anything else                    -> `<other-XXXXXX>/.../basename`

`XXXXXX` is `sha256(parent directory)[:6]`. That keeps every property the
evidence actually rests on -- two matches in the same directory get the same
token, two in different directories get different ones, and the count of
distinct locations is unchanged -- while a reader who does not already know a
path cannot recover it from the token. The operator can demonstrate any single
mapping on request by hashing the path; nobody can enumerate them.

The basename survives because it is doing evidential work and discloses
nothing: a judge reading the copy search needs to see that both exact matches
are files called `LICENSE`, which is why they are carved out of the verdict.

    python3 scripts/pathredact.py --in evidence/originality.json
    python3 scripts/pathredact.py --in a.json --out b.json --report

Controls live in tests/check_no_host_disclosure.py, which plants a leak and
requires this module to catch it before believing that it found none here.
"""
import argparse
import hashlib
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Locations that are the same on every Linux host and name nothing private.
SYSTEM_PREFIXES = (
    "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/opt/conda",
    "/proc", "/sys", "/dev", "/tmp", "/var/tmp", "/var/run", "/run",
)

# An absolute path of at least two components. Stops at the characters that
# terminate a path inside prose, a JSON string or a Python error message.
#
# The lookbehind matters more than it looks. Without it the pattern re-matches
# its own output -- `<repo>/evidence/x.json` contains an apparent absolute path
# starting at `evidence` -- and it chews the middle out of a URL, matching from
# `github.com` onwards inside `https://github.com/owner/name`. Excluding a
# preceding word character, `>`, `:`, `/` or `.` rules out both, and the
# control suite plants one of each. The `.` is for `..` segments: a real
# message carries `site-packages/pyarrow/../../../libarrow.so`, and without
# it the tail beginning at the first `..` reads as a fresh absolute path, and
# an already-redacted file reports itself as still leaking.
#
# Components are restricted to characters that appear in real path names. A
# looser class was tried first and it matched two-character runs inside the
# demo video: scanning a committed binary as text produces byte sequences that
# a permissive pattern reads as paths, and a floor that cries wolf on an mp4
# gets switched off. tests/check_no_host_disclosure.py re-checks the
# pre-redaction evidence with this pattern and requires every real path in it
# to still be found.
#
# One house rule follows from that suite: no path-shaped literal may appear in
# this file or in the suite itself. Both are tracked, both are scanned, and
# there is no exemption list -- so an example path goes in without its leading
# slash, and a fixture is assembled at run time.
PATH_RE = re.compile(r"(?<![\w>:/.])/[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+@%~-]+)+")


def _is_system(path):
    return any(path == p or path.startswith(p + "/") for p in SYSTEM_PREFIXES)


def _prefix_map():
    """Longest-first list of (real_prefix, placeholder).

    Built from the running interpreter and this file's own location, so the
    module carries no absolute path of its own.
    """
    workspace = os.path.dirname(REPO)
    pairs = [
        (REPO, "<repo>"),
        (workspace, "<workspace>"),
        (sys.prefix, "<python-prefix>"),
        (sys.base_prefix, "<python-prefix>"),
    ]
    out = []
    for p, tag in pairs:
        for variant in {p, os.path.realpath(p)}:
            # A prefix that IS an ordinary system location must never be
            # registered. Cloned into /tmp/x, the workspace above the repo is
            # /tmp, and registering it rewrites every /tmp path in the tree as
            # `<workspace>/...` -- the floor then reports the repository's own
            # clean-clone transcript as a leak. Caught by running this suite
            # against a fresh clone rather than in place; C13 keeps it caught.
            if variant and variant != "/" and not _is_system(variant):
                out.append((variant, tag))
    # Longest prefix wins, so <repo> is applied before <workspace>.
    out.sort(key=lambda t: len(t[0]), reverse=True)
    seen, uniq = set(), []
    for p, tag in out:
        if p not in seen:
            seen.add(p)
            uniq.append((p, tag))
    return uniq


def redact_path(path, prefixes=None):
    """Rewrite one absolute path. Relative paths are returned untouched."""
    if not path.startswith("/"):
        return path
    for prefix, tag in (prefixes if prefixes is not None else _prefix_map()):
        if path == prefix:
            return tag
        if path.startswith(prefix + "/"):
            return tag + path[len(prefix):]
    if _is_system(path):
        return path
    parent, base = os.path.split(path.rstrip("/"))
    token = hashlib.sha256(parent.encode("utf-8")).hexdigest()[:6]
    if not base:
        return "<other-%s>" % token
    return "<other-%s>/.../%s" % (token, base)


def redact_text(text, prefixes=None):
    """Rewrite every absolute path appearing anywhere in a string."""
    if prefixes is None:
        prefixes = _prefix_map()
    return PATH_RE.sub(lambda m: redact_path(m.group(0), prefixes), text)


def redact_obj(obj, prefixes=None):
    """Rewrite paths in every string of a JSON-shaped structure, keys included."""
    if prefixes is None:
        prefixes = _prefix_map()
    if isinstance(obj, str):
        return redact_text(obj, prefixes)
    if isinstance(obj, list):
        return [redact_obj(v, prefixes) for v in obj]
    if isinstance(obj, dict):
        return {redact_text(k, prefixes): redact_obj(v, prefixes)
                for k, v in obj.items()}
    return obj


def find_leaks(text, prefixes=None):
    """Every absolute path in `text` that redaction would still rewrite.

    Used by the control suite and by tests/check_no_host_disclosure.py. An
    empty list means the text is already clean; it does NOT mean the scanner
    looked, which is why every caller pairs it with a planted positive.
    """
    if prefixes is None:
        prefixes = _prefix_map()
    leaks = []
    for m in PATH_RE.finditer(text):
        raw = m.group(0)
        if redact_path(raw, prefixes) != raw:
            leaks.append(raw)
    return leaks


def emit_json(obj, path, indent=1, _redact=None):
    """Redact, verify, then write -- or refuse and write nothing.

    Every generator that walks this host funnels its output through here, so
    the guarantee is one function rather than three copies of a comment. The
    refusal is the point: if a path survives redaction, publishing the file is
    worse than failing the run, because the file is committed to a public
    repository and a leak there cannot be recalled.

    `_redact` exists so the control suite can hand in a redactor that misses,
    and watch the refusal fire. Nothing else should pass it.

    Returns 0 on write, 2 on refusal.
    """
    redactor = _redact or redact_obj
    text = json.dumps(redactor(obj), indent=indent, ensure_ascii=False) + "\n"
    leaks = find_leaks(text)
    if leaks:
        print("REFUSED: %d host path(s) survived redaction; nothing written to %s"
              % (len(leaks), path), file=sys.stderr)
        return 2
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return 0


def host_markers():
    """The shortest absolute prefixes that identify *this* machine.

    Text is scanned by pattern; a committed binary cannot be, because decoding
    an mp4 as text produces slash-separated two-character byte runs that any
    path pattern loose enough to be useful will match. So binaries are checked for these instead:
    the top two components of every location this process knows to be real --
    the repository, the workspace above it, the Python installation, the home
    directory. A binary that had a host path baked into it would contain one of
    these verbatim, and noise will not.

    Derived at run time from the running process, so no host path is written
    into this file.
    """
    seeds = [REPO, os.path.dirname(REPO), sys.prefix, sys.base_prefix,
             os.path.expanduser("~")]
    markers = set()
    for seed in seeds:
        if not seed or not seed.startswith("/"):
            continue
        parts = [c for c in os.path.realpath(seed).split("/") if c]
        if len(parts) >= 2:
            markers.add("/" + "/".join(parts[:2]))
    return sorted(markers)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", default=None,
                    help="default: rewrite --in in place")
    ap.add_argument("--report", action="store_true",
                    help="print what changed instead of staying silent")
    args = ap.parse_args(argv)

    dst = args.dst or args.src
    raw = open(args.src, encoding="utf-8").read()
    prefixes = _prefix_map()

    before = find_leaks(raw, prefixes)
    text = redact_text(raw, prefixes)
    after = find_leaks(text, prefixes)

    # Redaction rewrites path substrings in place; it must not reformat the
    # file or disturb a single number. Both halves are checked before the file
    # is written: the result still parses, and it is exactly what redacting the
    # parsed original would have produced. A measurement that changed while
    # being redacted would fail here rather than ship.
    doc_before = json.loads(raw)
    doc_after = json.loads(text)
    if doc_after != redact_obj(doc_before, prefixes):
        print("REFUSED: redaction changed something other than a path in %s"
              % args.src, file=sys.stderr)
        return 2

    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text)

    if args.report:
        print(json.dumps({
            "src": redact_path(os.path.abspath(args.src), prefixes),
            "paths_rewritten": len(before),
            "distinct_rewritten": len(set(before)),
            "leaks_remaining": len(after),
            "structure_unchanged_apart_from_paths": True,
        }, indent=1))
    return 1 if after else 0


if __name__ == "__main__":
    raise SystemExit(main())
