"""Controls for scripts/pathredact.py -- the host-path floor on a public repo.

Three evidence files here are produced by scanning this machine: the rule 2
copy search walks every project on it, and the dependency inventory walks the
Python installation. That is the right way to make those claims falsifiable,
but the raw output names directories belonging to work that has nothing to do
with this hackathon, and the output is committed somewhere anyone can read.

So absolute paths are rewritten on the way out. This file is what makes that
believable:

  A1 ACCEPT      no tracked file in this repository contains a path that
                 redaction would still rewrite
  C1 POSITIVE    the same scanner, run over a tree with a leak planted in it,
                 finds the leak. Without C1, A1 passes for a scanner that has
                 simply stopped looking
  C1b BINARY     and a host marker planted in a *binary* is found by the other
                 half of the scanner, which is the half A1 would otherwise be
                 trusting blind
  C2 PLACEHOLDER redacted output is not itself reported as a leak -- a checker
                 that fires on its own product is unusable
  C3 URL         `https://host/owner/name` is neither reported nor mangled
  C4 SYSTEM      /usr, /tmp and friends are the same on every host and stay
  C5 DOTDOT      `.../site-packages/pyarrow/../../../libarrow.so` inside an
                 already-redacted path is not a second, fresh path
  C6 DISTINCT    two files in one directory get the SAME token; two in
                 different directories get DIFFERENT ones. This is the property
                 the evidence rests on, so redaction may not flatten it
  C7 OPAQUE      no directory name from the original path survives in its token
  C8 IDEMPOTENT  redacting twice is redacting once
  C9 EMIT-OK     emit_json writes a clean document and reports success
  C10 EMIT-STOP  emit_json handed a redactor that MISSES writes nothing and
                 reports failure. C9 without C10 would pass for a guard that
                 always writes
  C11 NUMBERS    redaction moves no measurement: every number in a document
                 survives it unchanged
  C14 SELF       the redactor and this suite are themselves in the scanned set,
                 because a leak in either is the one least likely to be noticed
  C13 NO-SYSTEM  no ordinary system location is ever registered as a prefix,
                 which is what goes wrong when the repository is cloned under
                 /tmp and the directory above it is /tmp itself
  C12 REGRESSION the pattern still finds every path in the bytes that actually
                 leaked -- the five files as they stood at commit a2db334, the
                 tree an independent judge read. A pattern tightened until it
                 stops matching binary noise could be tightened until it stops
                 matching anything; this is what stops that

Note on the fixtures: every leaking path here is assembled from pieces at run
time and none appears as a literal in this file. None can -- A1 scans every
tracked file, this file and scripts/pathredact.py included, and there is no
exemption list, so a literal here would make the repository fail its own floor.
That is the floor working, not a workaround; C14 asserts that both files really
are in the scanned set rather than quietly skipped.

    python3 tests/check_no_host_disclosure.py
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.pathredact import (          # noqa: E402
    _is_system, _prefix_map, emit_json, find_leaks, host_markers,
    redact_obj, redact_path, redact_text,
)

results = []


def check(name, expect, ok, detail=""):
    results.append((name, expect, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {expect}" + (f"  [{detail}]" if detail else ""))


MARKERS = host_markers()


def scan_tree(root):
    """Every tracked file under `root`, mapped to the leaks it still carries.

    git ls-files rather than a walk, because what matters is what is
    *published*: an untracked scratch file is not in the repository.

    Text files are scanned by pattern. Binaries are scanned for the host
    markers instead -- see scripts/pathredact.host_markers -- because decoding
    the demo video as text yields two-character runs separated by slashes, and
    a pattern loose enough to catch a real path matches those too. Binaries are
    counted and named in the result rather than quietly dropped, so what was
    checked how is visible at the call site.
    """
    out = subprocess.run(["git", "-C", root, "ls-files", "-z"],
                         capture_output=True, text=True, check=True).stdout
    leaks, text_n, binary = {}, 0, []
    for rel in out.split("\0"):
        if not rel:
            continue
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            body = fh.read()
        if b"\0" in body[:8000]:                     # git's own heuristic
            binary.append(rel)
            hits = [m for m in MARKERS if m.encode() in body]
        else:
            text_n += 1
            hits = find_leaks(body.decode("utf-8", "ignore"))
        if hits:
            leaks[rel] = sorted(set(hits))
    return leaks, text_n, binary


# --- A1: the real tree is clean ---------------------------------------------
leaks, text_n, binary = scan_tree(REPO)
check("A1 accept", "no tracked file carries a redactable host path",
      not leaks and text_n > 0,
      f"text={text_n} binary={len(binary)}{tuple(binary) if binary else ''} "
      f"files_with_leaks={len(leaks)}"
      + (f" first={sorted(leaks)[0]}:{leaks[sorted(leaks)[0]][0]}" if leaks else ""))

# --- C1: the same scanner finds a planted leak ------------------------------
# Assembled, never written out: see the note in the module docstring.
PLANTED = os.sep.join(["", "srv", "some-other-project", "notes.md"])
tmp = tempfile.mkdtemp(prefix="upai_disclosure_ctrl_")
subprocess.run(["git", "init", "-q", tmp], check=True)
with open(os.path.join(tmp, "leaky.json"), "w", encoding="utf-8") as fh:
    json.dump({"found_at": PLANTED, "count": 1}, fh)
subprocess.run(["git", "-C", tmp, "add", "-A"], check=True)
# ...and a binary carrying a real host marker must be caught too, by the other
# half of the scanner. Written as bytes with a NUL so it is classified binary.
with open(os.path.join(tmp, "leaky.bin"), "wb") as fh:
    tail = os.sep.join(["", "some", "private", "thing"]).encode()
    fh.write(b"\x00\x01\x02" + MARKERS[0].encode() + tail + b"\x00")
subprocess.run(["git", "-C", tmp, "add", "-A"], check=True)
planted_leaks, planted_text, planted_binary = scan_tree(tmp)
check("C1 positive", "a planted host path is found by the same scanner",
      planted_leaks.get("leaky.json") == [PLANTED] and planted_text == 1,
      f"found={planted_leaks.get('leaky.json')}")
check("C1b positive-binary", "a host marker inside a binary is found",
      planted_leaks.get("leaky.bin") == [MARKERS[0]] and planted_binary == ["leaky.bin"],
      f"found={planted_leaks.get('leaky.bin')} classified={planted_binary}")

# --- C2: redacted output is not a leak --------------------------------------
redacted_forms = ["<repo>/evidence/originality.json",
                  "<workspace>/submission",
                  "<python-prefix>/lib/python3.13/site-packages/torch/__init__.py",
                  "<other-508830>/.../LICENSE"]
check("C2 placeholder", "redaction output is not itself reported as a leak",
      all(find_leaks(f) == [] for f in redacted_forms),
      f"{sum(1 for f in redacted_forms if find_leaks(f))} of {len(redacted_forms)} misread")

# --- C3: URLs survive untouched ---------------------------------------------
urls = ["https://github.com/jianwang-ntu/airlock-privacy-gate",
        "https://huggingface.co/distilbert-base-cased/resolve/main/config.json",
        "http://example.org/a/b/c"]
check("C3 url", "a URL is neither reported nor rewritten",
      all(find_leaks(u) == [] and redact_text(u) == u for u in urls),
      f"{sum(1 for u in urls if redact_text(u) != u)} of {len(urls)} mangled")

# --- C4: ordinary system locations survive ----------------------------------
system = ["/usr/bin/env", "/etc/hosts", "/tmp/airlock_deploy_cpu_0.2.json",
          "/lib/x86_64-linux-gnu/libstdc++.so.6"]
check("C4 system", "standard filesystem locations are left alone",
      all(find_leaks(p) == [] and redact_path(p) == p for p in system),
      f"{sum(1 for p in system if redact_path(p) != p)} of {len(system)} rewritten")

# --- C5: `..` inside an already-redacted path is not a fresh path -----------
dotdot = ("version `CXXABI_1.3.15' not found (required by <python-prefix>"
          "/lib/python3.13/site-packages/pyarrow/../../../libarrow.so.2100)")
check("C5 dotdot", "`..` segments do not read as a second absolute path",
      find_leaks(dotdot) == [], f"leaks={find_leaks(dotdot)}")

# --- C6: distinctness is preserved ------------------------------------------
same_a = os.sep.join(["", "srv", "proj-alpha", "build", "LICENSE"])
same_b = os.sep.join(["", "srv", "proj-alpha", "build", "README.md"])
other  = os.sep.join(["", "srv", "proj-beta", "build", "LICENSE"])
ta, tb, tc = (redact_path(p).split("/")[0] for p in (same_a, same_b, other))
check("C6 distinct", "same directory -> same token; different -> different",
      ta == tb and ta != tc, f"a={ta} b={tb} other={tc}")

# --- C7: no component of the original survives in the token -----------------
tok = redact_path(other)
check("C7 opaque", "no directory name from the path survives in its token",
      "proj-beta" not in tok and "srv" not in tok and tok.endswith("/LICENSE"),
      f"token={tok}")

# --- C8: redacting twice is redacting once ----------------------------------
doc = {"roots": [same_a, other], "note": f"walked {same_b} and /usr/bin/env"}
once = redact_obj(doc)
twice = redact_obj(once)
check("C8 idempotent", "redact(redact(x)) == redact(x)", once == twice,
      "" if once == twice else json.dumps(once)[:120])

# --- C9 / C10: the emit guard, both ways ------------------------------------
ok_path = os.path.join(tmp, "emitted_ok.json")
rc_ok = emit_json({"found_at": same_a, "n": 3}, ok_path)
wrote_ok = os.path.exists(ok_path)
check("C9 emit-ok", "a clean document is written and reported as written",
      rc_ok == 0 and wrote_ok and find_leaks(open(ok_path, encoding="utf-8").read()) == [],
      f"rc={rc_ok} written={wrote_ok}")

stop_path = os.path.join(tmp, "emitted_stop.json")
rc_stop = emit_json({"found_at": same_a, "n": 3}, stop_path, _redact=lambda o: o)
check("C10 emit-stop", "a redactor that misses causes a refusal, and no file",
      rc_stop == 2 and not os.path.exists(stop_path),
      f"rc={rc_stop} written={os.path.exists(stop_path)}")

# --- C11: no measurement moves ----------------------------------------------
numeric = {"files_walked": 5484328, "seconds": 64.3, "matches": 2,
           "max_jaccard": 0.1, "where": same_a,
           "nested": {"bytes": 2262703, "ratio": 0.0009, "at": other}}


def numbers(o):
    if isinstance(o, dict):
        return [n for v in o.values() for n in numbers(v)]
    if isinstance(o, list):
        return [n for v in o for n in numbers(v)]
    return [o] if isinstance(o, (int, float)) and not isinstance(o, bool) else []


check("C11 numbers", "redaction changes no number in the document",
      numbers(redact_obj(numeric)) == numbers(numeric),
      f"before={numbers(numeric)} after={numbers(redact_obj(numeric))}")

# --- C12: the pattern still catches what actually leaked ---------------------
# Byte-anchored to the tree the round-1 judge read, by sha, so tightening the
# pattern later cannot quietly blind it. Counts only -- naming the paths here
# would put them back into a tracked file and fail A1.
AUDITED = "a2db334"
EXPECTED = {
    "evidence/environment.json": (7, 7),
    "evidence/originality.json": (40, 37),
    "evidence/third_party.json": (38, 22),
    "ORIGINALITY.md": (3, 2),
    "scripts/measure_originality.py": (3, 3),
}
got, missing = {}, None
for rel, _ in EXPECTED.items():
    r = subprocess.run(["git", "-C", REPO, "show", f"{AUDITED}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        missing = rel
        break
    hits = find_leaks(r.stdout)
    got[rel] = (len(hits), len(set(hits)))
check("C12 regression",
      f"every host path present at {AUDITED} is still found",
      missing is None and got == EXPECTED,
      f"missing={missing}" if missing else
      f"got={got}" if got != EXPECTED else f"{sum(v[0] for v in got.values())} paths")

# --- C13: no ordinary system location is ever registered as a prefix --------
# Clone this repository into /tmp/x and the directory "above the workspace" is
# /tmp. Registering it rewrites every /tmp path in the tree and the floor
# reports the repository's own clean-clone transcript as a leak. C4 catches the
# symptom only on a host where the repo is not under /tmp; this catches the
# cause anywhere.
registered = [p for p, _ in _prefix_map()]
swallowed = [p for p in registered if _is_system(p)]
check("C13 no-system-prefix",
      "no /usr, /tmp or similar is registered as a redaction prefix",
      not swallowed and registered,
      f"registered={len(registered)} swallowed={swallowed}")

# --- C14: the floor's own source is inside the floor ------------------------
# The two files that implement redaction are the two where a leaked path would
# be least likely to be noticed, so the useful thing to assert is not that they
# pass but that they were LOOKED AT. An exemption added here later would show
# up as this control failing.
tracked = subprocess.run(["git", "-C", REPO, "ls-files"],
                         capture_output=True, text=True, check=True).stdout.split()
own = ["scripts/pathredact.py", "tests/check_no_host_disclosure.py"]
check("C14 self-scanned", "the redactor and this suite are themselves scanned",
      all(f in tracked for f in own)
      and all(f not in leaks for f in own) and len(tracked) > 50,
      f"tracked={len(tracked)} own_files_present={[f in tracked for f in own]}")

failed = [r for r in results if not r[2]]
print(f"\n{len(results) - len(failed)}/{len(results)} controls passed")
if failed:
    for name, expect, _, detail in failed:
        print(f"  FAILED {name}: {expect} [{detail}]")
sys.exit(1 if failed else 0)
