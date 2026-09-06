#!/usr/bin/env python3
"""Measure this repository against UPAI-Hackdays rule 2 (Originality).

Rule 2, as published on the hackathon's own Rules page, is three sentences:

  1. "All submissions must be original work created by the participating team."
  2. "Previously developed projects may not be submitted as-is."
  3. "Teams may use existing open-source libraries, APIs, and tools, but their
      contribution and implementation should be clearly demonstrated."

Sentence 2 is the only one of the three that is *falsifiable against this host*,
and it is the one this script exists for. It does not assert that nothing was
copied; it goes looking for a copy and reports what it found.

Two searches, because they fail differently:

  EXACT      every tracked file's sha256, against every file on the search
             roots whose *size* matches one of them. Catches a verbatim copy in
             either direction. Cheap: a full hash is only computed for a size
             collision.

  NEAR       normalised-line Jaccard against every text file under the focused
             root, through an inverted index so the cost is linear in candidate
             lines rather than quadratic. Catches a copy that was then edited,
             which EXACT cannot see.

Both searches report their own coverage -- files walked, directories that could
not be read -- because a search that silently walked nothing would otherwise
look exactly like a search that found nothing.

Writes evidence/originality.json. Host-dependent by construction: the search
roots are this machine's. On a machine that does not have them the status is
NOT_RUN, never PASS.

    python3 scripts/measure_originality.py [--roots ...] [--focus ...] [--out ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat as statmod
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Declared, not measured here: the hackathon's own clock, as Devpost reported
# it. Quoted so a clean clone is self-contained; asserted against the pinned
# capture of get_key_dates by submission/check_fields.py in the workspace.
# ---------------------------------------------------------------------------
HACKATHON = {
    "slug": "upai-hackdays",
    "devpost_id": 31216,
    "published_at": "2026-09-02T05:52:35Z",
    "submissions_start_at": "2026-09-07T04:00:00Z",
    "submissions_end_at": "2026-09-10T11:30:00Z",
    "source": "Devpost get_key_dates + get_hackathon_overview, captured "
              "2026-09-06T04:34:51Z; the capture is pinned in the workspace at "
              "rules_canonical/20260906T0430Z/mcp_surfaces.json",
}

RULE_2 = [
    "All submissions must be original work created by the participating team.",
    "Previously developed projects may not be submitted as-is.",
    "Teams may use existing open-source libraries, APIs, and tools, but their "
    "contribution and implementation should be clearly demonstrated.",
]

DEFAULT_ROOTS = ["/data/wj/wj_code", "/data/wj/anaconda/lib"]
DEFAULT_FOCUS = ["/data/wj/wj_code/dl_hackathon"]

TEXT_SUFFIXES = {".py", ".md", ".sh", ".txt"}
NEAR_MAX_BYTES = 512 * 1024
NEAR_MIN_LINE = 8
NEAR_REPORT = 0.30      # everything at or above this is listed
NEAR_FLAG = 0.50        # at or above this is called a probable copy
SKIP_DIR_NAMES = {".git", "__pycache__", ".mypy_cache", ".pytest_cache"}


def _is_self(path: str, selfdir: str) -> bool:
    """True when a match sits inside THIS job's own workspace.

    Rule 2 forbids submitting a *previously developed project*. Material under
    workspaces/upai-hackdays/ was written for this hackathon and is part of this
    entry, so a match there is not a prior project. It is still counted and
    listed, so the carve-out is visible rather than assumed.
    """
    rp = os.path.realpath(path)
    return rp == selfdir or rp.startswith(selfdir + os.sep)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_lines(text: str) -> set[str]:
    """Normalised line multiset -> set of line hashes.

    Blank lines and very short lines are dropped: `)`, `import os` and `}` are
    shared by every Python file ever written and would inflate every score.
    """
    out = set()
    for raw in text.splitlines():
        s = " ".join(raw.split())
        if len(s) < NEAR_MIN_LINE:
            continue
        out.add(hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:16])
    return out


# ---------------------------------------------------------------------------
# 1. Repository timeline, out of git
# ---------------------------------------------------------------------------
def repo_timeline() -> dict:
    fmt = "%H%x1f%aI%x1f%cI%x1f%an%x1f%ae%x1f%s"
    commits = []
    for line in git("log", "--reverse", f"--format={fmt}").splitlines():
        sha, adate, cdate, name, email, subject = line.split("\x1f")
        commits.append({
            "sha": sha,
            "authored_utc": datetime.fromisoformat(adate).astimezone(timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "committed_utc": datetime.fromisoformat(cdate).astimezone(timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "author_name": name,
            "author_email": email,
            "subject": subject,
        })
    first = commits[0]["authored_utc"]
    last = commits[-1]["authored_utc"]

    def secs(a: str, b: str) -> int:
        fa = datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        fb = datetime.strptime(b, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int((fb - fa).total_seconds())

    published = HACKATHON["published_at"]
    opens = HACKATHON["submissions_start_at"]
    return {
        "commit_count": len(commits),
        "commits": commits,
        "first_commit_utc": first,
        "last_commit_utc": last,
        "span_seconds": secs(first, last),
        "root_commit_is_first_content": len(git("rev-list", "--max-parents=0", "HEAD").split()) == 1,
        "vs_hackathon": {
            "hackathon_published_at": published,
            "seconds_after_publication": secs(published, first),
            "first_commit_after_publication": secs(published, first) > 0,
            "submissions_open_at": opens,
            "seconds_before_submissions_open": secs(first, opens),
            "first_commit_before_submissions_open": secs(first, opens) > 0,
        },
    }


# ---------------------------------------------------------------------------
# 2. Exact-copy search
# ---------------------------------------------------------------------------
def exact_scan(tracked: dict[str, dict], roots: list[Path], exclude: set[str],
               selfdir: str) -> dict:
    by_size: dict[int, list[str]] = {}
    for rel, meta in tracked.items():
        if meta["bytes"] == 0:
            continue
        by_size.setdefault(meta["bytes"], []).append(rel)
    by_sha = {meta["sha256"]: rel for rel, meta in tracked.items() if meta["bytes"]}

    walked = 0
    size_hits = 0
    unreadable: list[str] = []
    matches: list[dict] = []
    t0 = time.time()

    for root in roots:
        if not root.exists():
            unreadable.append(f"{root} (absent)")
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False,
                                                    onerror=lambda e: unreadable.append(str(e))):
            rp = os.path.realpath(dirpath)
            if any(rp == ex or rp.startswith(ex + os.sep) for ex in exclude):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    st = os.lstat(p)
                except OSError as exc:
                    unreadable.append(f"{p}: {exc}")
                    continue
                if not statmod.S_ISREG(st.st_mode):
                    continue
                walked += 1
                if st.st_size not in by_size:
                    continue
                size_hits += 1
                try:
                    digest = sha256_file(Path(p))
                except OSError as exc:
                    unreadable.append(f"{p}: {exc}")
                    continue
                if digest in by_sha:
                    matches.append({
                        "tracked_file": by_sha[digest],
                        "sha256": digest,
                        "bytes": st.st_size,
                        "found_at": p,
                        "found_mtime_utc": datetime.fromtimestamp(
                            st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "in_this_jobs_own_workspace": _is_self(p, selfdir),
                    })

    return {
        "roots": [str(r) for r in roots],
        "excluded_prefixes": sorted(exclude),
        "skipped_dir_names": sorted(SKIP_DIR_NAMES),
        "zero_byte_tracked_files_excluded": sorted(
            rel for rel, m in tracked.items() if m["bytes"] == 0),
        "zero_byte_note": "A zero-byte file matches every other zero-byte file "
                          "on the host and carries no authorship, so it is not "
                          "a copy question. Both are package markers.",
        "files_walked": walked,
        "size_collisions_hashed": size_hits,
        "unreadable_count": len(unreadable),
        "unreadable_sample": unreadable[:20],
        "self_workspace": selfdir,
        "matches": matches,
        "match_count": len(matches),
        "match_count_outside_this_job": sum(
            1 for m in matches if not m["in_this_jobs_own_workspace"]),
        "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 3. Near-duplicate search (inverted index; linear in candidate lines)
# ---------------------------------------------------------------------------
def near_scan(tracked_text: dict[str, set[str]], focus: list[Path],
              exclude: set[str], selfdir: str) -> dict:
    index: dict[str, list[str]] = {}
    for rel, lines in tracked_text.items():
        for h in lines:
            index.setdefault(h, []).append(rel)

    walked = 0
    compared = 0
    unreadable: list[str] = []
    hits: list[dict] = []
    # "nothing scored above 0.30" is a weak sentence on its own -- it is also
    # what a broken comparator says. The best score actually observed is kept
    # so the threshold can be seen to have headroom under it.
    top: list[tuple[float, dict]] = []
    t0 = time.time()

    for root in focus:
        if not root.exists():
            unreadable.append(f"{root} (absent)")
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False,
                                                    onerror=lambda e: unreadable.append(str(e))):
            rp = os.path.realpath(dirpath)
            if any(rp == ex or rp.startswith(ex + os.sep) for ex in exclude):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for fn in filenames:
                if os.path.splitext(fn)[1] not in TEXT_SUFFIXES:
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.lstat(p)
                    if not statmod.S_ISREG(st.st_mode) or st.st_size > NEAR_MAX_BYTES:
                        continue
                    text = Path(p).read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    unreadable.append(f"{p}: {exc}")
                    continue
                walked += 1
                cand = norm_lines(text)
                if not cand:
                    continue
                overlap: dict[str, int] = {}
                for h in cand:
                    for rel in index.get(h, ()):
                        overlap[rel] = overlap.get(rel, 0) + 1
                if not overlap:
                    continue
                compared += 1
                for rel, n in overlap.items():
                    union = len(cand) + len(tracked_text[rel]) - n
                    j = n / union if union else 0.0
                    if len(top) < 10 or j > top[-1][0]:
                        top.append((j, {"tracked_file": rel, "candidate": p,
                                        "jaccard": round(j, 4), "shared_lines": n,
                                        "candidate_lines": len(cand),
                                        "tracked_lines": len(tracked_text[rel]),
                                        "in_this_jobs_own_workspace": _is_self(p, selfdir)}))
                        top.sort(key=lambda t: -t[0])
                        del top[10:]
                    if j >= NEAR_REPORT:
                        hits.append({
                            "tracked_file": rel,
                            "candidate": p,
                            "jaccard": round(j, 4),
                            "shared_lines": n,
                            "candidate_lines": len(cand),
                            "tracked_lines": len(tracked_text[rel]),
                            "candidate_mtime_utc": datetime.fromtimestamp(
                                st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "flagged_probable_copy": j >= NEAR_FLAG,
                            "in_this_jobs_own_workspace": _is_self(p, selfdir),
                        })

    hits.sort(key=lambda h: -h["jaccard"])
    return {
        "max_jaccard_observed": round(top[0][0], 4) if top else None,
        "top_similarities": [t[1] for t in top],
        "focus_roots": [str(r) for r in focus],
        "excluded_prefixes": sorted(exclude),
        "suffixes": sorted(TEXT_SUFFIXES),
        "max_candidate_bytes": NEAR_MAX_BYTES,
        "min_normalised_line_chars": NEAR_MIN_LINE,
        "report_threshold": NEAR_REPORT,
        "flag_threshold": NEAR_FLAG,
        "tracked_text_files": len(tracked_text),
        "candidates_walked": walked,
        "candidates_sharing_a_line": compared,
        "unreadable_count": len(unreadable),
        "unreadable_sample": unreadable[:20],
        "self_workspace": selfdir,
        "hits": hits,
        "hit_count": len(hits),
        "flagged_count": sum(1 for h in hits if h["flagged_probable_copy"]),
        "flagged_count_outside_this_job": sum(
            1 for h in hits if h["flagged_probable_copy"]
            and not h["in_this_jobs_own_workspace"]),
        "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# 4. Rule 2 sentence 3: what was reused, and what we added on top
# ---------------------------------------------------------------------------
def reuse_declaration() -> dict:
    tp = json.loads((REPO / "evidence" / "third_party.json").read_text())
    libs = sorted(tp["python_libraries"])
    models = [{"id": m["id"], "license": m.get("license"), "role": m.get("role")}
              for m in tp["models"]]
    ds = tp["dataset"]
    return {
        "derived_from": "evidence/third_party.json (scripts/measure_third_party.py)",
        "libraries": [{"name": tp["python_libraries"][k]["name"],
                       "version": tp["python_libraries"][k]["version"],
                       "license": tp["python_libraries"][k]["license_declared"]}
                      for k in libs],
        "models": models,
        "dataset": {"id": ds["id"], "license": ds.get("license"), "role": ds.get("role")},
        "external_binaries": sorted(tp["external_binaries"]),
        "note": "Rule 2 sentence 3 permits exactly these and asks that our own "
                "contribution on top of them be demonstrated. The demonstration "
                "is the first_party block below plus evidence/detector_eval.json, "
                "evidence/gate_eval.json and evidence/posterior_ablation.json.",
    }


def first_party(tracked: dict[str, dict]) -> dict:
    py = {rel: m for rel, m in tracked.items() if rel.endswith(".py")}
    sloc = 0
    for rel in py:
        text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                sloc += 1
    areas: dict[str, dict] = {}
    for rel in py:
        top = rel.split("/")[0] if "/" in rel else "."
        areas.setdefault(top, {"files": 0})["files"] += 1
    return {
        "tracked_files": len(tracked),
        "tracked_bytes": sum(m["bytes"] for m in tracked.values()),
        "python_files": len(py),
        "python_sloc": sloc,
        "python_sloc_definition": "tracked .py lines that are neither blank nor "
                                  "comment-only. Volume, not quality.",
        "python_files_by_area": areas,
        "vendored_third_party_source_files": 0,
        "vendored_note": "No third-party source is vendored into this tree; every "
                         "library is a declared dependency installed from its "
                         "publisher. The EXACT scan below includes the site-packages "
                         "root precisely so that claim can fail.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--focus", nargs="*", default=DEFAULT_FOCUS)
    ap.add_argument("--self", dest="selfdir", default=str(REPO.parent),
                    help="this job's own workspace; matches inside it are "
                         "reported but are not prior projects")
    ap.add_argument("--out", default=str(REPO / "evidence" / "originality.json"))
    args = ap.parse_args()

    tracked: dict[str, dict] = {}
    for rel in git("ls-files").split("\n"):
        if not rel:
            continue
        p = REPO / rel
        if not p.is_file():
            continue
        st = p.stat()
        tracked[rel] = {"bytes": st.st_size,
                        "sha256": sha256_file(p) if st.st_size else ""}

    tracked_text = {}
    for rel, m in tracked.items():
        if os.path.splitext(rel)[1] in TEXT_SUFFIXES and m["bytes"]:
            lines = norm_lines((REPO / rel).read_text(encoding="utf-8", errors="replace"))
            if lines:
                tracked_text[rel] = lines

    exclude = {os.path.realpath(REPO)}

    selfdir = os.path.realpath(args.selfdir)
    timeline = repo_timeline()
    ex = exact_scan(tracked, [Path(r) for r in args.roots], exclude, selfdir)
    nr = near_scan(tracked_text, [Path(r) for r in args.focus], exclude, selfdir)

    # A search that walked nothing is not a clean search.
    scans_ran = ex["files_walked"] > 0 and nr["candidates_walked"] > 0
    unexplained_exact = [m for m in ex["matches"]
                         if os.path.basename(m["tracked_file"]) != "LICENSE"
                         and not m["in_this_jobs_own_workspace"]]
    if not scans_ran:
        status = "NOT_RUN"
    elif nr["flagged_count_outside_this_job"] == 0 and not unexplained_exact:
        status = "NO_COPY_FOUND"
    else:
        status = "REVIEW_REQUIRED"

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": "Evidence for UPAI-Hackdays rule 2 (Originality), derived from "
                "this repository's git history and from a search of this host. "
                "Nothing here is transcribed.",
        "head": git("rev-parse", "HEAD").strip(),
        "head_note": "describes the tree at `head`; committing this file "
                     "produces a new head, so `head` is expected to be an "
                     "ancestor of the branch tip.",
        "rule_2": RULE_2,
        "hackathon": HACKATHON,
        "repository_timeline": timeline,
        "first_party": first_party(tracked),
        "reuse_permitted_by_sentence_3": reuse_declaration(),
        "search_exact": ex,
        "search_near_duplicate": nr,
        "status": status,
        "status_definition": {
            "NO_COPY_FOUND": "both searches ran, walked files, and found no "
                             "match that is not the MIT licence text.",
            "REVIEW_REQUIRED": "a match was found that a human must read.",
            "NOT_RUN": "a search walked zero files -- the roots are not on this "
                       "machine. Never report this as a pass.",
        },
        "what_this_does_not_establish": [
            "It searches THIS host. It cannot see a copy taken from somewhere "
            "this machine has never held.",
            "The near-duplicate search covers .py/.md/.sh/.txt under the focus "
            "root only, at or below 512 KiB. A copied binary or notebook would "
            "be caught by EXACT only if byte-identical.",
            "It measures copying, not quality, and not whether the idea is new.",
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"status={status} exact_walked={ex['files_walked']} "
          f"exact_matches={ex['match_count']} near_walked={nr['candidates_walked']} "
          f"near_hits={nr['hit_count']} max_j={nr['max_jaccard_observed']} "
          f"flagged={nr['flagged_count']} "
          f"flagged_outside={nr['flagged_count_outside_this_job']} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
