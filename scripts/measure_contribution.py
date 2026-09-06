"""Who built this repository, measured out of git rather than asserted, and
written to evidence/contribution.json.

Requirements 7 of the UPAI-Hackdays brief asks us to "List all team members if
you participate with your friends and briefly mention their roles/contributions."
This is a solo entry, so the interesting part of that answer is not the list --
it is the *contributions*, and a claim about who wrote what is exactly the kind
of claim rule 7 of the event rules ("misrepresentation of work") punishes.

So nothing here is typed. The script reads:

  * the commit graph -- count, span, and every distinct author string. There
    are two author strings in this history and only one participant; the file
    reports both rather than collapsing them, because a judge running
    `git shortlog -sne` will see two and a submission that claimed one would
    look like it was hiding a collaborator;
  * the tracked file set and its Python line count, per top-level area, so the
    "what did you actually do" answer is a measurement and not an adjective;
  * whether the working tree is clean at the commit the submission pins.

It deliberately does NOT try to decide whether the two author strings are the
same human. That is not a fact this repository holds, and resolving it in our
own favour is the failure mode the whole project is about. It reports the
strings and lets the reader see them.

Stdlib only. Run: python3 scripts/measure_contribution.py
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EV = os.path.join(ROOT, "evidence")


def git(*args: str) -> str:
    """Run git in the repository root and return stdout, stripped."""
    out = subprocess.run(
        ["git", "-C", ROOT, *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return out.stdout.decode("utf-8", "replace").rstrip("\n")


def to_utc(iso: str) -> str:
    """git's %cI is offset-local; the record is kept in UTC like every other date."""
    dt = datetime.datetime.fromisoformat(iso)
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------- commits
raw = git("log", "--reverse", "--format=%H%x1f%an%x1f%ae%x1f%cI%x1f%s")
commits = []
for line in raw.split("\n"):
    sha, name, email, when, subject = line.split("\x1f")
    commits.append(
        {
            "sha": sha,
            "short_sha": sha[:7],
            "author_name": name,
            "author_email": email,
            "committed_utc": to_utc(when),
            "subject": subject,
        }
    )

authors = collections.Counter((c["author_name"], c["author_email"]) for c in commits)
author_strings = [
    {"name": n, "email": e, "commits": k} for (n, e), k in sorted(authors.items())
]

# ---------------------------------------------------------------- tracked files
# Read the TREE AT HEAD, not the working directory. The record names a commit,
# so every number in it must be re-derivable from that commit alone -- otherwise
# committing the record changes the thing the record describes, and the prose
# quoting it goes stale the moment it is saved.
HEAD = commits[-1]["sha"]
tracked = [p for p in git("ls-tree", "-r", "--name-only", HEAD).split("\n") if p]
py_files = [p for p in tracked if p.endswith(".py")]


def sloc(path: str) -> int:
    """Non-blank, non-comment-only lines in the blob at HEAD. A crude measure of
    volume, and named as one."""
    n = 0
    for line in git("show", f"{HEAD}:{path}").split("\n"):
        s = line.strip()
        if s and not s.startswith("#"):
            n += 1
    return n


def area(path: str) -> str:
    head = path.split("/")[0]
    return head if "/" in path else "(top level)"


by_area: dict[str, dict[str, int]] = {}
for p in py_files:
    a = by_area.setdefault(area(p), {"py_files": 0, "sloc": 0})
    a["py_files"] += 1
    a["sloc"] += sloc(p)

# ------------------------------------------------------------------ worktree
status = git("status", "--porcelain")
dirty = [l for l in status.split("\n") if l.strip()]

record = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    ),
    "what": "Provenance of this repository, read out of git. Supports Requirement 7 "
    "(team information / roles and contributions) and rule 7 (no misrepresentation "
    "of work). Every field is derived; none is typed.",
    "head": HEAD,
    "describes": "the tree at `head`. Committing this file produces a NEW head, so "
    "`head` here is expected to be an ancestor of the branch tip, and every count "
    "below is re-derivable from `head` with git alone.",
    "commit_count": len(commits),
    "first_commit_utc": commits[0]["committed_utc"],
    "last_commit_utc": commits[-1]["committed_utc"],
    "commits": commits,
    "author_strings": author_strings,
    "distinct_author_strings": len(author_strings),
    "author_strings_note": "Two strings, one participant. This file does not assert "
    "that the two strings are the same person -- it is not a fact the repository "
    "holds. It reports what `git shortlog -sne` reports.",
    "tracked_files": len(tracked),
    "python_files": len(py_files),
    "python_sloc": sum(sloc(p) for p in py_files),
    "python_sloc_definition": "lines that are neither blank nor a comment-only line, "
    "counted per tracked .py file. A crude measure of volume, not of quality.",
    "by_area": dict(sorted(by_area.items())),
    "worktree_clean": not dirty,
    "worktree_dirty_entries": dirty,
}

os.makedirs(EV, exist_ok=True)
path = os.path.join(EV, "contribution.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=1, ensure_ascii=False)
    fh.write("\n")

print(json.dumps(record, indent=1, ensure_ascii=False))
sys.exit(0)
