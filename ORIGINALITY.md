# Originality

UPAI-Hackdays **rule 2**, in the organiser's own words:

> All submissions must be original work created by the participating team.
>
> Previously developed projects may not be submitted as-is.
>
> Teams may use existing open-source libraries, APIs, and tools, but their
> contribution and implementation should be clearly demonstrated.

Three sentences making three different claims. This document answers them in
order. The second one is answered by **going and looking**, because it is the
only one of the three that can be tested rather than asserted, and an
originality statement that only asserts is worth nothing.

Everything below is produced by
[`scripts/measure_originality.py`](scripts/measure_originality.py) into
[`evidence/originality.json`](evidence/originality.json). Nothing here is typed
from memory.

---

## 1. It was built for this hackathon, and after the hackathon existed

| | UTC |
|---|---|
| UPAI-Hackdays published on Devpost | **2026-09-02T05:52:35Z** |
| first commit in this repository | **2026-09-06T06:03:14Z** |
| last commit measured here | **2026-09-06T10:21:00Z** |
| submissions open | **2026-09-07T04:00:00Z** |
| submissions close | **2026-09-10T11:30:00Z** |

The repository begins **346,239 s — 4d 0h 10m 39s — after** the hackathon was
published. It has **12** commits over **15,466 s (4h 17m 46s)**, one root
commit and no merges. There is no imported history, no squashed prior
repository, and no commit that predates the event's own announcement.

### The head start, stated plainly

**The first commit lands 79,006 s — 21h 56m 46s — before the submission window
opened.** That is said here rather than left to be noticed.

Rule 2 sets no build window. It is three sentences and none of them names one,
and neither does anything else the organiser publishes: the full rules body and
the Devpost `terms_and_conditions_text` were searched for *"during the
hackathon"*, *"during the event"*, *"built during"*, *"created during"*,
*"development period"* and *"build period"*, and none of those phrases occurs
anywhere. On the rules as published, building before the window opens is not a
breach.

It is still a fact a judge is entitled to weigh, and it is the one fact in this
document that runs against the entry, so it is disclosed rather than resolved in
our own favour. What rule 2 forbids is submitting a *previously developed
project*, and section 2 is that question tested.

### What git alone cannot settle

The first commit is **31 files and 4,529 insertions** in one go. That is exactly
the shape an *imported* project has, and git cannot tell "written, then
committed once" apart from "copied in from somewhere else". Commit timestamps
are also writable by whoever makes the commit.

So the timeline above is offered as context, not as proof. The proof is the
search.

---

## 2. It is not a previously developed project

Two searches of this machine, because they fail in different ways.

### Search A — a verbatim copy, anywhere on the host

Every tracked file's sha256, against every file on two roots — the whole of
`/data/wj/wj_code` (every project this machine holds, including the other
hackathon workspaces) and `/data/wj/anaconda/lib` (every installed library, so
that "nothing is vendored" can fail).

| | measured |
|---|---|
| files walked | **5,484,328** |
| files whose size collided with a tracked file, and were therefore hashed | **23,890** |
| byte-identical matches found | **2** |
| matches that are not the MIT licence text | **0** |
| directories that could not be read | **21** |

Both matches are `LICENSE` — 1,066 bytes of the standard MIT licence text, found
in two other hackathon workspaces on this machine:

```
workspaces/ai-infra-summit-hackathon/build/LICENSE
workspaces/ibm-bob-2-hackathon/evidence_20260905T2130Z/LICENSE
```

That is not authorship and it is not a project; it is the same boilerplate
licence, from the same author, in three places. It is reported rather than
filtered out, and it doubles as the search's own proof of life: a scan that
found *nothing at all* would be indistinguishable from a scan that never ran.

The 21 unreadable entries are permission-denied runtime directories
(`.gnupg`, `.ssh`, database volumes) inside third-party repositories cloned
under `dl_sim/`. They are counted, not swallowed.

### Search B — a copy that was then edited

Byte-identity is easy to defeat: change a few lines and search A goes quiet. So
every tracked text file is also compared by normalised-line Jaccard against
every `.py`/`.md`/`.sh`/`.txt` file under the hackathon tree, through an
inverted index. Blank lines and lines under 8 characters are dropped, so `pass`
and `import os` cannot inflate a score.

| | measured |
|---|---|
| tracked text files indexed | **40** |
| candidate files compared | **35,369** |
| candidates sharing at least one normalised line | **20,949** |
| **highest similarity to anything on this host** | **0.10** |
| candidates at or above the 0.30 reporting threshold | **0** |
| candidates at or above the 0.50 flag threshold | **0** |

The best score on the machine is **0.10**, between `scripts/fetch_model.sh` and
`pack_submission.sh` in another workspace — an unrelated script that zips a
submission directory. What the two share is `#!/usr/bin/env bash` and
`set -euo pipefail`; there is no third line in common. The threshold has an
order of magnitude of headroom under it. That is the number that makes "no copy
found" mean something: not that nothing crossed a line, but that nothing came
close to one.

Verdict recorded in the evidence file: **`NO_COPY_FOUND`**.

### The controls

The search's whole value is its ability to find a copy that *is* there, so that
is tested directly by
[`tests/check_originality_scan.py`](tests/check_originality_scan.py) —
**9 of 9 passing**, each against a tree built for the run:

- a verbatim copy of `airlock/gate.py` planted on the root **is** found;
- a copy of `airlock/detect.py` with 22 lines rewritten **is** flagged at 0.6972
  — and is correctly *missed* by search A, which is why search B exists;
- absent roots report `NOT_RUN`, never a pass;
- the MIT licence carve-out and the same-workspace carve-out are each paired
  with the opposite case, so neither can quietly swallow a real match.

### What this does not establish

- It searches **this** host. A copy taken from something this machine has never
  held is outside its reach.
- Search B's focus is the hackathon tree (35,369 files). An *edited* copy taken
  from elsewhere on the machine would be caught by search A only if it were
  byte-identical.
- It measures **copying**, not quality, and not whether the idea is new.
- The host is live: the walked-file count is a count of the moment — three runs
  inside one half hour differed by 2 in it.

---

## 3. What was reused, and what was built on top of it

Rule 2's third sentence permits existing libraries, APIs and tools, and asks
that our own contribution be demonstrated. The full inventory with licences is
[`THIRD_PARTY.md`](THIRD_PARTY.md); the short form:

| reused | | licence |
|---|---|---|
| base encoder | `distilbert-base-cased` | Apache-2.0 |
| answering models | `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| corpus | `gretelai/synthetic_pii_finance_multilingual` | Apache-2.0 |
| libraries | `torch`, `transformers`, `pyarrow`, `pandas`, `Pillow` | BSD / Apache / MIT-CMU |
| system binaries, fonts | 12 and 4, none redistributed | see `THIRD_PARTY.md` |

**No third-party source is vendored into this tree** — 0 tracked files match
anything under `/data/wj/anaconda/lib`, which is why that root is in search A.
What is tracked is **56 files, 2,262,703 bytes, 31 Python files, 4,199 lines**
that are neither blank nor comment-only: 13 files of product under `airlock/`,
14 of measurement under `scripts/`, 4 of controls under `tests/`.

The contribution on top of the reused parts is not a claim, it is the gap
between the reused thing and the shipped thing, measured on held-out data:

- **The base encoder cannot do this task at all.** `distilbert-base-cased` has no
  PII head; the detector is a 59-label BIO fine-tune of it, trained here in
  156.8 s. Its evidence is `evidence/detector_eval.json`.
- **Against a non-learned baseline**, the same pipeline with a regex and
  capitalised-bigram detector in place of the model scores **38.45%** identifier
  recall and leaks from **90.39%** of documents, against **93.96%** and
  **13.27%**.
- **The gate is not just "send less".** Withholding the same *number* of
  documents at random leaves the leak rate at the base rate — **10.10%** against
  Airlock's **4.67%** at identical 63.65% coverage.
- **The calibration is load-bearing**, not decoration: ranking on the model's
  posterior gives AUC **0.7498**, ranking on the count of surviving flagged
  tokens gives **0.4531**.

Every one of those figures is re-read from `evidence/` by
`tests/check_readme_numbers.py`, which fails if the prose and the evidence
disagree.
