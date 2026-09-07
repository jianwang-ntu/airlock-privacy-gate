#!/usr/bin/env python3
"""Re-derive Airlock's label vocabulary from the corpus and fail if it drifts.

`airlock/labels.py` says its `PII_TYPES` list "is read from the corpus rather
than invented here; `scripts/check_label_vocab.py` re-derives it from the
parquet files and fails if this list drifts." This is that script. Until it
existed, that sentence was an assurance a reader could not run, which is the
one thing this repository is not allowed to do.

What it re-derives, and from what:

  the set of distinct `label` values over every span in `pii_spans`, read out
  of `English_train.parquet` and `English_test.parquet` directly -- not out of
  the JSONL that `scripts/prepare_data.py` writes, and not out of the loader in
  `airlock/data.py`, which drops overlapping spans and so could hide a type
  that only ever appears inside an overlap.

House rule, the same one `tests/run_checks.py` keeps: every accepting check is
paired with a control that must FAIL for the check to mean anything. A check
whose control also passes is reported BROKEN and fails the run, because a check
that cannot fail is not a check.

It reads only span labels and counts. No document text is read, printed or
written by this script.

Run:   python3 scripts/check_label_vocab.py
       AIRLOCK_DATA_DIR=/path/to/parquet python3 scripts/check_label_vocab.py
Exit:  0  every check passed
       1  the vocabulary drifted, or a check is broken
       2  the corpus is not on this machine -- nothing was checked
"""
import collections
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from airlock.labels import (                      # noqa: E402
    ID2LABEL,
    IDENTIFYING,
    LABEL2ID,
    LABELS,
    NON_IDENTIFYING,
    PII_TYPES,
    type_of,
)

DATA = os.environ.get("AIRLOCK_DATA_DIR", os.path.join(ROOT, ".data_cache"))
SPLITS = {"train": "English_train.parquet", "test": "English_test.parquet"}

PASS, FAIL, BROKEN, SKIP = [], [], [], []


def check(name, ok, detail, control=None):
    """`control` is a zero-arg callable re-running the SAME assertion against a
    deliberately corrupted input. It must return False. If it returns True the
    assertion cannot fail and the check is void, whatever `ok` said."""
    ok = bool(ok)
    if control is not None:
        if bool(control()):
            BROKEN.append((name, "control passed on corrupted input"))
            print(f"BROKEN {name}: control passed on corrupted input -- "
                  f"this check cannot fail, so its PASS means nothing")
            return
    if ok:
        PASS.append(name)
        print(f"PASS  {name}: {detail}")
    else:
        FAIL.append((name, detail))
        print(f"FAIL  {name}: {detail}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive(paths):
    """Distinct span labels and their frequencies, straight off the parquet."""
    import pandas as pd

    counts = collections.Counter()
    per_split = {}
    for split, path in paths.items():
        df = pd.read_parquet(path, columns=["pii_spans"])
        seen = collections.Counter()
        for spans in df.pii_spans:
            rows = json.loads(spans) if isinstance(spans, str) else list(spans)
            for sp in rows:
                seen[sp["label"]] += 1
        per_split[split] = {"rows": int(len(df)), "spans": int(sum(seen.values())),
                            "distinct_labels": len(seen)}
        counts.update(seen)
    return counts, per_split


def main():
    paths = {s: os.path.join(DATA, f) for s, f in SPLITS.items()}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print("NOTHING CHECKED -- the corpus is not on this machine.")
        for p in missing:
            print(f"  absent: {p}")
        print("\nFetch gretelai/synthetic_pii_finance_multilingual (Apache-2.0,")
        print("English split) into that directory, or point AIRLOCK_DATA_DIR at it.")
        print("Exit 2 means 'not checked'. It does NOT mean the vocabulary is intact.")
        return 2

    # The bytes are pinned FIRST. If the corpus on this machine is not the
    # corpus the list was derived from, that has to be reported as a failed
    # check -- not discovered halfway through a parquet read as a traceback.
    manifest_path = os.path.join(DATA, "data_manifest.json")
    shas = {os.path.basename(p): sha256(p) for p in paths.values()}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            pinned = json.load(fh).get("files", {})
        for fname, got in shas.items():
            want = (pinned.get(fname) or {}).get("sha256")
            if want is None:
                SKIP.append((f"corpus/{fname}-sha256",
                             "not pinned in data_manifest.json"))
                print(f"SKIP  corpus/{fname}-sha256: not pinned in data_manifest.json")
                continue
            check(
                f"corpus/{fname}-is-the-pinned-bytes",
                got == want,
                f"sha256 {got[:16]}... vs pinned {want[:16]}...",
                control=lambda g=got, w=want: ("0" + g[1:]) == w,
            )
    else:
        SKIP.append(("corpus/sha256-pin", "data_manifest.json absent -- run "
                                          "scripts/prepare_data.py to write it"))
        print("SKIP  corpus/sha256-pin: data_manifest.json absent -- the vocabulary "
              "is still checked below, but the bytes it came from are not pinned")

    try:
        counts, per_split = derive(paths)
    except ImportError as e:
        print(f"NOTHING CHECKED -- pyarrow/pandas unavailable: {e}")
        print("Exit 2 means 'not checked'.")
        return 2
    except Exception as e:  # noqa: BLE001 -- an unreadable corpus is a FAIL, not a crash
        check(
            "corpus/parquet-is-readable",
            False,
            f"{type(e).__name__}: {e}",
        )
        print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} passed, {len(FAIL)} failed, "
              f"{len(BROKEN)} broken, {len(SKIP)} skipped -- the vocabulary was NOT "
              f"re-derived, so nothing here says the list is intact")
        return 1

    derived = set(counts)
    declared = set(PII_TYPES)

    # ---------------------------------------------------------------- the claim
    check(
        "vocab/declared-list-is-the-corpus-vocabulary",
        derived == declared,
        (f"{len(derived)} distinct labels over {sum(counts.values())} spans in "
         f"{sum(v['rows'] for v in per_split.values())} documents; "
         f"in corpus not declared: {sorted(derived - declared) or 'none'}; "
         f"declared not in corpus: {sorted(declared - derived) or 'none'}"),
        control=lambda: (derived | {"__drifted_type__"}) == declared,
    )
    check(
        "vocab/every-declared-type-is-attested",
        not (declared - derived),
        f"{len(declared)} declared, {len(declared & derived)} attested in the corpus",
        control=lambda: not ((declared | {"__unattested__"}) - derived),
    )
    check(
        "vocab/no-corpus-type-is-undeclared",
        not (derived - declared),
        f"{len(derived - declared)} corpus labels absent from PII_TYPES",
        control=lambda: not ((derived | {"__undeclared__"}) - declared),
    )

    # ------------------------------------------------- the list's own integrity
    dupes = [t for t, n in collections.Counter(PII_TYPES).items() if n > 1]
    check(
        "vocab/PII_TYPES-has-no-duplicates",
        not dupes,
        f"{len(PII_TYPES)} entries, {len(declared)} distinct; duplicates: {dupes or 'none'}",
        control=lambda: not [t for t, n in
                             collections.Counter(PII_TYPES + PII_TYPES[:1]).items() if n > 1],
    )
    check(
        "vocab/NON_IDENTIFYING-is-a-subset",
        NON_IDENTIFYING <= declared,
        f"{sorted(NON_IDENTIFYING)} all present in PII_TYPES",
        control=lambda: (NON_IDENTIFYING | {"__not_a_type__"}) <= declared,
    )
    check(
        "vocab/IDENTIFYING-is-the-complement",
        set(IDENTIFYING) == declared - NON_IDENTIFYING
        and len(IDENTIFYING) == len(declared) - len(NON_IDENTIFYING),
        f"{len(IDENTIFYING)} identifying = {len(declared)} declared "
        f"- {len(NON_IDENTIFYING)} non-identifying",
        control=lambda: set(IDENTIFYING) == declared,
    )

    # ------------------------------------------------------- the BIO tag set
    check(
        "tags/one-O-plus-two-tags-per-type",
        len(LABELS) == 2 * len(PII_TYPES) + 1 and LABELS[0] == "O",
        f"{len(LABELS)} labels for {len(PII_TYPES)} types",
        control=lambda: len(LABELS + ["B-__extra__"]) == 2 * len(PII_TYPES) + 1,
    )
    check(
        "tags/LABEL2ID-round-trips",
        all(ID2LABEL[LABEL2ID[l]] == l for l in LABELS)
        and len(LABEL2ID) == len(LABELS) == len(ID2LABEL),
        f"{len(LABEL2ID)} labels map to {len(ID2LABEL)} distinct ids and back",
        control=lambda: all(ID2LABEL[LABEL2ID[l]] == l for l in LABELS)
        and len(LABEL2ID) == len(LABELS) + 1,
    )
    check(
        "tags/type_of-recovers-the-declared-type",
        all(type_of(l) in declared for l in LABELS if l != "O")
        and type_of("O") == ""
        and {type_of(l) for l in LABELS if l != "O"} == declared,
        f"every one of {len(LABELS) - 1} tags strips back to a declared type",
        control=lambda: all(type_of(l) in (declared - {PII_TYPES[0]})
                            for l in LABELS if l != "O"),
    )

    # ------------------------------------------------------------------ record
    out_dir = os.path.join(ROOT, "evidence")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "label_vocab.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "what": "PII_TYPES in airlock/labels.py re-derived from the corpus parquet",
                "source": "gretelai/synthetic_pii_finance_multilingual (Apache-2.0), English split",
                "derived_from": sorted(os.path.basename(p) for p in paths.values()),
                "corpus_sha256": shas,
                "per_split": per_split,
                "declared_types": len(PII_TYPES),
                "derived_types": len(derived),
                "matches": sorted(derived) == sorted(declared),
                "in_corpus_not_declared": sorted(derived - declared),
                "declared_not_in_corpus": sorted(declared - derived),
                "span_counts_by_type": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                "non_identifying": sorted(NON_IDENTIFYING),
                "identifying": len(IDENTIFYING),
                "bio_labels": len(LABELS),
                "checks": {"pass": len(PASS), "fail": len(FAIL),
                           "broken": len(BROKEN), "skip": len(SKIP)},
                "note": "Span labels and counts only. No document text is read or written here.",
            },
            fh,
            indent=2,
        )

    total = len(PASS) + len(FAIL) + len(BROKEN)
    print(f"\n{len(PASS)}/{total} passed, {len(FAIL)} failed, {len(BROKEN)} broken, "
          f"{len(SKIP)} skipped -> {os.path.relpath(out, ROOT)}")
    for n, why in FAIL:
        print(f"  FAILED {n}: {why}")
    for n, why in BROKEN:
        print(f"  BROKEN {n}: {why}")
    return 1 if (FAIL or BROKEN) else 0


if __name__ == "__main__":
    sys.exit(main())
