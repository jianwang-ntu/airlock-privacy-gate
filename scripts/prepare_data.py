"""Convert the Gretel parquet splits to JSONL once.

Kept out of the library so that pyarrow -- which on some hosts cannot be
imported after torch has loaded the system libstdc++ -- is never on the
runtime import path. Records the sha256 of every input and output file.
"""
import hashlib
import json
import os
import sys

import pandas as pd

DATA = os.environ.get("AIRLOCK_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".data_cache"))
SPLITS = {"train": "English_train.parquet", "test": "English_test.parquet"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = {"source": "gretelai/synthetic_pii_finance_multilingual", "license": "apache-2.0", "files": {}}
    for split, fname in SPLITS.items():
        src = os.path.join(DATA, fname)
        dst = os.path.join(DATA, f"english_{split}.jsonl")
        df = pd.read_parquet(src)
        with open(dst, "w") as f:
            for text, spans, dt in zip(df.generated_text, df.pii_spans, df.document_type):
                f.write(json.dumps({"text": text, "spans": json.loads(spans) if isinstance(spans, str) else list(spans), "doc_type": dt}) + "\n")
        manifest["files"][fname] = {"sha256": sha256(src), "rows": len(df), "jsonl": os.path.basename(dst), "jsonl_sha256": sha256(dst)}
        print(f"{split}: {len(df)} rows -> {dst}")
    with open(os.path.join(DATA, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
