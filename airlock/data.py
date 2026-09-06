"""Corpus loading and char-span -> BIO-tag alignment.

The corpus is gretelai/synthetic_pii_finance_multilingual (Apache-2.0). Only
the English split is used. Each record carries `generated_text` and
`pii_spans`, a JSON list of {start, end, label} character offsets.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .labels import LABEL2ID

DATA_DIR = os.environ.get(
    "AIRLOCK_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".data_cache"),
)


@dataclass
class Doc:
    text: str
    spans: list  # [{"start": int, "end": int, "label": str}]
    doc_type: str = ""


def load_split(split: str, limit: int | None = None) -> list[Doc]:
    """split is 'train' or 'test'. Reads the JSONL written by scripts/prepare_data.py."""
    path = os.path.join(DATA_DIR, f"english_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing -- run: python3 scripts/prepare_data.py")
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    docs = []
    for row in rows:
        text, dt = row["text"], row.get("doc_type", "")
        parsed = list(row["spans"])
        parsed = [s for s in parsed if s["end"] > s["start"]]
        parsed.sort(key=lambda s: (s["start"], s["end"]))
        # drop spans that overlap an earlier kept span: BIO cannot express them
        kept, last_end = [], -1
        for s in parsed:
            if s["start"] >= last_end:
                kept.append(s)
                last_end = s["end"]
        docs.append(Doc(text=text, spans=kept, doc_type=dt))
    return docs


def encode_doc(tokenizer, doc: Doc, max_length: int = 384, stride: int = 96):
    """Tokenize into overlapping windows and project char spans onto tokens.

    Returns a list of dicts with input_ids, attention_mask, labels, offsets.
    A token is B-<type> when it is the first token overlapping that span in the
    document (token start <= span start), otherwise I-<type>.
    """
    enc = tokenizer(
        doc.text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )
    out = []
    n_windows = len(enc["input_ids"])
    for w in range(n_windows):
        offsets = enc["offset_mapping"][w]
        labels = []
        for (a, b) in offsets:
            if a == b:  # special token
                labels.append(-100)
                continue
            tag = "O"
            for s in doc.spans:
                if a < s["end"] and b > s["start"]:  # overlap
                    prefix = "B" if a <= s["start"] else "I"
                    tag = f"{prefix}-{s['label']}"
                    break
            labels.append(LABEL2ID[tag])
        out.append(
            {
                "input_ids": enc["input_ids"][w],
                "attention_mask": enc["attention_mask"][w],
                "labels": labels,
                "offsets": offsets,
            }
        )
    return out
