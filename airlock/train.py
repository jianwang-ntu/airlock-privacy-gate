"""Fine-tune the stage-1 identifier detector (token classification).

Plain PyTorch loop -- no Trainer -- so every number reported (steps, loss,
wall clock, examples seen) is produced by code in this file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .data import encode_doc, load_split
from .labels import ID2LABEL, LABEL2ID, LABELS

DEV_DOCS = 2000  # held out of train for threshold/calibration work; test stays untouched


class WindowSet(Dataset):
    def __init__(self, windows):
        self.w = windows

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return self.w[i]


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        att[i, :k] = torch.tensor(b["attention_mask"])
        lab[i, :k] = torch.tensor(b["labels"])
    return ids, att, lab


def build_windows(tokenizer, docs, max_length, stride):
    out = []
    for d in docs:
        out.extend(encode_doc(tokenizer, d, max_length=max_length, stride=stride))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="distilbert-base-cased")
    ap.add_argument("--out", default="models/detector")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base)
    docs = load_split("train", limit=args.limit)
    dev_docs = docs[-DEV_DOCS:] if args.limit is None else docs[-max(1, len(docs) // 10):]
    train_docs = docs[: len(docs) - len(dev_docs)]
    train_w = build_windows(tok, train_docs, args.max_length, args.stride)
    dev_w = build_windows(tok, dev_docs, args.max_length, args.stride)
    print(f"docs train={len(train_docs)} dev={len(dev_docs)} | windows train={len(train_w)} dev={len(dev_w)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(dev)

    pad = tok.pad_token_id
    dl = DataLoader(
        WindowSet(train_w), batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, pad), num_workers=2, drop_last=True,
    )
    dl_dev = DataLoader(
        WindowSet(dev_w), batch_size=args.batch, shuffle=False,
        collate_fn=lambda b: collate(b, pad), num_workers=2,
    )

    steps = int(len(dl) * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.06, anneal_strategy="linear"
    )
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)

    model.train()
    step, done, losses = 0, False, []
    while not done:
        for ids, att, lab in dl:
            ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
            with amp:
                out = model(input_ids=ids, attention_mask=att, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{steps} loss {sum(losses[-100:])/100:.4f} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if step >= steps:
                done = True
                break

    # held-out token loss, so the reported fit is not a training-set number
    model.eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for ids, att, lab in dl_dev:
            ids, att, lab = ids.to(dev), att.to(dev), lab.to(dev)
            with amp:
                out = model(input_ids=ids, attention_mask=att, labels=lab)
            k = int((lab != -100).sum())
            tot += out.loss.item() * k
            ntok += k
    dev_loss = tot / max(1, ntok)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    meta = {
        "base_model": args.base,
        "labels": len(LABELS),
        "train_docs": len(train_docs),
        "dev_docs": len(dev_docs),
        "train_windows": len(train_w),
        "dev_windows": len(dev_w),
        "steps": step,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_length": args.max_length,
        "stride": args.stride,
        "seed": args.seed,
        "final_train_loss_last100": sum(losses[-100:]) / min(100, len(losses)),
        "dev_token_loss": dev_loss,
        "wall_seconds": round(time.time() - t0, 1),
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
