"""What it costs to run Airlock's detector, per device, and writes
evidence/deploy_cost.json.

Every throughput figure this project has published so far was taken on an
L40S. That makes the scaling question unanswerable: a gate that needs a
datacentre GPU per tenant scales differently from one that runs on the box the
documents are already on. This measures the same forward pass on CPU at pinned
thread counts and on the GPU, over the same documents, so the ratio means
something.

One configuration per process -- torch's intra-op thread pool is set once and
a warm CUDA context in the same process would contaminate the CPU timing.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlock  # noqa: F401,E402  -- pyarrow shim; must precede torch on this host

BATCH = 8  # the batch airlock.detect.find_batch ships with


def measure_one(device: str, threads: int | None, limit: int, model: str) -> dict:
    import torch
    if threads is not None:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None

    from airlock.data import load_split
    from airlock.detect import Detector

    docs = load_split("test", limit=limit)
    texts = [d.text for d in docs]

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    det = Detector(model, device=device)

    # warm-up on a small slice, excluded from the timing: the first forward
    # pass pays for CUDA context creation / lazy kernel selection, and
    # charging that to throughput would understate every device unequally.
    det._token_scores(texts[:BATCH])
    if device == "cuda":
        torch.cuda.synchronize()

    n_windows = 0
    t0 = time.time()
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        enc = det.tok(chunk, return_overflowing_tokens=True, truncation=True,
                      max_length=det.max_length, stride=det.stride, padding=True)
        n_windows += len(enc["input_ids"])
        det._token_scores(chunk)
    if device == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    chars = sum(len(t) for t in texts)
    return {
        "device": device,
        "device_name": (torch.cuda.get_device_name(0) if device == "cuda"
                        else f"CPU, {threads} threads"),
        "torch_threads": threads if device == "cpu" else None,
        "documents": len(texts),
        "characters": chars,
        "mean_characters_per_document": round(chars / len(texts), 1),
        "windows": n_windows,
        "batch": BATCH,
        "seconds": round(secs, 2),
        "docs_per_second": round(len(texts) / secs, 2),
        "chars_per_second": round(chars / secs, 1),
        "peak_rss_mib": round(rss1 / 1024.0, 1),
        "rss_before_model_mib": round(rss0 / 1024.0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="evidence/deploy_cost.json")
    # single-config mode, used by the driver
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--emit")
    args = ap.parse_args()

    if args.device:
        r = measure_one(args.device, args.threads, args.limit, args.model)
        json.dump(r, open(args.emit, "w"), indent=2)
        return

    configs = [("cpu", 1), ("cpu", 4), ("cpu", 8), ("cuda", None)]
    runs = []
    for dev, th in configs:
        tmp = f"/tmp/airlock_deploy_{dev}_{th}.json"
        cmd = [sys.executable, os.path.abspath(__file__), "--device", dev,
               "--limit", str(args.limit), "--model", args.model, "--emit", tmp]
        if th is not None:
            cmd += ["--threads", str(th)]
        env = dict(os.environ)
        if dev == "cpu":
            # pin the whole numeric stack, not just torch's own pool: this box
            # has 384 cores and an unpinned CPU number here would describe
            # hardware nobody deploys a redaction sidecar on.
            for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[v] = str(th)
            env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(cmd, check=True, env=env,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        runs.append(json.load(open(tmp)))

    by = {f"{r['device']}_{r['torch_threads']}" if r["device"] == "cpu" else r["device"]: r
          for r in runs}
    gpu = by["cuda"]
    cpu4 = by["cpu_4"]

    params = 0
    meta_path = os.path.join(args.model, "train_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    params = meta.get("parameters", 0)
    weights = os.path.getsize(os.path.join(args.model, "model.safetensors"))

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what": ("Forward pass of the shipped detector over the same documents on each "
                 "device. This is airlock.evaluate's own measurement -- Detector._token_scores, "
                 "batch 8 -- not a separate benchmark path."),
        "split": "test",
        "documents": gpu["documents"],
        "model_parameters": params,
        "weights_bytes": weights,
        "weights_mib": round(weights / 1048576.0, 1),
        "runs": runs,
        "gpu_over_cpu4_speedup": round(gpu["docs_per_second"] / cpu4["docs_per_second"], 2),
        "cpu4_documents_per_core_hour": round(cpu4["docs_per_second"] * 3600.0 / 4, 1),
        "cpu4_documents_per_day": round(cpu4["docs_per_second"] * 86400.0),
        "not_claimed": ("Forward pass only, single process, no HTTP, no batching across "
                        "requests and no quantisation. Wall-clock on a shared host; the CPU "
                        "runs are pinned by OMP/MKL/OPENBLAS and torch.set_num_threads but "
                        "the box was not otherwise idle-guaranteed. Documents here average "
                        f"{gpu['mean_characters_per_document']} characters; throughput on "
                        "longer documents is lower because windows, not documents, are the "
                        "unit of work."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))
    for r in runs:
        print(f"  {r['device_name']:<28} {r['docs_per_second']:>8.2f} docs/s  "
              f"{r['seconds']:>6.2f}s  peak RSS {r['peak_rss_mib']} MiB")


if __name__ == "__main__":
    main()
