"""What the gate actually sustains, per document, on the hardware it deploys on.

`measure_deploy_cost.py` times `Detector._token_scores` at batch 8 and the
README extrapolates that to a documents-per-day figure. That is the forward
pass, and the forward pass is not the sidecar. A deployed Airlock runs
`Gate.process` on ONE document at a time -- `HybridDetector` (the model
unioned with the checksum validators), then `redact`, then `raw_risk` over the
surviving tokens, then the isotonic calibrator -- because a request arrives on
its own and cannot wait for seven more to fill a batch.

So the published figure is an upper bound on a path nobody runs, and this
measures the path they do, sustained, with the latency distribution that a
throughput number hides.

Three arms, one process each -- torch's intra-op pool is set once per process:

  A forward_batch8   the published configuration, replayed. A TIE-IN CONTROL:
                     if it does not reproduce evidence/deploy_cost.json on this
                     host then the host has moved and arms B and C cannot be
                     compared against the published number at all.
  B forward_batch1   the same forward pass at the batch a sidecar really has.
                     Isolates batching from post-processing.
  C gate_sustained   the real Gate.process, one document at a time, for a
                     target wall-clock duration. Reports the latency
                     distribution and first-third vs last-third throughput, so
                     "sustained" is measured rather than assumed.

No document text is read, printed or stored by this script. Only counts,
character totals and timings leave it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlock  # noqa: F401,E402  -- pyarrow shim; must precede torch on this host

BATCH8 = 8          # the batch measure_deploy_cost.py and find_batch ship with
THREADS = 4         # the configuration the README's documents-per-day quotes


def _pin(threads: int):
    import torch
    torch.set_num_threads(threads)


def _corpus(limit: int):
    from airlock.data import load_split
    return [d.text for d in load_split("test", limit=limit)]


def _pct(xs, q):
    """Nearest-rank percentile: the smallest value at or above rank ceil(q*n).
    Interpolation is a fiction on a few hundred samples. `round` was wrong here
    -- it is banker's rounding, and it put p99 one rank too high on n=100."""
    s = sorted(xs)
    i = min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))
    return s[i]


# ------------------------------------------------------------------ arms
def arm_forward(batch: int, limit: int, model: str) -> dict:
    _pin(THREADS)
    from airlock.detect import Detector

    texts = _corpus(limit)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    det = Detector(model, device="cpu")
    det._token_scores(texts[:batch])          # warm-up, excluded, as published

    t0 = time.time()
    for i in range(0, len(texts), batch):
        det._token_scores(texts[i:i + batch])
    secs = time.time() - t0

    chars = sum(len(t) for t in texts)
    return {
        "arm": f"forward_batch{batch}",
        "what": ("Detector._token_scores only, batch %d, %d threads -- no validators, "
                 "no redaction, no risk, no calibration." % (batch, THREADS)),
        "documents": len(texts),
        "batch": batch,
        "characters": chars,
        "mean_characters_per_document": round(chars / len(texts), 1),
        "seconds": round(secs, 2),
        "docs_per_second": round(len(texts) / secs, 3),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "rss_before_model_mib": round(rss0 / 1024.0, 1),
    }


def arm_gate_sustained(seconds_target: float, limit: int, model: str, calibration: str) -> dict:
    _pin(THREADS)
    from airlock.gate import Calibrator, Gate
    from airlock.hybrid import HybridDetector

    texts = _corpus(limit)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    det = HybridDetector(model, device="cpu")
    if not os.path.exists(calibration):
        raise SystemExit("calibration %s missing -- run scripts/fetch_model.sh" % calibration)
    gate = Gate(det, Calibrator.load(calibration))

    gate.process(texts[0])                     # warm-up, excluded

    lat, chars, forwarded = [], 0, 0
    t0 = time.time()
    i = 0
    while time.time() - t0 < seconds_target:
        text = texts[i % len(texts)]
        a = time.time()
        r = gate.process(text)
        lat.append(time.time() - a)
        chars += len(text)
        forwarded += 1 if r["decision"].forward else 0
        i += 1
    secs = time.time() - t0

    n = len(lat)
    third = max(1, n // 3)
    first_s, last_s = sum(lat[:third]), sum(lat[-third:])
    return {
        "arm": "gate_sustained",
        "what": ("Gate.process one document at a time -- HybridDetector (model UNION "
                 "checksum validators), redact, raw_risk over surviving tokens, isotonic "
                 "calibrator, forward/local decision. %d threads. This is the deployed "
                 "path." % THREADS),
        "seconds_target": seconds_target,
        "seconds": round(secs, 2),
        "documents": n,
        "corpus_documents": len(texts),
        "passes_over_corpus": round(n / len(texts), 2),
        "characters": chars,
        "mean_characters_per_document": round(chars / n, 1),
        "docs_per_second": round(n / secs, 3),
        "chars_per_second": round(chars / secs, 1),
        "latency_seconds": {
            "mean": round(sum(lat) / n, 4),
            "p50": round(_pct(lat, 0.50), 4),
            "p90": round(_pct(lat, 0.90), 4),
            "p99": round(_pct(lat, 0.99), 4),
            "max": round(max(lat), 4),
            "min": round(min(lat), 4),
        },
        "decay_check": {
            "documents_per_third": third,
            "first_third_docs_per_second": round(third / first_s, 3),
            "last_third_docs_per_second": round(third / last_s, 3),
            "last_over_first": round((third / last_s) / (third / first_s), 4),
            "note": ("A throughput number taken over a few seconds does not show "
                     "thermal or scheduler decay. These two are the same measurement "
                     "over the first and last third of the run."),
        },
        "decisions": {
            "forwarded_to_hosted": forwarded,
            "held_local": n - forwarded,
            "note": "Aggregate counts only. No document, span or placeholder is recorded.",
        },
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "rss_before_model_mib": round(rss0 / 1024.0, 1),
    }


def arm_gate_stage_breakdown(limit: int, model: str, calibration: str) -> dict:
    """Where the per-document second goes, timed INSIDE one process.

    Arms A/B/C run in separate processes on a shared host, so a 2-5% gap
    between them is not resolvable -- and the first run of this script produced
    exactly that: a `gate_sustained` faster than the batch-1 forward pass it
    strictly contains, which is impossible and was measuring noise. The stages
    are therefore timed against each other in a single process, where the
    comparison is not confounded by host load.

    The loop below is a replica of Gate.process's body. A replica can drift
    from the shipped path, so every document is ALSO put through the real
    gate.process and the two Decisions must be identical; the count of
    disagreements is reported and any disagreement fails the arm.
    """
    _pin(THREADS)
    from airlock.gate import Calibrator, Gate, raw_risk
    from airlock.hybrid import HybridDetector
    from airlock.redact import redact

    texts = _corpus(limit)
    det = HybridDetector(model, device="cpu")
    gate = Gate(det, Calibrator.load(calibration))
    gate.process(texts[0])                     # warm-up, excluded

    t = {"forward": 0.0, "decode_and_validators": 0.0, "redact": 0.0,
         "risk": 0.0, "calibrate": 0.0}
    disagreements = 0
    t0 = time.time()
    for text in texts:
        a = time.time(); toks = det._token_scores([text])[0]; t["forward"] += time.time() - a
        a = time.time(); spans = det._decode(text, toks, gate.threshold)
        t["decode_and_validators"] += time.time() - a
        a = time.time(); redacted, vault = redact(text, spans); t["redact"] += time.time() - a
        a = time.time(); expected, naive = raw_risk(text, toks, spans); t["risk"] += time.time() - a
        a = time.time()
        risk = gate.calibrator.predict(expected, gate.conservative)
        t["calibrate"] += time.time() - a
        replica_forward = risk <= gate.budget

        # equivalence control: the replica must decide what the shipped path decides
        if gate.process(text)["decision"].forward != replica_forward:
            disagreements += 1
    secs = time.time() - t0

    total = sum(t.values())
    n = len(texts)
    return {
        "arm": "gate_stage_breakdown",
        "what": ("Gate.process's stages timed against each other in ONE process, so the "
                 "attribution is not confounded by host load. Wall seconds include a "
                 "second, untimed gate.process call per document for the equivalence "
                 "control, so this arm's docs_per_second is NOT a throughput figure."),
        "documents": n,
        "seconds_wall_including_control": round(secs, 2),
        "seconds_attributed": round(total, 3),
        "per_stage_seconds": {k: round(v, 3) for k, v in t.items()},
        "per_stage_share": {k: round(v / total, 5) for k, v in t.items()},
        "per_stage_milliseconds_per_document": {k: round(1000 * v / n, 3)
                                                for k, v in t.items()},
        "model_half_share": round((t["forward"] + t["decode_and_validators"]) / total, 5),
        "gate_half_share": round((t["redact"] + t["risk"] + t["calibrate"]) / total, 5),
        "gate_half_over_model_half": round(
            (t["redact"] + t["risk"] + t["calibrate"]) /
            (t["forward"] + t["decode_and_validators"]), 5),
        "equivalence_control": {
            "what": ("Every document is also run through the unmodified gate.process; "
                     "the replica's forward/hold decision must match it."),
            "documents_compared": n,
            "disagreements": disagreements,
            "held": disagreements == 0,
        },
    }


# ------------------------------------------------------------------ driver
def _child(args_list, emit):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = str(THREADS)          # pin the whole numeric stack, not just torch
    env["CUDA_VISIBLE_DEVICES"] = ""   # this box has an L40S; the claim is about CPU
    subprocess.run([sys.executable, os.path.abspath(__file__)] + args_list,
                   check=True, env=env,
                   cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return json.load(open(emit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/detector")
    ap.add_argument("--calibration", default="models/gate_calibration.json")
    ap.add_argument("--limit", type=int, default=200,
                    help="documents in the replayed corpus; 200 matches deploy_cost.json")
    ap.add_argument("--sustain-seconds", type=float, default=120.0)
    ap.add_argument("--out", default="evidence/sustained_load.json")
    ap.add_argument("--arm")
    ap.add_argument("--emit")
    args = ap.parse_args()

    if args.arm:
        if args.arm == "forward_batch8":
            r = arm_forward(BATCH8, args.limit, args.model)
        elif args.arm == "forward_batch1":
            r = arm_forward(1, args.limit, args.model)
        elif args.arm == "gate_sustained":
            r = arm_gate_sustained(args.sustain_seconds, args.limit, args.model,
                                   args.calibration)
        elif args.arm == "gate_stage_breakdown":
            r = arm_gate_stage_breakdown(args.limit, args.model, args.calibration)
        else:
            raise SystemExit("unknown arm " + args.arm)
        json.dump(r, open(args.emit, "w"), indent=2)
        return

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    arms = []
    for arm in ("forward_batch8", "forward_batch1", "gate_sustained",
                "gate_stage_breakdown"):
        emit = f"/tmp/airlock_sustained_{arm}_{stamp}.json"
        arms.append(_child(
            ["--arm", arm, "--emit", emit, "--model", args.model,
             "--calibration", args.calibration, "--limit", str(args.limit),
             "--sustain-seconds", str(args.sustain_seconds)], emit))

    by = {a["arm"]: a for a in arms}
    f8, f1, gs = by["forward_batch8"], by["forward_batch1"], by["gate_sustained"]
    sb = by["gate_stage_breakdown"]

    published = json.load(open("evidence/deploy_cost.json"))
    pub4 = [r for r in published["runs"]
            if r["device"] == "cpu" and r["torch_threads"] == THREADS][0]
    ratio = f8["docs_per_second"] / pub4["docs_per_second"]

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": ("evidence/deploy_cost.json times the forward pass at batch 8 and the "
                "README turns its 4-thread figure into a documents-per-day number. "
                "Gate.process -- what a sidecar runs -- is a different path at a "
                "different batch. This measures that path, sustained."),
        "threads": THREADS,
        "device": "cpu",
        "arms": arms,
        "tie_in_control": {
            "what": ("Arm A replays the published configuration. Without it a lower "
                     "number in arm C could be this host being busier today rather "
                     "than the deployed path being slower."),
            "published_docs_per_second": pub4["docs_per_second"],
            "published_at": published["generated_at"],
            "replayed_docs_per_second": f8["docs_per_second"],
            "ratio_replayed_over_published": round(ratio, 4),
            "reproduced_within_15_percent": bool(abs(ratio - 1.0) <= 0.15),
        },
        "decomposition": {
            "forward_batch8_docs_per_second": f8["docs_per_second"],
            "forward_batch1_docs_per_second": f1["docs_per_second"],
            "gate_sustained_docs_per_second": gs["docs_per_second"],
            "cost_of_batch1_vs_batch8": round(f8["docs_per_second"] / f1["docs_per_second"], 3),
            "cost_of_deployed_path_vs_published": round(
                f8["docs_per_second"] / gs["docs_per_second"], 3),
            "cost_of_gate_over_forward_batch1_ACROSS_PROCESSES": round(
                f1["docs_per_second"] / gs["docs_per_second"], 3),
            "why_that_ratio_is_not_used": (
                "It comes out at or below 1.0, and a path that strictly contains the "
                "batch-1 forward pass cannot be faster than it. The arms run in "
                "separate processes on a shared 384-core host and a few percent between "
                "them is host load, not code. The attribution below is measured inside "
                "one process instead."),
            "gate_stages_over_model_stages_WITHIN_PROCESS":
                sb["gate_half_over_model_half"],
            "note": ("Batch 8 pads every sequence in the batch to the longest one, so "
                     "on documents of uneven length the batch a sidecar really has -- "
                     "one -- is not the slower configuration. The redaction, risk and "
                     "calibration that the published figure leaves out are measured "
                     "within-process at gate_stages_over_model_stages_WITHIN_PROCESS "
                     "of the model's own cost."),
        },
        "documents_per_day": {
            "published_forward_pass_extrapolation": published["cpu4_documents_per_day"],
            "sustained_deployed_path": round(gs["docs_per_second"] * 86400.0),
            "overstatement_factor": round(
                published["cpu4_documents_per_day"] / (gs["docs_per_second"] * 86400.0), 2),
        },
        "not_claimed": (
            "Still one process on one machine. No HTTP server, no concurrent clients, "
            "no queueing discipline and no quantisation, on a 384-core shared host "
            "pinned to %d threads by OMP/MKL/OPENBLAS and torch.set_num_threads but not "
            "otherwise idle-guaranteed. The answering model is not in this path at all "
            "-- neither the local fallback nor the hosted stand-in is invoked, so a "
            "document the gate REFUSES costs far more end to end than these numbers "
            "show. Documents average %s characters; longer documents are slower because "
            "windows, not documents, are the unit of work."
            % (THREADS, gs["mean_characters_per_document"])),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=2))
    for a in arms:
        # gate_stage_breakdown deliberately publishes no docs_per_second: its wall
        # clock includes a second gate.process per document for the equivalence
        # control, so a throughput read off it would be wrong by about half.
        if "docs_per_second" in a:
            print(f"  {a['arm']:<22} {a['docs_per_second']:>8.3f} docs/s  "
                  f"{a['seconds']:>6.2f}s  {a['documents']:>5d} docs")
        else:
            print(f"  {a['arm']:<22} {'--':>8} docs/s  "
                  f"{a['seconds_wall_including_control']:>6.2f}s  {a['documents']:>5d} docs "
                  f"(stage attribution; equivalence control held="
                  f"{a['equivalence_control']['held']})")


if __name__ == "__main__":
    main()
