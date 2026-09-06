"""Every number the README quotes, re-read from evidence/ and compared.

A write-up drifts from its evidence silently: the measurement moves, the prose
does not, and the false claim is always the flattering one. This check makes
that drift loud. It ends with a negative control -- a deliberately corrupted
evidence value that the checker must reject -- so a run that reports 0 mismatches
is reporting something.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, "evidence", name)) as f:
        return json.load(f)


def claims(det, gate, gaps, checks_total, deploy):
    h = det["by_threshold_hybrid"]["0.2"]
    m = det["by_threshold_model_only"]["0.2"]
    rx, val = det["regex_baseline"], det["checksum_validators_only"]
    rows = {r["budget"]: r for r in gate["risk_coverage"]}
    runs = {(r["device"], r["torch_threads"]): r for r in deploy["runs"]}
    dep = lambda k: f"{runs[k]['docs_per_second']:.2f}"
    b12, b045 = rows[0.12], rows[0.045]
    pct = lambda x: f"{x * 100:.2f}%"
    return [
        ("2,891 held-out documents", f"{det['documents']:,} held-out documents"),
        ("8,133", f"{h['gold_identifiers']:,}"),
        ("2,186 documents carry", f"{h['docs_with_identifiers']:,} documents carry"),
        ("| **93.96%** |", f"| **{pct(h['identifier_recall'])}** |"),
        ("| **13.27%** |", f"| **{pct(h['doc_leak_rate'])}** |"),
        ("| 26.72% |", f"| {pct(h['over_redaction_rate_upper_bound'])} |"),
        ("| model only | 93.98% | 13.22% | 25.56% |",
         f"| model only | {pct(m['identifier_recall'])} | {pct(m['doc_leak_rate'])} | "
         f"{pct(m['over_redaction_rate_upper_bound'])} |"),
        ("| regex + capitalised-bigram control | 38.45% | 90.39% | 78.71% |",
         f"| regex + capitalised-bigram control | {pct(rx['identifier_recall'])} | "
         f"{pct(rx['doc_leak_rate'])} | {pct(rx['over_redaction_rate_upper_bound'])} |"),
        ("| checksum validators only | 7.06% | 99.41% | 47.18% |",
         f"| checksum validators only | {pct(val['identifier_recall'])} | "
         f"{pct(val['doc_leak_rate'])} | {pct(val['over_redaction_rate_upper_bound'])} |"),
        ("in 13.27% of the documents", f"in {pct(h['doc_leak_rate'])} of the documents"),
        ("| forward everything | 100% | 10.03% |",
         f"| forward everything | 100% | {pct(gate['forward_everything_leak_rate'])} |"),
        ("| **Airlock, budget 0.12** | **63.65%** | **4.67%** |",
         f"| **Airlock, budget 0.12** | **{pct(b12['coverage'])}** | "
         f"**{pct(b12['observed_leak_rate_forwarded'])}** |"),
        ("| random gate, same coverage | 63.65% | 10.10% |",
         f"| random gate, same coverage | {pct(b12['coverage'])} | "
         f"{pct(b12['random_gate_same_coverage_leak_rate_mean'])} |"),
        ("| Airlock, budget 0.045 | 9.48% | 1.82% |",
         f"| Airlock, budget 0.045 | {pct(b045['coverage'])} | "
         f"{pct(b045['observed_leak_rate_forwarded'])} |"),
        ("| random gate, same coverage | 9.48% | 10.35% |",
         f"| random gate, same coverage | {pct(b045['coverage'])} | "
         f"{pct(b045['random_gate_same_coverage_leak_rate_mean'])} |"),
        ("AUC **0.7498**", f"AUC **{gate['ranking_auc_raw_statistic']}**"),
        ("error **0.0105**", f"error **{gate['ece_posterior_mean']}**"),
        ("**11/11** reliability deciles",
         f"**{gate['reliability_bins_where_upper_bound_is_conservative']}** reliability deciles"),
        ("65.2M parameters", f"{det['model']['parameters'] / 1e6:.1f}M parameters"),
        ("**156.8 s**", f"**{det['model']['wall_seconds']} s**"),
        ("2,492 steps over 24,019 documents",
         f"{det['model']['steps']:,} steps over {det['model']['train_docs']:,} documents"),
        ("**106.85 documents/second**",
         f"**{det['throughput']['model_docs_per_second']} documents/second**"),
        ("**135 email addresses",
         f"**{gaps['validator_hits_outside_any_gold_span']['email']} email addresses"),
        ("**44 of 83** card numbers",
         f"**{gaps['credit_card_spans_luhn_valid']} of {gaps['credit_card_spans']}** card numbers"),
        ("`last_name` alone is 52.56%",
         f"`last_name` alone is {pct(h['per_type']['last_name']['recall'])}"),
        ("(78 instances)", f"({h['per_type']['last_name']['gold']} instances)"),
        ("**19/19**", f"**{checks_total}/{checks_total}**"),
        # deployment cost -- the runs are keyed by (device, threads), not by order
        ("| CPU, 1 thread | **1.59** |", f"| CPU, 1 thread | **{dep(('cpu', 1))}** |"),
        ("| CPU, 4 threads | **4.18** |", f"| CPU, 4 threads | **{dep(('cpu', 4))}** |"),
        ("| CPU, 8 threads | **8.00** |", f"| CPU, 8 threads | **{dep(('cpu', 8))}** |"),
        ("| NVIDIA L40S | **103.94** |", f"| NVIDIA L40S | **{dep(('cuda', None))}** |"),
        ("**361,152 documents per day**",
         f"**{deploy['cpu4_documents_per_day']:,} documents per day**"),
        ("worth **24.87x**", f"worth **{deploy['gpu_over_cpu4_speedup']:.2f}x**"),
        ("**248.9 MiB** on disk", f"**{deploy['weights_mib']} MiB** on disk"),
        ("peaked at\n**1,485.4 MiB** resident",
         f"peaked at\n**{runs[('cpu', 4)]['peak_rss_mib']:,.1f} MiB** resident"),
        ("same 200 test\ndocuments", f"same {deploy['documents']} test\ndocuments"),
    ]


def run(readme_text, det, gate, gaps, checks_total, deploy):
    bad = []
    for quoted, derived in claims(det, gate, gaps, checks_total, deploy):
        if quoted != derived:
            bad.append((quoted, derived, "value moved"))
        elif quoted not in readme_text:
            bad.append((quoted, derived, "not present in README"))
    return bad


def main():
    readme = open(os.path.join(ROOT, "README.md")).read()
    det, gate, gaps = load("detector_eval.json"), load("gate_eval.json"), load("corpus_gaps.json")
    deploy = load("deploy_cost.json")
    sys.path.insert(0, ROOT)
    from tests.run_checks import CHECKS
    bad = run(readme, det, gate, gaps, len(CHECKS), deploy)
    for quoted, derived, why in bad:
        print(f"MISMATCH ({why}): README says {quoted!r}, evidence gives {derived!r}")
    total = len(claims(det, gate, gaps, len(CHECKS), deploy))
    print(f"{total - len(bad)}/{total} README claims match the evidence files")

    # negative control: corrupt one evidence value and require a mismatch
    import copy
    corrupted = copy.deepcopy(gate)
    corrupted["forward_everything_leak_rate"] = round(gate["forward_everything_leak_rate"] / 2, 4)
    ctrl = run(readme, det, corrupted, gaps, len(CHECKS), deploy)
    if not ctrl:
        print("CONTROL FAILED: halving the measured leak rate did not trip the checker")
        return 1
    print(f"control: halving forward_everything_leak_rate trips {len(ctrl)} claim(s) -- checker is live")

    # a control sited where the NEW claims are: the first one only moves a gate
    # value and would pass vacuously over the deployment-cost rows.
    dep_corrupt = copy.deepcopy(deploy)
    dep_corrupt["runs"] = [{**r, "docs_per_second": round(r["docs_per_second"] * 2, 2)}
                           for r in dep_corrupt["runs"]]
    ctrl2 = run(readme, det, gate, gaps, len(CHECKS), dep_corrupt)
    if not ctrl2:
        print("CONTROL FAILED: doubling every measured throughput did not trip the checker")
        return 1
    print(f"control: doubling every device throughput trips {len(ctrl2)} claim(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
