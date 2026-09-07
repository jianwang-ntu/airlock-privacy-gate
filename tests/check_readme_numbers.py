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


def _lib_order(envj):
    """The README lists the libraries runtime-first. The SET is what matters --
    order is fixed here so a missing or added library moves the string."""
    order = ["torch", "transformers", "pyarrow", "pandas", "PIL"]
    libs = set(envj["libraries"])
    named = [m for m in order if m in libs]
    rest = sorted(libs - set(order))
    return [("Pillow" if m == "PIL" else m) for m in named + rest]


def _netfiles(envj):
    """The files the import sweep flags as opening a network connection."""
    seen = envj["network_capability"]["stdlib_network_modules_imported"].values()
    return "`" + "`, `".join(sorted({f for fs in seen for f in fs})) + "`"


def _ev(ed, arm, lib):
    """A library version as one arm of env_drift.json recorded it."""
    return ed["arms"][arm]["installed"][lib]


def _c(ed, key):
    """A [pinned, unpinned] pair out of env_drift.json's drift_cost block."""
    return ed["drift_cost"][key]


def _g(x):
    return f"{x:g}"


def _budgets(ed, side):
    b = _c(ed, "budgets_violated")[side]
    return ", ".join(f"{x:g}" for x in b) if b else "none"


def claims(det, gate, gaps, checks_total, deploy, abl, envj, lma, lmac, fair, fairc,
           drop=None, envd=None):
    # `drop` and `envd` default to the files on disk so the existing call sites
    # are unchanged; the controls at the end of main() pass corrupted copies.
    d = drop if drop is not None else load("loader_span_drop.json")
    ed = envd if envd is not None else load("env_drift.json")
    h = det["by_threshold_hybrid"]["0.2"]
    m = det["by_threshold_model_only"]["0.2"]
    # ---- rule 8 fairness helpers
    _CN = {c["id"]: c for c in fair["contrasts"]}
    fh, fproj, finstr = h, fair["deployment_projection"], fair["instrument_check"]
    _WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
    _word = lambda n: _WORDS.get(n, str(n))
    _fpct = lambda x: f"{x * 100:.2f}%"
    _fr = lambda cid, side: _fpct(_CN[cid][side]["recall"])
    _fn = lambda cid, side: f"{_CN[cid][side]['spans']:,}"
    _fgap = lambda cid: f"{abs(_CN[cid]['recall_gap_a_minus_b']) * 100:.2f}"
    _fp = lambda cid: "%.2g" % _CN[cid]["holm_adjusted_p"]

    def _fci(cid, side):
        b = _CN[cid][side]["recall_ci95"]
        return f"{b[0] * 100:.2f}\u2013{b[1] * 100:.2f}"

    rx, val = det["regex_baseline"], det["checksum_validators_only"]
    rows = {r["budget"]: r for r in gate["risk_coverage"]}
    runs = {(r["device"], r["torch_threads"]): r for r in deploy["runs"]}
    ab = {r["statistic"]: r for r in abl["statistics"]}
    cov = "leak_rate_at_coverage_0.6365"
    dep = lambda k: f"{runs[k]['docs_per_second']:.2f}"
    b12, b045 = rows[0.12], rows[0.045]
    pct = lambda x: f"{x * 100:.2f}%"
    # the rule 3 counterfactual: three whole pipelines, keyed by arm name
    A = {a["arm"]: a for a in lma["arms"]}
    aS, aB, aR = A["S_shipped"], A["B_no_learned_model"], A["R_redaction_api"]
    cv = "%.4f" % lma["coverages_compared"][0]
    cS, cB, cR = aS["at_coverage"][cv], aB["at_coverage"][cv], aR["at_coverage"][cv]
    mB = aB["coverage_needed_to_match_shipped_exposure"][cv]
    mR = aR["coverage_needed_to_match_shipped_exposure"][cv]
    n_cand = len(aB["candidates"])
    n_stat = len({c["statistic"].replace(" (negated)", "") for c in aB["candidates"]})
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
        ("**21/21**", f"**{checks_total}/{checks_total}**"),
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
        # ---- the posterior ablation
        ("distilbert-base-cased, **29**\n  PII types as **59** BIO labels",
         f"distilbert-base-cased, **{(det['model']['labels'] - 1) // 2}**\n"
         f"  PII types as **{det['model']['labels']}** BIO labels"),
        ("| **sum of per-token posterior — shipped** | **0.7498** | **4.67%** |",
         f"| **sum of per-token posterior — shipped** | **{ab['posterior']['auc']}** | "
         f"**{pct(ab['posterior'][cov])}** |"),
        ("| count of surviving tokens (uncertainty discarded) | 0.4531 | 11.20% |",
         f"| count of surviving tokens (uncertainty discarded) | {ab['count']['auc']} | "
         f"{pct(ab['count'][cov])} |"),
        ("| flag at 0.01 (posterior → a yes/no) | 0.7337 | 5.22% |",
         f"| flag at 0.01 (posterior → a yes/no) | {ab['flag@0.01']['auc']} | "
         f"{pct(ab['flag@0.01'][cov])} |"),
        ("| flag at 0.05 | 0.7077 | 4.84% |",
         f"| flag at 0.05 | {ab['flag@0.05']['auc']} | {pct(ab['flag@0.05'][cov])} |"),
        ("| flag at the redaction threshold, 0.20 | 0.5207 | 9.51% |",
         f"| flag at the redaction threshold, 0.20 | {ab['flag@0.2']['auc']} | "
         f"{pct(ab['flag@0.2'][cov])} |"),
        ("| random ranking, 20 seeds | — | 10.10% |",
         f"| random ranking, 20 seeds | — | "
         f"{pct(ab['random ranking (20 seeds)'][cov])} |"),
        ("collapses to a coin flip: AUC 0.5207, three distinct values across 2,891\ndocuments",
         f"collapses to a coin flip: AUC {ab['flag@0.2']['auc']}, "
         f"three distinct values across {abl['documents']:,}\ndocuments"),
        ("against a 10.03% base rate", f"against a {pct(abl['base_leak_rate'])} base rate"),
        ("Token count alone scores 0.4531", f"Token count alone scores {ab['count']['auc']}"),
        ("a flag at 0.01 reaches 0.7337 and 5.22%",
         f"a flag at 0.01 reaches {ab['flag@0.01']['auc']} and {pct(ab['flag@0.01'][cov])}"),
        ("ahead by 0.0161 AUC and 0.55 points",
         f"ahead by {abl['auc_delta_posterior_minus_best_flag']} AUC and "
         f"{abs(ab['flag@0.01'][cov] - ab['posterior'][cov]) * 100:.2f} points"),
        ("lands on the same 0.7498 and the same 4.67% as the",
         f"lands on the same {gate['ranking_auc_raw_statistic']} and the same "
         f"{pct(rows[0.12]['observed_leak_rate_forwarded'])} as the"),
        # the suite size the README advertises. `checks_total` was threaded
        # through this function and then never used, so "19 controls" sat
        # unchecked while the suite grew to 21.
        ("python3 tests/run_checks.py        # 21 controls",
         f"python3 tests/run_checks.py        # {checks_total} controls"),
        # Requirements 5: the library list is the import graph, not a memory
        ("`torch`, `transformers`, `pyarrow`, `pandas`, `Pillow`, `scipy`",
         ", ".join(f"`{m}`" for m in _lib_order(envj))),
        ("ffmpeg", "ffmpeg" if "ffmpeg" in envj["external_binaries"] else "<absent>"),
        # ---- rule 8: who the system protects less. These rows are the only
        # ones that read fairness.json, so controls 7 and 8 below are sited on
        # it -- the six earlier controls would pass vacuously over all of them.
        ("catches 93.96% of identifiers", f"catches {_fpct(fh['identifier_recall'])} of identifiers"),
        ("6.04% falls on one kind of person",
         f"{_fpct(1 - fh['identifier_recall'])} falls on one kind of person"),
        ("all 8,133 gold", f"all {fair['population']['gold_identifying_spans']:,} gold"),
        ("along six pre-registered contrasts",
         f"along {_word(sum(1 for c in fair['contrasts'] if c['pre_registered']))} "
         f"pre-registered contrasts"),
        ("— 6 aggregate fields and all 25 per-type",
         f"— {finstr['aggregate_fields_compared']} aggregate fields and all "
         f"{finstr['per_type_cells_compared']} per-type"),
        ("**96.96%** of the time; one that is not, **87.54%** (n = 1,549, Holm p 1.4e-27).",
         f"**{_fr('C3', 'arm_b')}** of the time; one that is not, **{_fr('C3', 'arm_a')}** "
         f"(n = {_fn('C3', 'arm_a')}, Holm p {_fp('C3')})."),
        ("**62.28%** against **85.15%**", f"**{_fr('C5', 'arm_a')}** against **{_fr('C5', 'arm_b')}**"),
        ("**52.56%** of the time (n = 78, CI 41.62–63.26) against **85.53%**",
         f"**{_fr('C7', 'arm_a')}** of the time (n = {_fn('C7', 'arm_a')}, "
         f"CI {_fci('C7', 'arm_a')}) against **{_fr('C7', 'arm_b')}**"),
        ("a 32.97-point gap", f"a {_fgap('C7')}-point gap"),
        ("**57.98%** of held-out name",
         f"**{_fpct(fproj['share_of_person_name_spans_seen_in_training'])}** of held-out name"),
        ("turns 93.96% into a projected **91.49%**",
         f"turns {_fpct(fproj['measured_identifier_recall'])} into a projected "
         f"**{_fpct(fproj['projected_identifier_recall_if_no_name_were_seen_in_training'])}**"),
        ("redacted **97.60%**", f"redacted **{_fr('C4', 'arm_a')}**"),
        ("against **92.35%** for pure-ASCII names",
         f"against **{_fr('C4', 'arm_b')}** for pure-ASCII names"),
        ("**128** of the 1,939 distinct person-name surfaces",
         f"**{fair['population']['distinct_person_name_surfaces_leaked_at_least_once']}** of the "
         f"{fair['population']['distinct_person_name_surfaces']:,} distinct person-name surfaces"),
        ("**29/29**, including 200 null draws",
         f"**{fairc['passed']}/{fairc['total']}**, including {fairc['null_draws']} null draws"),
        ("(**1.50%** significant", f"(**{_fpct(fairc['null_false_positive_fraction'])}** significant"),
        # ---- rule 3: delete the learned model
        ("same 2,891 held-out documents to find out",
         f"same {lma['documents']:,} held-out documents to find out"),
        ("| **S — shipped** | model ∪ validators | Σ posterior over surviving tokens | **10.03%** |",
         f"| **S — shipped** | model ∪ validators | Σ posterior over surviving tokens | "
         f"**{pct(aS['base_leak_rate_all_documents'])}** |"),
        ("| B — no model anywhere | regex ∪ validators | best no-ML statistic | 68.35% |",
         f"| B — no model anywhere | regex ∪ validators | best no-ML statistic | "
         f"{pct(aB['base_leak_rate_all_documents'])} |"),
        ("| R — redaction API | model ∪ validators | best no-ML statistic | 10.03% |",
         f"| R — redaction API | model ∪ validators | best no-ML statistic | "
         f"{pct(aR['base_leak_rate_all_documents'])} |"),
        ("their statistic from twelve\ncandidates — six, and both signs of each",
         f"their statistic from {'twelve' if n_cand == 12 else n_cand}\n"
         f"candidates — {'six' if n_stat == 6 else n_stat}, and both signs of each"),
        ("At the shipped 63.65% coverage, all three forwarding the same 1,840 documents:",
         f"At the shipped {pct(float(cv))} coverage, all three forwarding the same "
         f"{cS['documents_forwarded']:,} documents:"),
        ("| **S — shipped** | Σ posterior | **0.7498** | **86** | **4.67%** | **2.97%** |",
         f"| **S — shipped** | Σ posterior | **{aS['ranking_auc']}** | "
         f"**{cS['leaking_documents_forwarded']:,}** | "
         f"**{pct(cS['leak_rate_among_forwarded'])}** | "
         f"**{pct(cS['whole_corpus_exposure'])}** |"),
        ("| B — no model | surviving capitalised words | 0.6493 | 1,129 | 61.36% | 39.05% |",
         f"| B — no model | {aB['oracle_statistic_chosen'].replace('_', ' ')} | "
         f"{aB['ranking_auc']} | {cB['leaking_documents_forwarded']:,} | "
         f"{pct(cB['leak_rate_among_forwarded'])} | {pct(cB['whole_corpus_exposure'])} |"),
        ("| R — redaction API | spans redacted | 0.5954 | 149 | 8.10% | 5.15% |",
         f"| R — redaction API | {aR['oracle_statistic_chosen'].replace('_', ' ')} | "
         f"{aR['ranking_auc']} | {cR['leaking_documents_forwarded']:,} | "
         f"{pct(cR['leak_rate_among_forwarded'])} | {pct(cR['whole_corpus_exposure'])} |"),
        ("the same 86 leaking documents forwarded —",
         f"the same {mB['shipped_leaking_documents_forwarded']:,} leaking documents forwarded —"),
        ("arm B may forward 236 documents (8.16% coverage) and arm R 1,182 (40.89%),\n"
         "against the shipped 63.65%.",
         f"arm B may forward {mB['max_documents_this_arm_can_forward']:,} documents "
         f"({pct(mB['max_coverage'])} coverage) and arm R "
         f"{mR['max_documents_this_arm_can_forward']:,} ({pct(mR['max_coverage'])}),\n"
         f"against the shipped {pct(float(cv))}."),
        ("Arm R's 0.5954 beats the 0.5207",
         f"Arm R's {aR['ranking_auc']} beats the {ab['flag@0.2']['auc']}"),
        ("0.6493 against a 68.54% random",
         f"{aB['ranking_auc']} against a "
         f"{pct(cB['random_gate_same_coverage_leak_rate'])} random"),
        ("a 68.35% base leak rate leaves nothing good enough to",
         f"a {pct(aB['base_leak_rate_all_documents'])} base leak rate leaves nothing "
         f"good enough to"),
        ("`tests/check_learned_model_ablation.py` → **30/30**",
         f"`tests/check_learned_model_ablation.py` → **{lmac['passed']}/{lmac['total']}**"),
        # ---- the loader's own discarded spans
        ("**72** of the **8,205** identifying annotations",
         f"**{d['discarded_by_overlap_rule']}** of the "
         f"**{d['identifying_annotations_in_corpus']:,}** identifying annotations"),
        ("leaving the **8,133** every recall number",
         f"leaving the **{d['identifying_annotations_scored']:,}** every recall number"),
        ("**44** of the 72 lie inside a span",
         f"**{d['discarded_covered_by_kept_spans']}** of the "
         f"{d['discarded_by_overlap_rule']} lie inside a span"),
        ("**28** reach past one, by **200** characters",
         f"**{d['discarded_not_covered']}** reach past one, by "
         f"**{d['uncovered_characters']}** characters"),
        ("Scored over all 8,205 with every discarded span counted as a miss,",
         f"Scored over all {d['identifying_annotations_in_corpus']:,} with every "
         f"discarded span counted as a miss,"),
        ("recall reads **93.14%** rather than 93.96% — **0.82** points",
         f"recall reads **{d['recall_if_all_discarded_are_misses'] * 100:.2f}%** rather "
         f"than {d['recall_reported'] * 100:.2f}% — "
         f"**{d['overstatement_percentage_points']:.2f}** points"),
        # ---- the environment the numbers came from (evidence/env_drift.json).
        # Two arms of the same code, weights, corpus, seed and torch build; only
        # the library versions move. Left column is this repository's pin.
        ("measured with **transformers " + _ev(ed, "reference", "transformers") + "**",
         f"measured with **transformers {_ev(ed, 'reference', 'transformers')}**"),
        ("resolved to **" + _ev(ed, "latest_unpinned", "transformers") + "** on "
         + ed["generated_at"][:10],
         f"resolved to **{_ev(ed, 'latest_unpinned', 'transformers')}** on "
         f"{ed['generated_at'][:10]}"),
        ("| identifier recall | **93.96%** | 86.63% |",
         f"| identifier recall | **{pct(_c(ed, 'identifier_recall')[0])}** | "
         f"{pct(_c(ed, 'identifier_recall')[1])} |"),
        ("| documents that still leak | **13.27%** | 26.94% |",
         f"| documents that still leak | **{pct(_c(ed, 'doc_leak_rate')[0])}** | "
         f"{pct(_c(ed, 'doc_leak_rate')[1])} |"),
        ("| gate ranking AUC | 0.7498 | 0.6273 |",
         f"| gate ranking AUC | {_g(_c(ed, 'gate_ranking_auc')[0])} | "
         f"{_g(_c(ed, 'gate_ranking_auc')[1])} |"),
        ("| calibration error (ECE) | 0.0105 | 0.112 |",
         f"| calibration error (ECE) | {_g(_c(ed, 'ece_posterior_mean')[0])} | "
         f"{_g(_c(ed, 'ece_posterior_mean')[1])} |"),
        ("| deciles where the certified bound holds | 11/11 | 2/11 |",
         "| deciles where the certified bound holds | "
         f"{_c(ed, 'reliability_bins_where_upper_bound_is_conservative')[0]} | "
         f"{_c(ed, 'reliability_bins_where_upper_bound_is_conservative')[1]} |"),
        ("| leak budgets violated | none | 0.045, 0.06, 0.09, 0.12, 0.16 |",
         f"| leak budgets violated | {_budgets(ed, 0)} | {_budgets(ed, 1)} |"),
        ("stops holding at 5 of the 9 budgets tested",
         f"stops holding at {len(_c(ed, 'budgets_violated')[1])} of the "
         f"{len(ed['arms']['reference']['headline']['budgets_tested'])} budgets tested"),
        ("`" + ed["coexistence_probe"]["transformers_requires_tokenizers"][0] + "`",
         "`" + ed["coexistence_probe"]["transformers_requires_tokenizers"][0] + "`"),
        # the import sweep's own verdict, and the file that moved it. Left
        # unchecked, a repository that quietly grows a second network caller
        # keeps reading as if it had not.
        ("**REVIEW** rather than **NO_NETWORK_CLIENT**",
         f"**{envj['network_capability']['verdict']}** rather than "
         "**NO_NETWORK_CLIENT**"),
        ("the file it flags is " + _netfiles(envj),
         "the file it flags is " + _netfiles(envj)),
    ]


def run(readme_text, det, gate, gaps, checks_total, deploy, abl, envj, lma, lmac,
        fair, fairc, drop=None, envd=None):
    bad = []
    for quoted, derived in claims(det, gate, gaps, checks_total, deploy, abl, envj,
                                  lma, lmac, fair, fairc, drop, envd):
        if quoted != derived:
            bad.append((quoted, derived, "value moved"))
        elif quoted not in readme_text:
            bad.append((quoted, derived, "not present in README"))
    return bad


def main():
    readme = open(os.path.join(ROOT, "README.md")).read()
    det, gate, gaps = load("detector_eval.json"), load("gate_eval.json"), load("corpus_gaps.json")
    deploy = load("deploy_cost.json")
    abl = load("posterior_ablation.json")
    envj = load("environment.json")
    lma = load("learned_model_ablation.json")
    lmac = load("learned_model_ablation_controls.json")
    fair = load("fairness.json")
    fairc = load("fairness_controls.json")
    sys.path.insert(0, ROOT)
    from tests.run_checks import CHECKS
    bad = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac, fair, fairc)
    for quoted, derived, why in bad:
        print(f"MISMATCH ({why}): README says {quoted!r}, evidence gives {derived!r}")
    total = len(claims(det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac, fair, fairc))
    print(f"{total - len(bad)}/{total} README claims match the evidence files")

    # negative control: corrupt one evidence value and require a mismatch
    import copy
    corrupted = copy.deepcopy(gate)
    corrupted["forward_everything_leak_rate"] = round(gate["forward_everything_leak_rate"] / 2, 4)
    ctrl = run(readme, det, corrupted, gaps, len(CHECKS), deploy, abl, envj, lma, lmac, fair, fairc)
    if not ctrl:
        print("CONTROL FAILED: halving the measured leak rate did not trip the checker")
        return 1
    print(f"control: halving forward_everything_leak_rate trips {len(ctrl)} claim(s) -- checker is live")

    # a control sited where the NEW claims are: the first one only moves a gate
    # value and would pass vacuously over the deployment-cost rows.
    dep_corrupt = copy.deepcopy(deploy)
    dep_corrupt["runs"] = [{**r, "docs_per_second": round(r["docs_per_second"] * 2, 2)}
                           for r in dep_corrupt["runs"]]
    ctrl2 = run(readme, det, gate, gaps, len(CHECKS), dep_corrupt, abl, envj, lma, lmac,
                fair, fairc)
    if not ctrl2:
        print("CONTROL FAILED: doubling every measured throughput did not trip the checker")
        return 1
    print(f"control: doubling every device throughput trips {len(ctrl2)} claim(s)")

    # a third control, sited on the ABLATION rows: neither control above touches
    # posterior_ablation.json, so both would pass vacuously over the new claims.
    # This one inverts the finding -- it makes the discarded-uncertainty variant
    # rank BETTER than the shipped statistic.
    abl_corrupt = copy.deepcopy(abl)
    for r in abl_corrupt["statistics"]:
        if r["statistic"] == "count":
            r["auc"] = 0.9111
            r["leak_rate_at_coverage_0.6365"] = 0.0101
    ctrl3 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl_corrupt, envj, lma, lmac, fair, fairc)
    if not ctrl3:
        print("CONTROL FAILED: inverting the ablation finding did not trip the checker")
        return 1
    print(f"control: inverting the count-statistic ablation trips {len(ctrl3)} claim(s)")
    # a fourth control, sited on the Requirements-5 rows: none of the three
    # above touches environment.json, so all three would pass vacuously over
    # the library list and the suite size.
    env_corrupt = copy.deepcopy(envj)
    env_corrupt["libraries"].pop("pyarrow")
    env_corrupt["external_binaries"].pop("ffmpeg")
    # and the flattering direction on the network sweep: a clean verdict with
    # nothing flagged, which is what a repository would look like if it stopped
    # reporting its own network callers.
    env_corrupt["network_capability"]["verdict"] = "NO_NETWORK_CLIENT"
    env_corrupt["network_capability"]["stdlib_network_modules_imported"] = {}
    ctrl4 = run(readme, det, gate, gaps, len(CHECKS) + 1, deploy, abl, env_corrupt, lma, lmac,
                fair, fairc)
    if len(ctrl4) < 3:
        print("CONTROL FAILED: dropping a library, a binary and moving the suite "
              f"size tripped only {len(ctrl4)} claim(s), expected 3")
        return 1
    print(f"control: dropping pyarrow + ffmpeg and moving the suite size "
          f"trips {len(ctrl4)} claim(s)")
    # a fifth control, sited on learned_model_ablation.json: none of the four
    # above reads that file, so all four pass vacuously over the rule 3 rows.
    # It inverts the finding -- the model-free arm is made SAFER than shipped --
    # and withdraws the oracle advantage at the same time.
    lma_corrupt = copy.deepcopy(lma)
    for a in lma_corrupt["arms"]:
        if a["arm"] == "B_no_learned_model":
            a["ranking_auc"] = 0.9312
            a["base_leak_rate_all_documents"] = 0.0101
            a["candidates"] = [c for c in a["candidates"]
                               if not c["statistic"].endswith("(negated)")]
    ctrl5 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma_corrupt, lmac,
                fair, fairc)
    if len(ctrl5) < 3:
        print("CONTROL FAILED: inverting the delete-the-model finding and withdrawing "
              f"the oracle advantage tripped only {len(ctrl5)} claim(s), expected 3")
        return 1
    print(f"control: inverting the delete-the-model finding trips {len(ctrl5)} claim(s)")
    # a sixth, on the controls file, because ctrl5 does not read it either
    lmac_corrupt = copy.deepcopy(lmac)
    lmac_corrupt["passed"] = lmac["passed"] - 1
    ctrl6 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac_corrupt,
                fair, fairc)
    if not ctrl6:
        print("CONTROL FAILED: a red ablation control did not trip the checker")
        return 1
    print(f"control: one red ablation control trips {len(ctrl6)} claim(s)")
    # a seventh, on fairness.json: none of the six above reads it, so all six
    # pass vacuously over every rule-8 row. It inverts the finding -- the unseen
    # arm is made SAFER than the seen arm -- which is the flattering direction.
    fair_corrupt = copy.deepcopy(fair)
    for c in fair_corrupt["contrasts"]:
        if c["id"] == "C3":
            c["arm_a"]["recall"] = 0.9891
            c["recall_gap_a_minus_b"] = 0.0195
        if c["id"] == "C7":
            c["arm_a"]["recall"] = 0.9012
    fair_corrupt["deployment_projection"][
        "projected_identifier_recall_if_no_name_were_seen_in_training"] = 0.9702
    ctrl7 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac,
                fair_corrupt, fairc)
    if len(ctrl7) < 3:
        print("CONTROL FAILED: inverting the fairness finding tripped only "
              f"{len(ctrl7)} claim(s), expected 3")
        return 1
    print(f"control: inverting the fairness finding trips {len(ctrl7)} claim(s)")
    # an eighth, on fairness_controls.json, because ctrl7 does not read it
    fairc_corrupt = copy.deepcopy(fairc)
    fairc_corrupt["passed"] = fairc["passed"] - 1
    fairc_corrupt["null_false_positive_fraction"] = 0.4
    ctrl8 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac,
                fair, fairc_corrupt)
    if len(ctrl8) < 2:
        print("CONTROL FAILED: a red fairness control and a broken null calibration "
              f"tripped only {len(ctrl8)} claim(s), expected 2")
        return 1
    print(f"control: a red fairness control + broken null calibration trips "
          f"{len(ctrl8)} claim(s)")
    # a ninth, on loader_span_drop.json: none of the eight above reads it, so
    # all eight pass vacuously over the discarded-span rows. It shrinks the
    # discard to nothing -- the flattering direction, and the one a reader
    # would be hurt by -- and must therefore be caught.
    drop_corrupt = copy.deepcopy(load("loader_span_drop.json"))
    drop_corrupt["discarded_by_overlap_rule"] = 0
    drop_corrupt["discarded_covered_by_kept_spans"] = 0
    drop_corrupt["discarded_not_covered"] = 0
    drop_corrupt["uncovered_characters"] = 0
    drop_corrupt["identifying_annotations_in_corpus"] = \
        drop_corrupt["identifying_annotations_scored"]
    drop_corrupt["recall_if_all_discarded_are_misses"] = drop_corrupt["recall_reported"]
    drop_corrupt["overstatement_percentage_points"] = 0.0
    ctrl9 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac,
                fair, fairc, drop_corrupt)
    if len(ctrl9) < 5:
        print("CONTROL FAILED: erasing the loader's discarded spans tripped only "
              f"{len(ctrl9)} claim(s), expected at least 5")
        return 1
    print(f"control: erasing the loader's discarded spans trips {len(ctrl9)} claim(s)")
    # a tenth, on env_drift.json: none of the nine above reads it, so all nine
    # pass vacuously over the environment table. It makes the unpinned arm cost
    # nothing -- the flattering direction, and the one that would let the pin be
    # quietly dropped -- and must therefore be caught.
    envd_corrupt = copy.deepcopy(load("env_drift.json"))
    cost = envd_corrupt["drift_cost"]
    for k in ("identifier_recall", "doc_leak_rate", "typed_recall", "identifiers_leaked",
              "docs_with_leak", "gate_ranking_auc", "ece_posterior_mean",
              "reliability_bins_where_upper_bound_is_conservative", "budgets_violated"):
        cost[k] = [cost[k][0], cost[k][0]]
    envd_corrupt["arms"]["latest_unpinned"]["installed"]["transformers"] = \
        envd_corrupt["arms"]["reference"]["installed"]["transformers"]
    ctrl10 = run(readme, det, gate, gaps, len(CHECKS), deploy, abl, envj, lma, lmac,
                 fair, fairc, None, envd_corrupt)
    if len(ctrl10) < 6:
        print("CONTROL FAILED: erasing the cost of an unpinned install tripped only "
              f"{len(ctrl10)} claim(s), expected at least 6")
        return 1
    print(f"control: erasing the cost of an unpinned install trips {len(ctrl10)} claim(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
