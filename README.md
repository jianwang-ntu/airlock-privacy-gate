# Airlock

**A calibrated privacy gate in front of hosted AI.**

Redaction tools tell you what they found. Airlock tells you what it might have
**missed** — as a probability, calibrated against measured outcomes — and
refuses to send the document when that number is above the operator's budget.
Those documents are answered by a small model on the operator's own machine
instead.

Built for **UPAI-Hackdays** (MLH), September 2026. Solo entry. Everything below
is measured by code in this repository and re-derivable from `evidence/`.

---

## Demo

**[`demo/airlock-demo.mp4`](demo/airlock-demo.mp4)** — 1 min 46 s, silent, 1080p.
A recording of both paths running end to end: one document the gate refuses and
answers locally, one it forwards to the hosted stand-in. It ends on the
integrity check reporting `trustworthy=False`, which is the system behaving as
designed and was left in.

The recording and the video are produced by `scripts/record_demo.py` and
`scripts/render_demo_video.py`, and the raw capture is in
`demo/demo_cast.json`. `demo/README.md` states exactly what the renderer
changed — timing only — and why the numbers on the cards cannot drift from
`evidence/`.

---

## The problem

A support team wants to use a hosted language model on its ticket queue. The
tickets contain names, account numbers, phone numbers and card details. Sending
them to a third party is a data-protection event — India's DPDP Act 2023 and
GDPR Article 5(1)(c) both require that you send the minimum, and both put the
liability on the *sender*.

The usual answer is a PII redaction step. The gap is that redaction is
**best effort and silent about its own failures**. It hands back a masked
document with no statement of residual risk, so the person accountable for the
decision has nothing to decide with.

That gap is not hypothetical. **Our own detector, at its shipped operating
point, still leaves an annotated identifier in 13.27% of the documents that
contain one.** A pipeline that redacts and forwards regardless is a pipeline
that leaks more than one document in eight and never says so.

## What Airlock does

```
document ──▶ detect ──▶ redact to typed placeholders ──▶ score what survived
                                                              │
                                        risk ≤ budget ────────┴──────── risk > budget
                                             │                              │
                                     hosted model                    local model
                                             │                              │
                                             └────▶ re-hydrate locally ◀────┘
                                                    + integrity check
```

1. **Detect** — a fine-tuned token classifier over 29 identifier types, unioned
   with checksum validators (Luhn, IBAN mod-97) for the identifiers that carry
   their own proof.
2. **Redact** — every occurrence of one identifier becomes the *same* typed
   placeholder (`[NAME_1]`, `[IBAN_2]`), so the outgoing text still reads as a
   document and the model can still reason over it. The mapping never leaves
   the machine.
3. **Score** — every token that survived into the outgoing text carries the
   detector's own probability that it was an identifier. Their sum is mapped
   onto observed leak frequency by isotonic regression fitted on documents held
   out of the detector's training.
4. **Decide** — forward, or answer locally. The gate decides on a one-sided 95%
   **upper** bound, never the point estimate.
5. **Re-hydrate** — the answer comes back with placeholders; the real values go
   in locally. If the model failed to return a placeholder, Airlock says so
   rather than handing back a document with an invented account number in it.

## What is measured

2,891 held-out documents, never used for training or calibration. 8,133
annotated identifiers; 2,186 documents carry at least one.

**Detection** (`evidence/detector_eval.json`), at the shipped threshold 0.2:

| detector | identifier recall | documents leaking ≥1 | over-redaction (upper bound) |
|---|---|---|---|
| **Airlock (model ∪ validators)** | **93.96%** | **13.27%** | 26.72% |
| model only | 93.98% | 13.22% | 25.56% |
| regex + capitalised-bigram control | 38.45% | 90.39% | 78.71% |
| checksum validators only | 7.06% | 99.41% | 47.18% |

**The gate** (`evidence/gate_eval.json`), redaction threshold 0.2:

| policy | documents forwarded | leak rate among forwarded |
|---|---|---|
| forward everything | 100% | 10.03% |
| **Airlock, budget 0.12** | **63.65%** | **4.67%** |
| random gate, same coverage | 63.65% | 10.10% |
| Airlock, budget 0.045 | 9.48% | 1.82% |
| random gate, same coverage | 9.48% | 10.35% |

The random-gate row is the control that carries the weight: withholding the
same *number* of documents at random leaves the leak rate at the base rate, so
the reduction comes from the ranking and not from sending less.

Ranking AUC **0.7498**. Expected calibration error **0.0105**. The upper bound
was conservative in **11/11** reliability deciles, and the observed leak rate
among forwarded documents was inside the budget at **every** operating point
tested.

**Cost.** The detector is 65.2M parameters, fine-tuned in **156.8 s** on one
L40S (2,492 steps over 24,019 documents), and scores **106.85 documents/second**
on that GPU. There is no per-document API cost and no third-party dependency in
the detection path.

**It does not need that GPU.** The vault of real values must never leave the
operator's machine, so the gate has to be able to run on the machine the
documents are already on. `scripts/measure_deploy_cost.py` runs the same
forward pass — `Detector._token_scores`, batch 8 — over the same 200 test
documents on each device:

| device | documents / second |
|---|---|
| CPU, 1 thread | **1.59** |
| CPU, 4 threads | **4.18** |
| CPU, 8 threads | **8.00** |
| NVIDIA L40S | **103.94** |

Four pinned CPU cores clear **361,152 documents per day** of continuous
detection; the GPU is worth **24.87x** and is an optimisation, not a
requirement. Weights are **248.9 MiB** on disk and the 4-core run peaked at
**1,485.4 MiB** resident. The GPU row here is an independent re-measurement of
the 106.85 above on a 200-document subset, and lands within 3% of it. Forward
pass only: no HTTP, no queueing, no quantisation, on a host that was not
idle-guaranteed.

`python3 tests/run_checks.py` → **21/21**, each accepting check paired with a
control that must fail for it to mean anything.

## What this is *not*

Stated first-hand, because two of these were found by controls in this
repository rather than by reading the metrics:

- **The corpus's own annotation is incomplete.** In 800 test documents the
  checksum validators found **135 email addresses the annotation does not
  mark**. So recall here is an *upper* bound on true recall and over-redaction
  an *upper* bound on false positives. Both are reported in that direction on
  purpose.
- **The validator layer changes the corpus headline by less than 0.1 points**,
  because only **44 of 83** card numbers in the corpus are Luhn-valid. It is in
  the system because the model alone left `4111 1111 1111 1111` — a valid,
  universally published test card — in the outgoing text, and a control caught
  it. Synthetic corpora under-represent exactly the case a checksum settles.
- **The risk statistic grows with document length.** A long clean document can
  outscore a short risky one; AUC 0.7498 is real discrimination, not a solved
  problem.
- **Recall is uneven by type.** `last_name` alone is 52.56% (78 instances);
  `iban`, `ipv4`, `ipv6`, `bban` and `driver_license_number` are 100%.
- **The calibration is threshold-specific and data-hungry.** Change the
  redaction threshold and it must be refitted. With 2,000 calibration documents
  the tightest certifiable risk is ~4.4%; below that budget Airlock forwards
  nothing rather than claiming a bound it cannot support.
- **No third-party API is called anywhere in this repository.** The forwarded
  leg was exercised against a larger model on the same machine standing in for
  the hosted one (`airlock/route.py::HostedStandIn`), so the loop is real
  end-to-end without anything leaving the host. Pointing it at a real provider
  is a one-function adapter that has not been exercised here.
- **One language, one domain.** English financial and support documents.
  Nothing here has been measured on real production text.

## Run it

```bash
pip install -r requirements.txt
# scripts/render_demo_video.py also needs the ffmpeg binary: apt install ffmpeg

bash scripts/fetch_model.sh        # the exact weights every number here was measured with
#   or
bash scripts/train_all.sh          # rebuild from the corpus: ~4 minutes on one L40S

python3 tests/run_checks.py        # 21 controls
python3 tests/check_readme_numbers.py   # every number below, re-read from evidence/
python3 tests/check_learned_model_ablation.py   # controls for the rule 3 counterfactual

python3 -m airlock.cli --text "Please wire GBP 12,400 to IBAN DE89370400440532013000 \
for Meera Subramanian; card 4111 1111 1111 1111 must not be charged."

python3 -m airlock.cli --file ticket.txt --ask "What is the customer asking for?"
```

```bash
python3 scripts/record_demo.py          # re-record the demo capture
python3 scripts/render_demo_video.py    # re-render demo/airlock-demo.mp4
```

`scripts/demo_run.py` runs two documents end to end — one the gate forwards,
one it refuses — and writes exactly what happened to `evidence/demo_run.json`,
including the answers and the integrity report.

## How AI is used

AI is the mechanism, not a wrapper around someone else's:

- a **fine-tuned transformer token classifier** (distilbert-base-cased, **29**
  PII types as **59** BIO labels, **25** of them treated as identifying and
  redacted) trained here from the base checkpoint — it is the only component
  that can find an identifier with no fixed shape;
- its **per-token posterior** is the input to the risk statistic, so the model's
  uncertainty is used as a quantity rather than thresholded away;
- **isotonic calibration** turns that statistic into a probability with a
  measured reliability curve;
- **two generative models** on the answering side, one standing in for the
  hosted model and one as the local fallback.

Remove the learned model and the *detector* collapses to the regex control in
the table above: 38.45% recall, 90.39% of documents leaking. What that costs the
**product** is measured separately, below.

### Is the posterior load-bearing, or just the decision?

`scripts/ablate_posterior.py` reads the *same* forward pass and the *same*
redaction spans five ways and ranks the test split by each, at fixed coverage —
forward the k lowest-risk documents, same k — because only the shipped
statistic has a calibrator and comparing budgets would compare calibrations
instead of rankings.

| how the model's output is read | AUC | leak rate among the 63.65% forwarded |
|---|---|---|
| **sum of per-token posterior — shipped** | **0.7498** | **4.67%** |
| count of surviving tokens (uncertainty discarded) | 0.4531 | 11.20% |
| flag at 0.01 (posterior → a yes/no) | 0.7337 | 5.22% |
| flag at 0.05 | 0.7077 | 4.84% |
| flag at the redaction threshold, 0.20 | 0.5207 | 9.51% |
| random ranking, 20 seeds | — | 10.10% |

**The gate runs on what is below the redaction threshold.** Use only the tokens
the model chose to mask — its verdict with the uncertainty discarded — and it
collapses to a coin flip: AUC 0.5207, three distinct values across 2,891
documents, against a 10.03% base rate. Token count alone scores 0.4531, *below*
chance, so the length objection to a length-sensitive statistic does not hold on
this corpus. Against us: a flag at 0.01 reaches 0.7337 and 5.22%, so the
continuous posterior is ahead by 0.0161 AUC and 0.55 points, not by an order of
magnitude. The AUCs are single-run point estimates with no interval attached and
the flag thresholds were not tuned.

This script never loads the calibrator; it re-sums the surviving-token
posteriors itself, and lands on the same 0.7498 and the same 4.67% as the
calibrated gate — an independent reproduction of both by a second code path.

### Delete the learned model: can the gate absorb it?

The row above is about the *detector*. The fair objection to it is that the gate
is exactly the thing meant to absorb a weak detector — let a no-ML pipeline
withhold what it cannot handle and it might reach the same safety at some lower
coverage. `scripts/ablate_learned_model.py` runs three whole pipelines over the
same 2,891 held-out documents to find out.

| arm | redaction | risk statistic | documents leaking |
|---|---|---|---|
| **S — shipped** | model ∪ validators | Σ posterior over surviving tokens | **10.03%** |
| B — no model anywhere | regex ∪ validators | best no-ML statistic | 68.35% |
| R — redaction API | model ∪ validators | best no-ML statistic | 10.03% |

Arm R is the serious competitor: buy redaction, treat it as a black box, build
the gate yourself. Its redaction is identical to arm S's, so the only thing it
lacks is the posterior. **Both no-ML arms pick their statistic from twelve
candidates — six, and both signs of each — by looking at the test labels.** That
is an oracle arm S is not given, and it is granted on purpose.

At the shipped 63.65% coverage, all three forwarding the same 1,840 documents:

| arm | statistic chosen | AUC | leaks forwarded | leak rate | exposure |
|---|---|---|---|---|---|
| **S — shipped** | Σ posterior | **0.7498** | **86** | **4.67%** | **2.97%** |
| B — no model | surviving capitalised words | 0.6493 | 1,129 | 61.36% | 39.05% |
| R — redaction API | spans redacted | 0.5954 | 149 | 8.10% | 5.15% |

Held to arm S's *absolute* exposure — the same 86 leaking documents forwarded —
arm B may forward 236 documents (8.16% coverage) and arm R 1,182 (40.89%),
against the shipped 63.65%. The gate does not absorb the missing model.

Two things here run against this project and are stated rather than buried.
Arm R's 0.5954 beats the 0.5207 this README previously cited when it called a
redaction API's output a coin flip: counting the spans a redactor returns is a
real if weak signal, and the earlier sentence generalised further than anything
measured. And arm B's ranking is not worthless — 0.6493 against a 68.54% random
gate at the same coverage. It is not the ranking that fails when the model is
deleted; it is that a 68.35% base leak rate leaves nothing good enough to
forward.

`tests/check_learned_model_ablation.py` → **30/30**, and the two that matter
most hand the harness a perfect ranking and a perfectly wrong one, so "the no-ML
gate scored badly" can be told apart from "the scorer scores everything badly".

## Attribution

The full inventory — every library, model, dataset, binary and font, each with
the licence read off this machine and off the publisher's own surface — is
**[`THIRD_PARTY.md`](THIRD_PARTY.md)**, derived by
`scripts/measure_third_party.py` into `evidence/third_party.json`. The short
version:

- Training corpus: [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual),
  Apache-2.0, English split only. Synthetic by construction — no real personal
  data was used, held or processed anywhere in this project.
- Base encoder: `distilbert-base-cased` (Apache-2.0). The detector we release
  is a fine-tune of it, so the released weights are Apache-2.0 too.
- Answering models: `Qwen/Qwen2.5-0.5B-Instruct` and
  `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0), downloaded by you, not shipped
  by us.
- Libraries, the complete set our own code imports — derived from the import
  graph by `scripts/measure_environment.py`, not typed from memory:
  `torch`, `transformers`, `pyarrow`, `pandas`, `Pillow`, `scipy`. Everything
  else is the Python standard library. `scipy` is analysis-only — it
  cross-checks this repository's own Fisher exact test and nothing in
  `airlock/` imports it. `ffmpeg` is a system binary, used only to render the
  demo video.
- **No hosted inference API is called anywhere in this project.** Measured, not
  asserted: 25 vendor client packages were looked for in the import graph and
  none is present, and no provider endpoint or credential appears in the
  source. The network is used in three places, all downloads and none of them
  inference — the corpus from `huggingface.co`, our own release weights from
  `github.com`, and the licence lookups in `scripts/measure_third_party.py`
  (`--offline` skips those). `transformers.from_pretrained` also fetches a
  checkpoint on first use if you hand it a hub id instead of a local path.
  After `scripts/fetch_model.sh` the gate needs no network at all.

## Originality

**[`ORIGINALITY.md`](ORIGINALITY.md)** answers the event's rule 2 sentence by
sentence, and answers the middle one — *"previously developed projects may not
be submitted as-is"* — by searching this machine for a copy of every tracked
file rather than by asserting there isn't one. It reports what the search
covered, what it found, and what it cannot see. `scripts/measure_originality.py`
produces it into `evidence/originality.json`;
`tests/check_originality_scan.py` is the set of controls that has to find a copy
that *is* there before "none found" is worth reading.

## Responsible AI

The purpose of this project is data minimisation, and it is built not to
overstate its own protection. Every headline number is an outcome measured on
held-out data, the uncertain ones are reported as bounds in the direction that
is *against* the system, and the gate refuses rather than guesses when it
cannot support a bound. It processes only synthetic documents in this
repository. It is a control that reduces exposure; it is not a compliance
guarantee, and nothing here should be read as legal advice.

**It has not been evaluated against an adversary.** Every number here comes
from a corpus that is not trying to defeat the detector. Airlock is a control
against accident, not against intent.

### Who this system protects less

A gate that catches 93.96% of identifiers is not thereby fair. If the missing
6.04% falls on one kind of person, those are the people it exposes and the
headline hides it. `scripts/measure_fairness.py` re-scores all 8,133 gold
identifiers in the held-out split through the shipped detector and splits them
along six pre-registered contrasts, with Wilson intervals and Holm-corrected
Fisher exact tests. It refuses to run unless it first reproduces
`evidence/detector_eval.json` exactly — 6 aggregate fields and all 25 per-type
cells. Full write-up in `evidence/fairness.json`.

**Protection depends on whether the model has met your name before.** A name
whose exact surface form is in the detector's gradient set is redacted
**96.96%** of the time; one that is not, **87.54%** (n = 1,549, Holm p 1.4e-27).
The obvious objection is span length — bare surnames are one token, full names
three — so the contrast was re-run with length held at one token, and the gap
**widens** to **62.28%** against **85.15%**. A bare surname is redacted
**52.56%** of the time (n = 78, CI 41.62–63.26) against **85.53%** for a
single-token given name, a 32.97-point gap. Rare names are, in the world,
disproportionately the names of people outside the majority naming convention
of the training data.

**The headline is flattered by the benchmark.** **57.98%** of held-out name
spans share an exact surface with a training document — an artifact of a
synthetic corpus that a real deployment would not enjoy. Repricing every name
span at the measured unseen rate turns 93.96% into a projected **91.49%**.

**One result ran against the hypothesis and is reported anyway.** Names with
non-ASCII characters were expected to fare worse; they are redacted **97.60%**
of the time against **92.35%** for pure-ASCII names. That should not be read as
"fair to international names" — the corpus is synthetic, and the axis on which
this system is unfair turned out to be rarity, not script.

**128** of the 1,939 distinct person-name surfaces in the held-out split leaked
at least once. No mitigation is implemented: the disparity is measured,
disclosed, and still present in the shipped system.
`tests/check_fairness_measurement.py` — **29/29**, including 200 null draws at
an identical leak rate to show the machinery does not manufacture gaps
(**1.50%** significant at α = 0.05), a planted 100%-vs-0% gap it must find, and
a corrupted-reference run it must refuse.

## Licence

MIT for this repository — see `LICENSE`.

The released weights are a different matter and are not covered by it: they are
a fine-tune of `distilbert-base-cased`, so `airlock-detector-v0.1.0.tar.gz` is
distributed under **Apache-2.0**, with the modification stated in
[`THIRD_PARTY.md`](THIRD_PARTY.md#1-what-we-redistribute-and-under-what-licence).
