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

`python3 tests/run_checks.py` → **19/19**, each accepting check paired with a
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

bash scripts/fetch_model.sh        # the exact weights every number here was measured with
#   or
bash scripts/train_all.sh          # rebuild from the corpus: ~4 minutes on one L40S

python3 tests/run_checks.py        # 19 controls
python3 tests/check_readme_numbers.py   # every number below, re-read from evidence/

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

- a **fine-tuned transformer token classifier** (distilbert-base-cased, 29
  identifier types, 59 BIO labels) trained here from the base checkpoint — it is
  the only component that can find an identifier with no fixed shape;
- its **per-token posterior** is the input to the risk statistic, so the model's
  uncertainty is used as a quantity rather than thresholded away;
- **isotonic calibration** turns that statistic into a probability with a
  measured reliability curve;
- **two generative models** on the answering side, one standing in for the
  hosted model and one as the local fallback.

Remove the learned model and the system collapses to the regex control in the
table above: 38.45% recall, 90.39% of documents leaking.

## Attribution

- Training corpus: [`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual),
  Apache-2.0, English split only. Synthetic by construction — no real personal
  data was used, held or processed anywhere in this project.
- Base encoder: `distilbert-base-cased` (Apache-2.0).
- Answering models: `Qwen/Qwen2.5-0.5B-Instruct` and
  `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0).
- Libraries: PyTorch, Hugging Face Transformers, pandas (data preparation only).

## Responsible AI

The purpose of this project is data minimisation, and it is built not to
overstate its own protection. Every headline number is an outcome measured on
held-out data, the uncertain ones are reported as bounds in the direction that
is *against* the system, and the gate refuses rather than guesses when it
cannot support a bound. It processes only synthetic documents in this
repository. It is a control that reduces exposure; it is not a compliance
guarantee, and nothing here should be read as legal advice.

## Licence

MIT — see `LICENSE`.
