# Third-party material, licences and attribution

UPAI-Hackdays rule 6 asks two separate things, and this file answers both:

> "Teams are responsible for ensuring they have the right to use all code,
> datasets, APIs, images, models, and other third-party resources included in
> their project."
> "Proper attribution must be provided where required."

Every row below was **read off this machine and off the publishers' own
surfaces** by `scripts/measure_third_party.py`, which writes
`evidence/third_party.json`. Nothing here is typed from memory, and a licence
that could not be read is reported as unknown rather than assumed permissive.
Re-derive it with:

```bash
python3 scripts/measure_third_party.py
```

That script makes **no network call**, on purpose: this repository contains no
network client, `evidence/environment.json` measures that over its whole Python
import graph, and the entry claims it. Adding an HTTP call here to look up a
licence would have traded a measured property of a privacy gate for one saved
`curl`. The publishers' own licence strings are therefore *pinned* in
`evidence/publisher_licenses.json`, and every row of that file carries the
one-line command that reproduces it.

---

## 1. What we redistribute, and under what licence

| Artefact | Contains | Licence |
|---|---|---|
| This repository | our own Python, README, demo documents, demo video | **MIT**, see `LICENSE` |
| Release asset `airlock-detector-v0.1.0.tar.gz` | `detector/` — `distilbert-base-cased` **fine-tuned by us**; `gate_calibration.json` — fitted by us | **Apache-2.0**, because the weights are a derivative of an Apache-2.0 model (§2) |

The repository licence is MIT; the released weights are not. Saying "MIT" over
the whole distribution would have been wrong, and it is the reason this file
exists rather than a line in the README.

**Statement of changes** (Apache-2.0 §4b): `distilbert-base-cased` was
fine-tuned for token classification over 59 BIO labels covering 29 identifier
types, for 2,492 steps / 2 epochs at lr 5e-5, by `airlock/train.py`. The
architecture head is replaced; the encoder is otherwise the published one. The
training run is recorded in `models/detector/train_meta.json`, which ships
inside the asset.

## 2. Models

| Model | Role | Licence | Gated | Redistributed by us |
|---|---|---|---|---|
| [`distilbert-base-cased`](https://huggingface.co/distilbert/distilbert-base-cased) | base encoder of the detector | `apache-2.0` | no | yes — fine-tuned, in the release asset |
| [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | local answering model behind the redacted prompt | `apache-2.0` | no | no — downloaded by the user |
| [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | larger local stand-in for the hosted model Airlock protects the operator from | `apache-2.0` | no | no — downloaded by the user |

Each licence is the publisher's own `cardData.license` from
`https://huggingface.co/api/models/<id>`, pinned in
`evidence/publisher_licenses.json` with the `curl` that reproduces it — not the
model README's prose, and not memory. The
model **ids** are not typed here either: the base checkpoint comes out of the
checkpoint's own `train_meta.json`, and the two answering models out of
`airlock/route.py:LOCAL_DEFAULT` / `HOSTED_STANDIN_DEFAULT`.

## 3. Dataset

**[`gretelai/synthetic_pii_finance_multilingual`](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)**
— Apache-2.0, not gated. English split only. Synthetic by construction: no real
personal data was used, held or processed anywhere in this project.

The licence is checked in both directions. `scripts/prepare_data.py` records
`"license": "apache-2.0"` into the data manifest at conversion time — but that
is a literal somebody typed, so `measure_third_party.py` fetches the
publisher's own value and compares. They **agree**. If they ever stop agreeing
the measurement says so instead of preferring ours.

The dataset card adds a qualifier that the `apache-2.0` tag alone would lose, so
the card's own `## License` section is pinned verbatim in
`evidence/publisher_licenses.json` and quoted rather than summarised:

> All data in this generated dataset is Apache 2.0 licensed and can be used for
> any purpose that is not harmful.

Training a detector whose only job is to keep identifiers out of a hosted
model's context is squarely within that. The card requests a citation, so:

```bibtex
@software{gretel-synthetic-pii-finance-multilingual-2024,
  author = {Watson, Alex and Meyer, Yev and Van Segbroeck, Maarten and
            Grossman, Matthew and Torbey, Sami and Mlocek, Piotr and Greco, Johnny},
  title  = {{Synthetic-PII-Financial-Documents-North-America}: A synthetic dataset
            for training language models to label and detect PII in domain specific formats},
  month  = {June},
  year   = {2024},
  url    = {https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual}
}
```

## 4. Python libraries

The set is the third-party top-level modules our own `.py` files import, found
by parsing the import graph — not read off `requirements.txt`, which is the
thing being checked. Each version and licence is read from the installed
distribution's own `METADATA` on this machine.

| Module | Distribution | Version | Licence as declared | Declared in | Licence text in the wheel |
|---|---|---|---|---|---|
| `torch` | torch | 2.10.0 | `BSD-3-Clause` | METADATA `License` | `LICENSE`, `NOTICE` |
| `transformers` | transformers | 5.5.4 | `Apache 2.0 License` | METADATA `License` | `LICENSE` |
| `pandas` | pandas | 2.3.3 | `BSD 3-Clause License` | first line of the full licence text in METADATA `License`; classifier `License :: OSI Approved :: BSD License` | none shipped |
| `pyarrow` | pyarrow | 21.0.0 | `Apache Software License` | METADATA `License` | `LICENSE.txt`, `NOTICE.txt` |
| `PIL` | pillow | 12.0.0 | `MIT-CMU` | METADATA `License-Expression` | `LICENSE` |

None of these is vendored — no third-party source file is copied into this
repository. They are installed from PyPI by `requirements.txt`, so their own
licence texts travel with them.

## 5. Programs used to build or run, and not redistributed

Twelve external binaries are invoked by our scripts: `bash`, `curl`, `cut`,
`dpkg`, `env`, `ffmpeg`, `git`, `grep`, `mkdir`, `python3`, `tail`, `tar`. All
are the host's own; none is shipped in this repository or in the release asset.
Two are worth naming individually:

- **ffmpeg 4.4.2** (`ffmpeg` package) renders `demo/airlock-demo.mp4`. Its
  Debian copyright file gives `LGPL-2.1+` for `Files: *`, but the flag that
  actually decides the binary's licence is the build configuration, and this
  build is **`--enable-gpl`** (`--enable-nonfree` false) — so the binary on
  this host is GPL. We do not redistribute it. The `.mp4` is the *output* of
  running it, encoded with `libx264`; running a GPL program does not place its
  output under the GPL.
- **git 2.34.1** (`GPL-2`) is read by `scripts/measure_contribution.py` to
  derive who wrote this repository. Also not redistributed.

## 6. Fonts

`scripts/render_demo_video.py` draws the demo frames with four DejaVu faces —
`DejaVuSans`, `DejaVuSans-Bold`, `DejaVuSansMono`, `DejaVuSansMono-Bold` — from
the host's `fonts-dejavu-core` package. Upstream <https://dejavu-fonts.github.io/>.
The `Files: *` stanza of its copyright file is the **Bitstream Vera** licence.

We do not redistribute the `.ttf` files. What ships is the demo video, which
contains rendered glyphs; the Bitstream Vera licence governs copies of the Font
Software, not documents set in it.

## 7. What is ours

- All Python under `airlock/`, `scripts/` and `tests/`.
- `models/gate_calibration.json` — fitted by `scripts/fit_gate.py` on documents
  held out of training.
- The two demo documents in `demo/docs/`, **written by hand** for this entry in
  the shape of the corpus rather than sampled from it; provenance in
  `demo/docs/README.md`. The card number they contain is
  `4111 1111 1111 1111`, the universally published Luhn-valid test value, and
  belongs to nobody.
- `demo/airlock-demo.mp4` and every measurement in `evidence/`.

## 8. APIs

No hosted inference API is called anywhere in this project. That is the whole
premise of Airlock — the hosted model is the threat model — so it is measured
rather than asserted: **25 vendor client packages were looked for in the import
graph** (`openai`, `anthropic`, `google`, `cohere`, `mistralai`, `boto3`,
`langchain`, `litellm`, `ollama`, `requests`, `httpx`, … the full list is in
`evidence/third_party.json`), and **none is imported**. No provider endpoint and
no credential appears in the source.

The network is nevertheless used, in exactly two places, both of them downloads
and neither of them inference. Naming them is the honest version of the claim:

| Where | What it fetches |
|---|---|
| `scripts/train_all.sh` | the corpus, from `huggingface.co` |
| `scripts/fetch_model.sh` | our own release weights, from `github.com`, checked against a pinned sha256 |

Both are shell scripts, run once, that shell out to `curl`. No Python line in
this repository can make a request: `evidence/environment.json` sweeps the
import graph for 13 stdlib network modules and finds **0**, and this file's own
producer is subject to that same sweep — which is why the licences above are
pinned rather than fetched.

Separately, `transformers.from_pretrained` downloads a checkpoint from
`huggingface.co` on first use when it is given a hub id rather than a local
path. After `scripts/fetch_model.sh`, the gate and detector run from
`models/` and need no network at all.

## 9. Recorded and not resolved

Three things this file will not round in our favour.

1. **H.264 patent licensing.** `demo/airlock-demo.mp4` is encoded with
   `libx264`. Copyright is settled — the output of a GPL encoder is ours — but
   AVC/H.264 is patent-encumbered, and distribution of encoded content is
   governed by a patent pool rather than by ffmpeg's copyright licence. We have
   not obtained or verified any such licence, which is the ordinary position of
   essentially every `.mp4` uploaded to a hackathon. Status: **UNKNOWN,
   logged**. If the organiser objects, the video re-renders to VP9 from the
   same frames with a one-line change to `scripts/render_demo_video.py`.
2. **Licence strings we did not normalise.** `transformers` and `pyarrow`
   declare their licences as free text — `Apache 2.0 License`,
   `Apache Software License` — not as SPDX identifiers. We quote what they
   declare and read the shipped licence text; we do not rewrite either into
   `Apache-2.0` on their behalf.
3. **pandas ships no licence file** inside its `.dist-info` on this host. Its
   name comes from the first line of the full licence text carried in the
   METADATA `License` field, plus the OSI classifier. That is weaker evidence
   than a shipped `LICENSE`, and is recorded as such.

---

*Generated evidence: `evidence/third_party.json`. Producer:
`scripts/measure_third_party.py`. Controls that re-derive every claim on this
page from that file, each paired with a negative control, live in the
submission checker for this entry.*
