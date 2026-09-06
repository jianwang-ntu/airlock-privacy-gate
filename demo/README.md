# The demo video

`airlock-demo.mp4` — 1920×1080, 1 min 46 s, silent, 1.9 MB.

It is a recording of Airlock running, not a picture of Airlock running.

## How it was made, exactly

Two scripts, both in this repository, both re-runnable:

```bash
python3 scripts/record_demo.py        # -> demo/demo_cast.json
python3 scripts/render_demo_video.py  # -> demo/airlock-demo.mp4
```

`record_demo.py` runs five commands on a pty on this machine and writes every
byte each of them returned into `demo_cast.json`, tagged with the step it
belongs to and the second at which it arrived. Nothing is typed by hand. The
only chunks the script writes itself are the `$ …` prompt lines, and the cast
marks those `src: "recorder"` so the recording can be audited. The five
commands and their exit codes are in `demo_cast.json` under `steps` — all five
returned 0.

`render_demo_video.py` draws those bytes as a terminal would, and adds the
title and number cards. Three things it does to the recording, all of them
changes to *timing* and none to content:

- gaps longer than 1.6 s — the detector and the answering model loading — are
  shortened to 1.6 s;
- when a command writes more than one screenful, each screen is held for 6 s so
  it can be read;
- the last screen of each command is held for a few seconds.

Order is preserved and no byte of output is altered, reordered or re-typed. The
renderer refuses to run if the recording contains terminal escape sequences,
because it would then be drawing something other than what the terminal showed.

The numbers on the cards are not typed into the renderer. They are read out of
`evidence/gate_eval.json`, `evidence/detector_eval.json` and
`evidence/corpus_gaps.json` at render time, so a card cannot drift away from
the measurement it quotes.

## What the run shows

| section | what you see |
|---|---|
| the document | a payment instruction carrying a name, an e-mail address, an IBAN, a phone number, an employee id and a card number |
| the gate refuses | 7 identifiers detected with their probabilities, the redacted text that *would* have left the machine, calibrated residual risk **0.151 above the 0.120 budget**, and the document answered locally instead |
| the gate forwards | a ticket with no personal identifier, risk 0.104 inside budget, answered by the hosted stand-in, re-hydrated, integrity `trustworthy=True` |
| the controls | `tests/run_checks.py` — 19 checks, each paired with a control |
| the numbers | `tests/check_readme_numbers.py` — every number in the README re-read from `evidence/` |

The refusing run ends on a warning rather than an answer: the local model did
not return two of the six placeholders, so Airlock reports
`trustworthy=False` and says the re-hydrated text must not be treated as
faithful. That is the system behaving as designed and it was left in.

## The documents

`docs/payment_instruction.txt` and `docs/support_ticket.txt` are byte-identical
to the two documents `scripts/demo_run.py` uses. Both are synthetic and written
by hand; no real personal data appears anywhere in this project. See
`docs/README.md`.
