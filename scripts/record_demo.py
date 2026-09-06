"""Record a real terminal session of Airlock running, for the demo video.

Every command in ``STEPS`` is executed here, on a pty, and every byte the
terminal received is written to ``demo/demo_cast.json`` with the offset in
seconds at which it arrived. Nothing is transcribed by hand and no output is
edited afterwards.

Two kinds of chunk are recorded and the cast says which is which:

  ``src: "recorder"``  the prompt line, written by this script so a viewer can
                       see the command. It is not shell output.
  ``src: "program"``   bytes the command actually wrote to the pty.

The renderer draws both, as a terminal would. The distinction is kept in the
cast so the recording can be audited.

    python3 scripts/record_demo.py            # writes demo/demo_cast.json
"""
from __future__ import annotations

import json
import os
import pty
import select
import subprocess
import sys
import time

COLS, ROWS = 128, 34

STEPS = [
    dict(section="the document",
         caption="A payment instruction. Name, e-mail, IBAN, phone, employee id, card number.",
         cmd="cat demo/docs/payment_instruction.txt"),
    dict(section="the gate refuses",
         caption="Detect, redact, score what survived -- then decide. Risk above budget: it stays home.",
         cmd=('python3 -m airlock.cli --file demo/docs/payment_instruction.txt '
              '--ask "Summarise the instruction and list what must be verified before paying."')),
    dict(section="the gate forwards",
         caption="A ticket with no personal identifier. Risk inside budget: it goes to the hosted model.",
         cmd=('python3 -m airlock.cli --file demo/docs/support_ticket.txt '
              '--ask "What is the customer asking for, and what should the agent check first?"')),
    dict(section="the controls",
         caption="19 checks, each paired with a control that must fail for it to mean anything.",
         cmd="python3 tests/run_checks.py"),
    dict(section="the numbers",
         caption="Every number in the README re-read from evidence/, with a control on the checker.",
         cmd="python3 tests/check_readme_numbers.py"),
]


def run_on_pty(cmd: str, env: dict, on_chunk) -> int:
    """Run ``cmd`` with its stdout/stderr on a pty; hand every chunk to on_chunk."""
    master, slave = pty.openpty()
    try:
        import fcntl
        import struct
        import termios
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    except Exception:
        pass
    proc = subprocess.Popen(["bash", "-c", cmd], stdin=subprocess.DEVNULL,
                            stdout=slave, stderr=slave, env=env, close_fds=True)
    os.close(slave)
    try:
        while True:
            r, _, _ = select.select([master], [], [], 0.05)
            if r:
                try:
                    data = os.read(master, 65536)
                except OSError:
                    break
                if not data:
                    break
                on_chunk(data)
            elif proc.poll() is not None:
                # drain anything left in the pty buffer
                while True:
                    r, _, _ = select.select([master], [], [], 0.05)
                    if not r:
                        break
                    try:
                        data = os.read(master, 65536)
                    except OSError:
                        data = b""
                    if not data:
                        break
                    on_chunk(data)
                break
    finally:
        os.close(master)
    return proc.wait()


def main(out_path="demo/demo_cast.json"):
    env = dict(os.environ)
    env.update(COLUMNS=str(COLS), LINES=str(ROWS), TERM="dumb",
               PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore",
               TRANSFORMERS_VERBOSITY="error", TOKENIZERS_PARALLELISM="false",
               HF_HUB_DISABLE_PROGRESS_BARS="1", NO_COLOR="1")
    env.pop("FORCE_COLOR", None)

    t0 = time.time()
    events, steps = [], []
    for i, step in enumerate(STEPS):
        start = time.time() - t0
        events.append(dict(t=round(start, 3), step=i, src="recorder",
                           data=f"$ {step['cmd']}\r\n"))
        rc = run_on_pty(step["cmd"], env,
                        lambda d: events.append(dict(t=round(time.time() - t0, 3), step=i,
                                                     src="program",
                                                     data=d.decode("utf-8", "replace"))))
        end = time.time() - t0
        steps.append(dict(index=i, section=step["section"], caption=step["caption"],
                          cmd=step["cmd"], returncode=rc,
                          started_at=round(start, 3), ended_at=round(end, 3)))
        print(f"[{step['section']}] rc={rc} {end - start:.1f}s", file=sys.stderr)

    payload = dict(
        version=1,
        what=("A real terminal recording. Each command in `steps` was executed on a pty on the "
              "machine named in `host`; every byte it wrote is in `events`, tagged with the "
              "step it belongs to and the second at which it arrived. Chunks with "
              "src='recorder' are the prompt lines this script "
              "printed; chunks with src='program' are the command's own output. Nothing was "
              "typed by hand and no output was edited."),
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=dict(cols=COLS, rows=ROWS, term="dumb"),
        duration=round(time.time() - t0, 3),
        steps=steps,
        events=events,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"wrote {out_path}: {len(events)} chunks, {payload['duration']:.1f}s")
    return 0 if all(s["returncode"] == 0 for s in steps) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
