"""Render demo/demo_cast.json into the demo video.

The terminal frames are drawn from the recorded bytes and nothing else. The
number cards are read out of evidence/*.json at render time, so a card cannot
drift away from the measurement it quotes.

What the renderer does to the recording, stated because it is a change:

  * gaps longer than MAX_GAP seconds (model loading) are shortened to MAX_GAP;
  * when a command writes more than one screenful, each screen is held for
    PAGE_DWELL seconds so it can be read;
  * a HOLD second pause is added after each command.

Order is preserved and no byte of output is altered, reordered or re-typed.
Refuses to run if the recording contains terminal escape sequences, because the
renderer would then be drawing something other than what the terminal showed.

    python3 scripts/render_demo_video.py            # -> demo/airlock-demo.mp4
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
COLS, PAGE_ROWS = 128, 36
MAX_GAP, PAGE_DWELL, HOLD = 1.6, 6.0, 3.2
# seconds the final screen of a step is held, when it needs longer than HOLD to read
HOLDS = {"the document": 6.0, "the gate forwards": 5.0, "the numbers": 4.5}

BG = (11, 16, 32)
PANEL = (13, 17, 23)
BORDER = (37, 45, 63)
FG = (215, 221, 232)
DIM = (129, 140, 163)
ACCENT = (122, 162, 247)
GREEN = (110, 231, 168)
AMBER = (240, 190, 100)
RED = (240, 130, 130)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TERM_SIZE, LINE_H = 19, 23
_f = {}


def font(path, size):
    key = (path, size)
    if key not in _f:
        _f[key] = ImageFont.truetype(path, size)
    return _f[key]


# ---------------------------------------------------------------- text layout

def wrap_rows(text, cols=COLS):
    """The rows a `cols`-wide terminal would show for this text."""
    out = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(8).split("\n"):
        if not line:
            out.append("")
            continue
        while len(line) > cols:
            out.append(line[:cols])
            line = line[cols:]
        out.append(line)
    return out


def para(draw, text, x, y, width, f, fill, leading=8):
    """Draw wrapped proportional text; return the y below it."""
    words, line = text.split(), ""
    for w in words:
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=f) > width and line:
            draw.text((x, y), line, font=f, fill=fill)
            y += f.size + leading
            line = w
        else:
            line = trial
    if line:
        draw.text((x, y), line, font=f, fill=fill)
        y += f.size + leading
    return y


# ------------------------------------------------------------------- chrome

def chrome(img, caption_section=None, caption_text=None, footer=None):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 84], fill=(16, 22, 42))
    d.line([(0, 84), (W, 84)], fill=BORDER, width=2)
    d.text((60, 24), "AIRLOCK", font=font(SANS_B, 34), fill=FG)
    d.text((240, 33), "a calibrated privacy gate in front of hosted AI",
           font=font(SANS, 24), fill=DIM)
    right = "UPAI-Hackdays 2026  ·  solo entry"
    d.text((W - 60 - d.textlength(right, font=font(SANS, 21)), 34), right,
           font=font(SANS, 21), fill=DIM)
    if caption_section:
        d.text((60, 104), caption_section.upper(), font=font(SANS_B, 22), fill=ACCENT)
        d.text((60 + d.textlength(caption_section.upper(), font=font(SANS_B, 22)) + 26, 104),
               caption_text or "", font=font(SANS, 22), fill=FG)
    if footer:
        d.text((60, H - 40), footer, font=font(SANS, 19), fill=DIM)
    return d


def terminal_frame(rows, section, caption, footer, highlight=()):
    img = Image.new("RGB", (W, H), BG)
    d = chrome(img, section, caption, footer)
    inner = int(COLS * font(MONO, TERM_SIZE).getlength("M"))
    w, h = inner + 80, PAGE_ROWS * LINE_H + 32
    px, py = (W - w) // 2, 150
    d.rectangle([px, py, px + w, py + h], fill=PANEL, outline=BORDER, width=2)
    x0, y0 = px + 40, py + 16
    f = font(MONO, TERM_SIZE)
    fb = font(MONO_B, TERM_SIZE)
    for i, row in enumerate(rows[:PAGE_ROWS]):
        y = y0 + i * LINE_H
        if row.startswith("$ "):
            d.text((x0, y), row, font=fb, fill=GREEN)
        elif any(k in row for k in highlight):
            d.text((x0, y), row, font=fb, fill=AMBER)
        elif row.startswith("PASS"):
            d.text((x0, y), row[:4], font=fb, fill=GREEN)
            d.text((x0 + f.getlength("PASS"), y), row[4:], font=f, fill=FG)
        else:
            d.text((x0, y), row, font=f, fill=FG)
    return img


# --------------------------------------------------------------------- cards

def card(title, lines, kicker=None, footer=None, body_font=30):
    img = Image.new("RGB", (W, H), BG)
    d = chrome(img, footer=footer)
    y = 210
    if kicker:
        d.text((160, y), kicker.upper(), font=font(SANS_B, 24), fill=ACCENT)
        y += 48
    d.text((160, y), title, font=font(SANS_B, 54), fill=FG)
    y += 104
    for line, colour in lines:
        y = para(d, line, 160, y, W - 320, font(SANS, body_font), colour, leading=12) + 22
    return img


def table_card(title, kicker, headers, rows, note, footer=None):
    img = Image.new("RGB", (W, H), BG)
    d = chrome(img, footer=footer)
    d.text((160, 190), kicker.upper(), font=font(SANS_B, 24), fill=ACCENT)
    d.text((160, 238), title, font=font(SANS_B, 50), fill=FG)
    f, fb = font(MONO, 27), font(MONO_B, 27)
    xs = [160, 900, 1330]
    y = 360
    for i, htxt in enumerate(headers):
        d.text((xs[i], y), htxt, font=fb, fill=DIM)
    y += 46
    d.line([(160, y), (W - 160, y)], fill=BORDER, width=2)
    y += 22
    for cells, colour, bold in rows:
        ft = fb if bold else f
        for i, c in enumerate(cells):
            d.text((xs[i], y), c, font=ft, fill=colour)
        y += 48
    y += 24
    para(d, note, 160, y, W - 320, font(SANS, 26), DIM, leading=10)
    return img


# ------------------------------------------------------------------ schedule

def build(cast, ev):
    """[(PIL image, seconds)] for the whole video."""
    repo = "github.com/jianwang-ntu/airlock-privacy-gate"
    foot_run = f"real run recorded {cast['recorded_at']}  ·  {repo}"
    seq = []

    seq.append((card(
        "It tells you what it might have missed.",
        [("Redaction tools report what they found. Airlock reports the probability that an "
          "identifier survived anyway — calibrated against measured outcomes — and refuses to "
          "send the document when that number is above the operator's budget.", FG),
         ("Those documents are answered by a small model on the operator's own machine instead.", DIM)],
        kicker="Airlock", footer=repo), 6.5))

    leak = ev["detector"]["shipped"]["doc_leak_rate"]
    seq.append((card(
        "The gap is that redaction is silent about its own failure.",
        [(f"Our own detector, at its shipped operating point, still leaves an annotated "
          f"identifier in {leak * 100:.2f}% of the documents that contain one.", AMBER),
         ("A pipeline that redacts and forwards regardless leaks better than one document in "
          "eight and never says so. The person accountable for the decision is handed a masked "
          "document and no statement of residual risk.", FG)],
        kicker="the problem", footer=repo), 9.0))

    # ---- the recorded run
    for step in cast["steps"]:
        chunks = [e for e in cast["events"] if e["step"] == step["index"]]
        rows, shown_pages, buf = [], 0, ""
        hl = ("route  ", "residual leak risk", "WARNING", "19/19", "27/27", "placeholders expected")
        for i, e in enumerate(chunks):
            buf += e["data"]
            rows = wrap_rows(buf)
            page = max(0, (len(rows) - 1) // PAGE_ROWS)
            nxt = chunks[i + 1]["t"] if i + 1 < len(chunks) else step["ended_at"]
            dwell = min(max(nxt - e["t"], 0.0), MAX_GAP)
            while shown_pages < page:   # a page just filled up: show it whole before moving on
                seq.append((terminal_frame(rows[shown_pages * PAGE_ROWS:(shown_pages + 1) * PAGE_ROWS],
                                           step["section"], step["caption"], foot_run, hl), PAGE_DWELL))
                shown_pages += 1
            seq.append((terminal_frame(rows[page * PAGE_ROWS:(page + 1) * PAGE_ROWS],
                                       step["section"], step["caption"], foot_run, hl),
                        max(dwell, 0.35)))
        # hold the last screen of the step
        page = max(0, (len(rows) - 1) // PAGE_ROWS)
        seq.append((terminal_frame(rows[page * PAGE_ROWS:(page + 1) * PAGE_ROWS],
                                   step["section"], step["caption"], foot_run, hl),
                    HOLDS.get(step["section"], HOLD)))

    # ---- what was measured
    g = ev["gate"]
    seq.append((table_card(
        "Withholding the same documents at random does nothing.",
        f"{g['documents']:,} held-out documents",
        ["policy", "forwarded", "leak rate"],
        [(["forward everything", "100%", f"{g['forward_all']*100:.2f}%"], FG, False),
         ([f"Airlock, budget {g['b']['budget']}", f"{g['b']['coverage']*100:.2f}%",
           f"{g['b']['leak']*100:.2f}%"], GREEN, True),
         (["random gate, same coverage", f"{g['b']['coverage']*100:.2f}%",
           f"{g['b']['rand']*100:.2f}%"], DIM, False),
         ([f"Airlock, budget {g['t']['budget']}", f"{g['t']['coverage']*100:.2f}%",
           f"{g['t']['leak']*100:.2f}%"], GREEN, True),
         (["random gate, same coverage", f"{g['t']['coverage']*100:.2f}%",
           f"{g['t']['rand']*100:.2f}%"], DIM, False)],
        f"The random row is the control that carries the weight: it holds back the same NUMBER "
        f"of documents and the leak rate stays at the base rate. Ranking AUC {g['auc']}. "
        f"Expected calibration error {g['ece']}. The one-sided upper bound was conservative in "
        f"{g['conservative']} reliability deciles.",
        footer=repo), 13.0))

    d_ = ev["detector"]
    seq.append((table_card(
        "The learned model is the part that works.",
        f"identifier recall, same {d_['documents']:,} documents",
        ["detector", "recall", "docs leaking"],
        [(["Airlock (model ∪ validators)", f"{d_['shipped']['recall']*100:.2f}%",
           f"{d_['shipped']['doc_leak_rate']*100:.2f}%"], GREEN, True),
         (["model only", f"{d_['model_only']['recall']*100:.2f}%",
           f"{d_['model_only']['doc_leak_rate']*100:.2f}%"], FG, False),
         (["regex + capitalised-bigram control", f"{d_['regex']['recall']*100:.2f}%",
           f"{d_['regex']['doc_leak_rate']*100:.2f}%"], RED, False),
         (["checksum validators only", f"{d_['validators']['recall']*100:.2f}%",
           f"{d_['validators']['doc_leak_rate']*100:.2f}%"], RED, False)],
        f"Remove the fine-tuned transformer and the system collapses to the regex control. The "
        f"validator layer moves the corpus headline by less than a tenth of a point; it is in "
        f"the system because the model alone left a valid, universally published test card in "
        f"the outgoing text and a control caught it. {d_['params'] / 1e6:.1f}M parameters, "
        f"fine-tuned here from {d_['base']} in {d_['train_seconds']} s on one "
        f"{d_['device'].split()[-1]}, {d_['docs_per_second']} documents per second, no "
        f"per-document API cost and no third-party dependency in the detection path.",
        footer=repo), 13.0))

    seq.append((card(
        "What this is not.",
        [(f"The corpus's own annotation is incomplete — the checksum validators found "
          f"{ev['corpus']['unmarked_emails']} e-mail addresses in {ev['corpus']['sampled']} test "
          f"documents that the annotation does not mark. So recall here is an upper bound and "
          f"over-redaction an upper bound. Both are reported in the direction that is against "
          f"the system.", FG),
         (f"The risk statistic grows with document length; AUC {g['auc']} is real discrimination, "
          "not a solved problem. English financial and support text only, all of it synthetic. "
          "No third-party API is called anywhere in the repository: the forwarded leg was "
          "exercised against a larger model on the same machine.", DIM)],
        kicker="stated first-hand", footer=repo, body_font=28), 12.0))

    seq.append((card(
        repo,
        [("Everything in this video was produced by code in that repository. The terminal "
          "frames are a recording of a real run; the numbers on the cards are read out of "
          "evidence/ when the video is rendered.", FG),
         ("MIT licensed. Built for UPAI-Hackdays, September 2026. Solo entry.", DIM)],
        kicker="Airlock", footer=repo), 7.0))
    return seq


# ------------------------------------------------------------------ evidence

def load_evidence():
    g = json.load(open("evidence/gate_eval.json"))
    d = json.load(open("evidence/detector_eval.json"))
    c = json.load(open("evidence/corpus_gaps.json"))
    thr = str(g["redaction_threshold"])

    def rc(budget):
        for r in g["risk_coverage"]:
            if abs(r["budget"] - budget) < 1e-9:
                return dict(budget=budget, coverage=r["coverage"],
                            leak=r["observed_leak_rate_forwarded"],
                            rand=r["random_gate_same_coverage_leak_rate_mean"])
        raise SystemExit(f"no risk_coverage row for budget {budget}")

    def det(block, key=thr):
        e = d[block][key] if key else d[block]
        return dict(recall=e["identifier_recall"], doc_leak_rate=e["doc_leak_rate"])

    return dict(
        gate=dict(forward_all=g["forward_everything_leak_rate"], b=rc(0.12), t=rc(0.045),
                  auc=g["ranking_auc_raw_statistic"], ece=g["ece_posterior_mean"],
                  conservative=g["reliability_bins_where_upper_bound_is_conservative"],
                  documents=g["documents"]),
        detector=dict(shipped=det("by_threshold_hybrid"), model_only=det("by_threshold_model_only"),
                      regex=det("regex_baseline", None), validators=det("checksum_validators_only", None),
                      params=d["model"]["parameters"], base=d["model"]["base_model"],
                      train_seconds=d["model"]["wall_seconds"], device=d["model"]["device"],
                      docs_per_second=d["throughput"]["model_docs_per_second"],
                      documents=d["documents"]),
        corpus=dict(unmarked_emails=c["validator_hits_outside_any_gold_span"]["email"],
                    sampled=c["sampled_documents"]),
    )


def main(cast_path="demo/demo_cast.json", out="demo/airlock-demo.mp4"):
    cast = json.load(open(cast_path, encoding="utf-8"))
    if any("\x1b" in e["data"] for e in cast["events"]):
        raise SystemExit("recording contains escape sequences; this renderer would misdraw it")
    seq = build(cast, load_evidence())
    tmp = tempfile.mkdtemp(prefix="airlock-video-")
    try:
        listing = os.path.join(tmp, "frames.txt")
        with open(listing, "w") as fh:
            for i, (img, secs) in enumerate(seq):
                p = os.path.join(tmp, f"f{i:04d}.png")
                img.save(p)
                fh.write(f"file '{p}'\nduration {secs:.3f}\n")
            fh.write(f"file '{p}'\n")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        total = sum(secs for _, secs in seq)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", listing, "-t", f"{total:.3f}", "-r", "30", "-vsync", "cfr", "-c:v", "libx264",
               "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", out]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out}: {len(seq)} frames, {sum(x for _, x in seq):.1f}s, "
          f"{os.path.getsize(out) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
