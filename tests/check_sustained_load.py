"""Every figure the README quotes from the load test, re-derived and located.

Same house rule as check_readme_numbers.py: a presence check cannot fail, so
each claim is paired with a negative control that moves the measured value in
evidence/sustained_load.json, recomputes the claim, and requires the NEW string
to be absent from the README. If a corrupted number is still findable, the
check was reading nothing.

The corruptions are derived from the artifact rather than pinned to literals --
a control pinned to a literal stops corrupting the moment the measurement moves.

    python3 tests/check_sustained_load.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "evidence", "sustained_load.json")
README = os.path.join(ROOT, "README.md")


def _flat(t):
    """Collapse runs of whitespace. A claim must survive the README being
    rewrapped; it must not survive a number changing."""
    return " ".join(t.split())


def arms(d):
    by = {a["arm"]: a for a in d["arms"]}
    return by["gate_sustained"], by["forward_batch8"], by["forward_batch1"], \
        by["gate_stage_breakdown"]


def claims(d):
    """label -> the exact string the README must contain. Formatting here is the
    formatting in README.md; if they drift the check goes red, which is the
    point."""
    gs, f8, f1, sb = arms(d)
    lat, dec, ti, dd = gs["latency_seconds"], gs["decay_check"], \
        d["tie_in_control"], d["documents_per_day"]
    ms, share = sb["per_stage_milliseconds_per_document"], sb["per_stage_share"]
    eqc = sb["equivalence_control"]
    return {
        "sustained_seconds": "documents in %s s" % gs["seconds"],
        "sustained_documents": "| **%d** |" % gs["documents"],
        "sustained_rate": "**%.2f docs/s**" % gs["docs_per_second"],
        "sustained_per_day": "**{:,} / day**".format(dd["sustained_deployed_path"]),
        "latency_percentiles": "**%d / %d / %d ms**" % (
            round(lat["p50"] * 1000), round(lat["p90"] * 1000), round(lat["p99"] * 1000)),
        "decay": "%s -> %s docs/s (**no decay**, %sx)" % (
            dec["first_third_docs_per_second"], dec["last_third_docs_per_second"],
            dec["last_over_first"]),
        "peak_rss": "**{:,.1f} MiB**".format(gs["peak_rss_mib"]),
        "published_per_day": "published {:,} / day is conservative".format(
            dd["published_forward_pass_extrapolation"]),
        "over_published": "**%sx** it" % round(
            dd["sustained_deployed_path"] / dd["published_forward_pass_extrapolation"], 3),
        "batch1_rate": "**%.2f** docs/s against" % f1["docs_per_second"],
        "batch8_rate": "**%.2f** at batch 8" % f8["docs_per_second"],
        "tie_in_published": "against the published **%s**" % ti["published_docs_per_second"],
        "tie_in_ratio": "(**%sx**)" % round(ti["ratio_replayed_over_published"], 3),
        "stage_forward": "forward **%s ms**" % ms["forward"],
        "stage_decode": "decode and validators **%s ms**" % ms["decode_and_validators"],
        "stage_redact": "redact **%s ms**" % ms["redact"],
        "stage_risk": "risk **%s ms**" % ms["risk"],
        "stage_calibrate": "calibrate **%s ms**" % ms["calibrate"],
        "forward_share": "**%.3f%%** of the cost" % (share["forward"] * 100),
        "other_share": "**%.3f%%** of it" % ((1 - share["forward"]) * 100),
        "equivalence": "**%d/%d**" % (eqc["documents_compared"] - eqc["disagreements"],
                                      eqc["documents_compared"]),
    }


# ------------------------------------------------------------ corruptions
# Each moves ONE measured value, derived from the value itself, and names the
# claim label it must dislodge.
def _set(d, path, fn):
    o = d
    for k in path[:-1]:
        o = o[k] if not isinstance(k, int) else o[k]
    o[path[-1]] = fn(o[path[-1]])


def _arm_index(d, arm):
    return [i for i, a in enumerate(d["arms"]) if a["arm"] == arm][0]


def corruptions(d):
    i_gs, i_f8, i_f1, i_sb = (_arm_index(d, a) for a in
                              ("gate_sustained", "forward_batch8", "forward_batch1",
                               "gate_stage_breakdown"))
    return [
        ("sustained_rate", ["arms", i_gs, "docs_per_second"], lambda v: round(v + 0.5, 3)),
        ("sustained_documents", ["arms", i_gs, "documents"], lambda v: v + 7),
        ("sustained_seconds", ["arms", i_gs, "seconds"], lambda v: round(v + 1.0, 2)),
        ("latency_percentiles", ["arms", i_gs, "latency_seconds", "p99"],
         lambda v: round(v * 1.5, 4)),
        ("decay", ["arms", i_gs, "decay_check", "last_over_first"],
         lambda v: round(v + 0.2, 4)),
        ("peak_rss", ["arms", i_gs, "peak_rss_mib"], lambda v: round(v + 100.0, 1)),
        ("sustained_per_day", ["documents_per_day", "sustained_deployed_path"],
         lambda v: v + 1000),
        ("published_per_day", ["documents_per_day", "published_forward_pass_extrapolation"],
         lambda v: v + 1000),
        ("batch1_rate", ["arms", i_f1, "docs_per_second"], lambda v: round(v + 0.5, 3)),
        ("batch8_rate", ["arms", i_f8, "docs_per_second"], lambda v: round(v + 0.5, 3)),
        ("tie_in_published", ["tie_in_control", "published_docs_per_second"],
         lambda v: round(v + 0.5, 2)),
        ("tie_in_ratio", ["tie_in_control", "ratio_replayed_over_published"],
         lambda v: round(v - 0.2, 4)),
        ("stage_forward", ["arms", i_sb, "per_stage_milliseconds_per_document", "forward"],
         lambda v: round(v + 10.0, 3)),
        ("stage_risk", ["arms", i_sb, "per_stage_milliseconds_per_document", "risk"],
         lambda v: round(v + 1.0, 3)),
        ("forward_share", ["arms", i_sb, "per_stage_share", "forward"],
         lambda v: round(v - 0.01, 5)),
        ("equivalence", ["arms", i_sb, "equivalence_control", "disagreements"],
         lambda v: v + 3),
    ]


# ------------------------------------------------------------- structural
def structural(d):
    """(name, ok, why) triples. Each is re-checked on a corrupted copy below."""
    gs, f8, f1, sb = arms(d)
    lat, ti, dd = gs["latency_seconds"], d["tie_in_control"], d["documents_per_day"]
    eqc, share = sb["equivalence_control"], sb["per_stage_share"]
    out = []
    out.append(("tie-in control reproduced the published run",
                ti["reproduced_within_15_percent"] is True
                and abs(ti["ratio_replayed_over_published"] - 1.0) <= 0.15,
                "replayed/published = %s" % ti["ratio_replayed_over_published"]))
    out.append(("equivalence control held: the replica decides what Gate.process decides",
                eqc["held"] is True and eqc["disagreements"] == 0
                and eqc["documents_compared"] > 0,
                "%d disagreements over %d documents"
                % (eqc["disagreements"], eqc["documents_compared"])))
    out.append(("stage shares sum to 1",
                abs(sum(share.values()) - 1.0) < 1e-4,
                "sum = %r" % sum(share.values())))
    out.append(("latency percentiles are ordered",
                lat["min"] <= lat["p50"] <= lat["p90"] <= lat["p99"] <= lat["max"],
                "%r" % lat))
    out.append(("the sustained arm actually sustained",
                gs["seconds"] >= 0.95 * gs["seconds_target"]
                and gs["documents"] > gs["corpus_documents"],
                "%ss of a %ss target, %d documents over a %d-document corpus"
                % (gs["seconds"], gs["seconds_target"], gs["documents"],
                   gs["corpus_documents"])))
    out.append(("documents/day is the sustained rate, not something else",
                dd["sustained_deployed_path"] == round(gs["docs_per_second"] * 86400.0),
                "%d vs %d" % (dd["sustained_deployed_path"],
                              round(gs["docs_per_second"] * 86400.0))))
    out.append(("every arm scored the same document population",
                f8["documents"] == f1["documents"] == gs["corpus_documents"]
                == sb["documents"],
                "%d / %d / %d / %d" % (f8["documents"], f1["documents"],
                                       gs["corpus_documents"], sb["documents"])))
    return out


STRUCTURAL_CONTROLS = [
    ("tie-in control reproduced the published run",
     ["tie_in_control", "reproduced_within_15_percent"], lambda v: False),
    ("equivalence control held: the replica decides what Gate.process decides",
     None, "equivalence"),
    ("latency percentiles are ordered", None, "latency"),
    ("the sustained arm actually sustained", None, "sustain"),
    ("documents/day is the sustained rate, not something else",
     ["documents_per_day", "sustained_deployed_path"], lambda v: v + 1),
    ("stage shares sum to 1", None, "shares"),
    ("every arm scored the same document population", None, "population"),
]


def _corrupt_structural(d, kind):
    i_gs = _arm_index(d, "gate_sustained")
    i_sb = _arm_index(d, "gate_stage_breakdown")
    i_f1 = _arm_index(d, "forward_batch1")
    if kind == "equivalence":
        d["arms"][i_sb]["equivalence_control"]["disagreements"] = 1
        d["arms"][i_sb]["equivalence_control"]["held"] = False
    elif kind == "latency":
        lat = d["arms"][i_gs]["latency_seconds"]
        lat["p50"], lat["p90"] = lat["p90"], lat["p50"]
    elif kind == "sustain":
        d["arms"][i_gs]["documents"] = d["arms"][i_gs]["corpus_documents"] - 1
    elif kind == "shares":
        d["arms"][i_sb]["per_stage_share"]["forward"] += 0.05
    elif kind == "population":
        d["arms"][i_f1]["documents"] += 1
    return d


def main():
    if not os.path.exists(EV):
        print("evidence/sustained_load.json absent -- run "
              "scripts/measure_sustained_load.py")
        return 1
    d = json.load(open(EV))
    readme = _flat(open(README, encoding="utf-8").read())
    ok, bad = 0, []

    # ---- 1. every claim is on the page
    cl = claims(d)
    for label, s in cl.items():
        if _flat(s) in readme:
            ok += 1
        else:
            bad.append(("claim/%s" % label, "not in README: %r" % s))

    # ---- 2. every claim has a control that dislodges it
    for label, path, fn in corruptions(d):
        c = json.loads(json.dumps(d))
        _set(c, path, fn)
        moved = claims(c)[label]
        if moved == cl[label]:
            bad.append(("control/%s" % label,
                        "corruption did not move the claim -- the control is inert"))
        elif _flat(moved) in readme:
            bad.append(("control/%s" % label,
                        "corrupted claim %r is STILL in the README" % moved))
        else:
            ok += 1

    # ---- 3. structural facts, each re-checked on a corrupted copy
    st = {n: (o, why) for n, o, why in structural(d)}
    for n, (o, why) in st.items():
        if o:
            ok += 1
        else:
            bad.append(("structural/%s" % n, why))
    for name, path, fn in STRUCTURAL_CONTROLS:
        c = json.loads(json.dumps(d))
        if path is None:
            c = _corrupt_structural(c, fn)
        else:
            _set(c, path, fn)
        still = {n: o for n, o, _ in structural(c)}[name]
        if still:
            bad.append(("structural-control/%s" % name,
                        "check still passes on a corrupted copy -- it asserts nothing"))
        else:
            ok += 1

    total = ok + len(bad)
    for n, why in bad:
        print("FAIL %s: %s" % (n, why))
    print("%d/%d passed, %d failed" % (ok, total, len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
