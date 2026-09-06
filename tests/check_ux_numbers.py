#!/usr/bin/env python3
"""Every number the "Using it" section quotes, re-read from evidence/ux.json.

Same job as check_readme_numbers.py and the same reason: a write-up drifts from
its measurement silently, and the drift is always flattering. Each claim here is
re-derived from the evidence file rather than copied from the prose, then looked
for in README.md. A claim that cannot be found is a mismatch.

Every assertion is paired with a negative control -- the same derivation re-run
against a deliberately corrupted copy of the evidence -- and the corrupted run
MUST lose at least one claim. A checker that passes on corrupt input is void and
is reported as broken rather than as green.

Run:  python3 tests/check_ux_numbers.py     (stdlib only)
Exit: 0 all pass, 1 any fail.
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_CLAIMS = 31


def load():
    with open(os.path.join(ROOT, "evidence", "ux.json"), encoding="utf-8") as f:
        return json.load(f)


def claims(ux):
    """(name, substring-that-must-appear-in-the-README) -- derived, never typed.

    Anything a reader could check against the tool is here. Prose that makes no
    numeric claim is deliberately absent: this file measures drift, not style.
    """
    cs, op, st = ux["cold_start"], ux["operator_path"], ux["streams"]
    t = ux["throughput_warm"]
    ep, sf, sk = (ux["error_paths"], ux["silent_fallback_calibration"],
                  ux["skipped_fetch_model"])
    b, a = ep["before"], ep["after"]
    out = [
        ("path.commands",
         f"**{op['commands_clean_clone_to_first_decision']} commands and "
         f"{op['flags_required_for_first_decision']} flag from a clean clone"),
        ("cold.median", f"median of **{cs['median']} s**"),
        ("cold.runs", f"{cs['runs']} runs ({cs['min']}–{cs['max']})"),
        ("thr.small", f"**{t[0]['docs_per_second']} documents/second** at "
                      f"{t[0]['characters']} characters"),
        ("thr.mid", f"**{t[1]['docs_per_second']}/s** at {t[1]['characters']:,}"),
        ("thr.large", f"**{t[2]['docs_per_second']}/s** at\n{t[2]['characters']:,}"),
        ("streams.stdout_lines", f"{st['decision_lines_on_stdout']} lines of decision"),
        ("streams.json_keys", f"carries {len(st['json_keys'])} keys"),
        ("err.baseline_ref", f"At `{ux['baseline_ref'][:7]}`"),
        ("err.before_total",
         f"**{ep['properties_held_before']} of {ep['properties_total']}**"),
        ("err.after_total",
         f"**{ep['properties_held_after']} of {ep['properties_total']}**"),
        ("err.no_source",
         f"exit {b['no_source']['returncode']}, {b['no_source']['stderr_lines']} lines"),
        ("err.two_sources",
         f"exit {b['two_sources']['returncode']}, {b['two_sources']['stderr_lines']} lines"),
        ("err.missing_file.before",
         f"exit {b['missing_file']['returncode']}, "
         f"**{b['missing_file']['stderr_lines']}-line traceback**"),
        ("err.missing_file.after",
         f"exit {a['missing_file']['returncode']}, "
         f"**{a['missing_file']['stderr_lines']} line** naming the path"),
        ("err.bad_model.before",
         f"exit {b['bad_model_dir']['returncode']}, "
         f"**{b['bad_model_dir']['stderr_lines']}-line traceback**"),
        ("err.bad_model.after",
         f"exit {a['bad_model_dir']['returncode']}, "
         f"**{a['bad_model_dir']['stderr_lines']} lines** naming `fetch_model.sh`"),
        ("skip.before_lines",
         f"printed {sk['before']['stderr_lines']} lines of Hugging Face traceback"),
        ("skip.after",
         f"exits {sk['after']['returncode']} in {sk['after']['stderr_lines']} lines and names it"),
        ("cal.before_rc",
         f"risk`, exit {sf['before']['default_path_absent']['returncode']}, no warning"),
        ("cal.doc_chars", f"the same {sf['after']['document_characters']}-character document"),
        ("cal.calibrated", f"calibrated risk is **{sf['after']['with_calibration']['risk']}**"),
        ("cal.naive",
         f"the naive one is **{sf['before']['default_path_absent']['risk']}**"),
        ("cal.route", f"`{sf['after']['with_calibration']['route']}` either way"),
        ("cal.explicit_rc",
         f"it exits {sf['after']['explicit_missing_path']['returncode']} instead of quietly"),
        ("weak.large_chars", f"**{t[2]['characters']:,} characters take {t[2]['seconds']} s**"
         .replace("**", "")),
        ("weak.cold", f"**Cold start is {cs['median']} s"),
    ]
    # the four options that carry defaults, each named
    for opt in op["options_with_defaults"]:
        out.append((f"path.default{opt}", f"`{opt}`"))
    return out


def corruptions(ux):
    """Each returns a copy with ONE measured value moved. A corrupted copy that
    loses no claim means the claim it should have moved is not being checked."""
    def bump(path, delta):
        d = copy.deepcopy(ux)
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = node[path[-1]] + delta
        return d

    return [
        ("cold start median +1 s", bump(["cold_start", "median"], 1.0)),
        ("small-doc throughput +10/s",
         bump(["throughput_warm", 0, "docs_per_second"], 10.0)),
        ("large-doc size +1000 chars",
         bump(["throughput_warm", 2, "characters"], 1000)),
        ("stdout decision lines +1", bump(["streams", "decision_lines_on_stdout"], 1)),
        ("error properties held after -1",
         bump(["error_paths", "properties_held_after"], -1)),
        ("bad --model traceback length -100",
         bump(["error_paths", "before", "bad_model_dir", "stderr_lines"], -100)),
        ("skipped-fetch stderr lines -100",
         bump(["skipped_fetch_model", "before", "stderr_lines"], -100)),
        ("calibrated risk +0.01",
         bump(["silent_fallback_calibration", "after", "with_calibration", "risk"], 0.01)),
        ("naive risk +0.01",
         bump(["silent_fallback_calibration", "before", "default_path_absent",
               "risk"], 0.01)),
        ("document length +1 char",
         bump(["silent_fallback_calibration", "after", "document_characters"], 1)),
    ]


def main():
    ux = load()
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()

    if "## Using it" not in readme:
        print("FAIL  README.md has no '## Using it' section")
        return 1

    # The README's table says these two paths are fixed. That is a claim about
    # the MEASUREMENT, not about the prose, so it is asserted directly rather
    # than by looking for a string -- a claim checked by the absence of a string
    # passes when the string was never going to be there.
    unfixed = [scen for scen in ("missing_file", "bad_model_dir")
               if ux["error_paths"]["after"][scen]["properties_held"] != 4]
    for scen in ("missing_file", "bad_model_dir"):
        held = ux["error_paths"]["after"][scen]["properties_held"]
        print(f"{'PASS' if held == 4 else 'FAIL'}  fixed.{scen}: {held}/4 properties hold")
    if unfixed:
        print(f"FAIL  README claims these paths are fixed and they are not: {unfixed}")
        return 1

    cl = claims(ux)
    missing = [(n, s) for n, s in cl if s and s not in readme]
    for n, s in cl:
        if s and s in readme:
            print(f"PASS  {n}")
    for n, s in missing:
        print(f"FAIL  {n}: {s!r} not in README.md")

    print(f"\n{len(cl) - len(missing)}/{len(cl)} claims re-derived and located")
    if len(cl) < MIN_CLAIMS:
        print(f"FAIL  only {len(cl)} claims checked, floor is {MIN_CLAIMS} -- a "
              f"shrinking checker is how a claim stops being checked")
        return 1

    # ---- negative controls
    void = []
    for label, bad in corruptions(ux):
        lost = [n for n, s in claims(bad) if s and s not in readme]
        base = {n for n, _ in missing}
        newly = [n for n in lost if n not in base]
        if newly:
            print(f"control: {label} -> {len(newly)} claim(s) go missing "
                  f"({', '.join(newly[:3])})")
        else:
            print(f"BROKEN control: {label} -> nothing moved; that value is not checked")
            void.append(label)

    if missing or void:
        print(f"\nFAILED: {len(missing)} mismatch(es), {len(void)} void control(s)")
        return 1
    print(f"\nOK: {len(cl)} claims, {len(corruptions(ux))} negative controls, "
          f"0 mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
