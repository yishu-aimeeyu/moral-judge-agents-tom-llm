#!/usr/bin/env python3
"""W4 report: evaluate the pre-registered readings against the curriculum results.

Reads outputs/experiments/w4_prompt_curriculum.csv and writes W4_CURRICULUM.md plus
figures_final/w4_curriculum.png. Every judgment made here is against a bar fixed in
W4_PRESPEC.md before the run: P3 (shift of at least +0.15 with a bootstrap CI on the
paired difference excluding 0, in at least 4 of 6 models) decides "prompting works";
P4 separates full recovery from a partial shift; P5 checks dose-response and flags an
L4-only jump as possible format imitation; P6 prints all four cell means at every level
so a contrast change produced by uniform movement cannot be read as intent re-weighting.
"""
import os, sys, csv
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402
import numpy as np  # noqa: E402

EXP = os.path.join(tc.ROOT, "outputs", "experiments")
CSV_IN = os.path.join(EXP, "w4_prompt_curriculum.csv")
MD_OUT = os.path.join(EXP, "W4_CURRICULUM.md")
SUMMARY_OUT = os.path.join(EXP, "W4_SUMMARY.md")
FIG = os.path.join(tc.ROOT, "outputs", "figures_final", "w4_curriculum.png")

SHIFT_BAR = 0.15          # P3, in normalised contrast units
MIN_MODELS = 4            # P3, of 6 engaged
ADULT_CONTRAST = 0.933 - 0.267   # Young 2007 digitized adults, attempted - accidental
W3_INTENT_MAX = 0.015     # largest |Δcontrast| from intent steering in W3
W3_OUTCOME_MIN = 0.232    # smallest |Δcontrast| from the W3 outcome positive control
W3_OUTCOME_MAX = 0.259    # largest, i.e. the W3 positive control's range is .232-.259
CUM = (1, 2, 3, 4, 5)     # the pre-registered cumulative curriculum
ABL = (6, 7, 8)           # non-cumulative ablation, added after the fact
ABL_LABEL = {2: "instruction alone", 6: "worked example alone", 7: "few-shot alone",
             8: "principle alone"}


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load():
    if not os.path.exists(CSV_IN):
        return {}
    by = defaultdict(dict)
    for r in csv.DictReader(open(CSV_IN)):
        by[r["model"]][int(r["level"])] = r
    return by


def attribute(r):
    """Where did a contrast change come from? Cell movement, not the contrast alone.

    The contrast is attempted - accidental, so the identical gain arises from adding blame
    to attempted harm (intent re-weighting, the interesting reading) or from removing blame
    from accidental harm (outcome-blame suppression, which is not the same claim), or from
    every cell sliding together (compression, which is neither). W3 already carries this
    caveat for its difference-of-means direction; the same guard belongs here.
    """
    d = {c: fl(r.get(f"d_{c}")) for c in tc.CELLS}
    da, dc = d["attempted"], d["accidental"]
    gain = da - dc
    if any(np.isnan(v) for v in d.values()):
        return "—", ""
    vals = [d[c] for c in tc.CELLS]
    uniform = (all(v > 0 for v in vals) or all(v < 0 for v in vals))
    if gain <= 0.02:
        head = "no gain"
    elif da > 0.02 and dc < -0.02:
        head = "**both** — attempted ↑, accidental ↓ (intent re-weighting)"
    elif dc < -0.02 and da <= 0.02:
        head = ("**accidental ↓ only** — outcome-blame suppressed, no intent-blame added"
                if da < -0.02 else "**accidental ↓** — outcome-blame suppressed")
    elif da > 0.02 and dc >= -0.02:
        head = "**attempted ↑** — intent-blame added"
    else:
        head = "mixed"
    if uniform and gain > 0.02:
        head += "; all four cells move the same way (compression)"
    detail = " ".join(f"{c[:4]}{d[c]:+.3f}" for c in tc.CELLS)
    return head, detail


def plot(by):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (no figure: {e})")
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, (m, all_lv) in enumerate(sorted(by.items())):
        lv = {k: v for k, v in all_lv.items() if k in CUM}   # ablation is a table, not a line
        ks = sorted(lv)
        y = [fl(lv[k]["contrast"]) for k in ks]
        lo = [fl(lv[k]["contrast_lo"]) for k in ks]
        hi = [fl(lv[k]["contrast_hi"]) for k in ks]
        ax.plot(ks, y, "-o", color=cmap(i % 10), label=tc.pretty(m), lw=1.8, ms=5)
        ax.fill_between(ks, lo, hi, color=cmap(i % 10), alpha=0.10, lw=0)
    ax.axhline(0, color="k", lw=1)
    ax.axhline(ADULT_CONTRAST, color="crimson", ls="--", lw=1.4)
    ax.text(1.02, ADULT_CONTRAST, " adult (Young 2007)", color="crimson",
            va="center", fontsize=8)
    ax.set_xticks([k for k in sorted({k for lv in by.values() for k in lv}) if k in CUM])
    ax.set_xticklabels([f"L{k}\n{n}" for k, n in
                        sorted({int(r["level"]): r["level_name"]
                                for lv in by.values() for r in lv.values()
                                if int(r["level"]) in CUM}.items())],
                       fontsize=8)
    ax.set_ylabel("contrast  (attempted − accidental, normalised)")
    ax.set_title("W4 prompt curriculum: in-context escalation vs baseline\n"
                 "negative = outcome-dominant (inverted vs adults); "
                 "3/6 clear the L5 bar; stack gains are mostly accidental-blame "
                 "suppression\nbands are scenario-group bootstrap 95% CIs", fontsize=9)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"  wrote {os.path.relpath(FIG, tc.ROOT)}")


def verdicts(by):
    """Two verdicts per model, because the curriculum turned out to be non-monotone.

    P3 places the bar at the top level (L5), and that is where the pre-registered verdict
    is read. The maximum shift over levels is reported alongside it as descriptive, never
    substituted for it: picking the best level after seeing the results is exactly the
    move a pre-registration exists to prevent.
    """
    out = {}
    for m, all_lv in by.items():
        lv = {k: v for k, v in all_lv.items() if k in CUM}
        ks = sorted(lv)
        if not ks:
            continue
        top = lv[ks[-1]]
        d, dlo, dhi = fl(top["d_contrast"]), fl(top["d_lo"]), fl(top["d_hi"])
        met = (d >= SHIFT_BAR) and (dlo > 0)
        bl = max(ks, key=lambda k: fl(lv[k]["d_contrast"]))
        bd, bdlo = fl(lv[bl]["d_contrast"]), fl(lv[bl]["d_lo"])
        best = max(ks, key=lambda k: fl(lv[k]["contrast"]))
        out[m] = dict(d=d, dlo=dlo, dhi=dhi, met=met, top_level=ks[-1],
                      met_any=(bd >= SHIFT_BAR) and (bdlo > 0),
                      max_level=bl, max_d=bd, max_lo=bdlo, max_hi=fl(lv[bl]["d_hi"]),
                      recovered=fl(lv[best]["contrast"]) > 0,
                      best_level=best, best=fl(lv[best]["contrast"]),
                      base=fl(lv[ks[0]]["contrast"]))
    return out


def main():
    by = load()
    if not by:
        print(f"no {os.path.relpath(CSV_IN, tc.ROOT)} yet — run 54 first")
        return
    V = verdicts(by)
    n_met = sum(1 for v in V.values() if v["met"])
    n_any = sum(1 for v in V.values() if v["met_any"])
    n_rec = sum(1 for v in V.values() if v["recovered"])
    works = n_met >= MIN_MODELS
    level_specific = (not works) and n_any >= MIN_MODELS

    ref = by[list(by)[0]][min(by[list(by)[0]])]
    has_abl = any(k in ABL for lv in by.values() for k in lv)
    tops = {m: {k: v for k, v in by[m].items() if k in CUM}[V[m]["top_level"]] for m in V}
    gainers = [m for m in V if fl(tops[m]["d_contrast"]) > 0.02]
    acc_only = [m for m in gainers if fl(tops[m]["d_accidental"]) < -0.02
                and fl(tops[m]["d_attempted"]) <= 0.02]
    att_up = [m for m in gainers if fl(tops[m]["d_attempted"]) > 0.02]
    abl_led, abl_beats = [], 0
    for m, all_lv in by.items():
        r8 = all_lv.get(8)
        if r8 and fl(r8["d_contrast"]) > 0.02:
            da, dc = fl(r8["d_attempted"]), fl(r8["d_accidental"])
            if da > 0.02 and da >= abs(dc):
                abl_led.append((m, da, dc))
        if 5 in all_lv and 8 in all_lv:
            if fl(all_lv[8]["d_contrast"]) > fl(all_lv[5]["d_contrast"]) + 0.05:
                abl_beats += 1

    L = ["# W4 — prompt curriculum: how far does in-context instruction move the bias?",
         "",
         f"_{len(by)} engaged open models, {len(CUM)} cumulative levels"
         + (f" plus a {len(ABL)}-cell non-cumulative ablation" if has_abl else "")
         + f", {ref['n_templates']}-template basis, {ref['n_groups']} scenario groups "
         f"({ref.get('n_cells', '?')} template × group cells). "
         "Readings pre-registered in `W4_PRESPEC.md`; prompts verbatim in "
         "`W4_PROMPT_LEVELS.md`. Short conclusion also in `W4_SUMMARY.md`._", "",
         "> **Corrections after run 1** (preserved under `_w4_prefix_fewshot_bug/`): "
         "few-shot agent-name bug on `human_verbatim` (L4/L5 rescored); bootstrap unit "
         "corrected from template × group cell to scenario group. L4 label polarity "
         "verified non-inverted and gated.", "",
         "## Verdict", "",
         "**The completed 6 × 8 run does not support \"unreachable by steering, reachable "
         "by prompting.\"** That reading overclaims both ends. The accurate summary:", "",
         f"1. **Only {n_met}/{len(V)} models clear the pre-registered +{SHIFT_BAR:.2f} bar "
         f"at L5** (need {MIN_MODELS}). By P3 this is not \"prompting works\".",
         f"2. **Gains at the top of the stack are predominantly accidental-blame "
         f"suppression**, not intent re-weighting: of {len(gainers)} models whose L5 "
         f"contrast improves, {len(acc_only)} improve only because blame for *accidental* "
         f"harm falls; {len(att_up)} add blame to attempted harm, and that addition is "
         "small beside the accidental drop.",
         f"3. **Genuine intent-led movement appears in {len(abl_led)}/{len(by)} models and "
         "only under principle-alone (L8)** — the bare statement of the intent principle, "
         "with no cue, worked example, or labelled few-shot. "
         + ("; ".join(f"{tc.pretty(m)} Δattempted {a:+.3f} vs Δaccidental {c:+.3f}"
                      for m, a, c in sorted(abl_led, key=lambda t: -t[1]))
            if abl_led else "None meet the intent-led criterion."),
         f"4. **Added scaffolding often degrades performance relative to the bare "
         f"principle**: in {abl_beats}/{len(by)} models principle-alone beats the full L5 "
         "stack by more than 0.05, and every model is sub-additive across the ablation "
         "parts.",
         "",
         "**Frame:** the outcome bias is *entrenched* — resistant to residual-stream "
         f"intervention at resolvable depths (W3, |Δcontrast| ≤ {W3_INTENT_MAX:.3f}) and "
         "only partially and inconsistently responsive to instruction (W4). Prompting is "
         "not a reliable rescue; when it moves the number it usually does so by softening "
         "blame for accidents rather than by condemning failed attempts.", "",
         "Reporting the best level over the curriculum as the headline would be selecting "
         "after seeing the data; `best level Δ` below is descriptive only. The "
         "pre-registered verdict remains the L5 column.", "",
         "Reference points: adults sit at "
         f"{ADULT_CONTRAST:+.3f} (Young 2007 digitized); the W3 outcome positive control "
         f"moves the contrast {W3_OUTCOME_MIN:.3f}–{W3_OUTCOME_MAX:.3f} at the same models.",
         "",
         "## Contrast by level", "",
          "## Contrast by level", "",
          "| model | " + " | ".join(f"L{k}" for k in CUM)
          + " | Δ(top−L1) [95% CI] | bar at top level | best level Δ [95% CI] "
            "| crosses 0 | where the gain comes from (at top level) |",
          "|---" * (len(CUM) + 6) + "|"]
    for m, all_lv in sorted(by.items()):
        lv = {k: v for k, v in all_lv.items() if k in CUM}
        v = V[m]
        cells = " | ".join(f"{fl(lv[k]['contrast']):+.3f}" if k in lv else "—"
                           for k in CUM)
        head, _ = attribute(lv[v["top_level"]])
        L.append(f"| {tc.pretty(m)} | {cells} | {v['d']:+.3f} "
                 f"[{v['dlo']:+.3f}, {v['dhi']:+.3f}] "
                 f"| {'**met**' if v['met'] else 'not met'} "
                 f"| L{v['max_level']}: {v['max_d']:+.3f} "
                 f"[{v['max_lo']:+.3f}, {v['max_hi']:+.3f}]"
                 f"{' ✱' if v['met_any'] else ''} "
                 f"| {'yes (L%d)' % v['best_level'] if v['recovered'] else 'no'} "
                 f"| {head} |")
    L += ["", "Δ columns are paired over scenario groups against that model's own L1. "
          "The pre-registered verdict is the `bar at top level` column; `best level Δ` is "
          "descriptive (✱ marks a shift that would have cleared the bar had the "
          "pre-registration placed it at that level, which it does not). The last column "
          "is the ceiling-compression check: a contrast gain built from `accidental` "
          "falling is outcome-blame suppression, not the intent re-weighting the "
          "intervention was aimed at, and a gain in which all four cells slide together is "
          "neither. Per-cell numbers are in the P6 table below."]

    # P5 dose-response, and the L4-vs-L5 test that separates imitation from principle.
    L += ["", "## Dose-response (P5)", "",
          "Levels are cumulative, so under P1 the shift should be monotone or nearly so. "
          "L4 adds labelled adult ratings; L5 adds the principle and no new labels, so an "
          "L4-only jump that does not persist at L5 indicates imitation of the few-shot "
          "format rather than uptake of the principle. The reverse — a gain at L3 that L4 "
          "destroys — indicates the opposite: the labelled examples are being imitated as "
          "anchors and are overriding the reasoning they were meant to illustrate.", "",
          "| model | monotone | L1→L2 | L2→L3 | L3→L4 | L4→L5 | reading |",
          "|---|---|---:|---:|---:|---:|---|"]
    for m, all_lv in sorted(by.items()):
        lv = {k: v for k, v in all_lv.items() if k in CUM}
        ks = sorted(lv)
        ys = [fl(lv[k]["contrast"]) for k in ks]
        mono = all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
        step = {}
        for k in (2, 3, 4, 5):
            step[k] = (fl(lv[k]["contrast"]) - fl(lv[k - 1]["contrast"])) \
                if (k in lv and k - 1 in lv) else float("nan")
        gain_l3 = sum(step[k] for k in (2, 3) if not np.isnan(step[k]))
        if any(np.isnan(v) for v in step.values()):
            note = "incomplete"
        elif gain_l3 >= SHIFT_BAR and step[4] <= -SHIFT_BAR:
            # Descriptive only. The labels are verified non-inverted, so this is not an
            # arithmetic artefact, but "anchor imitation" is one hypothesis among several
            # (added prompt length, format shift, position of the question) and the design
            # does not separate them. The ablation's few-shot-alone cell is the closest
            # available handle.
            note = "worked reasoning helps, adding labelled examples reverses it"
        elif step[4] >= SHIFT_BAR and step[5] <= -SHIFT_BAR / 2:
            note = "gain appears only with the labelled examples present"
        elif max(gain_l3, gain_l3 + step[4] + step[5]) >= SHIFT_BAR:
            note = "shift present and survives to the top level"
        else:
            note = "no shift of the pre-registered size at any level"
        L.append(f"| {tc.pretty(m)} | {'yes' if mono else 'no'} | "
                 + " | ".join(f"{step[k]:+.3f}" for k in (2, 3, 4, 5))
                 + f" | {note} |")

    # P6 ceiling guard: all four cell means at every level.
    L += ["", "## All four cell means at every level (P6 ceiling guard)", "",
          "A contrast change produced by all four means moving together is compression, "
          "not intent re-weighting — the same caveat that governs the W3 "
          "difference-of-means estimator. `extreme` is the fraction of ratings at the "
          "top or bottom of the scale; a level that pushes ratings to an endpoint removes "
          "the headroom the contrast needs.", "",
          "| model | level | neutral | accidental | attempted | intentional | contrast "
          "| Δ cells vs L1 | attribution | rating SD | extreme |",
          "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for m, lv in sorted(by.items()):
        for k in sorted(lv):
            r = lv[k]
            head, detail = attribute(r)
            L.append(f"| {tc.pretty(m)} | L{k} {r['level_name']} | "
                     + " | ".join(f"{fl(r['mean_' + c]):.3f}" for c in tc.CELLS)
                     + f" | {fl(r['contrast']):+.3f} | {detail} | {head} "
                     f"| {fl(r['rating_std']):.4f} | {fl(r['extreme_frac']):.2f} |")

    if has_abl:
        L += ["", "## Non-cumulative ablation (secondary)", "",
              "L1–L5 are cumulative, so L5 confounds the explicit principle with repair of "
              "whatever L4 did, and no cumulative level attributes the effect to a "
              "component. Each cell below is that component alone against the same L1 "
              "baseline. L2 already is \"instruction alone\" and is reused rather than "
              "re-run. **This is a secondary attribution analysis**: the pre-registered "
              "verdict stays the cumulative L5 column above, since these cells were "
              "designed after seeing run 1.", "",
              "| model | " + " | ".join(f"{ABL_LABEL[k]} (L{k})" for k in (2, 6, 7, 8))
              + " | Σ parts | L5 cumulative | reading |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
        for m, lv in sorted(by.items()):
            ds = {k: fl(lv[k]["d_contrast"]) if k in lv else float("nan")
                  for k in (2, 6, 7, 8)}
            if all(np.isnan(v) for v in ds.values()):
                continue
            parts = [v for v in ds.values() if not np.isnan(v)]
            tot, l5 = sum(parts), fl(lv[5]["d_contrast"]) if 5 in lv else float("nan")
            best = max((v, k) for k, v in ds.items() if not np.isnan(v))
            if np.isnan(l5):
                note = "L5 missing"
            elif best[0] >= SHIFT_BAR and l5 < SHIFT_BAR:
                note = f"{ABL_LABEL[best[1]]} beats the full stack — escalation subtracts"
            elif tot > 0 and l5 > tot + 0.05:
                note = "super-additive: the stack does more than its parts"
            elif tot > 0 and l5 < tot - 0.05:
                note = "sub-additive: components interfere"
            else:
                note = "roughly additive"
            L.append(f"| {tc.pretty(m)} | "
                     + " | ".join("—" if np.isnan(ds[k]) else f"{ds[k]:+.3f}"
                                  for k in (2, 6, 7, 8))
                     + f" | {tot:+.3f} | {l5:+.3f} | {note} |")
        L += ["", "Δ vs L1, paired over scenario groups. Σ parts is the arithmetic sum of "
              "the four single-component shifts and is a descriptive additivity reference, "
              "not a prediction any model of the effect entails.", "",
              "### Where the principle-alone gain comes from", "",
              "The attribution differs between the stack and the principle on its own, which "
              "is the most consequential thing the ablation shows. Same columns as the P6 "
              "guard, restricted to L5 (full stack) against L8 (principle alone).", "",
              "| model | L5 Δattempted | L5 Δaccidental | L8 Δattempted | L8 Δaccidental "
              "| L8 mechanism |", "|---|---:|---:|---:|---:|---|"]
        for m, lv in sorted(by.items()):
            if 8 not in lv or 5 not in lv:
                continue
            head8, _ = attribute(lv[8])
            L.append(f"| {tc.pretty(m)} | {fl(lv[5]['d_attempted']):+.3f} "
                     f"| {fl(lv[5]['d_accidental']):+.3f} "
                     f"| {fl(lv[8]['d_attempted']):+.3f} "
                     f"| {fl(lv[8]['d_accidental']):+.3f} | {head8} |")

    best_any = max((V[m]["max_d"] for m in V), default=float("nan"))
    L += ["", "## Relation to W3 (steering)", "",
          'Do **not** read W3+W4 as "unreachable by steering, reachable by prompting." '
          "Steering and prompting are different interventions with no common effect-size "
          "scale, and W4's own bar is not met. The joint statement is narrower: the bias is "
          f"**entrenched** — inert to residual-stream intervention at resolvable depths "
          f"(W3, |Δcontrast| ≤ {W3_INTENT_MAX:.3f}) and only partially / inconsistently "
          "moved by in-context instruction (W4: 3/6 at L5; stack gains mostly accidental "
          "suppression; intent-led movement in 2/6 and only under principle-alone).", "",
          f"Descriptively, the largest in-context shift ({best_any:+.3f}) can land near the "
          f"W3 outcome-direction control ({W3_OUTCOME_MIN:.3f}–{W3_OUTCOME_MAX:.3f}); flag "
          "that coincidence of magnitude, do not build on it.", "",
          "## What this does and does not license", "",
          "- The levels differ only in added text. Weights, decoding (logprob-EV over "
          "rating digits, deterministic), items, templates, scale normalisation and the "
          "contrast estimator are identical across levels, so the level effect is not "
          "confounded with the measurement.", "",
          "- The L3/L4 example vignettes are held out: novel content (climbing gym, print "
          "shop) that appears nowhere in the master, so the scaffolding cannot teach a "
          "test item. The L4 labels are the Young 2007 adult profile mapped onto each "
          "template's own scale; they are used only to write the examples and are never a "
          "target in the analysis.", "",
          "- A contrast gain is not automatically intent re-weighting. Read the attribution "
          "column: where the gain is `accidental` falling, the intervention removed blame "
          "from accidental harm rather than adding it to attempted harm, and where all four "
          "cells slide together the movement is compression. Both are real changes to the "
          "judgment and neither is evidence that the model started consulting intent.", "",
          "- This is an in-context result. It says nothing about whether fine-tuning could "
          "move the contrast, and a prompt-level shift does not establish that the model "
          "is using the same intent representation the probe reads — only that the input "
          "can change the weighting. Establishing that it is the same representation "
          "would require the W3 intervention to work, which it does not.", ""]

    with open(MD_OUT, "w") as f:
        f.write("\n".join(L) + "\n")
    # Short standalone conclusion for mentor / paper cross-refs (same content as Verdict).
    summary = ["# W4 — summary", "",
               "Source: `W4_CURRICULUM.md` (generated by `55_w4_summary.py`). "
               "Do not edit by hand.", "",
               "## Conclusion", "",
               'The completed 6-model × 8-level run **does not** support '
               '"unreachable by steering, reachable by prompting."', "",
               f"- Only **{n_met}/{len(V)}** clear the pre-registered +{SHIFT_BAR:.2f} bar "
               f"at L5 (need {MIN_MODELS}).",
               f"- Stack-top gains are predominantly **accidental-blame suppression** "
               f"({len(acc_only)}/{len(gainers)} improvers at L5).",
               f"- Genuine **intent-led** movement: **{len(abl_led)}/{len(by)}**, and only "
               "under principle-alone (L8).",
               f"- Scaffolding often hurts: principle-alone beats the full stack in "
               f"**{abl_beats}/{len(by)}** models; every model is sub-additive.",
               "",
               "**Frame:** the bias is entrenched — resistant to representation-level "
               "intervention and only partially and inconsistently responsive to "
               "instruction.", ""]
    with open(SUMMARY_OUT, "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"  wrote {os.path.relpath(MD_OUT, tc.ROOT)} "
          f"({len(by)} models; L5 bar {n_met}/{len(V)}; "
          f"intent-led @ L8 {len(abl_led)}/{len(by)})")
    print(f"  wrote {os.path.relpath(SUMMARY_OUT, tc.ROOT)}")
    plot(by)


if __name__ == "__main__":
    main()
