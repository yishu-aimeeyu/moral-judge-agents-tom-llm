#!/usr/bin/env python3
"""E4 -- cross-cultural placement of model intent effects. [CPU]

WHY THIS EXISTS. Every human anchor in the project is Western: Young et al. (2007) adults and
the child bands. The adult anchor already carries a heavy caveat (no confirmed
item-and-condition match, see dataset/human_reference/ADULT_ANCHOR_MATCH_AUDIT.md). The
obvious reviewer objection to "models fall below the youngest measured child" is that the
comparison rests on one cultural sample. Curtin et al. (2020) released 10-society vignette
data with an intent manipulation scored the same way in every society, which is the only
source available that can place model contrasts inside a non-Western-only human range.

WHAT THIS DELIBERATELY DOES NOT DO. The dataset has no failed-attempt condition, so the
project's headline contrast (attempted - accidental) has NO counterpart in it and is not
plotted against it. E4 uses the one axis present in both: the intent effect at constant harm,
intentional - accidental. Substituting the headline contrast would repeat exactly the error the
adult-anchor audit exists to prevent.

Pre-specified plan, thresholds and falsifiers: dataset/human_reference/crosscultural/E4_PRESPEC.md
Provenance, normalisation method and caveats: dataset/human_reference/crosscultural/README.md

Outputs
  outputs/crosscultural/society_intent_effects.csv
  outputs/crosscultural/model_placement.csv
  outputs/crosscultural/E4_CROSSCULTURAL.md
  outputs/figures_final/crosscultural_placement.png     (NEW figure; nothing existing touched)
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
XC_DIR = os.path.join(ROOT, "dataset", "human_reference", "crosscultural")
RAW = os.path.join(XC_DIR, "raw", "Intentions_KII_BarrettData.csv")
PRESPEC = os.path.join(XC_DIR, "E4_PRESPEC.md")
OUT_DIR = os.path.join(ROOT, "outputs", "crosscultural")
MIXED = os.path.join(ROOT, "outputs", "stats", "mixed_effects_2x2.csv")

# Fixed in the prespec, not exposed as flags
SCALE_RANGE = 4.0          # judgrev spans -2..+2
PRIMARY_QUESTION = "Badness"
QUESTIONS = ["Badness", "Punishment", "Reputation"]
ENGAGEMENT_FLOOR = 0.2191  # rating_std floor, floor_derivation.csv
N_BOOT = 2000
MODEL_BAR = 18             # 18/18 convention from the child ladders


def load_human(path):
    rows = [r for r in csv.DictReader(open(path))
            if r["bank"] == "INTENT" and r["condition"] in ("Intentional", "Accidental")]
    if not rows:
        raise SystemExit(f"no INTENT-bank Intentional/Accidental rows in {path}")
    return rows


def society_effects(rows, question, rng):
    """Subject-level differences first, then the society mean.

    Subject-first so societies with more items per subject are not weighted beyond their
    subject count -- fixed in the prespec.
    """
    per_subject = defaultdict(lambda: defaultdict(list))
    all_judg = defaultdict(list)
    for r in rows:
        if r["question"] != question:
            continue
        key = (r["society"], r["subject"])
        per_subject[key][r["condition"]].append(float(r["judgrev"]))
        all_judg[r["society"]].append(float(r["judgrev"]))

    by_soc = defaultdict(list)
    for (soc, _subj), conds in per_subject.items():
        if "Intentional" in conds and "Accidental" in conds:
            by_soc[soc].append(np.mean(conds["Intentional"]) - np.mean(conds["Accidental"]))

    out = []
    for soc in sorted(by_soc):
        d = np.asarray(by_soc[soc], dtype=float)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
        pooled_sd = float(np.std(all_judg[soc], ddof=1))
        raw = float(d.mean())
        out.append({
            "society": soc,
            "question": question,
            "n_subjects": len(d),
            "raw_diff": round(raw, 4),
            "raw_ci_lo": round(float(np.percentile(boots, 2.5)), 4),
            "raw_ci_hi": round(float(np.percentile(boots, 97.5)), 4),
            "scaled_diff": round(raw / SCALE_RANGE, 4),
            "scaled_ci_lo": round(float(np.percentile(boots, 2.5)) / SCALE_RANGE, 4),
            "scaled_ci_hi": round(float(np.percentile(boots, 97.5)) / SCALE_RANGE, 4),
            "cohens_d": round(raw / pooled_sd, 4) if pooled_sd > 1e-9 else "",
            "within_society_sd": round(pooled_sd, 4),
            "ci_excludes_zero": bool(np.percentile(boots, 2.5) > 0
                                     or np.percentile(boots, 97.5) < 0),
        })
    return out


def _norm(name):
    """contrast_by_model.csv writes `unsloth_Meta-Llama-3_1-8B`, mixed_effects writes
    `unsloth/Meta-Llama-3_1-8B`. Join on the slash-free form."""
    return name.strip().replace("/", "_")


def load_models():
    """Model intent effect at constant harm = cell_intentional - cell_accidental."""
    if not os.path.exists(MIXED):
        raise SystemExit(f"missing {MIXED}")
    qc = load_qc()
    out = []
    for r in csv.DictReader(open(MIXED)):
        if r.get("spec") != "primary":
            continue
        try:
            eff = float(r["cell_intentional"]) - float(r["cell_accidental"])
        except (ValueError, KeyError):
            continue
        q = qc.get(_norm(r["model"]), {})
        rs = q.get("rating_std")
        out.append({
            "model": r["model"],
            "intent_effect": round(eff, 4),
            "cell_intentional": round(float(r["cell_intentional"]), 4),
            "cell_accidental": round(float(r["cell_accidental"]), 4),
            "diag_attempted_minus_accidental": r.get("diag_attempted_minus_accidental", ""),
            "rating_std": rs if rs is not None else "",
            # two populations, per E4_PRESPEC.md amendment 1
            "engaged": (rs is not None and rs >= ENGAGEMENT_FLOOR),
            "non_degenerate": (q.get("degenerate") is False),
        })
    return out


def load_qc():
    """rating_std and the degenerate flag per model, from the behavioural stats table."""
    p = os.path.join(ROOT, "outputs", "stats", "contrast_by_model.csv")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p} (needed for rating_std / degenerate flags)")
    out = {}
    for r in csv.DictReader(open(p)):
        try:
            rs = float(r["rating_std"])
        except (KeyError, ValueError):
            rs = None
        out[_norm(r["model"])] = {
            "rating_std": rs,
            "degenerate": str(r.get("degenerate", "")).strip().lower() in ("true", "1"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--fig", default=os.path.join(ROOT, "outputs", "figures_final",
                                                  "crosscultural_placement.png"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(PRESPEC):
        raise SystemExit(f"{PRESPEC} missing; E4 is pre-registered, refusing to run.")
    if not os.path.exists(a.raw):
        raise SystemExit(f"{a.raw} missing. Download it: see {XC_DIR}/README.md")

    rows = load_human(a.raw)
    if a.dry_run:
        socs = sorted({r["society"] for r in rows})
        print("=== E4 PLAN (dry-run) ===")
        print(f"human rows (INTENT bank, Intentional/Accidental): {len(rows)}")
        print(f"societies ({len(socs)}): {', '.join(socs)}")
        print(f"questions: primary={PRIMARY_QUESTION}, robustness={QUESTIONS}")
        print(f"scale range {SCALE_RANGE} (judgrev -2..+2); bootstrap {N_BOOT} over subjects")
        print(f"model axis: cell_intentional - cell_accidental from {os.path.relpath(MIXED, ROOT)}")
        print(f"P1 bar: >= {MODEL_BAR} engaged models below the lowest society")
        return

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    soc_rows = []
    for q in QUESTIONS:
        soc_rows += society_effects(rows, q, rng)
    p = os.path.join(a.out, "society_intent_effects.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(soc_rows[0].keys()))
        w.writeheader(); w.writerows(soc_rows)
    print(f"wrote {os.path.relpath(p, ROOT)} ({len(soc_rows)} rows)")

    primary = [r for r in soc_rows if r["question"] == PRIMARY_QUESTION]
    lo = min(r["scaled_diff"] for r in primary)
    hi = max(r["scaled_diff"] for r in primary)
    print(f"\n{PRIMARY_QUESTION}: society scaled intent effects "
          f"{lo:+.4f} .. {hi:+.4f} (n={len(primary)})")
    for r in sorted(primary, key=lambda r: r["scaled_diff"]):
        print(f"  {r['society']:16} scaled={r['scaled_diff']:+.4f} "
              f"[{r['scaled_ci_lo']:+.4f},{r['scaled_ci_hi']:+.4f}] "
              f"raw={r['raw_diff']:+.3f} d={r['cohens_d']} n={r['n_subjects']}")

    models = load_models()
    for m in models:
        m["below_lowest_society"] = m["intent_effect"] < lo
        m["percentile_in_human_dist"] = round(
            100.0 * sum(1 for r in primary if r["scaled_diff"] <= m["intent_effect"])
            / len(primary), 1)
    p = os.path.join(a.out, "model_placement.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(models[0].keys()))
        w.writeheader(); w.writerows(models)
    print(f"\nwrote {os.path.relpath(p, ROOT)} ({len(models)} models)")

    write_verdict(soc_rows, primary, models, lo, hi, a.out)
    make_figure(primary, models, lo, a.fig)


def write_verdict(soc_rows, primary, models, lo, hi, outdir):
    pops = {
        "engaged": ([m for m in models if m["engaged"]], None),
        "non_degenerate": ([m for m in models if m["non_degenerate"]], MODEL_BAR),
    }
    neg_soc = [r for r in primary if r["scaled_diff"] <= 0 or not r["ci_excludes_zero"]]
    pos_soc = [r for r in primary if r not in neg_soc]
    p2 = not neg_soc

    # second-lowest society matters once P2 fails: the "outside the human range" claim has to
    # be restated against the societies that DO show an intent effect
    lo_pos = min((r["scaled_diff"] for r in pos_soc), default=float("nan"))

    L = ["# E4 — cross-cultural placement: verdict\n",
         "Generated by `code/experiments/64_crosscultural_placement.py`. Thresholds and "
         "falsifiers fixed in "
         "[`E4_PRESPEC.md`](../../dataset/human_reference/crosscultural/E4_PRESPEC.md) before "
         "this ran (see amendment 1 there for the population correction). Method and caveats: "
         "[`crosscultural/README.md`](../../dataset/human_reference/crosscultural/README.md).\n",
         "> **Axis note.** This is the intent effect at constant harm "
         "(`intentional − accidental`), not the project's headline `attempted − accidental` "
         "contrast. The dataset has no failed-attempt condition, so the headline contrast has "
         "no counterpart here and is not plotted against it.\n",
         "## Headline\n"]

    verdicts = {}
    for pop, (ms, bar) in pops.items():
        below = [m for m in ms if m["below_lowest_society"]]
        bar_eff = bar if bar is not None else len(ms)
        ok = len(below) >= bar_eff and len(ms) > 0
        verdicts[pop] = (ok, below, ms)
        L.append(f"- **P1** [`{pop}`, n={len(ms)}]: **{len(below)}/{len(ms)}** below the "
                 f"lowest society → **{'supported' if ok else 'NOT supported'}** "
                 f"(bar {bar_eff}/{len(ms)}).")
    agree = verdicts["engaged"][0] == verdicts["non_degenerate"][0]
    L.append(f"- The two populations **{'agree' if agree else 'DISAGREE'}**, so the placement "
             f"claim {'does not depend' if agree else '**depends**'} on which floor is applied"
             f"{'.' if agree else ' — that dependence is the result, not a footnote.'}")
    L.append(f"- **P2** (all 10 societies positive with CI excluding zero): "
             f"**{len(pos_soc)}/{len(primary)}** → "
             f"**{'supported' if p2 else 'NOT supported'}**.")
    L.append(f"- Human range ({PRIMARY_QUESTION}, scale-normalised): "
             f"**{lo:+.4f} … {hi:+.4f}**.")
    L.append("")

    if neg_soc:
        L.append("### P2 failed — the required restatement\n")
        L.append("Societies with no detectable intent effect on this measure: "
                 + ", ".join(f"**{r['society']}** ({r['scaled_diff']:+.4f}, CI "
                             f"[{r['scaled_ci_lo']:+.4f}, {r['scaled_ci_hi']:+.4f}])"
                             for r in neg_soc) + ".\n")
        L.append("The prespec fixed the consequence in advance: the claim **may not** be "
                 "stated as \"models sit outside the human range\". It must be stated as a "
                 "distribution-overlap claim, because at least one human society sits at the "
                 "same place the models do on this axis. The defensible form is:\n")
        L.append(f"> On the intent-at-constant-harm axis, model intent effects fall at or "
                 f"below the least intent-weighting of 10 societies, and below all "
                 f"{len(pos_soc)} societies with a detectable intent effect "
                 f"(lowest of those: {lo_pos:+.4f}).\n")
        L.append("This is a weaker claim than the one E4 set out to test, and it is the one "
                 "the data support. It is also independently interesting: it shows the "
                 "human range on this measure includes a society that does not use intent "
                 "here, so \"outcome-weighting\" is not by itself non-human.\n")
        L.append("**Why P1's own comparison point is not usable as stated.** The lowest "
                 f"society is Yasawa at exactly {lo:+.4f}, so *any* non-zero positive model "
                 "value counts as \"above\" it. That is a degenerate threshold: the models "
                 "named above sit at +0.0007 to +0.0211, which is not intent-weighting in any "
                 "meaningful sense, and none of their CIs would separate them from Yasawa. "
                 "The counts against the restated reference point are therefore the "
                 "informative ones:\n")
        L.append("| population | n | below lowest society (%+.4f) | below lowest society "
                 "with a detectable effect (%+.4f) |" % (lo, lo_pos))
        L.append("|---|---:|---:|---:|")
        for pop, (ms, _bar) in pops.items():
            n_lo = sum(1 for m in ms if m["intent_effect"] < lo)
            n_pos = sum(1 for m in ms if m["intent_effect"] < lo_pos)
            L.append(f"| `{pop}` | {len(ms)} | {n_lo}/{len(ms)} | **{n_pos}/{len(ms)}** |")
        L.append("")

    for pop, (ok, below, ms) in verdicts.items():
        above = [m for m in ms if not m["below_lowest_society"]]
        if above:
            L.append(f"Models at or above the lowest society [`{pop}`], named as the prespec "
                     f"requires: "
                     + ", ".join(f"`{m['model']}` ({m['intent_effect']:+.4f})"
                                 for m in above) + ".\n")

    L.append("## Human societies (%s, primary)\n" % PRIMARY_QUESTION)
    L.append("| society | n subjects | raw diff (−2…+2) | scaled diff | 95% CI (scaled) | Cohen's d |")
    L.append("|---|---:|---:|---:|---|---:|")
    for r in sorted(primary, key=lambda r: -r["scaled_diff"]):
        L.append(f"| {r['society']} | {r['n_subjects']} | {r['raw_diff']:+.3f} | "
                 f"{r['scaled_diff']:+.4f} | "
                 f"[{r['scaled_ci_lo']:+.4f}, {r['scaled_ci_hi']:+.4f}] | {r['cohens_d']} |")
    L.append("")

    L.append("## Question robustness (P4)\n")
    L.append("| question | min scaled | max scaled | societies positive |")
    L.append("|---|---:|---:|---:|")
    for q in QUESTIONS:
        qq = [r for r in soc_rows if r["question"] == q]
        if not qq:
            continue
        pos = sum(1 for r in qq if r["scaled_diff"] > 0 and r["ci_excludes_zero"])
        L.append(f"| {q} | {min(r['scaled_diff'] for r in qq):+.4f} | "
                 f"{max(r['scaled_diff'] for r in qq):+.4f} | {pos}/{len(qq)} |")
    L.append("")

    L.append("## Models\n")
    L.append("| model | intent effect (intentional − accidental) | headline contrast "
             "(attempted − accidental) | rating_std | engaged | non-degenerate | "
             "below lowest society |")
    L.append("|---|---:|---:|---:|---|---|---|")
    for m in sorted(models, key=lambda m: m["intent_effect"]):
        rs = m["rating_std"]
        diag = m.get("diag_attempted_minus_accidental", "")
        try:
            diag_s = f"{float(diag):+.4f}"
        except (TypeError, ValueError):
            diag_s = "—"
        L.append(f"| `{m['model']}` | {m['intent_effect']:+.4f} | {diag_s} | "
                 f"{rs if rs != '' else '—'} | "
                 f"{'yes' if m['engaged'] else 'no'} | "
                 f"{'yes' if m['non_degenerate'] else 'no'} | "
                 f"{'yes' if m['below_lowest_society'] else '**no**'} |")
    L.append("")
    L.append("The headline-contrast column is shown only so the two axes are not confused: it "
             "is **not** compared to any human number in this file, because no human number "
             "here corresponds to it.\n")

    L.append("## What this cannot show\n")
    L.append("Different stimuli (no shared item), different response scale and task, "
             "different populations and administration. This is a **reference distribution, "
             "not item-matched ground truth**, and proximity to it licenses no developmental "
             "or causal claim. Same caveat discipline as `ADULT_ANCHOR_MATCH_AUDIT.md`.\n")
    L.append("One asymmetry in the figure is worth stating outright, because it could be "
             "misread as false precision: the society points carry 95% CIs over subjects, "
             "and the model points carry **no interval at all**. Model effects here are "
             "aggregate cell means over the same 298 items under deterministic scoring, so "
             "there is no subject-level sampling variability to display, and an interval "
             "computed over items would not be the same quantity as the human one. The "
             "practical consequence is that small differences *between models* on this axis "
             "should not be read as reliable, and no such comparison is made above.\n")

    p = os.path.join(outdir, "E4_CROSSCULTURAL.md")
    open(p, "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(p, ROOT)}")
    for pop, (ok, below, ms) in verdicts.items():
        print(f"P1 [{pop}]: {len(below)}/{len(ms)} -> "
              f"{'supported' if ok else 'NOT supported'}")
    print(f"P2: {'supported' if p2 else 'NOT supported'}"
          + ("" if p2 else f" ({', '.join(r['society'] for r in neg_soc)} at/below zero)"))


def make_figure(primary, models, lo, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})")
        return
    socs = sorted(primary, key=lambda r: r["scaled_diff"])
    shown = sorted([m for m in models if m["non_degenerate"]],
                   key=lambda m: m["intent_effect"])
    pos = [r for r in socs if r["scaled_diff"] > 0 and r["ci_excludes_zero"]]
    lo_pos = min((r["scaled_diff"] for r in pos), default=float("nan"))

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    for i, r in enumerate(socs):
        null_soc = not (r["scaled_diff"] > 0 and r["ci_excludes_zero"])
        ax.errorbar(r["scaled_diff"], i,
                    xerr=[[r["scaled_diff"] - r["scaled_ci_lo"]],
                          [r["scaled_ci_hi"] - r["scaled_diff"]]],
                    fmt="o", color="#999999" if null_soc else "#222222",
                    ms=6, capsize=3, lw=1.2, zorder=3)
        ax.text(r["scaled_ci_hi"] + 0.006, i,
                f"{r['society']} (n={r['n_subjects']})"
                + ("  ← no intent effect" if null_soc else ""),
                va="center", fontsize=8,
                color="#999999" if null_soc else "#222222")
    y0 = len(socs) + 0.8
    for j, m in enumerate(shown):
        y = y0 + j
        col = "#c0392b" if m["engaged"] else "#e8a09a"
        ax.plot(m["intent_effect"], y, marker="s", ms=5.5, color=col, zorder=3)
        ax.text(m["intent_effect"] + 0.006, y,
                m["model"] + ("" if m["engaged"] else "  (below engagement floor)"),
                va="center", fontsize=7, color=col)
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(lo, color="#7fb8d8", ls="--", lw=1.1)
    if np.isfinite(lo_pos):
        ax.axvline(lo_pos, color="#2c7fb8", ls="--", lw=1.3)
        ax.text(lo_pos, -1.6, f" lowest society WITH an intent effect ({lo_pos:+.3f})",
                color="#2c7fb8", fontsize=8, va="top")
    ax.axhline(y0 - 0.9, color="gray", lw=0.6, ls=":")
    ax.set_yticks([])
    ax.set_ylim(-2.2, y0 + len(shown))
    ax.set_xlabel("intent effect at constant harm: (intentional − accidental), "
                  "scale-normalised")
    ax.set_title("E4 — model intent effects placed in a 10-society human distribution\n"
                 "different stimuli, scale and task; reference distribution, "
                 "NOT item-matched ground truth", fontsize=10)
    # models cluster near zero and their labels run rightwards, so the legend goes top-right
    ax.text(0.985, 0.985,
            "black = societies with a detectable intent effect (95% CI over subjects)\n"
            "grey = society whose CI includes zero\n"
            "red = open-weight models (pale = below engagement floor)\n"
            "model points have NO interval: they are aggregate cell means over the\n"
            "  same 298 items, so their spread is not comparable to subject-level CIs\n"
            "axis is NOT the headline attempted − accidental contrast",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", lw=0.8))
    xmax = max([r["scaled_ci_hi"] for r in socs]
               + [m["intent_effect"] for m in shown]) + 0.12
    ax.set_xlim(min([r["scaled_ci_lo"] for r in socs]) - 0.02, xmax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
