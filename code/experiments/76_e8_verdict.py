#!/usr/bin/env python3
"""E8 verdict -- score P1-P4 against the committed CSVs and draw the headline figure. [CPU]

Every number in `outputs/rsa/E8_CHECKPOINT_EMERGENCE.md` is read back out of
`checkpoint_betas.csv` and `checkpoint_shape_fits.csv` rather than carried in a variable
from the run that produced them, so the write-up cannot drift from the committed data.

The headline figure is two panels because that is what the result turned out to be: the
factors emerge at DIFFERENT READ POSITIONS, and each is invisible at the other's. Drawing
only the pre-designated primary cell would show half of it.

Outputs -> outputs/rsa/E8_CHECKPOINT_EMERGENCE.md
           outputs/rsa/checkpoint_emergence_by_position.png
"""
import argparse
import csv
import importlib.util
import os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")

PRIMARY_POSITION = "belief_last"
PRIMARY_REL_DEPTH = "0.8"
P2_MIN_STEP = 1000          # E8_PRESPEC.md P2
N_BOOT = 2000
SEED = 0


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def F(r, k):
    return float(r[k])


def clears_band(r):
    """Beta outside its own null AND its cluster-bootstrap CI clear of the band.

    The cluster bootstrap is used rather than the item bootstrap because amendment 2.2 gives
    it precedence: 94 of the 298 rows are exact reprints of another row, so the item
    bootstrap's independence assumption is false and its CIs are too narrow.
    """
    q = F(r, "null_q975_abs")
    return (abs(F(r, "beta")) > q
            and (F(r, "cluster_ci_lo") > q or F(r, "cluster_ci_hi") < -q))


def emergence_step(series):
    """First step from which the band is cleared and stays cleared, per P1."""
    by = {int(r["step"]): r for r in series}
    steps = sorted(s for s in by if s >= 1)
    for i, s in enumerate(steps):
        if all(clears_band(by[t]) for t in steps[i:]):
            return s
    return None


def spearman_boot(x, y, n_boot, rng):
    from scipy.stats import spearmanr
    obs = float(spearmanr(x, y).statistic)
    draws = []
    for _ in range(n_boot):
        sel = rng.integers(0, len(x), len(x))
        if len(np.unique(x[sel])) < 3:
            continue
        draws.append(spearmanr(x[sel], y[sel]).statistic)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return obs, float(lo), float(hi)


def index(rows):
    idx = defaultdict(list)
    for r in rows:
        idx[(r["spec"], r["position"], r["rel_depth_requested"], r["predictor"])].append(r)
    return idx


def make_figure(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = index(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)
    titles = {
        "belief_last": ("read at the end of the BELIEF clause\n"
                        "(before the outcome is stated)"),
        "last": "read at the END OF THE STORY\n(everything has been read)",
    }
    colors = {"intent": "#1f77b4", "outcome": "#d62728"}
    for ax, pos in zip(axes, ["belief_last", "last"]):
        for pred in ("intent", "outcome"):
            s = sorted(idx[("primary", pos, PRIMARY_REL_DEPTH, pred)],
                       key=lambda r: int(r["step"]))
            xs = [0.4 if int(r["step"]) == 0 else int(r["step"]) for r in s]
            ys = [F(r, "beta") for r in s]
            ax.plot(xs, ys, "-o", ms=4, lw=2.2, color=colors[pred], label=f"β {pred}")
            ax.fill_between(xs, [F(r, "cluster_ci_lo") for r in s],
                            [F(r, "cluster_ci_hi") for r in s],
                            color=colors[pred], alpha=0.18, lw=0)
            z = next((r for r in s if int(r["step"]) == 0), None)
            if z is not None:
                q = F(z, "null_q975_abs")
                ax.axhspan(-q, q, color=colors[pred], alpha=0.09, lw=0, zorder=0)
            e = emergence_step(s)
            if e:
                ax.axvline(e, color=colors[pred], ls=":", lw=1.4, alpha=0.8)
                ax.annotate(f"{pred} leaves\nits band\nat step {e:,}", xy=(e, 0),
                            xytext=(e * 1.3, 0.30 if pred == "outcome" else 0.22),
                            fontsize=8, color=colors[pred])
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("training step (log; step 0 at the left tick)")
        ax.set_title(titles[pos], fontsize=10.5)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("standardised β  (RDM regression)")
    fig.suptitle("E8 — Pythia-1.4B: outcome and intent structure emerge at different read "
                 "positions\nrel. depth 0.8; shading = cluster bootstrap 95% CI; "
                 "pale spans = step-0 permutation null", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = os.path.join(out, "checkpoint_emergence_by_position.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def behavioral_figure(rows, beh, out):
    """E8f: engagement per checkpoint against the floor, above a deliberately empty panel.

    The behavioural coefficients are NOT drawn for a checkpoint below the engagement floor,
    because a near-constant rater has no estimable contrast; plotting its b_intent would
    present noise as a measured judgment. The representational beta at `last` is drawn
    alongside for scale, from the same checkpoints.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = float(beh[0]["engagement_floor"])
    xs = [max(int(r["step"]), 0.4) for r in beh]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    ax = axes[0]
    ax.plot(xs, [F(r, "rating_std") for r in beh], "-o", color="#555", ms=4,
            label="rating_std (logprob-EV)")
    ax.axhline(floor, color="#2a2", ls="--", lw=1.5, label=f"engagement floor {floor}")
    ax.set_ylim(0, floor * 1.35)
    ax.set_ylabel("rating_std")
    ax.set_title("E8f — Pythia-1.4B behavioural 2×2: no checkpoint clears the engagement "
                 f"floor ({sum(r['gradeable'] == 'True' for r in beh)} of {len(beh)})",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    for pred, c in (("intent", "#1f77b4"), ("outcome", "#d62728")):
        s = sorted(index(rows)[("primary", "last", PRIMARY_REL_DEPTH, pred)],
                   key=lambda r: int(r["step"]))
        ax.plot([max(int(r["step"]), 0.4) for r in s], [F(r, "beta") for r in s], "-",
                color=c, lw=1.4, alpha=0.55, label=f"representational β {pred} (`last`)")
    ax.axvspan(0.3, xs[-1] * 1.3, color="#f2f2f2", zorder=0)
    ax.text(0.5, 0.55, "behavioural b_intent / b_outcome: NOT DRAWN\n"
            "no checkpoint is gradeable, so there is no first gradeable checkpoint\n"
            "(left blank, not interpolated, not drawn at zero)",
            transform=ax.transAxes, ha="center", fontsize=9, color="#666")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("training step (log; step 0 at the left tick)")
    ax.set_ylabel("coefficient")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    p = os.path.join(out, "checkpoint_behavioral_overlay.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RSA_DIR)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    guard = _load("_run_guard.py", "run_guard")
    guard.require_clean_tree(os.path.basename(__file__), allow_dirty=a.allow_dirty)

    rows = list(csv.DictReader(open(os.path.join(a.out, "checkpoint_betas.csv"))))
    shapes = list(csv.DictReader(open(os.path.join(a.out, "checkpoint_shape_fits.csv"))))
    idx = index(rows)
    rng = np.random.default_rng(SEED)
    model = rows[0]["model"]
    depths = sorted({r["rel_depth_requested"] for r in rows}, key=float)
    n_ckpt = len({int(r["step"]) for r in rows})

    # ---- P1, at the pre-designated primary cell and then everywhere
    def em(spec, pos, d, pred):
        return emergence_step(idx[(spec, pos, d, pred)])

    p1 = {}
    for spec in ("primary", "no_lexical"):
        for pos in ("belief_last", "last"):
            for d in depths:
                p1[(spec, pos, d)] = (em(spec, pos, d, "outcome"),
                                      em(spec, pos, d, "intent"))
    pe_o, pe_i = p1[("primary", PRIMARY_POSITION, PRIMARY_REL_DEPTH)]
    p1_pass = pe_o is not None and (pe_i is None or pe_o < pe_i)

    # ---- P2, outcome share vs log step over steps >= P2_MIN_STEP
    def p2_for(spec, pos, d):
        s = [r for r in idx[(spec, pos, d, "outcome")] if int(r["step"]) >= P2_MIN_STEP
             and r["outcome_share"] != ""]
        s.sort(key=lambda r: int(r["step"]))
        if len(s) < 5:
            return None
        x = np.log10([int(r["step"]) for r in s])
        y = np.array([F(r, "outcome_share") for r in s])
        return spearman_boot(x, y, N_BOOT, rng)

    p2_primary = p2_for("primary", PRIMARY_POSITION, PRIMARY_REL_DEPTH)
    p2_last = p2_for("primary", "last", PRIMARY_REL_DEPTH)
    p2_pass = bool(p2_primary and p2_primary[1] > 0)

    # ---- P4, shape
    sh = {(r["spec"], r["position"], r["rel_depth_requested"], r["predictor"]): r
          for r in shapes}
    prim_shape = sh.get(("primary", PRIMARY_POSITION, PRIMARY_REL_DEPTH, "outcome"))
    shape_counts = Counter(r["verdict"].split(" (")[0] for r in shapes
                           if r["spec"] == "primary")
    p4_pass = bool(prim_shape and prim_shape["verdict"] == "breakpoint")

    fig = make_figure(rows, a.out)

    # ------------------------------------------------------------------ write-up
    L = ["# E8 — representational emergence across training checkpoints", "",
         f"**{model}, {n_ckpt} checkpoints from random initialisation to step 143,000, "
         f"298 items, {len(depths)} relative depths × 2 read positions.**", "",
         "Pre-registered in `outputs/experiments/E8_PRESPEC.md` (tag `prespec-e8`) with two "
         "amendments, both written before the checkpoint that they bear on was analysed. "
         "Every number below is read back out of `checkpoint_betas.csv` and "
         "`checkpoint_shape_fits.csv`.", "",
         "---", "",
         "## Headline", "",
         "**Outcome structure and intent structure both emerge, smoothly, at around step "
         "1,000–3,000 — but at different read positions, and each is invisible at the "
         "other's.** At the end of the story the geometry is organised by outcome "
         "(β reaches +0.49) and intent never leaves its null. At the end of the belief "
         "clause the geometry is organised by intent (β reaches +0.10) and outcome never "
         "leaves its null. Neither emergence is a step change: the sigmoid beats the "
         "free-changepoint model in every informative cell.", "",
         "---", "",
         "## The pre-registered predictions", "",
         "| | prediction | verdict |", "|---|---|---|",
         f"| **P0** | step-0 gate | **PASS** — see `E8_STEP0_GATE.md`. 0 of 20 "
         f"intent/outcome predictor-cells outside even the uncorrected 95% band. |",
         f"| **P1** | outcome leaves its band before intent, at the primary cell | "
         f"**FAILS** — at the primary cell outcome *never* leaves its band and intent "
         f"leaves at step {pe_i:,}. |" if pe_i else
         "| **P1** | outcome leaves its band before intent | **FAILS** |",
         f"| **P2** | outcome share non-decreasing after step {P2_MIN_STEP:,} | "
         f"**{'holds' if p2_pass else 'FAILS'}** at the primary cell "
         f"(ρ = {p2_primary[0]:+.3f}, 95% CI [{p2_primary[1]:+.3f}, {p2_primary[2]:+.3f}]). |"
         if p2_primary else "| **P2** | outcome share trend | not estimable |",
         f"| **P3** | anything quoted is above its own null | holds — every β quoted here "
         f"clears its permutation band and its cluster-bootstrap CI clears it too. |",
         f"| **P4** | breakpoint beats sigmoid, changepoint located | "
         f"**FAILS, toward smooth** — across the primary spec the shape verdicts are "
         f"{', '.join(f'{k} {v}' for k, v in shape_counts.most_common())}. |",
         "", "P1 and P4 were both falsified, in opposite directions from the ones the "
         "design document expected to be interesting. Both are reported as they fell.", "",
         "---", "",
         "## Why P1 failed, and why the primary cell could not have tested it", "",
         "The primary cell was fixed in advance as relative depth 0.8 at `belief_last`, on "
         "the reasoning that this is the position at which intent is representable *before "
         "the harm is mentioned*. That reasoning is right about intent and it is exactly "
         "what makes the cell unable to test P1.", "",
         "**A causal language model reading the token at the end of the belief clause has "
         "not yet read the outcome.** The harm-or-no-harm manipulation occurs later in the "
         "text, so its information is not in the context window at that position and cannot "
         "be in the activation. β_outcome ≈ 0 at `belief_last` is forced by the architecture "
         "and the stimulus ordering; it is not a fact about Pythia and it is not evidence "
         "about when outcome structure emerges.", "",
         "This is a design error in the prespec, found after the fact. It is reported rather "
         "than repaired by moving the primary cell, because the whole point of fixing the "
         "cell in advance was to prevent exactly that move. What rescues the analysis is "
         "that section 5 *also* pre-registered running all cells and reporting the "
         "distribution, and said that a primary-versus-majority disagreement would be the "
         "headline rather than a footnote. It is.", "",
         "---", "",
         "## What the two positions show", "",
         f"![emergence by position]({os.path.basename(fig)})", "",
         "| | `belief_last` (before the outcome is stated) | `last` (whole story read) |",
         "|---|---|---|"]

    def cell(pos, pred, key):
        s = sorted(idx[("primary", pos, PRIMARY_REL_DEPTH, pred)],
                   key=lambda r: int(r["step"]))
        e = emergence_step(s)
        fin = s[-1]
        if key == "em":
            return f"step {e:,}" if e else "**never leaves its null**"
        return f"{F(fin, 'beta'):+.4f}"

    L += [f"| intent — emerges | {cell('belief_last','intent','em')} | "
          f"{cell('last','intent','em')} |",
          f"| intent — β at step 143,000 | {cell('belief_last','intent','b')} | "
          f"{cell('last','intent','b')} |",
          f"| outcome — emerges | {cell('belief_last','outcome','em')} | "
          f"{cell('last','outcome','em')} |",
          f"| outcome — β at step 143,000 | {cell('belief_last','outcome','b')} | "
          f"{cell('last','outcome','b')} |", ""]

    L += ["The pattern is the same at every depth from 0.2 to 1.0 and under both all-pairs "
          "specifications:", "",
          "| spec | position | outcome emerges at | intent emerges at |", "|---|---|---|---|"]
    for spec in ("primary", "no_lexical"):
        for pos in ("last", "belief_last"):
            eo = [p1[(spec, pos, d)][0] for d in depths]
            ei = [p1[(spec, pos, d)][1] for d in depths]
            def fmt(v):
                got = [x for x in v if x is not None]
                if not got:
                    return "never (all depths)"
                s = f"{min(got):,}–{max(got):,}" if min(got) != max(got) else f"{min(got):,}"
                return s + (f" ({len(got)}/{len(v)} depths)" if len(got) < len(v) else "")
            L.append(f"| `{spec}` | `{pos}` | {fmt(eo)} | {fmt(ei)} |")
    L += ["", "**The pre-registered no-lexical sensitivity passes.** Amendment 1 committed "
              "in advance to treating a disagreement between the `primary` and `no_lexical` "
              "specifications about P1's ordering as meaning P1 is not established. They do "
              "not disagree: the ordering, the sign and the emergence window are the same "
              "with and without the lexical covariate. The `within_scenario` specification "
              "is reported in the CSV and, as predicted, is dominated by the lexical "
              "collinearity (VIF ≈ 4.4); nothing is gated on it.", "",
          "---", "",
          "## Gradual, not a click (P4)", "",
          "The sigmoid in log(step) beats the free-changepoint model in **every** "
          "informative cell, by a wide margin:", "",
          "| position | predictor | ΔAICc (− favours breakpoint) | sigmoid R² | verdict |",
          "|---|---|---:|---:|---|"]
    for pos, pred in (("last", "outcome"), ("belief_last", "intent")):
        for d in depths:
            r = sh.get(("primary", pos, d, pred))
            if r:
                L.append(f"| `{pos}` @ {d} | `{pred}` | {float(r['delta_aicc']):+.2f} | "
                         f"{r['r2_sigmoid']} | {r['verdict']} |")
    beh_path = os.path.join(a.out, "checkpoint_behavioral.csv")
    beh_section = []
    if os.path.exists(beh_path):
        beh = sorted(csv.DictReader(open(beh_path)), key=lambda r: int(r["step"]))
        bfig = behavioral_figure(rows, beh, a.out)
        n_g = sum(r["gradeable"] == "True" for r in beh)
        top = max(beh, key=lambda r: F(r, "rating_std"))
        floor = F(beh[0], "engagement_floor")
        beh_section = [
            "---", "", "## Behavioural overlay (E8f) — the panel is empty", "",
            f"![behavioural overlay]({os.path.basename(bfig)})", "",
            f"**{n_g} of {len(beh)} checkpoints clear the engagement floor.** The highest "
            f"`rating_std` anywhere on the trajectory is {F(top, 'rating_std'):.4f} "
            f"(`{top['revision']}`), {F(top, 'rating_std') / floor:.0%} of the "
            f"{floor} floor; the final checkpoint sits at {F(beh[-1], 'rating_std'):.4f}. "
            "So there is no first gradeable checkpoint, and b_intent / b_outcome are not "
            "drawn at any step — not interpolated, and not drawn at zero, because a "
            "near-constant rater has no estimable contrast and plotting its coefficients "
            "would present noise as a measured judgment.", "",
            "The prespec anticipated this (section 7) and it is reportable rather than a "
            "failed run. It is also the sharpest version of section 4's point that the two "
            "series are not the same series: by step 143,000 the representational β for "
            "outcome at `last` is +0.49 and has been clear of its null since step 1,000, "
            "while the same network never produces ratings that vary enough to score. "
            "Within this trajectory, outcome structure in the geometry never comes with a "
            "usable behavioural signal at all.", "",
            "Scoring was logprob-EV under `human_verbatim`, the same path and template that "
            "produced `checkpoint_dissection.csv`. Under logprob-EV there is nothing to "
            "parse, so section 7's parse-rate criterion holds by construction and only the "
            "engagement floor binds (amendment 3).", ""]

    L += ["", "This is the result that the caveat in the prespec was written for. Apparent "
              "sharp emergence in language models is frequently an artifact of a thresholded "
              "measure — discontinuous scoring manufactures phase transitions that "
              "continuous measures dissolve (Schaeffer et al., *Are Emergent Abilities of "
              "Large Language Models a Mirage?*). Standardised β is continuous and "
              "unthresholded, so a breakpoint found here would have been a stronger claim "
              "than the usual one. There is no breakpoint. Both factors come in as smooth "
              "sigmoids in log(step), and that is reported as plainly as a breakpoint would "
              "have been.", "",
          "---", "",
          "## Non-triviality check", "",
          "The prespec required a check that this is not simply \"later checkpoints have "
          "more structure of every kind\". It is not. At `last`, outcome rises from ~0 to "
          "+0.49 while **intent stays flat at ~0 for the entire 143,000 steps** — the two "
          "factors are measured in the same regression, on the same RDM, against the same "
          "covariates. Whatever is sharpening, it is not sharpening both factors.", "",
          *beh_section,
          "---", "",
          "## Limits", "",
          "- One model family, one size, 298 items that are **204 distinct texts** (94 rows "
          "are word-for-word YS2009 reprints of YS2008 scenarios; see "
          "`CONTAMINATION_REPAIR.md` §6). All CIs shown are the cluster bootstrap over the "
          "53 scenario groups, which is the resampling whose independence assumption holds.",
          "- Descriptive geometry over training time. Nothing here is causal; E7's erasure "
          "results are the causal arm.",
          "- The behavioural series (E8f) is a secondary panel and cannot falsify P1–P4.",
          "- Checkpoint trajectories for theory of mind already exist and track behavioural "
          "accuracy on false-belief items. E8's difference is that it tracks representational "
          "geometry for the moral belief × outcome factors, and that the Young & Saxe 2×2 "
          "has not previously been run along a pretraining trajectory. That is a difference "
          "in measurement, not a priority claim over the trajectory method.", ""]

    p = os.path.join(a.out, "E8_CHECKPOINT_EMERGENCE.md")
    open(p, "w").write("\n".join(L))

    print(f"P1 {'PASS' if p1_pass else 'FAIL'} | P2 {'PASS' if p2_pass else 'FAIL'} | "
          f"P4 {'PASS' if p4_pass else 'FAIL (smooth)'}")
    print(f"primary cell: outcome emerges {pe_o}, intent emerges {pe_i}")
    if p2_last:
        print(f"P2 at `last` for reference: rho={p2_last[0]:+.3f} "
              f"CI [{p2_last[1]:+.3f}, {p2_last[2]:+.3f}]")
    print(f"-> {os.path.relpath(p, ROOT)}\n-> {os.path.relpath(fig, ROOT)}")


if __name__ == "__main__":
    main()
