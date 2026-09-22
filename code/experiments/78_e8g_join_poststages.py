#!/usr/bin/env python3
"""E8g join -- OLMo-2-7B pretraining trajectory and post-training stages on one axis. [CPU]

The design asks for the pretraining trajectory to be joined to the existing
`checkpoint_dissection.csv` stages (base -> SFT -> DPO -> Instruct). That CSV holds
BEHAVIOURAL coefficients only, so joining it to a representational beta would put two
different quantities on one line. Instead the post-training stages are put through the same
regression E8d ran on the pretraining checkpoints -- same predictors, same item order, same
`fit_cell()` from `74_e8_checkpoint_betas.py`, same relative depths -- using the activations
already cached in `outputs/acts/` for those four models. The representational series is then
genuinely continuous, and the behavioural coefficients from `checkpoint_dissection.csv` are
drawn in a separate panel, only where they clear the engagement floor.

TWO DIFFERENCES BETWEEN THE HALVES OF THE AXIS, stated rather than hidden:
  * The stage activations were extracted in July/August at float16 (the extractor's default
    then); the E8 checkpoints at float32. Both went through `01_extract_activations.py` on
    raw text, so the read positions and pooling are identical.
  * "base" is the released OLMo-2-1124-7B, which is stage 1 *plus* the stage-2 annealing
    mix. It is not the same weights as the last stage-1 revision (step 928,646), so the gap
    between the last pretraining point and "base" spans stage 2, which E8g did not sample.

The pretraining half has no step-0 gate (amendment 3): OLMo-2 publishes no random-init
checkpoint.

Outputs -> outputs/rsa/checkpoint_betas_OLMo-2-1124-7B_poststages.csv
           outputs/rsa/checkpoint_trajectory_OLMo-2-1124-7B_joined.png
"""
import argparse
import csv
import importlib.util
import os
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")
ACTS = os.path.join(ROOT, "outputs", "acts")
DISSECTION = os.path.join(ROOT, "outputs", "experiments", "checkpoint_dissection.csv")
BEH_PRE = os.path.join(RSA_DIR, "checkpoint_behavioral_OLMo-2-1124-7B.csv")

MODEL = "OLMo-2-1124-7B"
STAGES = [("base", "OLMo-2-1124-7B"), ("SFT", "OLMo-2-1124-7B-SFT"),
          ("DPO", "OLMo-2-1124-7B-DPO"), ("Instruct", "OLMo-2-1124-7B-Instruct")]
REL_DEPTHS = [0.2, 0.4, 0.6, 0.8, 1.0]      # E8 set minus the excluded 0.0 (amendment 2.1)
POSITIONS = ["belief_last", "last"]
PLOT_DEPTH = "0.8"
ENGAGEMENT_FLOOR = 0.2191


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E8 = _load("74_e8_checkpoint_betas.py", "e8_74")


def fit_task(args):
    stage, tag, position, layer, req, n_stack, acts, mats, groups, n_perm, n_boot = args
    n = acts.shape[0]
    T, iu = E8.pair_index_table(n)
    up = {k: v[iu] for k, v in mats.items()}
    within = up["scenario"] == 0
    masks = {s: (within if s == "within_scenario" else np.ones(len(within), bool))
             for s in E8.SPECS}
    res = E8.fit_cell(acts, up, mats, masks, n_perm, n_boot, groups, T, iu, E8.SEED)
    rows = []
    for spec, names in E8.SPECS.items():
        b_out, b_int = res[spec]["outcome"]["beta"], res[spec]["intent"]["beta"]
        share = E8.outcome_share(b_out, b_int)
        for nm in names:
            r = res[spec][nm]
            rows.append({
                "model": MODEL, "revision": stage, "stage": stage, "source_tag": tag,
                "position": position, "layer": layer,
                "rel_depth": round(layer / (n_stack - 1), 4), "rel_depth_requested": req,
                "spec": spec, "predictor": nm, "beta": round(r["beta"], 5),
                "boot_ci_lo": round(r["boot_lo"], 5), "boot_ci_hi": round(r["boot_hi"], 5),
                "cluster_ci_lo": round(r["cluster_lo"], 5),
                "cluster_ci_hi": round(r["cluster_hi"], 5),
                "null_q975_abs": round(r["null_q975_abs"], 5),
                "p_perm": round(r["p_perm"], 5),
                "inside_null": bool(abs(r["beta"]) <= r["null_q975_abs"]),
                "vif": round(r["vif"], 3),
                "outcome_share": round(share, 4) if share == share else "",
                "extract_dtype": "float16",
            })
    print(f"  {stage:8} {position:12} L{layer}", flush=True)
    return rows


def compute(out_csv, n_perm, n_boot, jobs):
    mats, order = E8.load_predictors()
    order_ids = [r["story_id"] for r in order]
    groups = np.array([r["scenario_group"] or r["scenario_id"] for r in order])
    tasks = []
    for stage, tag in STAGES:
        d = np.load(os.path.join(ACTS, f"{tag}.npz"), allow_pickle=True)
        sids = [str(s) for s in d["story_id"]]
        pos_of = {s: i for i, s in enumerate(sids)}
        reorder = np.array([pos_of[s] for s in order_ids])
        for position in POSITIONS:
            A = d[position]
            n_stack = A.shape[1]
            for req in REL_DEPTHS:
                L = int(round(req * (n_stack - 1)))
                tasks.append((stage, tag, position, L, req, n_stack,
                              A[reorder, L, :].astype(np.float64), mats, groups,
                              n_perm, n_boot))
        print(f"loaded {tag}", flush=True)
    with Pool(jobs) as p:
        chunks = p.map(fit_task, tasks)
    rows = [r for c in chunks for r in c]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out_csv, ROOT)} ({len(rows)} rows)")


def make_figure(pre_csv, post_csv, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = [r for r in csv.DictReader(open(pre_csv))
           if r["spec"] == "primary" and r["rel_depth_requested"] == PLOT_DEPTH]
    post = [r for r in csv.DictReader(open(post_csv))
            if r["spec"] == "primary" and str(float(r["rel_depth_requested"])) == PLOT_DEPTH]
    beh = {r["stage"]: r for r in csv.DictReader(open(DISSECTION))
           if r["family"] == "OLMo-2-7B"}
    beh_pre = (sorted(csv.DictReader(open(BEH_PRE)), key=lambda r: int(r["step"]))
               if os.path.exists(BEH_PRE) else [])

    steps = sorted({int(r["step"]) for r in pre})
    xmax = np.log10(steps[-1])
    stage_x = {s: xmax + 0.9 * (i + 1) for i, (s, _) in enumerate(STAGES)}

    colors = {"intent": "#1f77b4", "outcome": "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for col, pos in enumerate(["belief_last", "last"]):
        ax = axes[0, col]
        for pred in ("intent", "outcome"):
            a = sorted([r for r in pre if r["position"] == pos and r["predictor"] == pred],
                       key=lambda r: int(r["step"]))
            b = [next(r for r in post if r["position"] == pos and r["predictor"] == pred
                      and r["stage"] == s) for s, _ in STAGES]
            xa = [np.log10(int(r["step"])) for r in a]
            xb = [stage_x[r["stage"]] for r in b]
            ya = [float(r["beta"]) for r in a]
            yb = [float(r["beta"]) for r in b]
            ax.plot(xa, ya, "-o", color=colors[pred], lw=2.2, ms=4, label=f"β {pred}")
            ax.fill_between(xa, [float(r["cluster_ci_lo"]) for r in a],
                            [float(r["cluster_ci_hi"]) for r in a], color=colors[pred],
                            alpha=0.18, lw=0)
            ax.plot(xb, yb, "-s", color=colors[pred], lw=2.2, ms=5)
            ax.errorbar(xb, yb, yerr=[[y - float(r["cluster_ci_lo"]) for y, r in zip(yb, b)],
                                      [float(r["cluster_ci_hi"]) - y for y, r in zip(yb, b)]],
                        fmt="none", ecolor=colors[pred], alpha=0.6, capsize=3)
            # dashed bridge: stage-2 annealing was not sampled
            ax.plot([xa[-1], xb[0]], [ya[-1], yb[0]], ":", color=colors[pred], lw=1.2)
            for r, x in list(zip(a, xa)) + list(zip(b, xb)):
                if bool(r["inside_null"] == "True"):
                    ax.plot(x, float(r["beta"]), "o", mfc="white", mec=colors[pred], ms=5)
        ax.axhline(0, color="k", lw=0.7)
        ax.axvline(xmax + 0.45, color="#888", lw=1)
        ax.set_title({"belief_last": "read at the end of the BELIEF clause",
                      "last": "read at the END OF THE STORY"}[pos], fontsize=11)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="upper left")
        ax.text(0.99, 0.02, "open markers = inside the permutation null\n"
                "no step-0 gate: OLMo-2 has no random-init checkpoint",
                transform=ax.transAxes, fontsize=7, ha="right", color="#555")

        axb = axes[1, col]
        # one behavioural sequence along the whole axis: stage-1 revisions (if scored), then
        # the post-training stages from checkpoint_dissection.csv
        seq = [(np.log10(int(r["step"])), f"step {int(r['step']):,}", r) for r in beh_pre]
        seq += [(stage_x[s], s, beh[s]) for s, _ in STAGES if s in beh]
        seq.sort(key=lambda t: t[0])
        gi = next((i for i, (_, _, r) in enumerate(seq)
                   if float(r["rating_std"]) >= ENGAGEMENT_FLOOR), None)
        if gi is not None:
            # section 7: coefficients only from the first gradeable point onward, and only
            # at points that are themselves gradeable
            shown = [(x, r) for x, _, r in seq[gi:]
                     if float(r["rating_std"]) >= ENGAGEMENT_FLOOR]
            for pred, key in (("intent", "b_intent"), ("outcome", "b_outcome")):
                axb.plot([x for x, _ in shown], [float(r[key]) for _, r in shown], "-s",
                         color=colors[pred], lw=2, ms=4, label=f"behavioural {key}")
            xg = seq[gi][0]
            axb.axvline(xg - 0.12, color="#2a2", ls="--", lw=1)
            axb.text(xg - 0.1, 0.9, f"first gradeable:\n{seq[gi][1]}", fontsize=8,
                     color="#2a2", transform=axb.get_xaxis_transform(), va="top")
        xg_edge = (seq[gi][0] - 0.12) if gi is not None else stage_x[STAGES[-1][0]] + 0.3
        axb.axvspan(np.log10(steps[0]) - 0.2, xg_edge, color="#eee", zorder=0)
        axb.text(np.log10(steps[0]), 0.35, "below the engagement floor" +
                 ("" if beh_pre else " / stage 1 not scored") +
                 " —\nleft blank, not interpolated", fontsize=8, color="#777",
                 transform=axb.get_xaxis_transform())
        if seq:
            tw = axb.twinx()
            tw.plot([x for x, _, _ in seq], [float(r["rating_std"]) for _, _, r in seq], ":.",
                    color="#888", lw=1, ms=4)
            tw.axhline(ENGAGEMENT_FLOOR, color="#888", lw=0.8, ls="--")
            tw.set_ylim(0, max(0.45, max(float(r["rating_std"]) for _, _, r in seq) * 1.1))
            tw.set_ylabel("rating_std (grey)\nfloor dashed", fontsize=7, color="#666")
            tw.tick_params(labelsize=7, colors="#666")
        axb.axhline(0, color="k", lw=0.7)
        axb.axvline(xmax + 0.45, color="#888", lw=1)
        axb.set_ylabel("behavioural coefficient" if col == 0 else "")
        axb.legend(fontsize=8, loc="upper left")
        axb.grid(alpha=0.2)

    ticks = [np.log10(s) for s in steps] + list(stage_x.values())
    labels = [f"{s:,}" for s in steps] + [s for s, _ in STAGES]
    for ax in axes[1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("stage-1 pretraining step (log)            |   post-training stage")
    axes[0, 0].set_ylabel("standardised β  (RDM regression)")
    fig.suptitle(f"E8g — OLMo-2-7B from pretraining through post-training, rel. depth "
                 f"{PLOT_DEPTH}\nshading / bars = cluster bootstrap 95% CI; dotted = stage-2 "
                 f"annealing, not sampled", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=E8.N_PERM)
    ap.add_argument("--boot", type=int, default=E8.N_BOOT)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    post_csv = os.path.join(RSA_DIR, f"checkpoint_betas_{MODEL}_poststages.csv")
    pre_csv = os.path.join(RSA_DIR, f"checkpoint_betas_{MODEL}.csv")
    out_png = os.path.join(RSA_DIR, f"checkpoint_trajectory_{MODEL}_joined.png")
    if not a.plot_only:
        guard = _load("_run_guard.py", "run_guard")
        guard.require_clean_tree(os.path.basename(__file__),
                                 extra_deps=("code/experiments/74_e8_checkpoint_betas.py",),
                                 allow_dirty=a.allow_dirty)
        compute(post_csv, a.perm, a.boot, a.jobs)
    if os.path.exists(pre_csv):
        print(f"-> {os.path.relpath(make_figure(pre_csv, post_csv, out_png), ROOT)}")
    else:
        print(f"(figure waits for {os.path.relpath(pre_csv, ROOT)})")


if __name__ == "__main__":
    main()
