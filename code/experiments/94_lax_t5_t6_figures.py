#!/usr/bin/env python3
"""
94_lax_t5_t6_figures.py -- Lax 9/29: summary tables + figures for T5 (91) and T6 (92) outputs.

Best layer for T6 = the saved trajectory layer nearest the 02_probe last-token intent peak
(Qwen2.5-7B[-Instruct] L19 -> saved L18; OLMo-2-7B[-Instruct] L16). Llama-3.1-8B has no 02 probe run;
L16 (same relative depth as OLMo, both 32 layers) is used and flagged.

  .venv/bin/python code/experiments/94_lax_t5_t6_figures.py
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "lax_0929")
FAMILIES = {"Qwen2.5-7B": ("Qwen2.5-7B", "Qwen2.5-7B-Instruct"),
            "OLMo-2-7B": ("OLMo-2-1124-7B", "OLMo-2-1124-7B-Instruct"),
            "Llama-3.1-8B": ("Meta-Llama-3.1-8B", "Meta-Llama-3.1-8B-Instruct")}
BEST = {"Qwen2.5-7B": 18, "Qwen2.5-7B-Instruct": 18, "OLMo-2-1124-7B": 16, "OLMo-2-1124-7B-Instruct": 16,
        "Meta-Llama-3.1-8B": 16, "Meta-Llama-3.1-8B-Instruct": 16}
STAGES = ["background", "belief", "action", "outcome"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- T5
def t5():
    files = sorted(glob.glob(os.path.join(OUT, "T5", "*__layers.csv")))
    if not files:
        print("T5: no outputs")
        return
    L = pd.concat([pd.read_csv(f) for f in files])
    P = pd.concat([pd.read_csv(f.replace("__layers.csv", "__perm.csv")) for f in files])
    L["rel_depth"] = L.layer / L.groupby("model").layer.transform("max")
    L.to_csv(os.path.join(OUT, "template_heldout_layers.csv"), index=False)
    fixed = P[P.layer_kind == "fixed_0.65"].pivot_table(index=["model", "layer"], columns="measure",
                                                        values=["obs", "perm_p"]).reset_index()
    fixed.columns = ["_".join(c).strip("_") for c in fixed.columns]
    tf = L.groupby("model")[[c for c in L.columns if c.startswith("tfidf_")]].first().reset_index()
    peak = L[L.layer > 0].sort_values("construct").groupby("model").tail(1)[
        ["model", "layer", "within", "pair", "construct", "gap_construct"]].add_prefix("peak_").rename(
        columns={"peak_model": "model"})
    s = fixed.merge(tf, on="model").merge(peak, on="model")
    s.to_csv(os.path.join(OUT, "template_heldout_summary.csv"), index=False)
    P.to_csv(os.path.join(OUT, "template_heldout_perm.csv"), index=False)
    print(s.round(3).T.to_string())

    fams = [f for f, (b, i) in FAMILIES.items() if b in set(L.model) or i in set(L.model)]
    fig, axes = plt.subplots(2, len(fams), figsize=(5.5 * len(fams), 8), squeeze=False)
    for j, f in enumerate(fams):
        for m, ls in zip(FAMILIES[f], ("--", "-")):
            g = L[L.model == m]
            if g.empty:
                continue
            ax = axes[0, j]
            for col, c in (("within", "k"), ("pair", "tab:orange"), ("construct", "tab:blue")):
                ax.plot(g.rel_depth, g[col], ls, color=c, lw=1.2, label=f"{col} [{'base' if ls == '--' else 'instruct'}]")
            ax.axhline(g["tfidf_tfidf_word_1_2_within"].iloc[0], color="0.5", ls=":", lw=1, label="TF-IDF word (within)")
            ax.axhline(0.5, color="0.8", lw=0.6)
            ax.set_title(f"{f}: intent at last prompt token", fontsize=10)
            axes[1, j].plot(g.rel_depth, g.gap_construct, ls, color="tab:blue", label=f"within - construct [{'base' if ls == '--' else 'instruct'}]")
            axes[1, j].plot(g.rel_depth, g.gap_pair, ls, color="tab:orange", label=f"within - pair [{'base' if ls == '--' else 'instruct'}]")
        axes[1, j].axhline(0, color="k", lw=0.6)
        axes[1, j].set_xlabel("relative depth")
        axes[0, j].legend(fontsize=6)
        axes[1, j].legend(fontsize=6)
    axes[0, 0].set_ylabel("GroupKFold accuracy")
    axes[1, 0].set_ylabel("template-held-out gap")
    fig.suptitle("T5: intent decoding, trained on template set A, tested on disjoint set B (dashed = base)")
    fig.tight_layout()
    save(fig, "fig_T5_template_heldout")

    mats = sorted(glob.glob(os.path.join(OUT, "T5", "*__pair_matrices.npz")))
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4), squeeze=False)
    for ax, f in zip(axes[0], mats):
        d = np.load(f)
        m = os.path.basename(f).split("__")[0]
        Lf = int(s.loc[s.model == m, "layer"].iloc[0])
        im = ax.imshow(d[f"L{Lf}"], vmin=0.5, vmax=1, cmap="viridis")
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(d["templates"], rotation=90, fontsize=6)
        ax.set_yticklabels(d["templates"], fontsize=6)
        ax.set_title(f"{m} L{Lf}: train (row) x test (col)", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    save(fig, "fig_T5_pair_matrices")


# ------------------------------------------------------------------------- T6
def t6():
    files = sorted(glob.glob(os.path.join(OUT, "T6", "*__curve.csv")))
    if not files:
        print("T6: no outputs")
        return
    C = pd.concat([pd.read_csv(f) for f in files])
    Hh = pd.concat([pd.read_csv(f.replace("__curve.csv", "__hypotheses.csv")) for f in files])
    R = pd.concat([pd.read_csv(f.replace("__curve.csv", "__rsa.csv")) for f in files])
    N = pd.concat([pd.read_csv(f.replace("__curve.csv", "__n.csv")) for f in files])
    C.to_csv(os.path.join(OUT, "time_decoding_curves.csv"), index=False)
    Hh.to_csv(os.path.join(OUT, "time_decoding_hypotheses.csv"), index=False)
    R.to_csv(os.path.join(OUT, "time_decoding_rsa.csv"), index=False)
    N.to_csv(os.path.join(OUT, "time_decoding_n.csv"), index=False)
    hb = Hh[Hh.apply(lambda r: r.layer == BEST.get(r.model), axis=1)]
    hb.to_csv(os.path.join(OUT, "time_decoding_hypotheses_bestlayer.csv"), index=False)
    print(hb.pivot_table(index="model", columns="stat", values="value").round(4).to_string())

    nb = C.bin.max() + 1
    per = nb // 4
    fams = [f for f, (b, i) in FAMILIES.items() if b in set(C.model) or i in set(C.model)]
    fig, axes = plt.subplots(3, len(fams), figsize=(5.5 * len(fams), 11), squeeze=False)
    for j, f in enumerate(fams):
        for m, ls, col in zip(FAMILIES[f], ("--", "-"), ("tab:orange", "tab:blue")):
            Lb = BEST[m]
            for i, t in enumerate(("intent", "outcome")):
                g = C[(C.model == m) & (C.layer == Lb) & (C.target == t)].sort_values("bin")
                if g.empty:
                    continue
                ax = axes[i, j]
                ax.fill_between(g.bin, g.null_lo, g.null_hi, color=col, alpha=0.12)
                ax.plot(g.bin, g.acc, ls, color=col, lw=1.5, label=f"{m} L{Lb}")
                ax.plot(g.bin, g.surface_acc, ":", color="0.4", lw=1, label="TF-IDF (same text)" if ls == "-" else None)
                ax.set_ylim(0.35, 1.02)
                ax.set_title(f"{f}: {t}", fontsize=10)
            r = R[(R.model == m) & (R.layer == Lb)]
            for term, c2 in (("intent", "tab:green"), ("outcome", "tab:red"), ("interaction", "tab:purple")):
                g = r[r.term == term].sort_values("bin")
                if g.empty:
                    continue
                axes[2, j].plot(g.bin, g.beta, ls, color=c2, lw=1.2, label=f"{term} [{'base' if ls == '--' else 'instruct'}]")
                axes[2, j].fill_between(g.bin, g.lo, g.hi, color=c2, alpha=0.1)
        axes[2, j].set_title(f"{f}: 4x4 condition RDM ~ model RDMs (beta)", fontsize=10)
        axes[2, j].axhline(0, color="k", lw=0.6)
        for i in range(3):
            for k in range(1, 4):
                axes[i, j].axvline(k * per - 0.5, color="0.7", lw=0.6)
            axes[i, j].set_xticks([k * per + per / 2 - 0.5 for k in range(4)])
            axes[i, j].set_xticklabels(STAGES, fontsize=8)
            axes[i, j].legend(fontsize=6)
    fig.suptitle("T6: decoding over time on clause_offsets (shaded = within-scenario shuffle null 95%; dashed = base)")
    fig.tight_layout()
    save(fig, "fig_T6_time_decoding")

    tgs = sorted(glob.glob(os.path.join(OUT, "T6", "*__tg.npz")))
    fig, axes = plt.subplots(2, len(tgs), figsize=(3.3 * len(tgs), 6.6), squeeze=False)
    for j, f in enumerate(tgs):
        m = os.path.basename(f).split("__")[0]
        d = np.load(f)
        for i, t in enumerate(("intent", "outcome")):
            k = f"L{BEST[m]}_{t}"
            if k not in d:
                continue
            ax = axes[i, j]
            ax.imshow(d[k], origin="lower", vmin=0.3, vmax=1, cmap="RdBu_r")
            for b in range(1, 4):
                ax.axvline(b * per - 0.5, color="k", lw=0.4)
                ax.axhline(b * per - 0.5, color="k", lw=0.4)
            ax.set_title(f"{m} L{BEST[m]} {t}", fontsize=7)
            ax.set_xlabel("test bin", fontsize=7)
            ax.set_ylabel("train bin", fontsize=7)
    fig.suptitle("T6: temporal generalization (stages: background | belief | action | outcome)")
    fig.tight_layout()
    save(fig, "fig_T6_temporal_generalization")


if __name__ == "__main__":
    t5()
    t6()
