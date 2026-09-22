#!/usr/bin/env python3
"""E8 figure -- the observed RDMs themselves, at a handful of checkpoints. [CPU]

E8d reports the RDMs only through their regression betas. This draws the matrices: the
intent and outcome predictors, then the model's RDM at six checkpoints, at both read
positions, at relative depth 0.8. Items are in the E8a order (belief -> outcome ->
scenario), so the four conditions are four blocks.

Each observed RDM is shown as within-matrix percentile ranks, because the regression is on
ranks and the raw 1 - r scale drifts by an order of magnitude across training. Below each
full matrix is its 4x4 condition average (mean rank over all off-diagonal item pairs in the
two blocks); scenario and lexical similarity dominate the full matrix, and the condition
average is where the intent and outcome structure can be seen by eye. Descriptive only; no
statistic here that is not already in checkpoint_betas*.csv.

Outputs -> outputs/rsa/rdm_snapshots_<model>.png
"""
import argparse
import csv
import os

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")
PRED_DIR = os.path.join(RSA_DIR, "predictors")
CKPT = os.path.join(ROOT, "outputs", "acts_ckpt")
ACTS = os.path.join(ROOT, "outputs", "acts")
REL_DEPTH = 0.8
POSITIONS = [("belief_last", "end of BELIEF clause"), ("last", "end of STORY")]

SNAPSHOTS = {
    "pythia-1.4b": [("step0", "step 0\n(random init)"), ("step128", "step 128"),
                    ("step1000", "step 1,000"), ("step4000", "step 4,000"),
                    ("step16000", "step 16,000"), ("step143000", "step 143,000\n(final)")],
    "OLMo-2-1124-7B": [("stage1-step150-tokens1B", "step 150\n(1B tok)"),
                       ("stage1-step600-tokens3B", "step 600\n(3B tok)"),
                       ("stage1-step2000-tokens9B", "step 2,000\n(9B tok)"),
                       ("stage1-step22000-tokens93B", "step 22,000\n(93B tok)"),
                       ("stage1-step928646-tokens3896B", "step 928,646\n(3.9T tok)"),
                       ("@Instruct", "Instruct\n(post-trained)")],
}


def build_rdm(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
    nrm[nrm == 0] = 1e-12
    C = (Xc / nrm) @ (Xc / nrm).T
    return 1.0 - np.clip(C, -1.0, 1.0)


def load(model, rev, position, order_ids):
    if rev == "@Instruct":
        d = np.load(os.path.join(ACTS, f"{model}-Instruct.npz"), allow_pickle=True)
        A = d[position]
        L = int(round(REL_DEPTH * (A.shape[1] - 1)))
    else:
        d = np.load(os.path.join(CKPT, f"{model}@{rev}.npz"), allow_pickle=True)
        A = d[position]
        li = [int(x) for x in d["layer_idx"]]
        n_stack = {"pythia-1.4b": 25, "OLMo-2-1124-7B": 33}[model]
        L = li.index(int(round(REL_DEPTH * (n_stack - 1))))
    pos_of = {str(s): i for i, s in enumerate(d["story_id"])}
    X = A[[pos_of[s] for s in order_ids], L, :].astype(np.float64)
    return build_rdm(X)


def pct_rank(R):
    iu = np.triu_indices(len(R), 1)
    out = np.full(R.shape, np.nan)
    r = rankdata(R[iu]) / len(iu[0])
    out[iu] = r
    out.T[iu] = r
    return out


def condition_mean(Rr, cond, labels):
    M = np.zeros((len(labels), len(labels)))
    for a, ca in enumerate(labels):
        for b, cb in enumerate(labels):
            blk = Rr[np.ix_(cond == ca, cond == cb)]
            M[a, b] = np.nanmean(blk)
    return M


def cached_rdms(model, ids):
    """Raw RDMs in the E8a item order, computed once per model.

    The OLMo Instruct activations are a 377 MB compressed file, so re-sorting the figure
    must not mean re-reading it.
    """
    path = os.path.join(RSA_DIR, f"rdm_snapshots_cache_{model}.npz")
    keys = [f"{pos}|{rev}" for pos, _ in POSITIONS for rev, _ in SNAPSHOTS[model]]
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if list(d["ids"]) == ids and all(k in d.files for k in keys):
            return {k: d[k] for k in keys}
    out = {}
    for k in keys:
        pos, rev = k.split("|")
        out[k] = load(model, rev, pos, ids).astype(np.float32)
        print(f"  {pos:12} {rev}", flush=True)
    np.savez_compressed(path, ids=np.array(ids), **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1.4b", choices=list(SNAPSHOTS))
    ap.add_argument("--sort", default="intent", choices=["intent", "outcome"],
                    help="which factor forms the two big halves of the item order")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(csv.DictReader(open(os.path.join(PRED_DIR, "item_order.csv"))))
    ids = [r["story_id"] for r in order]
    rdms = cached_rdms(a.model, ids)
    # E8a order is belief -> outcome -> scenario; the outcome-first order is a stable re-sort,
    # so scenario order inside each block is unchanged.
    if a.sort == "outcome":
        perm = np.array(sorted(range(len(order)), key=lambda i: (
            order[i]["outcome_label"] != "harm", order[i]["intent_label"] != "guilty")))
    else:
        perm = np.arange(len(order))
    cond = np.array([order[i]["condition"] for i in perm])
    labels = list(dict.fromkeys(cond))
    bounds = [i for i in range(1, len(cond)) if cond[i] != cond[i - 1]]
    starts = [0] + bounds
    ends = bounds + [len(cond)]
    preds = {n: np.load(os.path.join(PRED_DIR, f"{n}.npy"))[np.ix_(perm, perm)]
             for n in ("intent", "outcome")}
    snaps = SNAPSHOTS[a.model]

    ncol = 2 + len(snaps)
    fig, axes = plt.subplots(4, ncol, figsize=(2.35 * ncol, 10.4),
                             gridspec_kw={"height_ratios": [1, 0.62, 1, 0.62]})
    short = {"intentional": "int", "attempted": "att", "accidental": "acc",
             "neutral": "neu"}

    def mat(ax, M, title=None, first_col=False):
        ax.imshow(M, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="w", lw=0.6)
            ax.axvline(b - 0.5, color="w", lw=0.6)
        edges = starts + [len(cond) - 1]
        ax.set_xticks(edges)
        ax.set_xticklabels([f"s{e + 1}" for e in edges], fontsize=5, rotation=90)
        if first_col:
            ax.set_yticks([(s + e - 1) / 2 for s, e in zip(starts, ends)])
            ax.set_yticklabels([f"{short[l]}\ns{s + 1}–s{e}" for l, s, e in
                                zip(labels, starts, ends)], fontsize=6)
        else:
            ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8.5)

    def small(ax, M, lo=0.25, hi=0.75):
        # one fixed scale for every model panel, so step-0 noise is not stretched into contrast
        ax.imshow(M, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.3,
                        color="w" if M[i, j] < (lo + hi) / 2 else "k")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([short.get(l, l) for l in labels], fontsize=6.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([short.get(l, l) for l in labels], fontsize=6.5)

    for p, (pos, pos_title) in enumerate(POSITIONS):
        rf, rs = 2 * p, 2 * p + 1
        for c, n in enumerate(("intent", "outcome")):
            mat(axes[rf, c], preds[n], f"HYPOTHESIS\n{n}" if p == 0 else None,
                first_col=(c == 0))
            small(axes[rs, c], condition_mean(preds[n].astype(float), cond, labels), 0, 1)
        for k, (rev, lab) in enumerate(snaps):
            Rr = pct_rank(rdms[f"{pos}|{rev}"].astype(np.float64)[np.ix_(perm, perm)])
            mat(axes[rf, 2 + k], Rr, lab if p == 0 else None)
            small(axes[rs, 2 + k], condition_mean(Rr, cond, labels))
        axes[rf, 0].set_ylabel(f"read at\n{pos_title}", fontsize=9)
        axes[rs, 0].set_ylabel("condition\nmean", fontsize=8)

    fig.suptitle(f"{a.model} — model RDMs at relative depth {REL_DEPTH} across training, "
                 f"next to the intent and outcome hypotheses\n"
                 f"both axes = the 298 stimuli, sorted "
                 f"{'belief → outcome' if a.sort == 'intent' else 'outcome → belief'} → "
                 f"scenario (blocks: {', '.join(labels)}); model RDMs as percentile rank "
                 f"of 1 − r; bright = dissimilar", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RSA_DIR, f"rdm_snapshots_{a.model}_{a.sort}first.png")
    fig.savefig(out, dpi=150)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
