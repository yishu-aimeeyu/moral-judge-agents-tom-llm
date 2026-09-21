#!/usr/bin/env python3
"""E8a -- the five predictor (hypothesis) matrices for the checkpoint trajectory. [CPU]

WHY THIS EXISTS AS ITS OWN SCRIPT. E8b-E8e each need the same five 298x298 design matrices in
the same item order, across ~26 checkpoints x 12 cells. Rebuilding them inside each consumer
invites the failure where two stages disagree about row order and the regression silently fits
a permuted design. So the matrices are built once, written to disk with the item order beside
them, and every later stage loads `load_predictors()` from here rather than recomputing.

WHAT IS AND IS NOT A COVARIATE. `intent`, `outcome` and `interaction` are the factors of
interest. `scenario` and `lexical` are covariates and nothing in E8 gates on them -- E1's P4
gated on the lexical predictor and excluded 36/36 cells, because in these stimuli the factors
are lexically realised (within a scenario group the four conditions differ by exactly the
clause that realises the manipulation), so a word-overlap predictor necessarily outranks the
binary predictor expressing the same thing. See outputs/experiments/E8_PRESPEC.md section 2.1.

Item order is `belief -> outcome -> scenario_id`, fixed in the prespec, so that `intent` reads
as two blocks on the diagonal and `outcome` as a checkerboard inside them. The order is a
presentational choice for the heatmaps and cannot change any correlation, but it is written to
`item_order.csv` because the downstream RDMs must be built in it.

Outputs -> outputs/rsa/predictors/
  intent.npy outcome.npy interaction.npy scenario.npy lexical.npy   (298x298 float32)
  item_order.csv            the row/column order, with the labels that produced it
  predictor_correlations.csv  pairwise Spearman on the lower triangle
  predictor_matrices.png    all five as heatmaps, block boundaries drawn
"""
import argparse
import csv
import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PRED_DIR = os.path.join(ROOT, "outputs", "rsa", "predictors")

# Factors of interest first, then the two covariates, in the order they are reported.
PREDICTORS = ["intent", "outcome", "interaction", "scenario", "lexical"]
COVARIATES = ["scenario", "lexical"]


def _load(fname, modname):
    """Import a sibling script whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sorted_rows(master_csv):
    """Master rows in `belief -> outcome -> scenario_id` order.

    `guilty` before `innocent` and `harm` before `no_harm` is arbitrary but fixed: what matters
    is that every E8 stage uses this one function, not that the blocks appear in any particular
    corner of the figure.
    """
    rows = list(csv.DictReader(open(master_csv)))
    key = lambda r: (r["intent_label"] != "guilty",
                     r["outcome_label"] != "harm",
                     r["scenario_id"])
    return sorted(rows, key=key)


def build(rows):
    """The five matrices, all 298x298, all zero on the diagonal."""
    rsa = _load("24_rsa_cka.py", "rsa_24")
    vp = _load("61_rdm_variance_partition.py", "vp_61")

    intent = np.array([1 if r["intent_label"] == "guilty" else 0 for r in rows])
    outcome = np.array([1 if r["outcome_label"] == "harm" else 0 for r in rows])
    inter = intent * outcome
    scen = np.array([r.get("scenario_group") or r["scenario_id"] for r in rows])

    mats = {
        "intent": (intent[:, None] != intent[None, :]).astype(float),
        "outcome": (outcome[:, None] != outcome[None, :]).astype(float),
        # the treatment-coded product term, matching E1's `interaction` exactly
        "interaction": np.abs(inter[:, None] - inter[None, :]).astype(float),
        "scenario": (scen[:, None] != scen[None, :]).astype(float),
        # tfidf_word_1_2 cosine distance, same vectoriser settings as 21_surface_baseline.py
        "lexical": vp.lexical_rdm([r["text"] for r in rows]),
    }
    for m in mats.values():
        np.fill_diagonal(m, 0.0)
    return mats, rsa.upper


def correlations(mats, upper):
    """Pairwise Spearman between predictors, on the shared upper triangle."""
    from scipy.stats import spearmanr
    vecs = {k: upper(v) for k, v in mats.items()}
    out = []
    names = PREDICTORS
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = float(spearmanr(vecs[a], vecs[b]).statistic)
            out.append({"predictor_a": a, "predictor_b": b, "spearman_r": round(r, 4),
                        "r_squared": round(r * r, 4),
                        "a_is_covariate": a in COVARIATES,
                        "b_is_covariate": b in COVARIATES})
    return out


def block_edges(rows):
    """Indices where the intent block and the intent x outcome sub-blocks change."""
    intent_key = [r["intent_label"] for r in rows]
    cell_key = [(r["intent_label"], r["outcome_label"]) for r in rows]
    big = [i for i in range(1, len(rows)) if intent_key[i] != intent_key[i - 1]]
    small = [i for i in range(1, len(rows)) if cell_key[i] != cell_key[i - 1]]
    return big, small


def make_figure(mats, rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    big, small = block_edges(rows)
    n = len(rows)
    titles = {
        "intent": "intent (belief factor)\nblock diagonal",
        "outcome": "outcome\ncheckerboard",
        "interaction": "interaction\n(intent x outcome cell)",
        "scenario": "scenario identity\nCOVARIATE",
        "lexical": "lexical (TF-IDF cosine)\nCOVARIATE",
    }
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    for ax, name in zip(axes, PREDICTORS):
        M = mats[name]
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=max(M.max(), 1e-9), interpolation="nearest")
        for e in small:
            ax.axhline(e - 0.5, color="#4fc3f7", lw=0.5, alpha=0.65)
            ax.axvline(e - 0.5, color="#4fc3f7", lw=0.5, alpha=0.65)
        for e in big:
            ax.axhline(e - 0.5, color="#ffffff", lw=1.6)
            ax.axvline(e - 0.5, color="#ffffff", lw=1.6)
        ax.set_title(titles[name], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    axes[0].set_ylabel(f"{n} items, sorted belief -> outcome -> scenario_id", fontsize=9)
    fig.suptitle("E8a predictor matrices  (white = intent block edge, blue = intent x outcome "
                 "cell edge)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(outdir, "predictor_matrices.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master",
                                                  "moral_2x2_master.csv"))
    ap.add_argument("--out", default=PRED_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    rows = sorted_rows(a.csv)
    if a.dry_run:
        print("=== E8a PREDICTOR PLAN (dry-run) ===")
        print(f"items={len(rows)}  predictors={PREDICTORS}  covariates={COVARIATES}")
        print(f"order  = belief -> outcome -> scenario_id")
        print(f"first  = {rows[0]['story_id']}   last = {rows[-1]['story_id']}")
        print(f"out    = {a.out}")
        return

    guard = _load("_run_guard.py", "run_guard")
    guard.require_clean_tree(os.path.basename(__file__), allow_dirty=a.allow_dirty)

    os.makedirs(a.out, exist_ok=True)
    mats, upper = build(rows)

    with open(os.path.join(a.out, "item_order.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["order", "story_id", "intent_label",
                                          "outcome_label", "scenario_id", "scenario_group",
                                          "condition"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"order": i, "story_id": r["story_id"],
                        "intent_label": r["intent_label"],
                        "outcome_label": r["outcome_label"],
                        "scenario_id": r["scenario_id"],
                        "scenario_group": r.get("scenario_group", ""),
                        "condition": r["condition"]})

    for name, M in mats.items():
        np.save(os.path.join(a.out, f"{name}.npy"), M.astype(np.float32))

    cors = correlations(mats, upper)
    with open(os.path.join(a.out, "predictor_correlations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cors[0].keys()))
        w.writeheader(); w.writerows(cors)

    fig_path = make_figure(mats, rows, a.out)

    io = next(c for c in cors
              if {c["predictor_a"], c["predictor_b"]} == {"intent", "outcome"})
    print(f"\n=== E8a: {len(rows)} items, {len(PREDICTORS)} predictors ===")
    print(f"item order: belief -> outcome -> scenario_id "
          f"({rows[0]['story_id']} ... {rows[-1]['story_id']})")
    print("\npairwise Spearman on the upper triangle:")
    for c in cors:
        flag = "  (covariate pair)" if c["a_is_covariate"] and c["b_is_covariate"] else ""
        print(f"  {c['predictor_a']:12} x {c['predictor_b']:12} "
              f"r={c['spearman_r']:+.4f}  r2={c['r_squared']:.4f}{flag}")
    print(f"\nORTHOGONALITY CHECK (prespec 2.1): intent x outcome r = {io['spearman_r']:+.4f}, "
          f"shared variance {io['r_squared'] * 100:.2f}%")
    print(f"-> {a.out}\n-> {os.path.relpath(fig_path, ROOT)}")


if __name__ == "__main__":
    main()
