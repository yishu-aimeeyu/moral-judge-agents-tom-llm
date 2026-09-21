#!/usr/bin/env python3
"""E8c + E8d -- the step-0 gate, and standardised betas along the training trajectory. [CPU]

For every checkpoint x layer x position, regress the observed 298x298 RDM on the five E8a
predictors and report standardised betas with a permutation null and a bootstrap CI. E8c is
the blocking gate at `step0`; E8d is the trajectory. They live in one script because the gate
is a property of the same fit, and splitting them would let the trajectory be computed before
the gate had been read.

  ---- E8c, the gate -------------------------------------------------------------------
  A randomly initialised network cannot have learned the moral belief x outcome factors. If
  it scores on them, the distance metric is reading an input-side artifact -- sequence
  length, tokenisation of the manipulated clause -- and every later checkpoint would inherit
  it. The rule, its Bonferroni correction over cells, and the consequence of failure are
  fixed in E8_PRESPEC.md section 4. This script refuses to write E8d's outputs when the gate
  fails, rather than leaving that to the reader's discipline.

  Expected at step 0 and NOT a failure: large `lexical` and `scenario` betas. A random
  network is close to a random projection, and random projections approximately preserve
  input similarity, so token-overlap structure survives into the hidden states. The gate is
  on `intent` and `outcome` only.

  ---- E8d, the trajectory -------------------------------------------------------------
  Three specifications per cell, all pre-registered in amendment 1:
    primary          all five predictors
    no_lexical       lexical dropped -- the sensitivity that E8a's within-scenario
                     correlations force, since lexical is near-collinear with outcome
                     (r=+0.82) but not with intent (r=+0.13) among the 821 within-scenario
                     pairs, so "same covariate" is not "same treatment"
    within_scenario  only those 821 pairs, scenario dropped as constant. Expected to be
                     undecidable between outcome and lexical. Reported; gates nothing.

  Primary quantity is the OUTCOME SHARE |b_out| / (|b_out| + |b_int|), a bounded monotone
  remap of the b_outcome:b_intent ratio. The raw ratio is 0/0 at step 0 and near-singular
  through early training, which is exactly where the dynamics are; the share is finite
  everywhere. The raw ratio is reported too, blank wherever its denominator is inside its
  own null, because a ratio with a null denominator is not a measurement.

WHY THE PERMUTATION IS CHEAP. Permuting stimulus identity permutes rows and columns of the
RDM together, which permutes the multiset of upper-triangle entries without changing it. So
the rank-transform is computed once and the permuted response is a gather through a
precomputed pair-index table, instead of rebuilding and re-ranking a 44,253-vector 1000 times
per cell. This is an exact identity, not an approximation.

Outputs -> outputs/rsa/
  checkpoint_betas.csv        every cell x spec x predictor
  checkpoint_betas.png        E8d figure, step-0 null band shaded
  E8_STEP0_GATE.md            the E8c verdict, written whether it passes or fails
"""
import argparse
import csv
import glob
import importlib.util
import os
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")
PRED_DIR = os.path.join(RSA_DIR, "predictors")
ACTS_DIR = os.path.join(ROOT, "outputs", "acts_ckpt")

# E8_PRESPEC.md sections 3-5. Constants, not flags: a sweep must not quietly differ from
# the pre-registered one.
FACTORS = ["intent", "outcome", "interaction"]
COVARIATES = ["scenario", "lexical"]
ALL_PREDICTORS = FACTORS + COVARIATES
SPECS = {
    "primary": ALL_PREDICTORS,
    "no_lexical": FACTORS + ["scenario"],
    "within_scenario": FACTORS + ["lexical"],
}
POSITIONS = ["belief_last", "last"]
PRIMARY_POSITION = "belief_last"
PRIMARY_REL_DEPTH = 0.8
# The depths E8b was asked for. Used only to name the primary cell by the depth the prespec
# fixed, since the realised depth of a kept layer is quantised by the model's stack size.
LAYERS_REL_EXPECTED = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Amendment 2.1. At a single-token read position the layer-0 activation is a token-embedding
# lookup, and every one of the 298 vignettes ends in "." (295) or ")." (3): 2 distinct values
# across 298 items, at every checkpoint. The RDM has two entries, the null is atomic, and the
# fit is to "do these items end in the same token". Extracted, then excluded here so the
# decision is visible in the code rather than hidden in the sampling.
EXCLUDED_REL_DEPTHS = [0.0]
GATE_ALPHA = 0.05            # family-wise, Bonferroni over cells; section 4
N_PERM_GATE = 2000
N_PERM = 1000
N_BOOT = 1000
SEED = 0


def _load(fname, modname, subdir="experiments"):
    base = HERE if subdir == "experiments" else os.path.join(ROOT, "code")
    spec = importlib.util.spec_from_file_location(modname, os.path.join(base, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ small numerics
def rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def zrank(v):
    r = rank(v)
    sd = r.std()
    return (r - r.mean()) / (sd if sd > 1e-12 else 1.0)


def zcol(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


def ols_betas(X, y):
    """Standardised betas via the normal equations. X columns are already z-scored."""
    XtX = X.T @ X
    try:
        return np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, y, rcond=None)[0]


def vifs(X):
    """Variance inflation per column, on the z-scored design (no intercept column)."""
    out = []
    k = X.shape[1]
    for i in range(k):
        other = np.delete(X, i, axis=1)
        b = np.linalg.lstsq(other, X[:, i], rcond=None)[0]
        resid = X[:, i] - other @ b
        r2 = 1.0 - float(resid @ resid) / float(X[:, i] @ X[:, i])
        out.append(1.0 / max(1.0 - r2, 1e-12))
    return out


# ------------------------------------------------------------------ predictors / RDMs
def load_predictors():
    """The five E8a matrices and the item order they were built in."""
    order = list(csv.DictReader(open(os.path.join(PRED_DIR, "item_order.csv"))))
    mats = {n: np.load(os.path.join(PRED_DIR, f"{n}.npy")).astype(np.float64)
            for n in ALL_PREDICTORS}
    return mats, order


def pair_index_table(n):
    """`T[i,j]` = position of pair (i,j) in the upper-triangle vector. Symmetric."""
    iu = np.triu_indices(n, 1)
    T = np.zeros((n, n), dtype=np.int64)
    T[iu] = np.arange(len(iu[0]))
    T = T + T.T
    return T, iu


def build_rdm(X):
    """1 - Pearson r across features. Identical to `24_rsa_cka.build_rdm`."""
    Xc = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
    nrm[nrm == 0] = 1e-12
    C = (Xc / nrm) @ (Xc / nrm).T
    return 1.0 - np.clip(C, -1.0, 1.0)


def permute_within_groups(base, groups, rng):
    out = np.array(base, copy=True)
    for g in np.unique(groups):
        w = np.where(groups == g)[0]
        out[w] = out[rng.permutation(w)]
    return out


# ------------------------------------------------------------------ one cell, all specs
def fit_cell(acts, up, mats, masks, n_perm, n_boot, groups, T, iu, seed):
    """All three specs for one (layer, position) cell: betas, permutation null, bootstrap CI.

    The three specs share a cell, so the expensive parts are done once and reused:

      * The bootstrap resamples ITEMS (pairs are not independent, so pairs cannot be
        resampled), which means a fresh 298x298 RDM per draw. That RDM does not depend on
        which predictors are in the design, so it is built once per draw and all three
        specs are fitted against it.
      * The permutation null gathers rather than rebuilds. Permuting stimulus identity
        permutes rows and columns of the RDM together, which permutes the multiset of
        upper-triangle entries without changing it, so for the all-pairs specs the ranked
        response under a permutation is exactly the precomputed ranks reindexed. That is an
        identity, not an approximation. The within-scenario spec re-ranks, because there the
        mask selects a subset and rank-within-subset is not rank-then-subset.
    """
    n = acts.shape[0]
    rdm = build_rdm(acts)
    y_vec = rdm[iu]
    yr_all = zrank(y_vec)          # ranks over all pairs, reusable under permutation
    base = np.arange(n)

    fitted = {}
    for spec, names in SPECS.items():
        mask = masks[spec]
        all_pairs = mask.all()
        y = yr_all if all_pairs else zrank(y_vec[mask])
        X = np.column_stack([zrank(up[nm][mask]) for nm in names])
        fitted[spec] = dict(names=names, mask=mask, all_pairs=all_pairs,
                            X=X, beta=ols_betas(X, y), vif=vifs(X),
                            null=np.empty((n_perm, len(names))),
                            boots=np.empty((n_boot, len(names))))

    # ---- permutation null
    rng = np.random.default_rng(seed)
    for k in range(n_perm):
        p = permute_within_groups(base, groups, rng)
        idx = T[p[iu[0]], p[iu[1]]]
        for spec, f in fitted.items():
            if f["all_pairs"]:
                yp = yr_all[idx]                       # gather: exact, no re-rank needed
            else:
                yp = zrank(y_vec[idx][f["mask"]])      # rank within the subset
            f["null"][k] = ols_betas(f["X"], yp)

    # ---- two bootstraps, per amendment 2.2. The pre-registered one resamples items; the
    # cluster one resamples the 53 scenario groups, which is the resampling whose
    # independence assumption is actually true here (94 of the 298 rows are exact reprints
    # of another row, and the conditions within a scenario are not independent either).
    # Where the two disagree about whether a beta clears its band, the cluster one governs.
    clusters = [np.where(groups == g)[0] for g in np.unique(groups)]
    draws = {}
    for kind in ("item", "cluster"):
        rng = np.random.default_rng(seed + (1 if kind == "item" else 2))
        acc = {spec: np.empty((n_boot, len(f["names"]))) for spec, f in fitted.items()}
        ok = 0
        for _ in range(n_boot):
            if kind == "item":
                sel = rng.integers(0, n, n)
            else:
                pick = rng.integers(0, len(clusters), len(clusters))
                sel = np.concatenate([clusters[c] for c in pick])
            yb_full, cols, iu_b = resampled_design(acts, mats, sel)
            mb_within = cols["scenario"] == 0
            if mb_within.sum() < 30:
                continue
            staged, good = {}, True
            for spec, f in fitted.items():
                mb = np.ones(len(yb_full), dtype=bool) if f["all_pairs"] else mb_within
                Xb = np.column_stack([zrank(cols[nm][mb]) for nm in f["names"]])
                if min(Xb.std(axis=0)) < 1e-12:
                    good = False
                    break
                staged[spec] = ols_betas(Xb, zrank(yb_full[mb]))
            if not good:
                continue
            for spec, b in staged.items():
                acc[spec][ok] = b
            ok += 1
        draws[kind] = (acc, ok)

    out = {}
    for spec, f in fitted.items():
        out[spec] = {}
        for i, nm in enumerate(f["names"]):
            nl = f["null"][:, i]
            rec = dict(
                beta=float(f["beta"][i]), vif=float(f["vif"][i]),
                null_mean=float(nl.mean()),
                null_q975_abs=float(np.percentile(np.abs(nl), 97.5)),
                null_lo=float(np.percentile(nl, 2.5)),
                null_hi=float(np.percentile(nl, 97.5)),
                p_perm=float((np.sum(np.abs(nl) >= abs(f["beta"][i])) + 1) / (n_perm + 1)),
            )
            for kind, prefix in (("item", "boot"), ("cluster", "cluster")):
                acc, ok = draws[kind]
                bl = acc[spec][:ok, i] if ok else np.array([np.nan])
                rec[f"{prefix}_lo"] = float(np.nanpercentile(bl, 2.5))
                rec[f"{prefix}_hi"] = float(np.nanpercentile(bl, 97.5))
                rec[f"n_{prefix}_ok"] = ok
            out[spec][nm] = rec
    return out


def resampled_design(acts, mats, sel):
    """RDM and predictor columns for one bootstrap draw. The RDM does not depend on the
    design, so it is built once and reused across the three specs."""
    m = len(sel)
    iu_b = np.triu_indices(m, 1)
    yb = build_rdm(acts[sel])[iu_b]
    cols = {nm: mats[nm][np.ix_(sel, sel)][iu_b] for nm in ALL_PREDICTORS}
    return yb, cols, iu_b


def outcome_share(b_out, b_int):
    d = abs(b_out) + abs(b_int)
    return float(abs(b_out) / d) if d > 1e-12 else float("nan")


# ------------------------------------------------------------------ per-checkpoint worker
def process_checkpoint(args):
    npz_path, n_perm, n_boot, seed, mats, order_ids, groups = args
    d = np.load(npz_path, allow_pickle=True)
    tag = os.path.basename(npz_path)[:-4]
    model, _, revision = tag.partition("@")
    step = int("".join(c for c in revision if c.isdigit()) or -1)

    sids = [str(s) for s in d["story_id"]]
    pos_in_npz = {s: i for i, s in enumerate(sids)}
    missing = [s for s in order_ids if s not in pos_in_npz]
    if missing:
        raise SystemExit(f"{tag}: {len(missing)} items missing from the npz, e.g. {missing[:3]}")
    reorder = np.array([pos_in_npz[s] for s in order_ids])
    layer_idx = [int(x) for x in d["layer_idx"]] if "layer_idx" in d.files else None

    n = len(order_ids)
    T, iu = pair_index_table(n)
    up = {k: v[iu] for k, v in mats.items()}
    within = up["scenario"] == 0
    all_pairs = np.ones(len(within), dtype=bool)

    masks = {s: (within if s == "within_scenario" else all_pairs) for s in SPECS}

    rows = []
    for position in POSITIONS:
        if position not in d.files:
            continue
        A_all = d[position][reorder]           # [n_items, n_kept_layers, hidden]
        n_kept = A_all.shape[1]
        # The realised depth of a kept layer is not the depth that was requested: Pythia-1.4B
        # has a 25-deep stack, so the 0.8 request lands on layer 19 at a true depth of 0.792.
        # The primary cell is designated by the REQUESTED depth, which is what the prespec
        # fixed, and the realised depth is recorded beside it so neither can be mistaken.
        requested = (LAYERS_REL_EXPECTED if layer_idx and n_kept == len(LAYERS_REL_EXPECTED)
                     else [None] * n_kept)
        for li in range(n_kept):
            true_layer = layer_idx[li] if layer_idx else li
            depth_max = (layer_idx[-1] if layer_idx else n_kept - 1) or 1
            rel = round(true_layer / depth_max, 4)
            req = requested[li]
            if req is not None and any(abs(req - x) < 1e-9 for x in EXCLUDED_REL_DEPTHS):
                continue
            acts = A_all[:, li, :].astype(np.float64)
            res = fit_cell(acts, up, mats, masks, n_perm, n_boot, groups, T, iu, seed)
            for spec, names in SPECS.items():
                b_out = res[spec]["outcome"]["beta"]
                b_int = res[spec]["intent"]["beta"]
                int_in_null = abs(b_int) <= res[spec]["intent"]["null_q975_abs"]
                share = outcome_share(b_out, b_int)
                ratio = "" if int_in_null or abs(b_int) < 1e-12 else round(
                    abs(b_out) / abs(b_int), 4)
                for nm in names:
                    r = res[spec][nm]
                    rows.append({
                        "model": model, "revision": revision, "step": step,
                        "position": position, "layer": true_layer,
                        "rel_depth": rel,
                        "rel_depth_requested": "" if req is None else req,
                        "spec": spec, "predictor": nm,
                        "beta": round(r["beta"], 5),
                        "boot_ci_lo": round(r["boot_lo"], 5),
                        "boot_ci_hi": round(r["boot_hi"], 5),
                        "cluster_ci_lo": round(r["cluster_lo"], 5),
                        "cluster_ci_hi": round(r["cluster_hi"], 5),
                        "null_mean": round(r["null_mean"], 5),
                        "null_ci_lo": round(r["null_lo"], 5),
                        "null_ci_hi": round(r["null_hi"], 5),
                        "null_q975_abs": round(r["null_q975_abs"], 5),
                        "p_perm": round(r["p_perm"], 5),
                        "inside_null": bool(abs(r["beta"]) <= r["null_q975_abs"]),
                        "vif": round(r["vif"], 3),
                        "outcome_share": round(share, 4) if share == share else "",
                        "outcome_intent_ratio": ratio,
                        "n_perm": n_perm, "n_boot_ok": r["n_boot_ok"],
                        "n_cluster_ok": r["n_cluster_ok"],
                        "is_primary_cell": bool(position == PRIMARY_POSITION
                                                and req is not None
                                                and abs(req - PRIMARY_REL_DEPTH) < 1e-9),
                    })
    print(f"  {tag}: {len(rows)} rows", flush=True)
    return rows


# ------------------------------------------------------------------ E8c gate
def gate_verdict(rows, outdir, n_cells):
    """E8c. Bonferroni-corrected over cells, per E8_PRESPEC.md section 4."""
    step0 = [r for r in rows if r["step"] == 0 and r["spec"] == "primary"]
    if not step0:
        return None
    # Exactly the rule in E8_PRESPEC.md section 4: |beta| at or below the
    # 100*(1 - alpha/(2*n_cells))-th percentile of |null|. p_perm is already the |null|
    # exceedance rate, so the rule is p_perm > alpha/(2*n_cells). Writing the two-sided
    # correction on an already-absolute null is conservative; it is what was pre-registered
    # and it is not loosened here.
    thresh = GATE_ALPHA / (2.0 * n_cells)
    pct = 100.0 * (1.0 - thresh)
    fails, checked = [], []
    for r in step0:
        if r["predictor"] not in ("intent", "outcome"):
            continue
        passed = r["p_perm"] > thresh
        checked.append((r, passed))
        if not passed:
            fails.append(r)
    uncorrected = sum(1 for r, _ in checked if r["inside_null"] is False)
    expected_uncorrected = 2 * n_cells * 0.05
    return dict(pct=pct, thresh=thresh, n_cells=n_cells, fails=fails, checked=checked,
                uncorrected=uncorrected, expected_uncorrected=expected_uncorrected,
                passed=not fails, step0=step0)


def write_gate(g, outdir):
    L = ["# E8c — step-0 sanity gate", "",
         f"**Verdict: {'PASS' if g['passed'] else 'FAIL — E8d is blocked'}**", "",
         "The rule, its Bonferroni correction and the consequence of failure were fixed in "
         "`outputs/experiments/E8_PRESPEC.md` section 4 before any checkpoint was "
         "downloaded. At random initialisation a network cannot have learned the moral "
         "belief × outcome factors, so `intent` and `outcome` must both sit inside their own "
         "within-scenario permutation null in every cell.", "",
         f"- cells: **{g['n_cells']}** (layer × position), so the per-cell two-sided "
         f"threshold is the {g['pct']:.2f}th percentile of the null "
         f"(family-wise α = {GATE_ALPHA})",
         f"- cells exceeding the *uncorrected* 95% band: **{g['uncorrected']}** of "
         f"{2 * g['n_cells']} predictor-cells, against **{g['expected_uncorrected']:.1f}** "
         f"expected by chance on a clean pipeline",
         f"- cells failing the corrected rule: **{len(g['fails'])}**",
         "",
         f"Relative depth {EXCLUDED_REL_DEPTHS} is excluded per amendment 2.1: at a "
         f"single-token read position the layer-0 activation is a token-embedding lookup, "
         f"and every vignette ends in `.` or `).`, so it takes 2 distinct values across 298 "
         f"items at every checkpoint. The first run of this gate included it and also "
         f"passed (0 of 24 predictor-cells outside the uncorrected band); dropping cells "
         f"tightens a Bonferroni family rather than loosening it.", ""]
    if g["fails"]:
        L += ["## Failing cells", "",
              "| position | layer | rel depth | predictor | β | null q97.5(|β|) | p_perm |",
              "|---|---:|---:|---|---:|---:|---:|"]
        for r in g["fails"]:
            L.append(f"| `{r['position']}` | {r['layer']} | {r['rel_depth']} | "
                     f"`{r['predictor']}` | {r['beta']:+.4f} | {r['null_q975_abs']:.4f} | "
                     f"{r['p_perm']:.4f} |")
        L += ["", "**E8d is not run.** Per the prespec the two suspects, in the order they "
                  "are to be checked, are sequence-length leakage (the four conditions in a "
                  "group differ in length, and last-token pooling over a length-correlated "
                  "activation norm can encode that) and tokenisation of the manipulated "
                  "clause. There is no version of E8 in which a failed gate is reported as "
                  "early emergence.", ""]
    L += ["## All step-0 betas, including the covariates", "",
          "`lexical` and `scenario` are **expected to be large here and that is not a "
          "failure**: a random network is close to a random projection, and random "
          "projections approximately preserve input similarity, so token-overlap structure "
          "survives into the hidden states. This was stated in the prespec before the "
          "numbers existed, so that a large step-0 lexical β could not later be misread as "
          "contamination. The gate is on `intent` and `outcome` only.", "",
          "| position | layer | predictor | β | null q97.5(\\|β\\|) | inside null | p_perm |",
          "|---|---:|---|---:|---:|---|---:|"]
    for r in sorted(g["step0"], key=lambda r: (r["position"], r["layer"],
                                               ALL_PREDICTORS.index(r["predictor"]))):
        mark = "yes" if r["inside_null"] else "**no**"
        L.append(f"| `{r['position']}` | {r['layer']} | `{r['predictor']}` | "
                 f"{r['beta']:+.4f} | {r['null_q975_abs']:.4f} | {mark} | {r['p_perm']:.4f} |")
    L.append("")
    p = os.path.join(outdir, "E8_STEP0_GATE.md")
    open(p, "w").write("\n".join(L))
    return p


# ------------------------------------------------------------------ figure
def make_figure(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prim = [r for r in rows if r["spec"] == "primary" and r["is_primary_cell"]]
    if not prim:
        print("  figure skipped: no primary-cell rows")
        return None
    colors = {"intent": "#1f77b4", "outcome": "#d62728", "interaction": "#9467bd",
              "scenario": "#7f7f7f", "lexical": "#bcbd22"}
    steps = sorted({r["step"] for r in prim})
    pos_steps = [s for s in steps if s > 0]
    x0 = min(pos_steps) / 2.0 if pos_steps else 0.5   # where step 0 is drawn on a log axis

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6),
                             gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axes[0]
    for name in ALL_PREDICTORS:
        pts = sorted([r for r in prim if r["predictor"] == name], key=lambda r: r["step"])
        xs = [x0 if r["step"] == 0 else r["step"] for r in pts]
        ys = [r["beta"] for r in pts]
        cov = name in COVARIATES
        ax.plot(xs, ys, "-o", ms=3.5, lw=2.0 if not cov else 1.2,
                ls="-" if not cov else "--", color=colors[name], alpha=0.95 if not cov else 0.6,
                label=name + (" (covariate)" if cov else ""))
        # the cluster bootstrap is shaded, not the item one: amendment 2.2 gives it
        # precedence, because 94 of the 298 rows are exact reprints of another row
        ax.fill_between(xs, [r["cluster_ci_lo"] for r in pts],
                        [r["cluster_ci_hi"] for r in pts],
                        color=colors[name], alpha=0.13, lw=0)
        # the step-0 null band, shaded across the whole axis as the prespec requires
        z = next((r for r in pts if r["step"] == 0), None)
        if z is not None and name in ("intent", "outcome"):
            ax.axhspan(-z["null_q975_abs"], z["null_q975_abs"],
                       color=colors[name], alpha=0.07, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.axvline(x0 * 2 / 1.4, color="#bbb", lw=0.8, ls=":")
    ax.text(x0, ax.get_ylim()[1], " step 0\n (random init)", fontsize=7,
            va="top", ha="center", color="#555")
    ax.set_xlabel("training step (log; step 0 drawn at the left tick)")
    ax.set_ylabel("standardised β")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_title(f"E8d — β trajectory, primary cell "
                 f"(rel depth {PRIMARY_REL_DEPTH}, `{PRIMARY_POSITION}`)\n"
                 f"shaded bands = cluster bootstrap 95% CI (53 scenario groups); "
                 f"pale spans = step-0 permutation null", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    for spec, style in (("primary", "-o"), ("no_lexical", "--s")):
        pts = sorted([r for r in rows
                      if r["spec"] == spec and r["is_primary_cell"]
                      and r["predictor"] == "outcome"], key=lambda r: r["step"])
        xs = [x0 if r["step"] == 0 else r["step"] for r in pts]
        ys = [r["outcome_share"] if r["outcome_share"] != "" else np.nan for r in pts]
        ax2.plot(xs, ys, style, ms=3.5, lw=1.6, label=spec)
    ax2.set_xscale("log")
    ax2.axhline(0.5, color="k", lw=0.8, ls=":")
    ax2.text(x0, 0.505, " equal weighting", fontsize=7, color="#555")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("training step (log)")
    ax2.set_ylabel(r"outcome share  $|\beta_{out}|/(|\beta_{out}|+|\beta_{int}|)$")
    ax2.set_title("primary quantity, with the pre-registered\nno-lexical sensitivity",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    p = os.path.join(outdir, "checkpoint_betas.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default=ACTS_DIR)
    ap.add_argument("--out", default=RSA_DIR)
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--perm-gate", type=int, default=N_PERM_GATE)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--gate-only", action="store_true",
                    help="E8c only: fit step0 and write the gate verdict, nothing else")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    out_csv = os.path.join(a.out, "checkpoint_betas.csv")
    if a.plot_only:
        rows = list(csv.DictReader(open(out_csv)))
        for r in rows:
            for k in ("beta", "boot_ci_lo", "boot_ci_hi", "cluster_ci_lo", "cluster_ci_hi",
                      "null_mean", "null_ci_lo", "null_ci_hi", "null_q975_abs", "p_perm",
                      "rel_depth", "vif"):
                r[k] = float(r[k])
            r["step"] = int(r["step"]); r["layer"] = int(r["layer"])
            r["is_primary_cell"] = r["is_primary_cell"] in ("True", "true", "1")
            r["inside_null"] = r["inside_null"] in ("True", "true", "1")
            if r["outcome_share"] != "":
                r["outcome_share"] = float(r["outcome_share"])
        print(f"-> {make_figure(rows, a.out)}")
        return

    npzs = sorted(glob.glob(os.path.join(a.acts, "*.npz")),
                  key=lambda p: int("".join(c for c in os.path.basename(p).split("@")[-1]
                                            if c.isdigit()) or -1))
    if not npzs:
        raise SystemExit(f"no checkpoint activations in {a.acts} — run E8b first")
    step0 = [p for p in npzs if p.endswith("@step0.npz")]
    if not step0:
        raise SystemExit(f"no @step0.npz in {a.acts} — the E8c gate cannot be evaluated, "
                         f"and E8_PRESPEC.md section 4 blocks E8d until it is")

    if a.dry_run:
        print("=== E8c/E8d PLAN (dry-run) ===")
        print(f"checkpoints : {len(npzs)}  ({os.path.basename(npzs[0])} .. "
              f"{os.path.basename(npzs[-1])})")
        d = np.load(npzs[0], allow_pickle=True)
        nl = d["last"].shape[1]
        pos = [p for p in POSITIONS if p in d.files]
        print(f"cells       : {nl} layers x {len(pos)} positions = {nl * len(pos)}")
        print(f"specs       : {list(SPECS)}")
        print(f"perms       : {a.perm_gate} at step0, {a.perm} elsewhere; boot {a.boot}")
        print(f"gate        : Bonferroni over {nl * len(pos)} cells, family-wise "
              f"alpha={GATE_ALPHA}")
        print(f"primary cell: rel depth {PRIMARY_REL_DEPTH}, position {PRIMARY_POSITION}")
        print(f"jobs        : {a.jobs}")
        return

    guard = _load("_run_guard.py", "run_guard")
    guard.require_clean_tree(os.path.basename(__file__), allow_dirty=a.allow_dirty)

    mats, order = load_predictors()
    order_ids = [r["story_id"] for r in order]
    groups = np.array([r["scenario_group"] or r["scenario_id"] for r in order])

    if a.gate_only:
        npzs = step0
    tasks = [(p, a.perm_gate if p.endswith("@step0.npz") else a.perm,
              a.boot, SEED, mats, order_ids, groups) for p in npzs]

    print(f"=== E8c/E8d: {len(tasks)} checkpoint(s), {a.jobs} worker(s) ===", flush=True)
    if a.jobs > 1 and len(tasks) > 1:
        with Pool(a.jobs) as pool:
            chunks = pool.map(process_checkpoint, tasks)
    else:
        chunks = [process_checkpoint(t) for t in tasks]
    rows = [r for c in chunks for r in c]

    n_cells = len({(r["position"], r["layer"]) for r in rows})
    g = gate_verdict(rows, a.out, n_cells)
    gate_path = write_gate(g, a.out)
    print(f"\n=== E8c GATE: {'PASS' if g['passed'] else 'FAIL'} === -> "
          f"{os.path.relpath(gate_path, ROOT)}")
    for r, ok in g["checked"]:
        if not ok:
            print(f"  FAIL {r['position']} L{r['layer']} {r['predictor']}: "
                  f"beta={r['beta']:+.4f} p={r['p_perm']:.4f}")

    if not g["passed"]:
        raise SystemExit("[E8c] gate failed — E8d outputs are not written, per "
                         "E8_PRESPEC.md section 4. See " + os.path.relpath(gate_path, ROOT))
    if a.gate_only:
        print("[E8c] gate passed. Re-run without --gate-only for E8d.")
        return

    fields = [k for k in rows[0] if k != "null_abs"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out_csv, ROOT)} ({len(rows)} rows)")
    fp = make_figure(rows, a.out)
    if fp:
        print(f"-> {os.path.relpath(fp, ROOT)}")


if __name__ == "__main__":
    main()
