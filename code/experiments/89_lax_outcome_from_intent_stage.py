#!/usr/bin/env python3
"""
89_lax_outcome_from_intent_stage.py -- Lax 9/29 T4: is OUTCOME decodable from the intention stage?

Protocol = 02_probe.py: logistic probe, GroupKFold(5) by scenario_group, within-scenario label
permutation null (--perm), TF-IDF surface baseline (21_surface_baseline.span_text) on the SAME
truncated text with the same folds.

Part A  existing activations outputs/acts/*.npz, poolings belief_last / action_last (end of the
        belief / action clause, 02's clause mask: belief_verb + manual spans only). Every layer is
        scanned; the PRIMARY layer is fixed at relative depth 0.65 (not selected on the data), the
        peak layer is reported as secondary and its p is selection-biased.
        All YS2008/YS2009 stories state the foreshadow (outcome-determining fact) BEFORE the belief,
        so Part A measures "outcome information in the text so far", not prediction.
Part B  token trajectories (81, $TOM_TRAJ_DIR) at the last token of the belief text, original vs
        no_foreshadow variant (stage_landmarks.no_foreshadow deletes the foreshadow sentences), at
        every saved depth. In no_foreshadow the text up to that token is identical across outcome
        partners, so surface = chance by construction and any decodability is an expectation.

The fold projection (standardize + row-space SVD) does not depend on labels, so it is computed once
per layer and only the logistic fit is repeated per permutation -- identical predictions to 02.

  .venv/bin/python code/experiments/89_lax_outcome_from_intent_stage.py --perm 1000
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "outputs", "lax_0929")
N_JOBS = int(os.environ.get("JOBLIB_N_JOBS", "2"))
FIXED_DEPTH = 0.65


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P02 = _load("p02", os.path.join(ROOT, "code", "02_probe.py"))
S21 = _load("s21", os.path.join(HERE, "21_surface_baseline.py"))


def fold_projections(X, groups):
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    out = []
    for tr, te in GroupKFold(n_splits=5).split(X, None, groups):
        sc = StandardScaler().fit(X[tr])
        A, B = P02._rowspace_project(sc.transform(X[tr]), sc.transform(X[te]))
        out.append((tr, te, A, B))
    return out


def cv_from_proj(proj, y, n=None):
    """-> (acc, oof P(y=1)). Same fallback as 02.group_cv_acc for degenerate folds."""
    from sklearn.linear_model import LogisticRegression
    p = np.full(len(y) if n is None else n, np.nan)
    accs = []
    for tr, te, A, B in proj:
        if A.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            accs.append(max(y[te].mean(), 1 - y[te].mean()))
            p[te] = y[tr].mean()
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(A, y[tr])
        p[te] = clf.predict_proba(B)[:, 1]
        accs.append(clf.score(B, y[te]))
    return float(np.mean(accs)), p


def perm_null(proj, y, groups, n_perm, seed=0):
    from joblib import Parallel, delayed
    obs, _ = cv_from_proj(proj, y)
    rng = np.random.default_rng(seed)
    perms = [P02.permute_within_groups(y, groups, rng) for _ in range(n_perm)]
    null = np.array(Parallel(n_jobs=N_JOBS)(delayed(lambda yp: cv_from_proj(proj, yp)[0])(yp) for yp in perms))
    return obs, float(null.mean()), float(np.percentile(null, 95)), float((np.sum(null >= obs) + 1) / (n_perm + 1))


def tfidf_cv(texts, y, groups):
    from sklearn.feature_extraction.text import TfidfVectorizer
    res = {}
    for name, kw in S21.FEATURE_SETS.items():
        try:
            X = np.asarray(TfidfVectorizer(**kw).fit_transform(texts).todense())
        except ValueError:
            res[name] = (0.5, np.full(len(y), 0.5))
            continue
        res[name] = cv_from_proj(fold_projections(X, groups), y)
    return res


# ------------------------------------------------------------------------- Part A
def part_a(n_perm, only=None):
    lab = P02.load_labels(os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ok = P02.load_clause_mask(os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    offs = S21.load_offsets(os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    summ, layers_rows, items = [], [], []
    surf_cache = {}
    for npz in sorted(glob.glob(os.path.join(ROOT, "outputs", "acts", "*.npz"))):
        tag = os.path.basename(npz)[:-4]
        if only and only not in tag:
            continue
        d = np.load(npz, allow_pickle=True)
        sids_all = [str(s) for s in d["story_id"]]
        for pooling in ("belief_last", "action_last"):
            keep = [i for i, s in enumerate(sids_all) if s in lab and s in ok]
            sids = [sids_all[i] for i in keep]
            acts = d[pooling][keep]
            y_o = np.array([lab[s]["outcome_label"] == "harm" for s in sids], int)
            y_i = np.array([lab[s]["intent_label"] == "guilty" for s in sids], int)
            groups = np.array([lab[s]["scenario_group"] for s in sids])
            if pooling not in surf_cache:
                texts = [S21.span_text(lab[s], offs, pooling) for s in sids]
                surf_cache[pooling] = {t: tfidf_cv(texts, yy, groups) for t, yy in (("outcome", y_o), ("intent", y_i))}
            surf = surf_cache[pooling]
            nL = acts.shape[1]
            fixed = int(round(FIXED_DEPTH * (nL - 1)))
            projs, scan = {}, []
            for L in range(nL):
                pr = fold_projections(acts[:, L].astype(np.float32), groups)
                ao, po = cv_from_proj(pr, y_o)
                ai, _ = cv_from_proj(pr, y_i)
                scan.append((L, ao, ai, po))
                layers_rows.append(dict(model=tag, pooling=pooling, layer=L, outcome_acc=ao, intent_acc=ai))
                if L == fixed:
                    projs[L] = pr
            peak = max(scan[1:], key=lambda r: r[1])[0]
            if peak not in projs:
                projs[peak] = fold_projections(acts[:, peak].astype(np.float32), groups)
            for kind, L in (("fixed_0.65", fixed), ("peak", peak)):
                obs, nm, p95, p = perm_null(projs[L], y_o, groups, n_perm)
                summ.append(dict(model=tag, pooling=pooling, layer_kind=kind, layer=L, n_layers=nL, n=len(sids),
                                 outcome_acc=obs, null_mean=nm, null_p95=p95, perm_p=p, n_perm=n_perm,
                                 tfidf_word_acc=surf["outcome"]["tfidf_word_1_2"][0],
                                 tfidf_char_acc=surf["outcome"]["tfidf_char_3_5"][0],
                                 gap_vs_best_tfidf=obs - max(surf["outcome"]["tfidf_word_1_2"][0],
                                                             surf["outcome"]["tfidf_char_3_5"][0]),
                                 intent_acc_same_layer=scan[L][2]))
                print(f"A {tag:28} {pooling:12} {kind:10} L{L:<3} outcome={obs:.3f} p={p:.4f} "
                      f"tfidf={surf['outcome']['tfidf_word_1_2'][0]:.3f}/{surf['outcome']['tfidf_char_3_5'][0]:.3f}",
                      flush=True)
            po = scan[fixed][3]
            pw = surf["outcome"]["tfidf_word_1_2"][1]
            for r, s in enumerate(sids):
                items.append(dict(model=tag, pooling=pooling, layer=fixed, story_id=s,
                                  scenario_group=groups[r], source=lab[s]["source"],
                                  condition=lab[s]["condition"], outcome_true=y_o[r],
                                  p_harm=round(float(po[r]), 4), correct=int((po[r] > 0.5) == y_o[r]),
                                  tfidf_word_p_harm=round(float(pw[r]), 4),
                                  tfidf_correct=int((pw[r] > 0.5) == y_o[r])))
    return pd.DataFrame(summ), pd.DataFrame(layers_rows), pd.DataFrame(items)


# ------------------------------------------------------------------------- Part B
def part_b(n_perm, traj, only=None):
    import stage_landmarks as SL
    lab = {r["story_id"]: r for r in SL.load_rows()}
    rows, items = [], []
    for stem_npz in sorted(glob.glob(os.path.join(traj, "*__story__chatoff.npz"))):
        stem = stem_npz[:-4]
        model, variant = os.path.basename(stem).split("__")[:2]
        if only and only not in model:
            continue
        meta = np.load(stem + ".npz", allow_pickle=False)
        H = np.load(stem + ".H.npy", mmap_mode="r")
        sids = [str(s) for s in meta["story_id"]]
        texts = [str(t) for t in meta["text"]]
        lm = SL.landmarks([dict(lab[s], text=t) for s, t in zip(sids, texts)])
        ptr, ce = meta["tok_ptr"], meta["char_e"]
        keep, tok, cut = [], [], []
        for i, s in enumerate(sids):
            m = lm[s]
            if not m["complete"] or lab[s]["source"] == "YS2011":
                continue
            a, b = ptr[i], ptr[i + 1]
            ends = ce[a:b]
            st = np.where(ends >= 0)[0]
            j = np.searchsorted(ends[st], m["intent_off"], side="right") - 1
            keep.append(i)
            tok.append(a + int(st[max(j, 0)]))
            cut.append(int(ends[st[max(j, 0)]]))
        ks = [sids[i] for i in keep]
        y = np.array([lab[s]["outcome_label"] == "harm" for s in ks], int)
        groups = np.array([lab[s]["scenario_group"] for s in ks])
        tt = [texts[i][:c] for i, c in zip(keep, cut)]
        surf = tfidf_cv(tt, y, groups)
        n_identical = sum(len({t for s2, t in zip(ks, tt) if lab[s2]["scenario_group"] == g and
                                lab[s2]["source"] == lab[s]["source"] and
                                lab[s2]["intent_label"] == lab[s]["intent_label"]}) == 1
                          for s, g in zip(ks, groups))
        for li, L in enumerate(meta["layer_idx"]):
            X = np.asarray(H[np.array(tok), li], dtype=np.float32)
            pr = fold_projections(X, groups)
            if n_identical == len(ks):
                # every outcome pair shares its prefix, so a causal model's states are identical:
                # accuracy is 0.5 by construction and a permutation null adds nothing
                obs, _ = cv_from_proj(pr, y)
                nm, p95, p = 0.5, 0.5, 1.0
            else:
                obs, nm, p95, p = perm_null(pr, y, groups, n_perm)
            _, po = cv_from_proj(pr, y)
            rows.append(dict(model=model, variant=variant, layer=int(L), n=len(ks), outcome_acc=obs,
                             null_mean=nm, null_p95=p95, perm_p=p, n_perm=n_perm,
                             tfidf_word_acc=surf["tfidf_word_1_2"][0], tfidf_char_acc=surf["tfidf_char_3_5"][0],
                             frac_items_prefix_identical_to_outcome_partner=n_identical / len(ks)))
            print(f"B {model:28} {variant:14} L{int(L):<3} outcome={obs:.3f} p={p:.4f} "
                  f"tfidf={surf['tfidf_word_1_2'][0]:.3f} identical_prefix={n_identical / len(ks):.2f}", flush=True)
            for r, s in enumerate(ks):
                items.append(dict(model=model, variant=variant, layer=int(L), story_id=s, scenario_group=groups[r],
                                  source=lab[s]["source"], condition=lab[s]["condition"], outcome_true=y[r],
                                  p_harm=round(float(po[r]), 4)))
    return pd.DataFrame(rows), pd.DataFrame(items)


def plot(sa, la, sb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, pooling in zip(axes[:2], ("belief_last", "action_last")):
        g = la[la.pooling == pooling]
        for m, gm in g.groupby("model"):
            ax.plot(gm.layer / gm.layer.max(), gm.outcome_acc, lw=1, label=m)
        s = sa[(sa.pooling == pooling)].iloc[0]
        ax.axhline(s.tfidf_word_acc, color="k", ls="--", lw=1, label="TF-IDF word (same text)")
        ax.axhline(s.tfidf_char_acc, color="k", ls=":", lw=1, label="TF-IDF char")
        ax.axhline(0.5, color="0.6", lw=0.6)
        ax.set_title(f"A: outcome decoded at {pooling} (existing acts)", fontsize=9)
        ax.set_xlabel("relative depth")
        ax.set_ylabel("GroupKFold accuracy")
    axes[0].legend(fontsize=5.5, ncol=2)
    ax = axes[2]
    if len(sb):
        for (m, v), g in sb.groupby(["model", "variant"]):
            ax.plot(g.layer / g.layer.max(), g.outcome_acc, "-o" if v == "original" else "--s", ms=3, lw=1,
                    label=f"{m} [{v}]")
            ax.fill_between(g.layer / g.layer.max(), g.null_mean, g.null_p95, color="0.85", alpha=0.3)
        ax.axhline(0.5, color="0.6", lw=0.6)
        ax.legend(fontsize=5.5)
    ax.set_title("B: outcome at last belief token, original vs foreshadow deleted\n(grey = permutation null mean..95%)",
                 fontsize=9)
    ax.set_xlabel("relative depth")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--traj", default=os.environ.get(
        "TOM_TRAJ_DIR", f"/orcd/scratch/orcd/007/{os.environ.get('USER', '')}/acts_traj"))
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on model tag (smoke tests)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    sa, la, ia = part_a(a.perm, a.only)
    sa.to_csv(os.path.join(out, "outcome_from_intent_stage_summary.csv"), index=False)
    la.to_csv(os.path.join(out, "outcome_from_intent_stage_layers.csv"), index=False)
    ia.to_csv(os.path.join(out, "outcome_from_intent_stage_items.csv"), index=False)
    sb = pd.DataFrame()
    if not a.skip_b:
        sb, ib = part_b(a.perm, a.traj, a.only)
        sb.to_csv(os.path.join(out, "outcome_from_intent_stage_noforeshadow.csv"), index=False)
        ib.to_csv(os.path.join(out, "outcome_from_intent_stage_noforeshadow_items.csv"), index=False)
    plot(sa, la, sb, os.path.join(out, "fig_T4_outcome_from_intent_stage"))


if __name__ == "__main__":
    main()
