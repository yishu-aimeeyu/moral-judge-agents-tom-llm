#!/usr/bin/env python3
"""
92_lax_time_decoding_clauses.py -- Lax 9/29 T6: decoding over time aligned to clause_offsets.csv.

Re-analysis of 81's token trajectories (bare story, chat off, original text); no new activations.
Time axis: 10 bins in each of background [0, belief_start), belief [belief_start, belief_end),
action [belief_end, action_end), outcome [action_end, outcome_end) -- clause_offsets.csv spans,
02's clause mask (belief_verb + manual), stories whose four stage ends are not strictly increasing
are dropped and counted. Bin j of a stage = last token whose span ends at or before its char cut.
NB: in YS2008/YS2009 the foreshadow sentence lies inside "background" under this alignment.

Per saved layer (81 keeps relative depths 0.25/0.5/0.65/0.8/1.0, which include or neighbour the
02 best intent layer) and target (intent, outcome):
  * per-bin GroupKFold accuracy + within-scenario shuffle-null band (--perm, 2.5-97.5%) and p
  * TF-IDF surface accuracy on the text up to the same char (82.surface_curve)
  * temporal generalization matrix (82.temporal_generalization)
  * hypothesis statistics with scenario_group bootstrap CIs:
      intent_rise_belief     acc(last belief bin) - acc(first belief bin)
      intent_slope_late      slope of intent acc over action+outcome bins (plateau => ~0)
      intent_late_minus_belief_end   mean(action+outcome) - acc(last belief bin)
      outcome_slope_pre      slope of outcome acc over belief+action bins (flat => ~0)
      outcome_jump           acc(last outcome bin) - acc(last action bin)
  * 4x4 condition RDM per bin: within-scenario pairwise distances (z-scored within scenario),
    regressed on intent, outcome and interaction (intent x outcome conjunction) model RDMs;
    scenario bootstrap CIs.

  .venv/bin/python code/experiments/92_lax_time_decoding_clauses.py --model Qwen2.5-7B --perm 100
"""
import os, sys, argparse, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "outputs", "lax_0929", "T6")
N_JOBS = int(os.environ.get("JOBLIB_N_JOBS", "2"))
STAGES = ["background", "belief", "action", "outcome"]
CELLS = ["neutral", "accidental", "attempted", "intentional"]
# (intent, outcome, intent x outcome). With +-1 interaction coding every pair of cells differs in exactly
# two of the three, so the three "differs" RDMs sum to a constant and are not separately identifiable;
# the 0/1 product (only intentional = 1) gives an independent interaction RDM.
CODE = {"neutral": (0, 0, 0), "accidental": (0, 1, 0), "attempted": (1, 0, 0), "intentional": (1, 1, 1)}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P02 = _load("p02", "code/02_probe.py")
D82 = _load("d82", "code/experiments/82_temporal_decoding.py")


def clause_positions(meta, lab, n_bins):
    co = pd.read_csv(os.path.join(ROOT, "dataset", "master", "clause_offsets.csv")).set_index("story_id")
    ok = P02.load_clause_mask(os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    sids = [str(s) for s in meta["story_id"]]
    ptr, ce = meta["tok_ptr"], meta["char_e"]
    keep, pos, cuts, dropped = [], [], [], {"mask": 0, "order": 0}
    for i, s in enumerate(sids):
        if s not in ok or s not in lab:
            dropped["mask"] += 1
            continue
        r = co.loc[s]
        ends = [0, int(r.belief_start), int(r.belief_end), int(r.action_end), int(r.outcome_end)]
        if any(ends[k] >= ends[k + 1] for k in range(4)):
            dropped["order"] += 1
            continue
        a, b = ptr[i], ptr[i + 1]
        te = ce[a:b]
        st = np.where(te >= 0)[0]
        p, c = [], []
        for k in range(4):
            for j in range(n_bins):
                cut = ends[k] + int(round((j + 1) / n_bins * (ends[k + 1] - ends[k])))
                jj = np.searchsorted(te[st], cut, side="right") - 1
                p.append(a + int(st[max(jj, 0)]))
                c.append(cut)
        keep.append(i)
        pos.append(p)
        cuts.append(c)
    return np.array(keep), np.array(pos), np.array(cuts), dropped


def fold_proj(Xt, folds):
    out = []
    for tr, te in folds:
        mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
        Z = (Xt[tr] - mu) / sd
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        V = Vt[S > 1e-6 * S.max()].T if S.max() > 0 else np.zeros((Z.shape[1], 0))
        out.append((tr, te, Z @ V, ((Xt[te] - mu) / sd) @ V))
    return out


def acc_from_proj(proj, y):
    from sklearn.linear_model import LogisticRegression
    c = np.zeros(len(y), bool)
    for tr, te, A, B in proj:
        if A.shape[1] == 0 or len(np.unique(y[tr])) < 2:
            c[te] = (y[te] == int(y[tr].mean() > 0.5))
            continue
        c[te] = LogisticRegression(max_iter=3000, C=1.0).fit(A, y[tr]).predict(B) == y[te]
    return c


def null_band(X, y, groups, folds, n_perm, seed):
    """X [n, T, H] -> (lo [T], hi [T], null [n_perm, T])."""
    from joblib import Parallel, delayed
    T = X.shape[1]
    projs = Parallel(n_jobs=N_JOBS)(delayed(fold_proj)(X[:, t], folds) for t in range(T))
    rng = np.random.default_rng(seed)
    perms = [P02.permute_within_groups(y, groups, rng) for _ in range(n_perm)]

    def one(yp):
        return [acc_from_proj(projs[t], yp).mean() for t in range(T)]

    null = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(yp) for yp in perms))
    return np.percentile(null, 2.5, 0), np.percentile(null, 97.5, 0), null


def rsa3(X, rows, n_boot, rng):
    n, T, _ = X.shape
    by = {}
    for r, row in enumerate(rows):
        by.setdefault(row["scenario_group"] + "|" + row["source"], {})[row["condition"]] = r
    scen = [v for v in by.values() if len(v) == 4]
    pairs = [(a, b) for i, a in enumerate(CELLS) for b in CELLS[i + 1:]]
    D = np.array([[float(CODE[a][k] != CODE[b][k]) for k in range(3)] for a, b in pairs])
    D = D - D.mean(0)
    beta = np.full((T, len(scen), 3), np.nan)
    rdm = np.zeros((T, 4, 4))
    for t in range(T):
        Z = (X[:, t] - X[:, t].mean(0)) / (X[:, t].std(0) + 1e-6)
        acc, cnt = np.zeros((4, 4)), 0
        for s, cells in enumerate(scen):
            d = np.array([np.linalg.norm(Z[cells[a]] - Z[cells[b]]) for a, b in pairs])
            if d.max() < 1e-2 * np.sqrt(Z.shape[1]):
                continue
            dz = (d - d.mean()) / (d.std() + 1e-9)
            beta[t, s] = np.linalg.lstsq(D, dz, rcond=None)[0]
            for (a, b), v in zip(pairs, dz):
                i, j = CELLS.index(a), CELLS.index(b)
                acc[i, j] += v
                acc[j, i] += v
            cnt += 1
        rdm[t] = acc / max(cnt, 1)
    m = np.nanmean(beta, 1)
    boot = np.array([np.nanmean(beta[:, rng.integers(0, len(scen), len(scen))], 1) for _ in range(n_boot)])
    return m, np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0), rdm


def hyp_stats(corr_i, corr_o, st, ix):
    ai, ao = corr_i[:, ix].mean(1), corr_o[:, ix].mean(1)
    late = np.concatenate([st["action"], st["outcome"]])
    pre = np.concatenate([st["belief"], st["action"]])
    return dict(intent_rise_belief=ai[st["belief"][-1]] - ai[st["belief"][0]],
                intent_slope_late=np.polyfit(np.arange(len(late)), ai[late], 1)[0],
                intent_late_minus_belief_end=ai[late].mean() - ai[st["belief"][-1]],
                outcome_slope_pre=np.polyfit(np.arange(len(pre)), ao[pre], 1)[0],
                outcome_jump=ao[st["outcome"][-1]] - ao[st["action"][-1]])


def main():
    from sklearn.model_selection import GroupKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--traj", default=D82.TRAJ)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="subset of saved layer indices")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D82.LAB = D82.labels()
    lab = D82.LAB
    stem = D82.find_stem(a.traj, a.model, "original", "story", "off")
    meta, H = D82.load_traj(stem)
    keep, pos, cuts, dropped = clause_positions(meta, lab, a.bins)
    sids = [str(meta["story_id"][i]) for i in keep]
    rows = [lab[s] for s in sids]
    y = {"intent": np.array([r["intent_label"] == "guilty" for r in rows], int),
         "outcome": np.array([r["outcome_label"] == "harm" for r in rows], int)}
    groups = np.array([r["scenario_group"] for r in rows])
    folds = list(GroupKFold(n_splits=5).split(sids, None, groups))
    T = pos.shape[1]
    stage = np.repeat(np.arange(4), a.bins)
    st = {s: np.where(stage == k)[0] for k, s in enumerate(STAGES)}
    layers = [int(L) for L in meta["layer_idx"]]
    print(f"{a.model}: {len(keep)} stories kept, dropped {dropped}, {T} bins, layers {layers}", flush=True)
    texts = [str(meta["text"][i]) for i in keep]
    surf = {t: D82.surface_curve(texts, cuts, y[t], folds) for t in y}
    rng = np.random.default_rng(0)
    curve, hyp, rsa_rows, tg = [], [], [], {}
    for li, L in enumerate(layers):
        if a.layers and L not in a.layers:
            continue
        X = np.asarray(H[pos.ravel(), li], dtype=np.float32).reshape(len(keep), T, -1)
        P = {t: D82.temporal_generalization(X, y[t], groups, folds) for t in y}
        corr = {t: (P[t][np.arange(T), np.arange(T)] > 0.5) == y[t][None, :] for t in y}
        for t in y:
            tg[f"L{L}_{t}"] = ((P[t] > 0.5) == y[t][None, None, :]).mean(2)
            lo, hi, null = null_band(X, y[t], groups, folds, a.perm, seed=li)
            obs = corr[t].mean(1)
            for b in range(T):
                curve.append(dict(model=a.model, layer=L, target=t, bin=b, stage=STAGES[stage[b]], acc=obs[b],
                                  null_lo=lo[b], null_hi=hi[b],
                                  perm_p=(np.sum(null[:, b] >= obs[b]) + 1) / (a.perm + 1),
                                  surface_acc=float(((surf[t][b] > 0.5) == y[t]).mean())))
        full = hyp_stats(corr["intent"], corr["outcome"], st, np.arange(len(keep)))
        boots = [hyp_stats(corr["intent"], corr["outcome"], st, ix) for ix in D82.boot_groups(groups, a.boot, rng)]
        sc = {t: (surf[t] > 0.5) == y[t][None, :] for t in y}
        sfull = hyp_stats(sc["intent"], sc["outcome"], st, np.arange(len(keep)))
        for k, v in full.items():
            bv = np.array([b[k] for b in boots])
            hyp.append(dict(model=a.model, layer=L, stat=k, value=v, lo=np.percentile(bv, 2.5),
                            hi=np.percentile(bv, 97.5), surface_value=sfull[k]))
        m, lo, hi, rdm = rsa3(X, rows, a.boot, rng)
        for b in range(T):
            for j, nm in enumerate(("intent", "outcome", "interaction")):
                rsa_rows.append(dict(model=a.model, layer=L, bin=b, stage=STAGES[stage[b]], term=nm,
                                     beta=m[b, j], lo=lo[b, j], hi=hi[b, j]))
        tg[f"L{L}_rdm"] = rdm
        h = {r["stat"]: r for r in hyp[-5:]}
        print(f"  L{L}: rise={h['intent_rise_belief']['value']:+.3f} late_slope={h['intent_slope_late']['value']:+.4f} "
              f"[{h['intent_slope_late']['lo']:+.4f},{h['intent_slope_late']['hi']:+.4f}] "
              f"out_slope_pre={h['outcome_slope_pre']['value']:+.4f} out_jump={h['outcome_jump']['value']:+.3f}",
              flush=True)
    tag = f"{a.model}__clauses{a.bins}"
    pd.DataFrame(curve).to_csv(os.path.join(a.out, f"{tag}__curve.csv"), index=False)
    pd.DataFrame(hyp).to_csv(os.path.join(a.out, f"{tag}__hypotheses.csv"), index=False)
    pd.DataFrame(rsa_rows).to_csv(os.path.join(a.out, f"{tag}__rsa.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, f"{tag}__tg.npz"), stage=stage, cells=np.array(CELLS), **tg)
    pd.DataFrame([dict(model=a.model, n_kept=len(keep), **{f"dropped_{k}": v for k, v in dropped.items()})]).to_csv(
        os.path.join(a.out, f"{tag}__n.csv"), index=False)


if __name__ == "__main__":
    main()
