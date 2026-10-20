#!/usr/bin/env python3
"""
91_lax_template_heldout_decoding.py -- Lax 9/29 T5: end-of-prompt INTENT decoding with templates held out.

Input: 90's <model>__<template>__chat<on|off>.npz (last prompt token, every layer, 7 FACTORIAL templates).
Every split is crossed with GroupKFold(5) by scenario_group: test stories never share a scenario with
training stories, AND (for the cross conditions) are read under templates never seen in training.

  within     train template t, test template t                          (the within-template baseline)
  pair       train template a, test template b != a                     (7 x 7 matrix, off-diagonal mean)
  construct  train on 5 templates, test on the 2 wordings of the held-out construct
             (blame_w1/2, wrong_w1/2, punish_w1/2; human_verbatim always in training) -- disjoint sets A / B
  gap        within - pair, within - construct, per layer

Permutation null (02_probe.permute_within_groups on intent, the same permutation for every template of
a story) at the fixed layer (relative depth 0.65) and the peak construct layer; the peak p is
selection-biased. TF-IDF baseline (21 feature sets) on the full prompt text with the same splits.

  .venv/bin/python code/experiments/91_lax_template_heldout_decoding.py --model Qwen2.5-7B --perm 1000
"""
import os, glob, argparse, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "lax_0929", "T5")
N_JOBS = int(os.environ.get("JOBLIB_N_JOBS", "2"))
CONSTRUCTS = {"blame": ["blame_w1", "blame_w2"], "wrong": ["wrong_w1", "wrong_w2"],
              "punish": ["punish_w1", "punish_w2"]}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P02 = _load("p02", "code/02_probe.py")
B03 = _load("b03", "code/03_behavioral.py")
S21 = _load("s21", "code/experiments/21_surface_baseline.py")
TEMPLATES = B03.FACTORIAL_TEMPLATES


def load(model, tpl_dir, chat):
    X, sids = {}, None
    for t in TEMPLATES:
        pats = [f"{model}__{t}__chat{c}.npz" for c in (["on", "off"] if chat == "auto" else [chat])]
        hit = next((os.path.join(tpl_dir, p) for p in pats if os.path.exists(os.path.join(tpl_dir, p))), None)
        if hit is None:
            raise SystemExit(f"missing {pats[0]} in {tpl_dir}")
        d = np.load(hit, allow_pickle=False)
        s = [str(x) for x in d["story_id"]]
        assert sids is None or s == sids, "story order differs between templates"
        sids, X[t] = s, d["last"]
        chat_used = bool(d["chat"])
    return X, sids, chat_used


class Proj:
    """Label-independent part of the 02 probe (standardize + row-space projection), cached."""

    def __init__(self, Xtr, Xtests):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Z = sc.transform(Xtr)
        if Z.shape[1] > Z.shape[0]:
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            V = Vt[S > 1e-10].T
        else:
            V = np.eye(Z.shape[1])
        self.A = Z @ V
        self.B = [sc.transform(x) @ V for x in Xtests]

    def fit_score(self, ytr, ytests):
        from sklearn.linear_model import LogisticRegression
        if self.A.shape[1] == 0 or len(np.unique(ytr)) < 2:
            return [max(y.mean(), 1 - y.mean()) for y in ytests]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(self.A, ytr)
        return [clf.score(b, y) for b, y in zip(self.B, ytests)]


def build_projs(X, L, folds):
    """X[t] [n, layers, H]; returns {'pair': {(fold, a): Proj tested on all 7}, 'construct': {(fold, c): Proj}}"""
    pair, cons = {}, {}
    for f, (tr, te) in enumerate(folds):
        for a in TEMPLATES:
            pair[(f, a)] = Proj(X[a][tr, L].astype(np.float32), [X[b][te, L].astype(np.float32) for b in TEMPLATES])
        for c, held in CONSTRUCTS.items():
            train_t = [t for t in TEMPLATES if t not in held]
            Xtr = np.concatenate([X[t][tr, L] for t in train_t]).astype(np.float32)
            cons[(f, c)] = (Proj(Xtr, [X[t][te, L].astype(np.float32) for t in held]), len(train_t))
    return pair, cons


def evaluate(pair, cons, folds, y):
    M = np.zeros((len(TEMPLATES), len(TEMPLATES)))
    for f, (tr, te) in enumerate(folds):
        for i, a in enumerate(TEMPLATES):
            M[i] += np.array(pair[(f, a)].fit_score(y[tr], [y[te]] * len(TEMPLATES))) / len(folds)
    cacc = {}
    for c, held in CONSTRUCTS.items():
        s = []
        for f, (tr, te) in enumerate(folds):
            P, k = cons[(f, c)]
            s.append(np.mean(P.fit_score(np.tile(y[tr], k), [y[te]] * len(held))))
        cacc[c] = float(np.mean(s))
    off = ~np.eye(len(TEMPLATES), dtype=bool)
    return dict(within=float(np.diag(M).mean()), pair=float(M[off].mean()),
                construct=float(np.mean(list(cacc.values()))), **{f"construct_{c}": v for c, v in cacc.items()}), M


def tfidf(texts, y, folds):
    """texts[t] list of prompt strings; same within / pair / construct splits on TF-IDF features."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    res = {}
    for name, kw in S21.FEATURE_SETS.items():
        within, pair, cons = [], [], []
        for tr, te in folds:
            for a in TEMPLATES:
                v = TfidfVectorizer(**kw)
                clf = LogisticRegression(max_iter=3000).fit(v.fit_transform([texts[a][i] for i in tr]), y[tr])
                for b in TEMPLATES:
                    acc = clf.score(v.transform([texts[b][i] for i in te]), y[te])
                    (within if a == b else pair).append(acc)
            for c, held in CONSTRUCTS.items():
                tt = [t for t in TEMPLATES if t not in held]
                v = TfidfVectorizer(**kw)
                A = v.fit_transform([texts[t][i] for t in tt for i in tr])
                clf = LogisticRegression(max_iter=3000).fit(A, np.tile(y[tr], len(tt)))
                cons.append(np.mean([clf.score(v.transform([texts[t][i] for i in te]), y[te]) for t in held]))
        res[name] = dict(within=np.mean(within), pair=np.mean(pair), construct=np.mean(cons))
    return res


def main():
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model tag, e.g. Qwen2.5-7B")
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--tpl-dir", default=os.environ.get(
        "TOM_TPL_DIR", f"/orcd/scratch/orcd/007/{os.environ.get('USER', '')}/acts_templates"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    X, sids, chat_used = load(a.model, a.tpl_dir, a.chat)
    lab = P02.load_labels(os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    y = np.array([lab[s]["intent_label"] == "guilty" for s in sids], int)
    groups = np.array([lab[s]["scenario_group"] for s in sids])
    folds = list(GroupKFold(n_splits=5).split(sids, None, groups))
    nL = X[TEMPLATES[0]].shape[1]
    tag = f"{a.model}__chat{'on' if chat_used else 'off'}"
    print(f"{tag}: {len(sids)} stories, {nL} layers, {N_JOBS} jobs", flush=True)

    def layer(L):
        pair, cons = build_projs(X, L, folds)
        r, M = evaluate(pair, cons, folds, y)
        return L, r, M

    res = Parallel(n_jobs=N_JOBS)(delayed(layer)(L) for L in range(nL))
    rows, mats = [], {}
    for L, r, M in sorted(res, key=lambda t: t[0]):
        rows.append(dict(model=a.model, chat=chat_used, layer=L, **r,
                         gap_pair=r["within"] - r["pair"], gap_construct=r["within"] - r["construct"]))
        mats[f"L{L}"] = M
    df = pd.DataFrame(rows)

    texts = {t: [B03.build_prompt(lab[s]["text"], t, lab[s]["source"])[0] for s in sids] for t in TEMPLATES}
    surf = tfidf(texts, y, folds)
    for name, r in surf.items():
        for k, v in r.items():
            df[f"tfidf_{name}_{k}"] = v

    fixed = int(round(0.65 * (nL - 1)))
    peak = int(df.iloc[1:].sort_values("construct").iloc[-1]["layer"])
    prows = []
    for kind, L in (("fixed_0.65", fixed), ("peak_construct", peak)):
        pair, cons = build_projs(X, L, folds)
        obs, _ = evaluate(pair, cons, folds, y)
        rng = np.random.default_rng(0)
        perms = [P02.permute_within_groups(y, groups, rng) for _ in range(a.perm)]
        null = Parallel(n_jobs=N_JOBS)(delayed(lambda yp: evaluate(pair, cons, folds, yp)[0])(yp) for yp in perms)
        for k in ("within", "pair", "construct"):
            nv = np.array([n[k] for n in null])
            prows.append(dict(model=a.model, chat=chat_used, layer_kind=kind, layer=L, measure=k, obs=obs[k],
                              null_mean=nv.mean(), null_p95=np.percentile(nv, 95),
                              perm_p=(np.sum(nv >= obs[k]) + 1) / (a.perm + 1), n_perm=a.perm))
        gap_null = np.array([n["within"] - n["construct"] for n in null])
        prows.append(dict(model=a.model, chat=chat_used, layer_kind=kind, layer=L, measure="gap_construct",
                          obs=obs["within"] - obs["construct"], null_mean=gap_null.mean(),
                          null_p95=np.percentile(gap_null, 95),
                          perm_p=(np.sum(gap_null >= obs["within"] - obs["construct"]) + 1) / (a.perm + 1),
                          n_perm=a.perm))
        print(f"  perm {kind} L{L}: " + "  ".join(f"{r['measure']}={r['obs']:.3f} p={r['perm_p']:.4f}"
                                                for r in prows[-4:]), flush=True)
    df.to_csv(os.path.join(a.out, f"{tag}__layers.csv"), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(a.out, f"{tag}__perm.csv"), index=False)
    np.savez_compressed(os.path.join(a.out, f"{tag}__pair_matrices.npz"), templates=np.array(TEMPLATES), **mats)
    print(df[["layer", "within", "pair", "construct", "gap_pair", "gap_construct"]].round(3).to_string(index=False))
    print({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in surf.items()})


if __name__ == "__main__":
    main()
