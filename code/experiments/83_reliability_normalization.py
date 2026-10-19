#!/usr/bin/env python3
"""
83_reliability_normalization.py -- R1-R4: are the behavioural results robust to how the
ratings are normalized, how reliable is each story, and what drives the clusters?

Uses existing raw_<model>.csv ratings only (no GPU). Four analyses:

R1 NORMALIZATION. The headline contrast (attempted - accidental) is computed in
   scale-endpoint min-max space. Logprob-expected ratings are compressed toward the scale
   midpoint, so small endpoint-normalized contrasts may be calibration, not judgment. Each
   model x template is re-normalized five ways and the contrast / interaction recomputed:
     endpoint   (x - s_min)/(s_max - s_min)                 -- current
     observed   min-max over that model x template's own 298 ratings
     z_global   z-score over all items of that model x template
     z_source   z-score within source (removes the per-source scale and wording)
     rank       percentile rank within model x template
   Reported: Spearman of the model ordering between schemes, and sign flips.

R2 SPLIT-HALF RELIABILITY per source. "Raters" are split into random halves, item means
   correlated across stories, Spearman-Brown corrected, over --splits random splits:
     templates  raters = prompt templates within one model   (prompt-wording reliability)
     models     raters = models, template-averaged          (cross-model agreement)
     humans     raters = participants, if --human-subjects is given (subject,story_id,rating)
   Each is reported raw AND within-condition (condition means removed per source), because
   the 2x2 condition effect alone makes any rater set look reliable; the within-condition
   number is whether raters agree on WHICH stories are worse.

R3 PER-STORY CONSISTENCY. For every story: SD across templates (averaged over models) and
   SD across models; sorted, with a test of whether inconsistency concentrates in a source,
   condition or scenario.

R4 CLUSTERING and VARIANCE SHARES. Stories clustered on their (model x template) rating
   profile, models clustered on their 298-story profile; adjusted Rand index against
   condition / source / intent / outcome (stories) and base-vs-instruct / family (models).
   Plus per-model sum-of-squares shares for condition, scenario (an item prior that is not
   in the manipulation), template, condition x template and residual.

Output: outputs/analysis/reliability/*.csv, *.png, SUMMARY.md
"""
import os, re, glob, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "analysis", "reliability")
FACTORIAL = ["human_verbatim", "blame_w1", "blame_w2", "wrong_w1", "wrong_w2",
             "punish_w1", "punish_w2"]
RAW_GLOBS = ["outputs/behavior/raw_*.csv", "outputs/agents/behavior_v3/raw_*.csv",
             "outputs/agents/behavior/raw_*.csv"]
DEGENERATE = {"mock_smoke"}


def model_safe(name):
    return re.sub(r"[^\w\-]", "_", name)


def load_ratings(globs, templates, extra_globs=()):
    """First file per model wins across `globs` (the same model can appear in several run
    folders); `extra_globs` files are always added, as extra templates of their model."""
    seen, frames = set(), []
    for g in globs:
        for f in sorted(glob.glob(os.path.join(ROOT, g))):
            d = pd.read_csv(f)
            if d.empty:
                continue
            m = str(d["model"].iloc[0])
            if m in seen or model_safe(m) in DEGENERATE or "mock" in m:
                continue
            seen.add(m)
            frames.append(d)
    for g in extra_globs:
        frames += [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, g)))]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["template"].isin(templates)]
    d = d.groupby(["model", "template", "story_id", "source", "condition", "intent_label",
                   "outcome_label"], as_index=False)[["raw_rating", "norm_rating"]].mean()
    # a constant or fragmentary rater has no item ordering to be reliable about
    st = d.groupby(["model", "template"])["norm_rating"].agg(["std", "count"])
    bad = st[(st["std"].fillna(0) < 1e-9) | (st["count"] < 200)]
    for (m, t), r in bad.iterrows():
        print(f"  dropped {m} [{t}]: n={int(r['count'])}, sd={r['std']:.3g}")
    d = d.set_index(["model", "template"]).drop(bad.index).reset_index()
    reg = pd.read_csv(os.path.join(ROOT, "dataset", "model_registry.csv"))
    reg["key"] = reg["model_id"].map(model_safe)
    cls = dict(zip(reg["key"], reg["class"]))
    prov = dict(zip(reg["key"], reg["provider"]))
    d["class"] = d["model"].map(lambda m: cls.get(model_safe(m), "instruct" if "nstruct" in m else "?"))
    d["family"] = d["model"].map(lambda m: prov.get(model_safe(m), m.split("/")[0]))
    master = pd.read_csv(os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"),
                         usecols=["story_id", "scenario_group"])
    return d.merge(master, on="story_id", how="left")


# ------------------------------------------------------------------------- R1 normalization
def renormalize(d):
    d = d.copy()
    d["endpoint"] = d["norm_rating"]
    # endpoint-normalized ratings are already direction-aligned (higher = more blame); the
    # other schemes are built from them so a reversed scale cannot flip their sign
    e = d.groupby(["model", "template"])["endpoint"]
    lo, hi = e.transform("min"), e.transform("max")
    d["observed"] = (d["endpoint"] - lo) / (hi - lo).replace(0, np.nan)
    d["z_global"] = (d["endpoint"] - e.transform("mean")) / e.transform("std").replace(0, np.nan)
    es = d.groupby(["model", "template", "source"])["endpoint"]
    d["z_source"] = (d["endpoint"] - es.transform("mean")) / es.transform("std").replace(0, np.nan)
    d["rank"] = e.rank(pct=True)
    return d


SCHEMES = ["endpoint", "observed", "z_global", "z_source", "rank"]


def cell_effects(d, col):
    m = d.groupby("condition")[col].mean()
    return dict(contrast=m["attempted"] - m["accidental"],
                intent=(m["attempted"] + m["intentional"] - m["neutral"] - m["accidental"]) / 2,
                outcome=(m["accidental"] + m["intentional"] - m["neutral"] - m["attempted"]) / 2,
                interaction=(m["intentional"] - m["attempted"]) - (m["accidental"] - m["neutral"]))


def r1(d, out):
    from scipy.stats import spearmanr
    full = d[d["source"].isin(["YS2008", "YS2009"])]
    rows = []
    for (m, t), g in full.groupby(["model", "template"]):
        for s in SCHEMES:
            rows.append(dict(model=m, template=t, scheme=s, cls=g["class"].iloc[0],
                             **{k: round(v, 4) for k, v in cell_effects(g, s).items()}))
    for m, g in full.groupby("model"):
        for s in SCHEMES:
            rows.append(dict(model=m, template="ALL_MEAN", scheme=s, cls=g["class"].iloc[0],
                             **{k: round(v, 4) for k, v in cell_effects(g, s).items()}))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "R1_effects_by_normalization.csv"), index=False)
    a = r[r["template"] == "ALL_MEAN"].pivot(index="model", columns="scheme", values="contrast")
    summ = []
    for s in SCHEMES[1:]:
        rho = spearmanr(a["endpoint"], a[s]).statistic
        flips = int((np.sign(a["endpoint"]) != np.sign(a[s])).sum())
        per_t = r[r["template"] != "ALL_MEAN"].pivot_table(index=["model", "template"],
                                                           columns="scheme", values="contrast")
        flips_t = int((np.sign(per_t["endpoint"]) != np.sign(per_t[s])).sum())
        summ.append(dict(scheme=s, spearman_vs_endpoint=round(rho, 3),
                         model_sign_flips=flips, model_x_template_sign_flips=flips_t,
                         n_models=len(a), n_model_templates=len(per_t)))
    s = pd.DataFrame(summ)
    s.to_csv(os.path.join(out, "R1_normalization_summary.csv"), index=False)
    return a, s


# ------------------------------------------------------------------------- R2 reliability
def split_half(M, n_splits, rng):
    """M: [items, raters] (NaN allowed). -> (mean SB-corrected r, 2.5%, 97.5%)."""
    k = M.shape[1]
    if k < 2:
        return np.nan, np.nan, np.nan
    rs = []
    for _ in range(n_splits):
        p = rng.permutation(k)
        a, b = np.nanmean(M[:, p[:k // 2]], 1), np.nanmean(M[:, p[k // 2:]], 1)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
            continue
        r = np.corrcoef(a[ok], b[ok])[0, 1]
        rs.append(2 * r / (1 + r))
    if not rs:
        return np.nan, np.nan, np.nan
    rs = np.array(rs)
    return rs.mean(), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def _within_condition(wide, cond):
    return wide.sub(wide.groupby(cond).transform("mean"))


def r2(d, out, n_splits, human_path):
    rng = np.random.default_rng(0)
    rows = []

    def add(kind, who, src, wide, cond):
        for mode, W in (("raw", wide), ("within_condition", _within_condition(wide, cond))):
            m, lo, hi = split_half(W.to_numpy(float), n_splits, rng)
            rows.append(dict(rater_type=kind, who=who, source=src, mode=mode,
                             n_items=len(W), n_raters=W.shape[1],
                             split_half_sb=round(m, 3), lo=round(lo, 3), hi=round(hi, 3)))

    for src, ds in d.groupby("source"):
        cond = ds.drop_duplicates("story_id").set_index("story_id")["condition"]
        for m, g in ds.groupby("model"):
            w = g.pivot(index="story_id", columns="template", values="z_global")
            add("templates", m, src, w, cond.loc[w.index])
        w = ds.groupby(["story_id", "model"])["z_global"].mean().unstack()
        add("models", "all", src, w, cond.loc[w.index])
        for c, gc in ds.groupby("class"):
            w = gc.groupby(["story_id", "model"])["z_global"].mean().unstack()
            add("models", f"class={c}", src, w, cond.loc[w.index])
    if human_path:
        h = pd.read_csv(human_path)
        lab = d.drop_duplicates("story_id").set_index("story_id")[["source", "condition"]]
        h = h.join(lab, on="story_id")
        for src, hs in h.groupby("source"):
            w = hs.pivot_table(index="story_id", columns="subject", values="rating")
            add("humans", "participants", src, w, lab.loc[w.index, "condition"])
    else:
        print("R2: no --human-subjects file; human split-half skipped")
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(out, "R2_split_half.csv"), index=False)
    return r


# ------------------------------------------------------------------------- R3 per story
def r3(d, out):
    from scipy.stats import kruskal
    sd_t = d.groupby(["model", "story_id"])["z_global"].std().groupby("story_id").mean()
    item = d.groupby(["story_id", "model"])["z_global"].mean().unstack()
    sd_m = item.std(1)
    meta = d.drop_duplicates("story_id").set_index("story_id")[
        ["source", "condition", "scenario_group"]]
    s = meta.join(pd.DataFrame(dict(mean_z=item.mean(1), sd_across_templates=sd_t,
                                    sd_across_models=sd_m)))
    s = s.sort_values("sd_across_models", ascending=False)
    s["rank_inconsistent"] = np.arange(1, len(s) + 1)
    s.round(4).to_csv(os.path.join(out, "R3_story_consistency.csv"))
    tests = []
    for col in ("sd_across_templates", "sd_across_models"):
        for fac in ("source", "condition", "scenario_group"):
            groups = [g[col].dropna().to_numpy() for _, g in s.groupby(fac) if len(g) > 1]
            H, p = kruskal(*groups)
            tests.append(dict(measure=col, factor=fac, kruskal_H=round(H, 2), p=float(f"{p:.3g}"),
                              **{f"mean_{k}": round(v, 3) for k, v in
                                 s.groupby(fac)[col].mean().items()} if fac != "scenario_group" else {}))
    t = pd.DataFrame(tests)
    t.to_csv(os.path.join(out, "R3_consistency_tests.csv"), index=False)
    return s, t


# ------------------------------------------------------------------------- R4 clusters
def r4(d, out, k_story):
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import adjusted_rand_score as ari
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    res = []
    X = d.pivot_table(index="story_id", columns=["model", "template"], values="z_global")
    X = X.dropna(axis=1, thresh=int(0.9 * len(X))).apply(lambda c: c.fillna(c.mean()))
    meta = d.drop_duplicates("story_id").set_index("story_id").loc[X.index]
    for resid in (False, True):
        Y = X.sub(X.groupby(meta["condition"]).transform("mean")) if resid else X
        lab = fcluster(linkage(Y.to_numpy(), "ward"), k_story, "maxclust")
        for f in ("condition", "source", "intent_label", "outcome_label", "scenario_group"):
            res.append(dict(what="stories", residualized_condition=resid, k=k_story, factor=f,
                            ARI=round(ari(meta[f], lab), 3)))
    P = d[d.source.isin(["YS2008", "YS2009"])].groupby(["model", "story_id"])["z_global"].mean().unstack()
    corr = np.corrcoef(P.fillna(0).to_numpy())
    mm = d.drop_duplicates("model").set_index("model").loc[P.index]
    Z = linkage(P.fillna(0).to_numpy(), "average", metric="correlation")
    for k in (2, 3, 4):
        lab = fcluster(Z, k, "maxclust")
        for f in ("class", "family"):
            res.append(dict(what="models", residualized_condition=False, k=k, factor=f,
                            ARI=round(ari(mm[f], lab), 3)))
    pd.DataFrame(res).to_csv(os.path.join(out, "R4_cluster_alignment.csv"), index=False)

    shares = []
    for m, g in d[d.source.isin(["YS2008", "YS2009"])].groupby("model"):
        if g["template"].nunique() < 2:
            continue
        # endpoint, not z_global: z-scoring per template would delete the template main effect
        fit = smf.ols("endpoint ~ C(condition) * C(template) + C(scenario_group)", data=g).fit()
        a = anova_lm(fit, typ=2)
        tot = a["sum_sq"].sum()
        sh = (a["sum_sq"] / tot).rename(lambda s: s.replace("C(", "").replace(")", ""))
        shares.append(dict(model=m, cls=g["class"].iloc[0], **{k: round(v, 3) for k, v in sh.items()}))
    pd.DataFrame(shares).to_csv(os.path.join(out, "R4_variance_shares.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    order = dendrogram(Z, no_plot=True)["leaves"]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    names = [f"{P.index[i].split('/')[-1]} [{mm['class'].iloc[i]}]" for i in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(names, fontsize=6)
    ax.set_title("model x model correlation of 288-story rating profiles (z_global, template mean)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "R4_model_profile_clusters.png"), dpi=150)
    return pd.DataFrame(res), pd.DataFrame(shares)


def write_summary(out, s1, r2t, s3, t3, c4, v4):
    def md(df):
        try:
            return df.to_markdown(index=False)
        except ImportError:                      # tabulate is optional
            return "```\n" + df.to_string(index=False) + "\n```"
    lines = ["# Reliability / normalization summary (83_reliability_normalization.py)", "",
             "## R1 normalization: does the model ordering of the contrast survive?", md(s1), "",
             "## R2 split-half reliability (Spearman-Brown), template-raters median per source",
             md(r2t[r2t.rater_type == "templates"].groupby(["source", "mode"])["split_half_sb"]
                .median().reset_index()), "",
             "### model-raters", md(r2t[r2t.rater_type != "templates"]), "",
             "## R3 least consistent stories (top 15 by SD across models)",
             md(s3.head(15).reset_index().round(3)), "", md(t3), "",
             "## R4 cluster alignment (ARI)", md(c4), "",
             "## R4 variance shares", md(v4)]
    open(os.path.join(out, "SUMMARY.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--templates", nargs="+", default=FACTORIAL + ["participant_exact"])
    ap.add_argument("--raw-globs", nargs="+", default=RAW_GLOBS)
    ap.add_argument("--extra-globs", nargs="*", default=["outputs/behavior_participant_exact/raw_*.csv"],
                    help="additional raw files merged into the same model (e.g. participant_exact)")
    ap.add_argument("--human-subjects", default=None,
                    help="CSV with columns subject,story_id,rating (blame direction, any scale)")
    ap.add_argument("--splits", type=int, default=1000)
    ap.add_argument("--k-story", type=int, default=4)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_ratings(a.raw_globs, a.templates, a.extra_globs)
    print(f"{d['model'].nunique()} models, templates {sorted(d['template'].unique())}")
    d = renormalize(d)
    a1, s1 = r1(d, a.out); print(s1.to_string(index=False))
    r2t = r2(d, a.out, a.splits, a.human_subjects)
    s3, t3 = r3(d, a.out)
    c4, v4 = r4(d, a.out, a.k_story)
    write_summary(a.out, s1, r2t, s3, t3, c4, v4)
    print(f"-> {a.out}")
