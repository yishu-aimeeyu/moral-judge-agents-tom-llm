#!/usr/bin/env python3
"""
88_lax_variance_reliability.py -- Lax 9/29 T2 (rating variance per story) and T3 (split-half reliability).

T2  open-weight models (7 factorial templates each), endpoint-normalized ratings in [0, 1]:
      var_tpl   variance across templates within a model, averaged over models (+ IQR)
      var_mdl   variance across models within a template, averaged over templates (+ IQR)
      var_all   variance across all model x template cells (+ IQR)
    Top / bottom 15 stories, boxplots by source x condition, OLS of each variance on source,
    condition, word count and clause lengths (words in the clause_offsets.csv spans), SEs clustered
    by scenario_group. Closed models are excluded: they were run on one factorial template only.
    MOR4 x model variance correlation is NOT run: MOR4 story numbers are not mapped to story_ids (86).

T3  Spearman split-half, Spearman-Brown corrected, --splits random splits; median and 95% interval of
    the split distribution; per-item SD of the half-mean difference across splits.
      humans    MOR4 subjects with any response (86); items = story numbers (YS2008 design).
                raw: story mean over whatever condition each subject saw (condition rotates across
                subjects, so this mixes conditions); within_condition: rating minus that condition's
                mean, the per-story residual -> the noise ceiling. cell: story x condition cells.
      templates the 7 templates within one model                (z within model x template x source)
      models    open models within the base class / instruct class, template-averaged z
      samples   the 5 samples within model x template of the closed-model sampling runs
    Per source (YS2008, YS2009, YS2011), raw and within-condition (condition means removed).
    No per-item human data exist for YS2009 / YS2011 or for the master YS2008 texts themselves.

  .venv/bin/python code/experiments/88_lax_variance_reliability.py
"""
import os, sys, glob, importlib.util, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "lax_0929")
SOURCES = ["YS2008", "YS2009", "YS2011"]
CONDS = ["neutral", "accidental", "attempted", "intentional"]
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M83 = _load("m83", os.path.join(HERE, "83_reliability_normalization.py"))


def iqr(x):
    return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)


# ------------------------------------------------------------------------- T2
def clause_lengths(master):
    co = pd.read_csv(os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    txt = master.set_index("story_id")["text"]
    out = []
    for _, r in co.iterrows():
        t = txt.get(r.story_id)
        if not isinstance(t, str):
            continue
        row = dict(story_id=r.story_id, background_words=len(t[:int(r.belief_start)].split()))
        for c in ("belief", "action", "outcome"):
            s, e = r[f"{c}_start"], r[f"{c}_end"]
            row[f"{c}_words"] = len(t[int(s):int(e)].split()) if pd.notna(s) and pd.notna(e) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def story_variance(d, master):
    g = d.groupby(["story_id", "model"])["norm_rating"]
    tpl = g.agg(var="var", iqr=iqr).groupby("story_id").mean().add_suffix("_tpl")
    g = d.groupby(["story_id", "template"])["norm_rating"]
    mdl = g.agg(var="var", iqr=iqr).groupby("story_id").mean().add_suffix("_mdl")
    al = d.groupby("story_id")["norm_rating"].agg(var="var", iqr=iqr, mean="mean").add_suffix("_all")
    v = tpl.join(mdl).join(al).reset_index()
    v = v.merge(master[["story_id", "source", "scenario_group", "condition", "word_count"]], on="story_id")
    return v.merge(clause_lengths(master), on="story_id", how="left")


def variance_regression(v):
    import statsmodels.formula.api as smf
    rows = []
    rhs = "C(source) + C(condition, Treatment('neutral')) + word_count + belief_words + action_words + outcome_words"
    for y in ("var_tpl", "var_mdl", "var_all"):
        dd = v.dropna(subset=[y, "belief_words", "action_words", "outcome_words"])
        f = smf.ols(f"{y} ~ {rhs}", dd).fit(cov_type="cluster", cov_kwds={"groups": dd["scenario_group"]})
        for term in f.params.index:
            rows.append(dict(outcome=y, term=term, coef=f.params[term], se=f.bse[term], p=f.pvalues[term],
                             n=int(f.nobs), r2=f.rsquared))
    return pd.DataFrame(rows)


def plot_variance(v, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
    for ax, (y, t) in zip(axes, [("var_tpl", "across templates (within model)"),
                                 ("var_mdl", "across models (within template)"),
                                 ("var_all", "across all model x template cells")]):
        data, labels = [], []
        for s in SOURCES:
            for c in CONDS:
                x = v[(v.source == s) & (v.condition == c)][y].dropna()
                if len(x):
                    data.append(x)
                    labels.append(f"{s}\n{c}")
        ax.boxplot(data, showfliers=True, flierprops=dict(markersize=2))
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(f"rating variance {t}", fontsize=9)
    fig.suptitle("T2: per-story rating variance (endpoint-normalized, 20 open-weight models)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- T3
def spearman_split_half(M, n_splits, rng, min_items=5):
    """M [items, raters] with NaN. Returns dict(median, lo, hi, n_ok, mean_items) and per-item SD of
    the half-mean difference over splits."""
    from scipy.stats import spearmanr
    k = M.shape[1]
    if k < 2:
        return None, None
    rs, nit, diffs = [], [], []
    for _ in range(n_splits):
        p = rng.permutation(k)
        a, b = np.nanmean(M[:, p[:k // 2]], 1), np.nanmean(M[:, p[k // 2:]], 1)
        diffs.append(a - b)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < min_items or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
            continue
        r = spearmanr(a[ok], b[ok]).statistic
        rs.append(2 * r / (1 + r) if r > -1 else np.nan)
        nit.append(ok.sum())
    sd_diff = np.nanstd(np.array(diffs), 0)
    if not rs:
        return dict(median=np.nan, lo=np.nan, hi=np.nan, n_ok_splits=0, mean_items=0), sd_diff
    rs = np.array(rs)
    return dict(median=np.nanmedian(rs), lo=np.nanpercentile(rs, 2.5), hi=np.nanpercentile(rs, 97.5),
                n_ok_splits=len(rs), mean_items=float(np.mean(nit))), sd_diff


def _wc(W, cond):
    return W.sub(W.groupby(cond.to_numpy()).transform("mean"))


def model_reliability(d, n_splits, rng):
    d = d.copy()
    e = d.groupby(["model", "template", "source"])["norm_rating"]
    d["z"] = (d["norm_rating"] - e.transform("mean")) / e.transform("std").replace(0, np.nan)
    rows, per_item = [], []
    for src in SOURCES:
        ds = d[d.source == src]
        cond = ds.drop_duplicates("story_id").set_index("story_id")["condition"]

        def add(kind, who, W):
            for mode, X in (("raw", W), ("within_condition", _wc(W, cond.loc[W.index]))):
                s, sd = spearman_split_half(X.to_numpy(float), n_splits, rng)
                if s is None:
                    continue
                rows.append(dict(rater_type=kind, who=who, source=src, mode=mode, n_items=len(W),
                                 n_raters=W.shape[1], **s))
                if mode == "within_condition":
                    per_item.append(pd.DataFrame(dict(rater_type=kind, who=who, source=src,
                                                      item=W.index, sd_half_diff=sd)))
        for m, g in ds.groupby("model"):
            if g["template"].nunique() == 7:
                add("templates", m, g.pivot(index="story_id", columns="template", values="z"))
        for c in ("base", "instruct"):
            gc = ds[(ds["class"] == c) & (ds.groupby("model")["template"].transform("nunique") == 7)]
            add("models", f"{c} ({gc.model.nunique()})",
                gc.groupby(["story_id", "model"])["z"].mean().unstack())
    return pd.DataFrame(rows), pd.concat(per_item)


def sample_reliability(n_splits, rng, master):
    lab = master.set_index("story_id")[["source", "condition"]]
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "outputs/agents/behavior_v3/raw_*.csv"))):
        d = pd.read_csv(f)
        if d["sample"].nunique() < 2:
            continue
        for (m, t, src), g in d.groupby(["model", "template", "source"]):
            W = g.pivot_table(index="story_id", columns="sample", values="norm_rating")
            if len(W) < 10:
                continue
            cond = lab.loc[W.index, "condition"]
            for mode, X in (("raw", W), ("within_condition", _wc(W, cond))):
                s, _ = spearman_split_half(X.to_numpy(float), n_splits, rng)
                if s is None:
                    continue
                rows.append(dict(rater_type="samples", who=f"{m} [{t}]", source=src, mode=mode,
                                 n_items=len(W), n_raters=W.shape[1],
                                 frac_items_identical_across_samples=float((W.nunique(1) == 1).mean()), **s))
    return pd.DataFrame(rows)


def human_reliability(n_splits, rng):
    t = pd.read_csv(os.path.join(OUT, "mor4_trials.csv"), dtype={"subject": str})
    t = t[t.rating_raw != 0]
    t["resid"] = t.rating_1to4 - t.groupby("condition").rating_1to4.transform("mean")
    rows, per_item = [], []
    specs = {"raw": ("story_num", "rating_1to4"), "within_condition": ("story_num", "resid")}
    for mode, (idx, val) in specs.items():
        W = t.pivot_table(index=idx, columns="subject", values=val)
        s, sd = spearman_split_half(W.to_numpy(float), n_splits, rng)
        rows.append(dict(rater_type="humans (MOR4)", who=f"{W.shape[1]} subjects", source="YS2008 design (MOR4)",
                         mode=mode, n_items=len(W), n_raters=W.shape[1], **s))
        per_item.append(pd.DataFrame(dict(rater_type="humans (MOR4)", mode=mode, item=W.index, sd_half_diff=sd,
                                          n_ratings=W.notna().sum(1).to_numpy())))
    t["cell"] = t.story_num.astype(str) + "_" + t.condition_name
    W = t.pivot_table(index="cell", columns="subject", values="rating_1to4")
    s, _ = spearman_split_half(W.to_numpy(float), n_splits, rng, min_items=5)
    rows.append(dict(rater_type="humans (MOR4)", who=f"{W.shape[1]} subjects", source="YS2008 design (MOR4)",
                     mode="story_x_condition_cells", n_items=len(W), n_raters=W.shape[1], **s))
    return pd.DataFrame(rows), pd.concat(per_item)


def plot_reliability(r, human, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["templates", "models", "samples"]
    rw = r[r["mode"] == "within_condition"]
    hv = human[human["mode"] == "within_condition"].iloc[0]
    fig, axes = plt.subplots(len(SOURCES), len(cols), figsize=(14, 9), sharex=True)
    for i, src in enumerate(SOURCES):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            g = rw[(rw.source == src) & (rw.rater_type == c)].sort_values("median")
            y = np.arange(len(g))
            ax.errorbar(g["median"], y, xerr=[g["median"] - g["lo"], g["hi"] - g["median"]],
                        fmt="o", ms=3, lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels([w.split("/")[-1][:38] for w in g["who"]], fontsize=5.5)
            ax.axvline(0, color="k", lw=0.5)
            if src == "YS2008":
                ax.axvspan(hv["lo"], hv["hi"], color="tab:red", alpha=0.12)
                ax.axvline(hv["median"], color="tab:red", lw=1.2,
                           label=f"MOR4 humans {hv['median']:.2f}")
                ax.legend(fontsize=6, loc="lower left")
            else:
                ax.text(0.02, 0.98, "no per-item human data", transform=ax.transAxes, fontsize=6,
                        va="top", color="0.4")
            if i == 0:
                ax.set_title(f"raters = {c}", fontsize=9)
            if j == 0:
                ax.set_ylabel(src)
    for ax in axes[-1]:
        ax.set_xlabel("split-half Spearman-Brown (within condition)")
    fig.suptitle("T3: split-half reliability per source (median, 95% split interval)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    master = pd.read_csv(os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    d = M83.load_ratings(M83.RAW_GLOBS, M83.FACTORIAL)
    open_w = d[d.groupby("model")["template"].transform("nunique") == 7]
    print("T2 models:", open_w.model.nunique())

    v = story_variance(open_w, master)
    v.to_csv(os.path.join(OUT, "story_variance.csv"), index=False)
    keep = ["story_id", "source", "condition", "word_count", "var_all", "iqr_all", "var_tpl", "var_mdl", "mean_all"]
    s = v.sort_values("var_all")
    tb = pd.concat([s.tail(15)[::-1].assign(rank="top"), s.head(15).assign(rank="bottom")])[["rank"] + keep]
    tb.to_csv(os.path.join(OUT, "story_variance_top_bottom15.csv"), index=False)
    print(tb.round(4).to_string(index=False))
    sc = v.groupby(["source", "condition"])[["var_tpl", "var_mdl", "var_all"]].agg(["mean", "median", "count"])
    sc.to_csv(os.path.join(OUT, "story_variance_by_source_condition.csv"))
    print(sc.round(4).to_string())
    reg = variance_regression(v)
    reg.to_csv(os.path.join(OUT, "story_variance_regression.csv"), index=False)
    print(reg.round(4).to_string(index=False))
    plot_variance(v, os.path.join(OUT, "fig_T2_story_variance"))

    rm, pim = model_reliability(d, a.splits, rng)
    rs = sample_reliability(a.splits, rng, master)
    rh, pih = human_reliability(a.splits, rng)
    r = pd.concat([rh, rm, rs], ignore_index=True)
    r.to_csv(os.path.join(OUT, "split_half_reliability.csv"), index=False)
    pd.concat([pih, pim]).to_csv(os.path.join(OUT, "split_half_per_item_sd.csv"), index=False)
    agg = r.groupby(["rater_type", "source", "mode"])["median"].agg(["median", "min", "max", "count"])
    agg.to_csv(os.path.join(OUT, "split_half_summary.csv"))
    print(rh.round(3).to_string(index=False))
    print(agg.round(3).to_string())
    plot_reliability(r, rh, os.path.join(OUT, "fig_T3_split_half"))


if __name__ == "__main__":
    main()
