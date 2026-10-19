#!/usr/bin/env python3
"""
87_lax_normalization_audit.py -- Lax 9/29 T1: normalization audit + item clustering.

(1) scale_audit.csv: source x template -> native scale, anchor wording, direction, parsed from
    03_behavioral.build_prompt / PARA_TEMPLATES, plus what participants saw (raw_text appendices,
    human_reference/README.md, MOR4 run files).
(2) Intent-reliance contrast (attempted - accidental, YS2008+YS2009, template-averaged) per model
    under four normalizations, 95% CI from a scenario_group bootstrap:
      endpoint     (x - s_min)/(s_max - s_min) per template scale  -- the current headline
      global_mm    one min-max per model over all its endpoint ratings (all templates, sources)
      z_mts        z-score within model x template x source
      rank_mts     percentile rank within model x template x source
    Model ordering (Spearman vs endpoint), sign flips, and the instruct - base difference for every
    matched pair and for the class means.
(3) Clustering of the 298 items on their model x template rating profile (open-weight models):
    Ward linkage + metric MDS (UMAP is not installed in .venv), silhouette and ARI for source,
    native scale, condition, intent, outcome; repeated with human_verbatim-only features (native
    scale differs by source) and factorial-only features (common 1-7 scale).

  .venv/bin/python code/experiments/87_lax_normalization_audit.py
"""
import os, re, sys, importlib.util, warnings
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "outputs", "lax_0929")
SCHEMES = ["endpoint", "global_mm", "z_mts", "rank_mts"]
SEED = 0
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.argv, argv = [path], sys.argv
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


M83 = _load("m83", os.path.join(HERE, "83_reliability_normalization.py"))
B = _load("beh", os.path.join(ROOT, "code", "03_behavioral.py"))


# ------------------------------------------------------------------------- (1) scale audit
def _anchors(q):
    a = re.findall(r"(\d+)\s*\(([^)]*)\)", q)
    if a:
        return a[0][1], a[-1][1]
    a = re.findall(r"(\d+)\s*=\s*([a-z ]+)", q)
    return (a[0][1].strip(), a[-1][1].strip()) if a else ("(none)", "(none)")


def scale_audit():
    rows = []
    for src in ["YS2008", "YS2009", "YS2011"]:
        for t in B.FACTORIAL_TEMPLATES:
            q, lo, hi = B.build_prompt("Agent did a thing.", t, src)
            q = q.split("\n\n", 1)[1]
            a_lo, a_hi = _anchors(q)
            rows.append(dict(rater="model", source=src, template=t, scale_min=lo, scale_max=hi,
                             construct=B.TEMPLATE_META[t]["construct"], question=q,
                             anchor_low=a_lo, anchor_high=a_hi,
                             direction="higher = more condemnation"))
    human = [
        ("YS2008", "participants (YS2008 appendix)", 1, 3, "Forbidden", "Permissible",
         "higher = LESS condemnation (reversed vs model human_verbatim)", "dataset/raw_text/YS2008.txt"),
        ("YS2009", "participants (YS2009 appendix)", 1, 4, "None", "A Lot",
         "higher = more blame", "dataset/raw_text/YS2009.txt"),
        ("YS2011", "participants (YS2011 Exp.1)", 1, 7, "not at all", "very",
         "higher = more wrong", "dataset/raw_text/YS2011*.txt / 84_participant_exact_prompts.py"),
        ("adult anchor", "Young et al. 2007 Exp.1 cell means", 1, 4, "forbidden", "permissible",
         "higher = LESS condemnation", "dataset/human_reference/README.md"),
        ("MOR4 (YS2008 design)", "MOR4 scanner button box", 50, 52, "(not recorded)", "(not recorded)",
         "inferred from data: higher key = more condemnation (all-neutral cell ~ lowest key, SD 0.26)",
         "86_mor4_trials.py / mor4_condition_means.csv"),
    ]
    for src, t, lo, hi, al, ah, d, f in human:
        rows.append(dict(rater="human", source=src, template=t, scale_min=lo, scale_max=hi,
                         construct="as run", question=f, anchor_low=al, anchor_high=ah, direction=d))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- (2) normalizations
def renorm(d):
    d = d.copy()
    d["endpoint"] = d["norm_rating"]
    g = d.groupby("model")["endpoint"]
    d["global_mm"] = (d["endpoint"] - g.transform("min")) / (g.transform("max") - g.transform("min"))
    k = ["model", "template", "source"]
    e = d.groupby(k)["endpoint"]
    d["z_mts"] = (d["endpoint"] - e.transform("mean")) / e.transform("std").replace(0, np.nan)
    d["rank_mts"] = e.rank(pct=True)
    return d


def contrast_table(d):
    """per model x scheme x scenario_group: template-averaged attempted, accidental means."""
    f = d[d["source"].isin(["YS2008", "YS2009"]) & d["condition"].isin(["attempted", "accidental"])]
    rows = []
    for s in SCHEMES:
        p = f.groupby(["model", "scenario_group", "condition"])[s].agg(["sum", "count"]).reset_index()
        p["scheme"] = s
        rows.append(p)
    return pd.concat(rows)


def boot_contrast(p, n_boot=2000, seed=SEED):
    """p: rows for one model x scheme. Contrast = mean(attempted) - mean(accidental) over items."""
    w = p.pivot_table(index="scenario_group", columns="condition", values=["sum", "count"], fill_value=0)
    S = w["sum"][["attempted", "accidental"]].to_numpy()
    N = w["count"][["attempted", "accidental"]].to_numpy()
    est = S[:, 0].sum() / N[:, 0].sum() - S[:, 1].sum() / N[:, 1].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(n_boot, len(S)))
    s, n = S[idx].sum(1), N[idx].sum(1)
    bs = s[:, 0] / n[:, 0] - s[:, 1] / n[:, 1]
    return est, bs


PAIRS = [("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"), ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
         ("Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"), ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
         ("Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct"),
         ("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-Instruct"),
         ("mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3"),
         ("unsloth/Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct"),
         ("unsloth/gemma-2-9b", "unsloth/gemma-2-9b-it")]


def normalization_robustness(d, n_boot):
    from scipy.stats import spearmanr
    ct = contrast_table(renorm(d))
    cls = d.groupby("model")["class"].first()
    ntpl = d.groupby("model")["template"].nunique()
    rows, boots = [], {}
    for (m, s), p in ct.groupby(["model", "scheme"]):
        est, bs = boot_contrast(p, n_boot)
        boots[(m, s)] = bs
        rows.append(dict(model=m, cls=cls[m], n_templates=int(ntpl[m]), scheme=s, contrast=est,
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    r = pd.DataFrame(rows)
    # within each scheme, put contrasts on a common unit (contrast / SD of that scheme's values
    # across the model's items) is NOT done: schemes are compared on ordering and sign only.
    wide = r.pivot(index="model", columns="scheme", values="contrast")
    summ = []
    for s in SCHEMES:
        summ.append(dict(scheme=s, spearman_vs_endpoint=spearmanr(wide["endpoint"], wide[s]).statistic,
                         sign_flips_vs_endpoint=int((np.sign(wide["endpoint"]) != np.sign(wide[s])).sum()),
                         flipped_models=";".join(wide.index[np.sign(wide["endpoint"]) != np.sign(wide[s])]),
                         n_models=len(wide)))
    pairs = []
    for s in SCHEMES:
        for b, i in PAIRS:
            if (b, s) not in boots or (i, s) not in boots:
                continue
            diff = boots[(i, s)] - boots[(b, s)]
            pairs.append(dict(scheme=s, base=b, instruct=i,
                              diff=wide.loc[i, s] - wide.loc[b, s],
                              ci_lo=np.percentile(diff, 2.5), ci_hi=np.percentile(diff, 97.5)))
        bm = [m for m in wide.index if cls[m] == "base" and "[chat off]" not in m]
        im = [m for m in wide.index if cls[m] == "instruct" and ntpl[m] == 7]
        diff = np.mean([boots[(m, s)] for m in im], 0) - np.mean([boots[(m, s)] for m in bm], 0)
        pairs.append(dict(scheme=s, base=f"mean of {len(bm)} base", instruct=f"mean of {len(im)} open instruct",
                          diff=wide.loc[im, s].mean() - wide.loc[bm, s].mean(),
                          ci_lo=np.percentile(diff, 2.5), ci_hi=np.percentile(diff, 97.5)))
    pr = pd.DataFrame(pairs)
    pr["sign"] = np.sign(pr["diff"]).astype(int)
    pr["ci_excludes_0"] = (pr["ci_lo"] > 0) | (pr["ci_hi"] < 0)
    return r, pd.DataFrame(summ), pr


def plot_normalization(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = r[r.scheme == "endpoint"].sort_values("contrast")["model"].tolist()
    fig, axes = plt.subplots(1, len(SCHEMES), figsize=(4 * len(SCHEMES), 0.28 * len(order) + 1.5), sharey=True)
    for ax, s in zip(axes, SCHEMES):
        g = r[r.scheme == s].set_index("model").loc[order]
        y = np.arange(len(order))
        col = ["tab:orange" if c == "base" else "tab:blue" for c in g["cls"]]
        ax.errorbar(g["contrast"], y, xerr=[g["contrast"] - g["ci_lo"], g["ci_hi"] - g["contrast"]],
                    fmt="none", ecolor="0.6", lw=1)
        ax.scatter(g["contrast"], y, c=col, s=18, zorder=3)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(s)
        ax.set_xlabel("attempted - accidental")
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([m.split("/")[-1] for m in order], fontsize=7)
    fig.suptitle("Intent-reliance contrast by normalization (95% scenario-bootstrap CI; orange = base)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- (3) clustering
def item_matrix(d, templates):
    x = d[d["template"].isin(templates)].pivot_table(index="story_id", columns=["model", "template"],
                                                    values="norm_rating")
    return x.dropna(axis=1, thresh=int(0.95 * len(x))).apply(lambda c: c.fillna(c.mean()))


def clustering(d, master):
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.manifold import MDS
    from scipy.spatial.distance import pdist, squareform
    lab = master.set_index("story_id")
    scale = {"YS2008": "1-3", "YS2009": "1-4", "YS2011": "1-7"}
    feats = {"all_7_templates": B.FACTORIAL_TEMPLATES, "human_verbatim_only": ["human_verbatim"],
             "factorial_1to7_only": [t for t in B.FACTORIAL_TEMPLATES if t != "human_verbatim"]}
    rows, embeds = [], {}
    for name, tp in feats.items():
        x = item_matrix(d, tp)
        L = lab.loc[x.index]
        z = (x - x.mean()) / x.std().replace(0, 1)
        D = squareform(pdist(z.to_numpy(), "correlation"))
        Z = linkage(z.to_numpy(), "ward")
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, normalized_stress="auto")
        embeds[name] = (pd.DataFrame(mds.fit_transform(D), index=x.index, columns=["mds1", "mds2"]), Z)
        labels = {"source": L["source"], "native_scale": L["source"].map(scale), "condition": L["condition"],
                  "intent": L["intent_label"], "outcome": L["outcome_label"]}
        for lname, y in labels.items():
            k = y.nunique()
            cl = fcluster(Z, k, "maxclust")
            rows.append(dict(features=name, n_features=x.shape[1], label=lname, k=k,
                             silhouette=silhouette_score(D, y.to_numpy(), metric="precomputed"),
                             ward_ARI=adjusted_rand_score(y, cl)))
    return pd.DataFrame(rows), embeds


def plot_clusters(embeds, master, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    lab = master.set_index("story_id")
    names = list(embeds)
    fig, axes = plt.subplots(len(names), 4, figsize=(18, 4.2 * len(names)))
    for r, n in enumerate(names):
        e, Z = embeds[n]
        L = lab.loc[e.index]
        for c, col in enumerate(["source", "condition", "outcome_label"]):
            ax = axes[r, c]
            for v, g in e.groupby(L[col].to_numpy()):
                ax.scatter(g.mds1, g.mds2, s=10, label=v)
            ax.set_title(f"{n}: MDS colored by {col}" + (" (= native scale)" if col == "source" else ""), fontsize=9)
            ax.legend(fontsize=7, markerscale=1.5)
        dendrogram(Z, ax=axes[r, 3], no_labels=True, color_threshold=None,
                   link_color_func=lambda k: "0.4")
        axes[r, 3].set_title(f"{n}: Ward dendrogram", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=130)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sa = scale_audit()
    sa.to_csv(os.path.join(OUT, "scale_audit.csv"), index=False)
    print(sa[["rater", "source", "template", "scale_min", "scale_max", "anchor_low", "anchor_high"]].to_string())

    d = M83.load_ratings(M83.RAW_GLOBS, M83.FACTORIAL)
    co = M83.load_ratings(["outputs/behavior_chatoff/raw_*.csv"], M83.FACTORIAL)
    co["model"] = co["model"] + " [chat off]"
    co["class"] = "base"
    dd = pd.concat([d, co], ignore_index=True)

    r, summ, pr = normalization_robustness(dd, a.n_boot)
    r.to_csv(os.path.join(OUT, "normalization_robustness.csv"), index=False)
    summ.to_csv(os.path.join(OUT, "normalization_ordering.csv"), index=False)
    pr.to_csv(os.path.join(OUT, "normalization_base_vs_instruct.csv"), index=False)
    plot_normalization(r, os.path.join(OUT, "fig_T1_normalization"))
    print(summ.to_string(index=False))
    print(pr.round(3).to_string(index=False))

    master = pd.read_csv(os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    open_w = d[d.groupby("model")["template"].transform("nunique") == 7]
    cs, embeds = clustering(open_w, master)
    cs.to_csv(os.path.join(OUT, "clustering_silhouette.csv"), index=False)
    for n, (e, _) in embeds.items():
        e.to_csv(os.path.join(OUT, f"clustering_mds_{n}.csv"))
    plot_clusters(embeds, master, os.path.join(OUT, "fig_T1_clustering"))
    print(cs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
