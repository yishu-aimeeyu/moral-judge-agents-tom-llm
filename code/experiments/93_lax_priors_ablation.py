#!/usr/bin/env python3
"""
93_lax_priors_ablation.py -- Lax 9/29 T7: model priors not in the prompt, and what besides the prompt
drives the ratings.

--build    writes master-format stimulus CSVs (same story_id / labels, ablated text) to
           outputs/lax_0929/T7/stimuli_<ablation>.csv. Cut points from stage_landmarks (text diffs
           against the intent- / outcome-flipped partner), not clause_offsets:
             no_story         "<Agent>." only -- agent = 03.extract_agent(full text); the rating question
                              is added by 03.build_prompt as usual (factorial questions say "the agent")
             background_only  text before the foreshadow (or before the belief sentence if none)
             belief_removed   belief sentence(s) deleted (intent-differing span, sentence-expanded)
             outcome_removed  text up to the outcome sentence (YS2008/YS2009; YS2011 has no outcome partner)
           NB: belief_removed and outcome_removed keep the foreshadow sentence.
           Scoring is 03_behavioral.py --csv <stimuli> --out_dir outputs/lax_0929/T7/behavior_<ablation>
           --templates <FACTORIAL> (logprob, chat auto), see code/submit_lax_t7.sh.
--analyze  prior strength = |norm rating - 0.5| per ablation; context sensitivity = full - ablated
           (per item x template, and the attempted - accidental contrast per ablation); base vs instruct.
           Variance decomposition (OLS sums of squares, Type I in the stated order + unique drop-in-R2)
           of the 20 open models' full ratings over story (scenario_group), condition, template, scale,
           family, base/instruct and their key interactions; and of full + ablated ratings with ablation.

  .venv/bin/python code/experiments/93_lax_priors_ablation.py --build
  .venv/bin/python code/experiments/93_lax_priors_ablation.py --analyze
"""
import os, sys, glob, argparse, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "outputs", "lax_0929", "T7")
ABLATIONS = ["no_story", "background_only", "belief_removed", "outcome_removed"]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B03 = _load("b03", "code/03_behavioral.py")
M83 = _load("m83", "code/experiments/83_reliability_normalization.py")


# ------------------------------------------------------------------------- build
def build():
    import stage_landmarks as SL
    rows = SL.load_rows()
    lm = SL.landmarks(rows)
    os.makedirs(OUT, exist_ok=True)
    out = {a: [] for a in ABLATIONS}
    audit = []
    for r in rows:
        t, m = r["text"], lm[r["story_id"]]
        if m["intent_on"] is None:
            continue
        b_on, b_off = SL._sentence_bounds(t, m["intent_on"], m["intent_off"])
        texts = {"no_story": B03.extract_agent(t) + ".",
                 "background_only": t[:m["fore_on"] if m["fore_on"] is not None else b_on].strip(),
                 "belief_removed": (t[:b_on] + t[b_off:].lstrip()).strip()}
        if m["outcome_on"] is not None and m["complete"]:
            texts["outcome_removed"] = t[:m["outcome_on"]].strip()
        for a, x in texts.items():
            if x:
                out[a].append(dict(r, text=x))
        audit.append(dict(story_id=r["story_id"], source=r["source"], condition=r["condition"],
                          agent=B03.extract_agent(t), n_chars_full=len(t),
                          **{f"n_chars_{a}": len(texts.get(a, "")) for a in ABLATIONS}))
    cols = list(rows[0].keys())
    for a, rr in out.items():
        pd.DataFrame(rr)[cols].to_csv(os.path.join(OUT, f"stimuli_{a}.csv"), index=False)
        print(f"{a}: {len(rr)} stories -> stimuli_{a}.csv")
    pd.DataFrame(audit).to_csv(os.path.join(OUT, "stimuli_audit.csv"), index=False)
    ex = [r for r in rows if r["story_id"] == "YS2008-COFFEE-attempted"][0]["story_id"]
    for a in ABLATIONS:
        d = pd.read_csv(os.path.join(OUT, f"stimuli_{a}.csv"))
        print(f"--- {a} [{ex}]\n{d.loc[d.story_id == ex, 'text'].iloc[0]}")


# ------------------------------------------------------------------------- analyze
def load_ablation(a):
    files = glob.glob(os.path.join(OUT, f"behavior_{a}", "raw_*.csv"))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[d.template.isin(B03.FACTORIAL_TEMPLATES)]
    return d.groupby(["model", "template", "story_id", "source", "condition"], as_index=False)["norm_rating"].mean()


def sums_of_squares(df, y, terms):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    f = smf.ols(f"{y} ~ " + " + ".join(terms), df).fit()
    a = anova_lm(f, typ=1)
    tot = ((df[y] - df[y].mean()) ** 2).sum()
    a["share_typeI"] = a["sum_sq"] / tot
    uniq = {}
    for t in terms:
        r2 = smf.ols(f"{y} ~ " + " + ".join(x for x in terms if x != t), df).fit().rsquared
        uniq[t] = f.rsquared - r2
    a["unique_R2_drop"] = [uniq.get(i, np.nan) for i in a.index]
    a.index.name = "term"
    return a.reset_index(), f.rsquared


def analyze():
    full = M83.load_ratings(M83.RAW_GLOBS, M83.FACTORIAL)
    full = full[full.groupby("model")["template"].transform("nunique") == 7]
    meta = full.groupby("model")[["class", "family"]].first()
    scale = {"YS2008": "1-3", "YS2009": "1-4", "YS2011": "1-7"}
    full["scale"] = np.where(full.template == "human_verbatim", full.source.map(scale), "1-7")

    terms = ["C(scenario_group)", "C(condition)", "C(template)", "C(scale)", "C(family)", "C(Q('class'))",
             "C(condition):C(Q('class'))", "C(condition):C(template)", "C(template):C(Q('class'))",
             "C(condition):C(family)"]
    vd, r2 = sums_of_squares(full, "norm_rating", terms)
    vd["R2_model"] = r2
    vd.to_csv(os.path.join(OUT, "variance_decomposition_full.csv"), index=False)
    print(vd[["term", "df", "share_typeI", "unique_R2_drop"]].round(4).to_string(index=False))

    rows, ctx, cons = [], [], []
    kept = {}
    for a in ABLATIONS:
        d = load_ablation(a)
        if d is None:
            print(f"[missing] no ratings for {a}")
            continue
        kept[a] = d
    models = sorted(set().union(*[set(d.model) for d in kept.values()])) if kept else []
    for a, d in kept.items():
        m = d.merge(full[["model", "template", "story_id", "norm_rating"]], on=["model", "template", "story_id"],
                    suffixes=("_abl", "_full"))
        m["class"] = m.model.map(meta["class"])
        m["prior_strength"] = (m.norm_rating_abl - 0.5).abs()
        m["context_sensitivity"] = m.norm_rating_full - m.norm_rating_abl
        m["abs_context_sensitivity"] = m.context_sensitivity.abs()
        m["ablation"] = a
        rows.append(m)
        for (mod, cond), g in m.groupby(["model", "condition"]):
            ctx.append(dict(ablation=a, model=mod, cls=meta.loc[mod, "class"], condition=cond,
                            mean_ablated=g.norm_rating_abl.mean(), mean_full=g.norm_rating_full.mean(),
                            prior_strength=g.prior_strength.mean(),
                            context_sensitivity=g.context_sensitivity.mean(),
                            abs_context_sensitivity=g.abs_context_sensitivity.mean(), n=len(g)))
        f = m[m.source.isin(["YS2008", "YS2009"])]
        for mod, g in f.groupby("model"):
            c = g.groupby("condition")[["norm_rating_abl", "norm_rating_full"]].mean()
            if {"attempted", "accidental"} <= set(c.index):
                cons.append(dict(ablation=a, model=mod, cls=meta.loc[mod, "class"],
                                 contrast_ablated=c.loc["attempted", "norm_rating_abl"] - c.loc["accidental", "norm_rating_abl"],
                                 contrast_full=c.loc["attempted", "norm_rating_full"] - c.loc["accidental", "norm_rating_full"],
                                 harm_effect_ablated=(c.loc[["accidental", "intentional"], "norm_rating_abl"].mean()
                                                      - c.loc[["neutral", "attempted"], "norm_rating_abl"].mean()),
                                 intent_effect_ablated=(c.loc[["attempted", "intentional"], "norm_rating_abl"].mean()
                                                        - c.loc[["neutral", "accidental"], "norm_rating_abl"].mean())))
    if not rows:
        print("no ablation ratings yet; variance decomposition of full ratings only")
        return
    allm = pd.concat(rows)
    allm.to_csv(os.path.join(OUT, "ablation_items.csv"), index=False)
    pd.DataFrame(ctx).to_csv(os.path.join(OUT, "ablation_prior_context_by_condition.csv"), index=False)
    cs = pd.DataFrame(cons)
    cs.to_csv(os.path.join(OUT, "ablation_contrasts.csv"), index=False)
    summ = allm.groupby(["ablation", "class"])[["prior_strength", "context_sensitivity", "abs_context_sensitivity"]] \
        .agg(["mean", "std"]).round(4)
    summ.to_csv(os.path.join(OUT, "ablation_summary_by_class.csv"))
    print(summ.to_string())
    print(cs.round(3).to_string(index=False))

    rng = np.random.default_rng(0)
    diff_rows = []
    sg = allm.merge(full[["story_id", "scenario_group"]].drop_duplicates(), on="story_id")
    groups = sg.scenario_group.unique()
    for a, g in sg.groupby("ablation"):
        for k in ("prior_strength", "abs_context_sensitivity"):
            per = g.groupby(["scenario_group", "class"])[k].mean().unstack()
            est = per["instruct"].mean() - per["base"].mean()
            bs = [per.loc[rng.choice(per.index, len(per))].pipe(lambda p: p["instruct"].mean() - p["base"].mean())
                  for _ in range(2000)]
            diff_rows.append(dict(ablation=a, measure=k, instruct_minus_base=est,
                                  ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                                  n_models=g.model.nunique(), n_scenarios=len(per)))
    dr = pd.DataFrame(diff_rows)
    dr.to_csv(os.path.join(OUT, "ablation_base_vs_instruct.csv"), index=False)
    print(dr.round(4).to_string(index=False))

    ab = pd.concat([full.assign(ablation="full")[["model", "template", "story_id", "condition", "scenario_group",
                                                  "norm_rating", "class", "family", "ablation"]],
                    sg.rename(columns={"norm_rating_abl": "norm_rating"}).assign(family=lambda x: x.model.map(meta["family"]))
                    [["model", "template", "story_id", "condition", "scenario_group", "norm_rating", "class", "family",
                      "ablation"]]])
    ab = ab[ab.model.isin(models)]
    vt = ["C(scenario_group)", "C(condition)", "C(template)", "C(ablation)", "C(Q('class'))",
          "C(condition):C(ablation)", "C(ablation):C(Q('class'))", "C(condition):C(ablation):C(Q('class'))"]
    va, r2 = sums_of_squares(ab, "norm_rating", vt)
    va["R2_model"] = r2
    va.to_csv(os.path.join(OUT, "variance_decomposition_with_ablation.csv"), index=False)
    print(va[["term", "df", "share_typeI", "unique_R2_drop"]].round(4).to_string(index=False))
    plot(pd.DataFrame(ctx), cs, os.path.join(ROOT, "outputs", "lax_0929", "fig_T7_priors"))


def plot(ctx, cs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    abls = [a for a in ABLATIONS if a in set(ctx.ablation)]
    fig, axes = plt.subplots(2, len(abls), figsize=(4.2 * len(abls), 7.5), squeeze=False)
    conds = ["neutral", "accidental", "attempted", "intentional"]
    for j, a in enumerate(abls):
        g = ctx[ctx.ablation == a]
        ax = axes[0, j]
        for mod, gm in g.groupby("model"):
            gm = gm.set_index("condition").reindex(conds)
            ax.plot(conds, gm.mean_ablated, "--o" if gm["cls"].iloc[0] == "base" else "-o", ms=3,
                    label=mod.split("/")[-1])
        ax.axhline(0.5, color="0.6", lw=0.6)
        ax.set_ylim(0, 1)
        ax.set_title(f"{a}: mean rating (dashed = base)", fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax = axes[1, j]
        c = cs[cs.ablation == a]
        x = np.arange(len(c))
        ax.bar(x - 0.2, c.contrast_full, 0.4, label="full story", color="0.7")
        ax.bar(x + 0.2, c.contrast_ablated, 0.4, label="ablated", color="tab:blue")
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("/")[-1] for m in c.model], rotation=60, fontsize=6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("attempted - accidental", fontsize=9)
    axes[0, 0].legend(fontsize=6)
    axes[1, 0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path}.{ext}", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    if a.analyze:
        analyze()
