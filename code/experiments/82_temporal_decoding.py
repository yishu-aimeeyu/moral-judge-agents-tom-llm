#!/usr/bin/env python3
"""
82_temporal_decoding.py -- D2: when does intent become decodable, and does it stay?

Reads token trajectories from 81_extract_token_trajectories.py and resamples every story onto
a common time axis: each stage (stage_landmarks.STAGES) is cut into --bins equal steps,
landmarked per story from where its text diverges from its intent- / outcome-flipped
partner. Stage order is identical for every YS2008/YS2009 story:
  background | foreshadow | belief | action | outcome | question

At every (layer, bin) a logistic decoder is trained for intent (guilty vs innocent) and
outcome (harm vs no harm) with GroupKFold over scenario_group, so the 4 cells of a scenario
never straddle train and test. Trained decoders are applied to EVERY other bin, which gives
the temporal-generalization matrix (train t, test t'; King & Dehaene 2014): a square block =
one stable code, a narrow diagonal = a code that keeps changing.

Pre-registered readouts (hypotheses.csv), per layer:
  intent_pre_onset     intent acc before the belief stage. Text is identical across intent
                       conditions there, so this must be ~0.5; anything else is a bug.
  intent_rise          acc(end of belief) - acc(start of belief)
  intent_late_slope    slope of intent acc across action+outcome bins (per bin), with a
                       scenario bootstrap CI. Plateau hypothesis: CI contains 0.
  outcome_in_belief    outcome acc during the belief stage. With foreshadow this is high
                       BY CONSTRUCTION (the outcome was already stated); use --variant
                       no_foreshadow, where it is mechanically 0.5.
  outcome_intention_stage_slope
                       slope of outcome acc across belief+action bins: is the foreshadowed
                       outcome held flat while intent is being read, or overwritten by it?
  harm_expectation     decoder trained on OUTCOME at the end of the story, applied during the
                       belief stage: mean P(harm | guilty) - P(harm | innocent). In
                       no_foreshadow the text says nothing about outcome yet, so a positive
                       value is the model's own prior that bad intent leads to harm.
  surface_*            the same readouts from TF-IDF on the text up to that bin: the claim
                       is the GAP between model and surface, not the raw accuracy.

`--cross` instead trains on one prompt's final token and tests on another's (same model and
variant): diagonal minus off-diagonal = intent information tied to the question wording.

  python code/experiments/82_temporal_decoding.py --model Qwen2.5-7B --prompt story
  python code/experiments/82_temporal_decoding.py --model Qwen2.5-7B --variant no_foreshadow
  python code/experiments/82_temporal_decoding.py --model Qwen2.5-7B --cross \
         --prompts story human_verbatim participant_exact blame_w1 wrong_w2 punish_w1
"""
import os, sys, csv, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import stage_landmarks as SL

TRAJ = os.environ.get("TOM_TRAJ_DIR", f"/orcd/scratch/orcd/007/{os.environ.get('USER', '')}/acts_traj")
OUT = os.path.join(ROOT, "outputs", "decoding_time")
TARGETS = ("intent", "outcome")


# ----------------------------------------------------------------------------- loading
def find_stem(traj, model, variant, prompt, chat):
    if chat == "auto":
        # 81 --chat auto writes chat=on exactly when the tokenizer has a template, so an
        # existing chat=on file is the auto condition; chat=off is the explicit raw control
        on = os.path.join(traj, f"{model}__{variant}__{prompt}__chaton.npz")
        chat = "on" if os.path.exists(on) else "off"
    pat = os.path.join(traj, f"{model}__{variant}__{prompt}__chat{chat}.npz")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(f"no trajectory matches {pat}")
    if len(hits) > 1:
        raise SystemExit(f"ambiguous ({len(hits)}): pass --chat on|off\n  " + "\n  ".join(hits))
    return hits[0][:-4]


def load_traj(stem):
    meta = np.load(stem + ".npz", allow_pickle=False)
    H = np.load(stem + ".H.npy", mmap_mode="r")
    return meta, H


def labels():
    lab = {r["story_id"]: r for r in SL.load_rows()}
    return lab


def bin_positions(meta, n_bins):
    """-> (story_idx [n], pos [n, 6*n_bins] absolute token rows, stage_of_bin [6*n_bins])."""
    sids = [str(s) for s in meta["story_id"]]
    rows = [dict(LAB[s], text=str(t)) for s, t in zip(sids, meta["text"])]
    lm = SL.landmarks(rows)
    ptr, ce = meta["tok_ptr"], meta["char_e"]
    keep, pos = [], []
    for i, s in enumerate(sids):
        m = lm[s]
        if not m["complete"]:
            continue
        a, b = ptr[i], ptr[i + 1]
        ends = ce[a:b]
        story_tok = np.where(ends >= 0)[0]
        first, last_story = story_tok[0], story_tok[-1]

        def onset(c):      # last token that has NOT yet seen char c
            return int(story_tok[np.searchsorted(ends[story_tok], c, side="right") - 1]) \
                if c > ends[first] else first

        def upto(c):       # last token whose span ends at or before c
            j = np.searchsorted(ends[story_tok], c, side="right") - 1
            return int(story_tok[max(j, 0)])

        fore = onset(m["fore_on"]) if m["fore_on"] is not None else onset(m["intent_on"])
        bounds = [first, fore, onset(m["intent_on"]), upto(m["intent_off"]),
                  onset(m["outcome_on"]), last_story, b - a - 1]
        if any(bounds[k] > bounds[k + 1] for k in range(len(bounds) - 1)):
            continue
        p = []
        for k in range(len(SL.STAGES)):
            lo, hi = bounds[k], bounds[k + 1]
            p += [a + lo + int(round((j + 1) / n_bins * (hi - lo))) for j in range(n_bins)]
        keep.append(i)
        pos.append(p)
    stage = np.repeat(np.arange(len(SL.STAGES)), n_bins)
    return np.array(keep), np.array(pos), stage


def bin_char_ends(meta, keep, pos):
    """Story-text char reached at each bin (for the surface baseline)."""
    ce = meta["char_e"]
    out = np.zeros(pos.shape, dtype=int)
    for r, i in enumerate(keep):
        L = len(str(meta["text"][i]))
        for j, t in enumerate(pos[r]):
            out[r, j] = ce[t] if ce[t] >= 0 else L
    return out


# ----------------------------------------------------------------------------- decoding
def _fit(Xtr, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    _, S, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[S > 1e-6 * S.max()].T                           # row space: exact, see 02_probe.py
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Z @ V, y)
    return lambda X: clf.predict_proba(((X - mu) / sd) @ V)[:, 1]


def temporal_generalization(X, y, groups, folds):
    """X [n, T, H] -> P [T_train, T_test, n] out-of-fold P(y=1)."""
    n, T, _ = X.shape
    P = np.full((T, T, n), np.nan, dtype=np.float32)
    for tr, te in folds:
        Xte = X[te].reshape(len(te) * T, -1)
        for t in range(T):
            if len(np.unique(y[tr])) < 2 or np.allclose(X[tr, t].std(0), 0):
                P[t, :, te] = 0.5
                continue
            f = _fit(X[tr, t], y[tr])
            P[t][:, te] = f(Xte).reshape(len(te), T).T
    return P


def surface_curve(texts, char_end, y, folds):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    n, T = char_end.shape
    P = np.full((T, n), 0.5, dtype=np.float32)
    for t in range(T):
        docs = [texts[r][:char_end[r, t]] for r in range(n)]
        for tr, te in folds:
            try:
                v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
                A = v.fit_transform([docs[i] for i in tr])
                clf = LogisticRegression(max_iter=3000).fit(A, y[tr])
                P[t, te] = clf.predict_proba(v.transform([docs[i] for i in te]))[:, 1]
            except ValueError:     # empty vocabulary at the first bins
                pass
    return P


def surface_harm_expectation(texts, char_end, y_out, y_int, folds, t_train, t_test):
    """harm_expectation from TF-IDF: outcome decoder fit on text up to t_train, read at t_test."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    n = len(texts)
    p = np.full((len(t_test), n), np.nan)
    for tr, te in folds:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        A = v.fit_transform([texts[i][:char_end[i, t_train]] for i in tr])
        clf = LogisticRegression(max_iter=3000).fit(A, y_out[tr])
        for k, t in enumerate(t_test):
            p[k, te] = clf.predict_proba(v.transform([texts[i][:char_end[i, t]] for i in te]))[:, 1]
    p = p.mean(0)
    return float(np.mean([p[(y_int == 1) & (y_out == o)].mean() - p[(y_int == 0) & (y_out == o)].mean()
                          for o in (0, 1)]))


def rsa_betas(X, lab_rows, n_boot, rng):
    """Per bin: distance between cells of a scenario ~ intent_diff + outcome_diff.

    Distances are z-scored within scenario so scenario-level offsets (length, topic) cancel;
    betas are then the extra distance a pair gets for differing in intent / outcome.
    """
    n, T, _ = X.shape
    by_sc = {}
    for r, row in enumerate(lab_rows):
        by_sc.setdefault(row["scenario_group"] + "|" + row["source"], {})[row["condition"]] = r
    scen = [v for v in by_sc.values() if len(v) == 4]
    pairs = [(a, b) for i, a in enumerate(SL_CELLS) for b in SL_CELLS[i + 1:]]
    di = np.array([SL_I[a] != SL_I[b] for a, b in pairs], float)
    do = np.array([SL_O[a] != SL_O[b] for a, b in pairs], float)
    D = np.stack([di - di.mean(), do - do.mean()], 1)
    out = np.zeros((T, len(scen), 2))
    for t in range(T):
        Z = (X[:, t] - X[:, t].mean(0)) / (X[:, t].std(0) + 1e-6)
        for s, cells in enumerate(scen):
            d = np.array([np.linalg.norm(Z[cells[a]] - Z[cells[b]]) for a, b in pairs])
            # identical prefixes differ only by kernel nondeterminism across sequence
            # lengths; z-scoring that noise would manufacture structure out of nothing
            if d.max() < 1e-2 * np.sqrt(Z.shape[1]):
                continue
            d = (d - d.mean()) / (d.std() + 1e-9)
            out[t, s] = np.linalg.lstsq(D, d, rcond=None)[0]
    boot = np.array([out[:, rng.integers(0, len(scen), len(scen))].mean(1) for _ in range(n_boot)])
    return out.mean(1), np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)


SL_CELLS = ["neutral", "accidental", "attempted", "intentional"]
SL_I = {"neutral": 0, "accidental": 0, "attempted": 1, "intentional": 1}
SL_O = {"neutral": 0, "accidental": 1, "attempted": 0, "intentional": 1}


def boot_groups(groups, n_boot, rng):
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n_boot):
        yield np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])


# ----------------------------------------------------------------------------- main modes
def run_time(a):
    from sklearn.model_selection import GroupKFold
    stem = find_stem(a.traj, a.model, a.variant, a.prompt, a.chat)
    meta, H = load_traj(stem)
    keep, pos, stage = bin_positions(meta, a.bins)
    sids = [str(meta["story_id"][i]) for i in keep]
    rows = [LAB[s] for s in sids]
    y = {"intent": np.array([r["intent_label"] == "guilty" for r in rows], int),
         "outcome": np.array([r["outcome_label"] == "harm" for r in rows], int)}
    groups = np.array([r["scenario_group"] for r in rows])
    folds = list(GroupKFold(n_splits=a.folds).split(pos, y["intent"], groups))
    layers = [int(L) for L in meta["layer_idx"]]
    T = pos.shape[1]
    tag = os.path.basename(stem)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    print(f"{tag}: {len(keep)} stories, {T} bins, layers {layers}", flush=True)

    texts = [str(meta["text"][i]) for i in keep]
    cend = bin_char_ends(meta, keep, pos)
    surf = {t: surface_curve(texts, cend, y[t], folds) for t in TARGETS} if a.surface else {}

    curve, hyp, story_rows, rsa_rows = [], [], [], []
    tg_store = {}
    st = {s: np.where(stage == k)[0] for k, s in enumerate(SL.STAGES)}
    pre = np.concatenate([st["background"], st["foreshadow"]])
    late = np.concatenate([st["action"], st["outcome"]])
    mid_ = np.concatenate([st["belief"], st["action"]])         # intention stage, pre-outcome
    for li, L in enumerate(layers):
        X = np.asarray(H[pos.ravel(), li], dtype=np.float32).reshape(len(keep), T, -1)
        P = {t: temporal_generalization(X, y[t], groups, folds) for t in TARGETS}
        corr = {t: ((P[t][np.arange(T), np.arange(T)] > 0.5) == y[t][None, :]) for t in TARGETS}
        tg = {t: ((P[t] > 0.5) == y[t][None, None, :]).mean(2) for t in TARGETS}
        tg_store[f"L{L}_intent"], tg_store[f"L{L}_outcome"] = tg["intent"], tg["outcome"]
        for t in TARGETS:
            for b in range(T):
                row = dict(layer=L, target=t, bin=b, stage=SL.STAGES[stage[b]],
                           acc=round(float(corr[t][b].mean()), 4))
                if surf:
                    row["surface_acc"] = round(float(((surf[t][b] > 0.5) == y[t]).mean()), 4)
                curve.append(row)

        # harm expectation: outcome decoder trained at the final story bin, read at every bin
        t_out = st["outcome"][-1]
        ph = P["outcome"][t_out]                                   # [T, n]

        def stat(ix):
            """Readouts on story subset ix; harm expectation is compared WITHIN true outcome."""
            ci, co = corr["intent"][:, ix], corr["outcome"][:, ix]
            acc_i, acc_o = ci.mean(1), co.mean(1)
            slope = np.polyfit(np.arange(len(late)), acc_i[late], 1)[0]
            slope_o = np.polyfit(np.arange(len(mid_)), acc_o[mid_], 1)[0]
            phx = ph[:, ix]
            gg, ii, oo = y["intent"][ix] == 1, y["intent"][ix] == 0, y["outcome"][ix]
            he = np.mean([phx[st["belief"]][:, gg & (oo == o)].mean() - phx[st["belief"]][:, ii & (oo == o)].mean()
                          for o in (0, 1)])
            return dict(intent_pre_onset=acc_i[pre].mean(),
                        intent_rise=acc_i[st["belief"][-1]] - acc_i[st["belief"][0]],
                        intent_late_mean=acc_i[late].mean(), intent_late_slope=slope,
                        outcome_in_belief=co[st["belief"]].mean(),
                        outcome_intention_stage_slope=slope_o, harm_expectation=he)
        full = stat(np.arange(len(keep)))
        boots = [stat(ix) for ix in boot_groups(groups, a.boot, rng)]
        h = dict(model=a.model, variant=a.variant, prompt=a.prompt, layer=L, n=len(keep))
        for k, v in full.items():
            bv = np.array([b[k] for b in boots])
            h[k] = round(float(v), 4)
            h[k + "_lo"], h[k + "_hi"] = (round(float(np.percentile(bv, q)), 4) for q in (2.5, 97.5))
        if surf:
            sc = {t: ((surf[t] > 0.5) == y[t][None, :]) for t in TARGETS}
            h["surface_intent_late_mean"] = round(float(sc["intent"][late].mean()), 4)
            h["surface_outcome_in_belief"] = round(float(sc["outcome"][st["belief"]].mean()), 4)
            if li == 0:
                surf_he = surface_harm_expectation(texts, cend, y["outcome"], y["intent"], folds,
                                                   t_out, st["belief"])
            h["surface_harm_expectation"] = round(surf_he, 4)
        hyp.append(h)

        for r, s in enumerate(sids):
            for k, sname in enumerate(SL.STAGES):
                ix = st[sname]
                story_rows.append(dict(
                    story_id=s, scenario_group=groups[r], source=rows[r]["source"],
                    condition=rows[r]["condition"], layer=L, stage=sname,
                    p_intent_correct=round(float(np.where(y["intent"][r], P["intent"][ix, ix, r],
                                                          1 - P["intent"][ix, ix, r]).mean()), 4),
                    p_outcome_correct=round(float(np.where(y["outcome"][r], P["outcome"][ix, ix, r],
                                                           1 - P["outcome"][ix, ix, r]).mean()), 4),
                    p_harm_from_end_decoder=round(float(ph[ix, r].mean()), 4)))

        if a.rsa:
            m, lo, hi = rsa_betas(X, rows, a.boot, rng)
            for b in range(T):
                for j, nm in enumerate(("intent", "outcome")):
                    rsa_rows.append(dict(layer=L, bin=b, stage=SL.STAGES[stage[b]], term=nm,
                                         beta=round(float(m[b, j]), 4), lo=round(float(lo[b, j]), 4),
                                         hi=round(float(hi[b, j]), 4)))
        print(f"  L{L}: intent pre={full['intent_pre_onset']:.3f} late={full['intent_late_mean']:.3f} "
              f"slope={full['intent_late_slope']:+.4f}  outcome@belief={full['outcome_in_belief']:.3f} "
              f"harm_exp={full['harm_expectation']:+.3f}", flush=True)

    _write(os.path.join(a.out, f"{tag}__curve.csv"), curve)
    _write(os.path.join(a.out, f"{tag}__hypotheses.csv"), hyp)
    _write(os.path.join(a.out, f"{tag}__per_story.csv"), story_rows)
    if rsa_rows:
        _write(os.path.join(a.out, f"{tag}__rsa.csv"), rsa_rows)
    np.savez_compressed(os.path.join(a.out, f"{tag}__tg.npz"), stage=stage, **tg_store)
    plot_time(tag, curve, tg_store, stage, layers, a.bins, a.out, rsa_rows)


def run_cross(a):
    from sklearn.model_selection import GroupKFold
    data = {}
    for p in a.prompts:
        meta, H = load_traj(find_stem(a.traj, a.model, a.variant, p, a.chat))
        last = meta["tok_ptr"][1:] - 1
        data[p] = (meta, np.asarray(H[last], dtype=np.float32))     # [n, L, H]
    ids = [str(s) for s in data[a.prompts[0]][0]["story_id"]]
    for p in a.prompts:
        if [str(s) for s in data[p][0]["story_id"]] != ids:
            raise SystemExit(f"{p}: story order differs from {a.prompts[0]}")
    rows = [LAB[s] for s in ids]
    groups = np.array([r["scenario_group"] for r in rows])
    y = {"intent": np.array([r["intent_label"] == "guilty" for r in rows], int),
         "outcome": np.array([r["outcome_label"] == "harm" for r in rows], int)}
    folds = list(GroupKFold(n_splits=a.folds).split(ids, y["intent"], groups))
    layers = [int(L) for L in data[a.prompts[0]][0]["layer_idx"]]
    out = []
    for li, L in enumerate(layers):
        for t in TARGETS:
            for ptr_ in a.prompts:
                for pte in a.prompts:
                    hit = np.zeros(len(ids))
                    for tr, te in folds:
                        f = _fit(data[ptr_][1][tr, li], y[t][tr])
                        hit[te] = (f(data[pte][1][te, li]) > 0.5) == y[t][te]
                    out.append(dict(model=a.model, variant=a.variant, layer=L, target=t,
                                    train_prompt=ptr_, test_prompt=pte, acc=round(hit.mean(), 4)))
        print(f"  L{L} done", flush=True)
    os.makedirs(a.out, exist_ok=True)
    chat = "on" if bool(data[a.prompts[0]][0]["chat"]) else "off"
    p = os.path.join(a.out, f"{a.model}__{a.variant}__chat{chat}__cross_prompt.csv")
    _write(p, out)
    print(f"-> {p}")


# ----------------------------------------------------------------------------- output
def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}", flush=True)


def plot_time(tag, curve, tg, stage, layers, n_bins, out, rsa_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = len(stage)
    edges = [k * n_bins - 0.5 for k in range(len(SL.STAGES) + 1)]
    mid = layers[len(layers) // 2]
    n_pan = 4 if rsa_rows else 3
    fig, ax = plt.subplots(1, n_pan, figsize=(5.2 * n_pan, 4.2))
    cmap = plt.get_cmap("viridis")
    for k, t in enumerate(TARGETS):
        for j, L in enumerate(layers):
            v = [r["acc"] for r in curve if r["layer"] == L and r["target"] == t]
            ax[k].plot(v, color=cmap(j / max(1, len(layers) - 1)), lw=1.6, label=f"layer {L}")
        sv = [r.get("surface_acc") for r in curve if r["layer"] == layers[0] and r["target"] == t]
        if sv[0] is not None:
            ax[k].plot(sv, "k--", lw=1.2, label="TF-IDF surface")
        ax[k].axhline(0.5, color="gray", lw=0.8)
        ax[k].set_title(f"{t} decoding over time")
        ax[k].set_ylim(0.3, 1.02)
        ax[k].set_ylabel("cross-scenario accuracy")
    ax[0].legend(fontsize=7, loc="lower right")
    from matplotlib.colors import TwoSlopeNorm
    im = ax[2].imshow(tg[f"L{mid}_intent"], origin="lower", cmap="RdBu_r",
                      norm=TwoSlopeNorm(vcenter=0.5, vmin=0.2, vmax=1.0))
    ax[2].set_title(f"intent temporal generalization (L{mid})")
    ax[2].set_xlabel("test bin"); ax[2].set_ylabel("train bin")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    if rsa_rows:
        for nm, c in (("intent", "C3"), ("outcome", "C0")):
            rr = [r for r in rsa_rows if r["layer"] == mid and r["term"] == nm]
            ax[3].plot([r["beta"] for r in rr], color=c, label=nm)
            ax[3].fill_between(range(T), [r["lo"] for r in rr], [r["hi"] for r in rr], color=c, alpha=0.2)
        ax[3].axhline(0, color="gray", lw=0.8)
        ax[3].set_title(f"RSA: distance ~ intent + outcome (L{mid})")
        ax[3].legend(fontsize=7)
    for a_ in ax:
        for e in edges[1:-1]:
            a_.axvline(e, color="gray", lw=0.5, ls=":")
            if a_ is ax[2]:
                a_.axhline(e, color="gray", lw=0.5, ls=":")
        a_.set_xticks([(edges[k] + edges[k + 1]) / 2 for k in range(len(SL.STAGES))])
        a_.set_xticklabels(SL.STAGES, rotation=35, fontsize=7)
    ax[2].set_yticks(ax[2].get_xticks()); ax[2].set_yticklabels(SL.STAGES, fontsize=7)
    fig.suptitle(tag, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, f"{tag}__time.png")
    fig.savefig(p, dpi=150)
    print(f"-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", required=True, help="model tag, e.g. Qwen2.5-7B")
    ap.add_argument("--variant", default="original", choices=["original", "no_foreshadow"])
    ap.add_argument("--prompt", default="story")
    ap.add_argument("--chat", default="auto", choices=["auto", "on", "off"],
                    help="auto = the condition 81 --chat auto produced (on if it exists)")
    ap.add_argument("--bins", type=int, default=6, help="bins per stage")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-surface", dest="surface", action="store_false")
    ap.add_argument("--no-rsa", dest="rsa", action="store_false")
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--prompts", nargs="+", default=["story", "human_verbatim", "participant_exact",
                                                      "blame_w1", "wrong_w2", "punish_w1"])
    ap.add_argument("--traj", default=TRAJ)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    LAB = labels()
    run_cross(a) if a.cross else run_time(a)
