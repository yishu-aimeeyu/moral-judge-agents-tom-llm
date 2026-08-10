#!/usr/bin/env python3
"""E1 -- layerwise hypothesis-RDM variance partitioning. [CPU]

WHY THIS EXISTS. `24_rsa_cka.py` R4 already correlates each model RDM with an intent RDM and
an outcome RDM, but it reports the two as separate marginal correlations. Marginal
correlations cannot say which factor ORGANISES the geometry, because the 2x2 factors are
correlated with scenario identity and with sheer word overlap: a model whose space is
organised purely by lexical similarity scores respectably on both hypothesis RDMs. E1
competes four predictors in one regression and reports each one's unique variance, so
"the geometry is outcome-organised while the intent probe peaks" becomes a number with a
permutation null attached.

Predictors (all 298x298, upper triangle, rank-transformed):
  intent       pair differs on intent_label (the belief factor)
  outcome      pair differs on outcome_label
  interaction  |I_i*O_i - I_j*O_j|, the treatment-coded intent x outcome product term
  lexical      cosine distance between tfidf_word_1_2 vectors, same vectoriser settings as
               21_surface_baseline.py

Pre-specified predictions P1-P4, thresholds and falsifiers: outputs/rsa/E1_PRESPEC.md,
written before the first run and never overwritten.

Outputs -> outputs/rsa/
  rdm_variance_partition.csv
  rdm_variance_partition_layerwise.png
  E1_VARIANCE_PARTITION.md
"""
import argparse
import csv
import glob
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")
PRESPEC = os.path.join(RSA_DIR, "E1_PRESPEC.md")

# P1's "late layers" and the sign-test bar, fixed in the prespec rather than passed as flags
LATE_DEPTH_MIN = 0.75
SIGN_TEST_BAR = 7          # of 8 models; 6/8 is p=0.145 and does not clear
PREDICTORS = ["intent", "outcome", "interaction", "lexical"]


def _load_rsa_module():
    """24_rsa_cka.py starts with a digit, so it cannot be imported normally.

    Reusing build_rdm / upper / permute_within_groups is the point: E1 must partition the
    same RDMs the existing RSA outputs are built from, not a re-implementation of them.
    """
    spec = importlib.util.spec_from_file_location(
        "rsa_24", os.path.join(HERE, "24_rsa_cka.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)


class Partition:
    """Semipartial rho^2 for each predictor, with the reduced-model QRs precomputed.

    Only y changes across permutations, so the design factorisations are built once. That is
    what makes 1000 permutations per (model, layer) affordable on CPU.
    """

    def __init__(self, preds):
        self.names = list(preds)
        cols = [zscore(rank(preds[n])) for n in self.names]
        n = len(cols[0])
        self.X = np.column_stack([np.ones(n)] + cols)
        self.full = np.linalg.qr(self.X)
        self.reduced = {}
        for i, name in enumerate(self.names):
            keep = [0] + [j + 1 for j in range(len(self.names)) if j != i]
            self.reduced[name] = (self.X[:, keep], np.linalg.qr(self.X[:, keep]))

    @staticmethod
    def _r2(qr, y, sst):
        q, _ = qr
        resid = y - q @ (q.T @ y)
        return 1.0 - float(resid @ resid) / sst

    def fit(self, y_raw):
        y = zscore(rank(y_raw))
        sst = float(y @ y)
        r2_full = self._r2(self.full, y, sst)
        out = {"r2_full": r2_full}
        for name in self.names:
            out[name] = r2_full - self._r2(self.reduced[name][1], y, sst)
        return out


def build_predictors(lab, sids, rsa):
    intent = np.array([1 if lab[s]["intent_label"] == "guilty" else 0 for s in sids])
    outcome = np.array([1 if lab[s]["outcome_label"] == "harm" else 0 for s in sids])
    inter = intent * outcome
    scen = np.array([lab[s].get("scenario_group") or lab[s]["scenario_id"] for s in sids])

    up = rsa.upper
    preds = {
        "intent": up((intent[:, None] != intent[None, :]).astype(float)),
        "outcome": up((outcome[:, None] != outcome[None, :]).astype(float)),
        "interaction": up(np.abs(inter[:, None] - inter[None, :]).astype(float)),
        "lexical": up(lexical_rdm([lab[s]["text"] for s in sids])),
    }
    nuisance_scenario = up((scen[:, None] != scen[None, :]).astype(float))
    return preds, nuisance_scenario, scen


def residualise(v, on):
    """v with `on` regressed out, in rank space -- see E1_PRESPEC.md amendment 1.

    TF-IDF distance between these vignettes is almost the scenario-identity matrix (the four
    conditions in a group differ by one clause), so an un-residualised lexical predictor
    tests scenario membership rather than surface form.
    """
    y, x = zscore(rank(v)), zscore(rank(on))
    beta = float(x @ y) / float(x @ x)
    return y - beta * x


def lexical_rdm(texts):
    """Cosine distance on tfidf_word_1_2 -- identical settings to 21_surface_baseline.py."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    spec = importlib.util.spec_from_file_location(
        "surface_21", os.path.join(HERE, "21_surface_baseline.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    kw = m.FEATURE_SETS["tfidf_word_1_2"]
    X = TfidfVectorizer(**kw).fit_transform(texts).toarray()
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1e-12
    Xn = X / nrm
    return 1.0 - Xn @ Xn.T


def probe_acc_by_layer(tag, probe_dir, suffix):
    p = os.path.join(probe_dir, f"{tag}_probe{suffix}.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        out.setdefault(r["target"], {})[int(r["layer"])] = float(r["cv_acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--acts", default=os.path.join(ROOT, "outputs", "acts"))
    ap.add_argument("--probe", default=os.path.join(ROOT, "outputs", "probe"))
    ap.add_argument("--out", default=RSA_DIR)
    ap.add_argument("--pooling", default="mean",
                    choices=["last", "mean", "belief_last", "action_last"])
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()

    rsa = _load_rsa_module()
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.pooling == "last" else f"_{a.pooling}"
    lab = {r["story_id"]: r for r in csv.DictReader(open(a.csv))}
    npzs = sorted(glob.glob(os.path.join(a.acts, "*.npz")))
    if not npzs:
        raise SystemExit(f"no activations in {a.acts}")

    out_csv = os.path.join(a.out, "rdm_variance_partition.csv")
    if a.plot_only:
        rows = list(csv.DictReader(open(out_csv)))
        for r in rows:
            for k in ("unique_rho2", "r2_full", "null_mean", "null_q95",
                      "null_ci_lo", "null_ci_hi", "rel_depth", "intent_probe_acc",
                      "outcome_probe_acc"):
                r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
            r["layer"] = int(r["layer"])
            r["is_peak"] = r["is_peak"] in ("True", "true", "1")
        make_figure(rows, a.out)
        write_verdict(rows, a.out, a.pooling, a.perm)
        return

    if a.dry_run:
        print("=== E1 VARIANCE PARTITION PLAN (dry-run) ===")
        print(f"pooling={a.pooling}  perms={a.perm}  models={len(npzs)}")
        print(f"predictors={PREDICTORS} (+ scenario nuisance as secondary model)")
        total = 0
        for n in npzs:
            tag = os.path.basename(n)[:-4]
            d = np.load(n, allow_pickle=True)
            nL = d[a.pooling].shape[1]
            pk = rsa.peak_intent_layer(tag, a.probe, suffix)
            layers = sorted({int(round(r * (nL - 1))) for r in rsa.REL_DEPTHS} |
                            ({pk} if pk is not None else set()))
            total += len(layers)
            print(f"  {tag:26} layers={nL:3} peak={pk} partition_layers={layers}")
        print(f"\n{total} (model, layer) partitions x 2 model variants, "
              f"{a.perm} permutations each")
        print(f"P1 bar: outcome unique > intent unique at rel_depth >= {LATE_DEPTH_MIN} "
              f"in >= {SIGN_TEST_BAR}/{len(npzs)} models")
        print("\nLaunch:\n  JOBNAME=e1 CPUS=16 bash engaging/submit_cpu.sh "
              f"\"python code/experiments/61_rdm_variance_partition.py --perm {a.perm}\"")
        return

    if not os.path.exists(PRESPEC):
        raise SystemExit(
            f"{PRESPEC} is missing. E1 is pre-registered; refusing to produce results "
            "without the prespec on disk.")

    rng = np.random.default_rng(0)
    rows = []
    for npz in npzs:
        tag = os.path.basename(npz)[:-4]
        A, sids = rsa.load_model(npz, lab, a.pooling)
        nL = A.shape[1]
        pk = rsa.peak_intent_layer(tag, a.probe, suffix)
        layers = sorted({int(round(r * (nL - 1))) for r in rsa.REL_DEPTHS} |
                        ({pk} if pk is not None else set()))
        preds, nuis_scen, scen = build_predictors(lab, sids, rsa)
        accs = probe_acc_by_layer(tag, a.probe, suffix)

        variants = {
            "primary": Partition(preds),
            "plus_scenario": Partition({**preds, "scenario": nuis_scen}),
            # amendment 1: lexical residualised on scenario, so P4 tests surface form
            "lexical_resid": Partition({**{k: v for k, v in preds.items() if k != "lexical"},
                                        "scenario": nuis_scen,
                                        "lexical": residualise(preds["lexical"], nuis_scen)}),
        }

        print(f"\n=== {tag} (layers={nL}, peak={pk}) ===", flush=True)
        for L in layers:
            rdm = rsa.build_rdm(A[:, L, :].astype(np.float64))
            y = rsa.upper(rdm)
            obs = {name: part.fit(y) for name, part in variants.items()}

            null = {v: {p: np.empty(a.perm) for p in PREDICTORS} for v in variants}
            base = np.arange(rdm.shape[0])
            for i in range(a.perm):
                p = rsa.permute_within_groups(base, scen, rng)
                yp = rsa.upper(rdm[np.ix_(p, p)])
                for vname, part in variants.items():
                    fit = part.fit(yp)
                    for name in PREDICTORS:
                        null[vname][name][i] = fit[name]

            for name in PREDICTORS:
                for variant, fitres in obs.items():
                    nl = null[variant][name]
                    rows.append({
                        "model": tag, "layer": L,
                        "rel_depth": round(L / max(nL - 1, 1), 3),
                        "is_peak": L == pk,
                        "model_variant": variant,
                        "predictor": name,
                        "unique_rho2": round(fitres[name], 6),
                        "r2_full": round(fitres["r2_full"], 6),
                        "null_mean": round(float(nl.mean()), 6),
                        "null_q95": round(float(np.percentile(nl, 95)), 6),
                        "null_ci_lo": round(float(np.percentile(nl, 2.5)), 6),
                        "null_ci_hi": round(float(np.percentile(nl, 97.5)), 6),
                        "above_null_q95": bool(fitres[name] > np.percentile(nl, 95)),
                        "intent_probe_acc": accs.get("intent", {}).get(L, ""),
                        "outcome_probe_acc": accs.get("outcome", {}).get(L, ""),
                        "n_perm": a.perm,
                    })
            o, orl = obs["primary"], obs["lexical_resid"]
            print(f"  L{L:<3} d={L/max(nL-1,1):.2f}{' PEAK' if L == pk else '    '} "
                  f"R2={o['r2_full']:.3f} | intent={o['intent']:.4f} "
                  f"outcome={o['outcome']:.4f} inter={o['interaction']:.4f} "
                  f"lex={o['lexical']:.4f} || resid-variant lex={orl['lexical']:.4f}",
                  flush=True)

    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(out_csv, ROOT)} ({len(rows)} rows)")

    for r in rows:
        for k in ("intent_probe_acc", "outcome_probe_acc"):
            r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
    make_figure(rows, a.out)
    write_verdict(rows, a.out, a.pooling, a.perm)


# ------------------------------------------------------------------ verdict
def _verdict_counts(rows, variant="primary"):
    """P1/P2 counts plus the P3/P4 gates, computed the way the prespec defines them."""
    models = sorted({r["model"] for r in rows})
    late, peak, lex_flags = {}, {}, []
    for m in models:
        mr = [r for r in rows if r["model"] == m and r["model_variant"] == variant]
        by_layer = {}
        for r in mr:
            by_layer.setdefault(r["layer"], {})[r["predictor"]] = r
        late_layers = [L for L, d in by_layer.items()
                       if d["intent"]["rel_depth"] >= LATE_DEPTH_MIN]
        peak_layers = [L for L, d in by_layer.items() if d["intent"]["is_peak"]]

        def _cmp(layers):
            res = []
            for L in sorted(layers):
                d = by_layer[L]
                lex_dom = (d["lexical"]["unique_rho2"] > d["intent"]["unique_rho2"] and
                           d["lexical"]["unique_rho2"] > d["outcome"]["unique_rho2"])
                if lex_dom:
                    lex_flags.append((m, L))
                    continue          # P4: layer excluded, exclusion reported
                res.append({
                    "layer": L,
                    "intent": d["intent"]["unique_rho2"],
                    "outcome": d["outcome"]["unique_rho2"],
                    "outcome_gt_intent": d["outcome"]["unique_rho2"] > d["intent"]["unique_rho2"],
                    "both_above_null": (d["outcome"]["above_null_q95"] in (True, "True") and
                                        d["intent"]["above_null_q95"] in (True, "True")),
                })
            return res

        late[m] = _cmp(late_layers)
        peak[m] = _cmp(peak_layers)
    return models, late, peak, lex_flags


def _tally(models, d):
    """Split models into pass / fail / undecidable.

    A model whose every layer was P4-excluded has an EMPTY comparison set. Counting it as a
    failure would turn "we could not look" into "the effect went the other way", which is the
    reverse of what the prespec is for; amendment 1 fixes the reporting rule.
    """
    passed = [m for m in models if d[m] and all(x["outcome_gt_intent"] for x in d[m])]
    failed = [m for m in models if d[m] and not all(x["outcome_gt_intent"] for x in d[m])]
    undecid = [m for m in models if not d[m]]
    return passed, failed, undecid


def _all_layer_rows(rows, model, variant, late_only=False):
    """Every layer's numbers, including P4-excluded ones, flagged rather than dropped.

    The exclusion decides what the P1 count may be built from; it should not hide the numbers
    a reader needs to check the exclusion.
    """
    mr = [r for r in rows if r["model"] == model and r["model_variant"] == variant]
    by_layer = {}
    for r in mr:
        by_layer.setdefault(r["layer"], {})[r["predictor"]] = r
    out = []
    for L in sorted(by_layer):
        d = by_layer[L]
        if late_only and d["intent"]["rel_depth"] < LATE_DEPTH_MIN:
            continue
        out.append({
            "layer": L,
            "intent": d["intent"]["unique_rho2"],
            "outcome": d["outcome"]["unique_rho2"],
            "lexical": d["lexical"]["unique_rho2"],
            "outcome_gt_intent": d["outcome"]["unique_rho2"] > d["intent"]["unique_rho2"],
            "both_above_null": (d["outcome"]["above_null_q95"] in (True, "True") and
                                d["intent"]["above_null_q95"] in (True, "True")),
            "lex_dom": (d["lexical"]["unique_rho2"] > d["intent"]["unique_rho2"] and
                        d["lexical"]["unique_rho2"] > d["outcome"]["unique_rho2"]),
        })
    return out


def _m1_counts(rows, models, variant, mode):
    """M1: the ordering without P4's gate, with P3's permutation requirement kept.

    Unique variance is semipartial, so intent and outcome are each already adjusted for the
    lexical predictor; the size of `lexical` does not threaten the comparison between them.
    See E1_PRESPEC.md amendment 2. `mode` is "late" or "peak".
    """
    passed, failed, undecid = [], [], []
    for m in models:
        if mode == "late":
            xs = _all_layer_rows(rows, m, variant, late_only=True)
        else:
            peaks = {r["layer"] for r in rows
                     if r["model"] == m and r["is_peak"] in (True, "True")}
            xs = [x for x in _all_layer_rows(rows, m, variant) if x["layer"] in peaks]
        usable = [x for x in xs if x["both_above_null"]]
        if not usable:
            undecid.append(m)
        elif all(x["outcome_gt_intent"] for x in usable):
            passed.append(m)
        else:
            failed.append(m)
    return passed, failed, undecid


def _amended_section(rows, models, n_models):
    """Amendment 1's variant verdict, then amendment 2's gate-free M1 verdict."""
    out = []
    _, late, peak, lex_flags = _verdict_counts(rows, variant="lexical_resid")
    a_pass, a_fail, a_und = _tally(models, late)
    b_pass, b_fail, b_und = _tally(models, peak)
    out += ["## Amendment 1 — `lexical_resid` variant (P4 still applied)\n",
            "Lexical distance residualised on scenario identity.\n",
            _verdict_line(f"P1 at relative depth ≥ {LATE_DEPTH_MIN}",
                          a_pass, a_fail, a_und, n_models),
            _verdict_line("P2 at the peak-intent layer", b_pass, b_fail, b_und, n_models),
            ""]
    if len(a_und) == n_models:
        out += ["Residualising does not rescue the control: the intent and outcome "
                "manipulations are *realised lexically* in these vignettes — within a "
                "scenario group the four conditions differ by the clause that makes the act "
                "intended or the harm occur — so a fine-grained word-overlap predictor "
                "captures more unique variance than the binary factor it expresses. That is a "
                "property of the stimulus design, not of the models. See amendment 2.\n"]

    out.append("## M1 (amended, not pre-registered) — lexical-adjusted ordering\n")
    out.append("P4's gate dropped, P3's permutation requirement kept, bar unchanged at "
               f"{SIGN_TEST_BAR}/{n_models}. Unique variance is semipartial, so intent and "
               "outcome are each already computed net of the lexical predictor; the size of "
               "`lexical` does not bear on the comparison between them.\n")
    for variant in ("primary", "lexical_resid"):
        p_late = _m1_counts(rows, models, variant, "late")
        p_peak = _m1_counts(rows, models, variant, "peak")
        out.append(f"**`{variant}` variant**")
        out.append(_verdict_line(f"M1 late layers (relative depth ≥ {LATE_DEPTH_MIN})",
                                 *p_late, n_models))
        out.append(_verdict_line("M1 at the peak-intent layer", *p_peak, n_models))
        out.append("")
    return "\n".join(out)


def _verdict_line(label, passed, failed, undecid, n_models):
    if len(undecid) == n_models:
        return (f"- **{label}**: **undecidable** — every layer was excluded by P4 "
                f"(surface-dominated), so the comparison set is empty in this variant. "
                f"This is not evidence in either direction.")
    verdict = ("supported" if len(passed) >= SIGN_TEST_BAR else "NOT supported")
    extra = f", {len(undecid)} undecidable" if undecid else ""
    return (f"- **{label}**: **{len(passed)}/{n_models}** models "
            f"({len(failed)} against{extra}) → **{verdict}** "
            f"(bar {SIGN_TEST_BAR}/{n_models}).")


def write_verdict(rows, outdir, pooling, n_perm):
    models, late, peak, lex_flags = _verdict_counts(rows)
    n_models = len(models)
    p1_pass, p1_fail, p1_und = _tally(models, late)
    p2_pass, p2_fail, p2_und = _tally(models, peak)
    p1 = len(p1_pass) >= SIGN_TEST_BAR and len(p1_und) < n_models

    L = []
    L.append("# E1 — layerwise hypothesis-RDM variance partitioning: verdict\n")
    L.append(f"Generated by `code/experiments/61_rdm_variance_partition.py` "
             f"(pooling `{pooling}`, {n_perm} permutations). Thresholds and falsifiers were "
             f"fixed in [`E1_PRESPEC.md`](E1_PRESPEC.md) before this ran; amendment 1 there "
             f"records the surface-control fix and the reporting rule used below.\n")
    L.append("## Headline — pre-registered `primary` variant\n")
    L.append(_verdict_line(f"P1 (outcome unique ρ² > intent unique ρ² at relative depth "
                           f"≥ {LATE_DEPTH_MIN})", p1_pass, p1_fail, p1_und, n_models))
    L.append(_verdict_line("P2 (same ordering at each model's own peak-intent layer)",
                           p2_pass, p2_fail, p2_und, n_models))
    if lex_flags:
        n_layers_flagged = len(set(lex_flags))
        L.append(f"- **P4** (surface control) excluded **{n_layers_flagged}** "
                 f"model×layer cell(s) as lexical-dominated. TF-IDF distance between these "
                 f"vignettes is nearly the scenario-identity matrix, which is why: see "
                 f"amendment 1.")
    else:
        L.append("- **P4** (surface control): no layer was lexical-dominated; no exclusions.")
    L.append("")
    L.append(_amended_section(rows, models, n_models))
    L.append("")

    for variant, title in (("primary", "pre-registered `primary` variant"),
                           ("lexical_resid", "amended `lexical_resid` variant")):
        L.append(f"## Late layers (relative depth ≥ {LATE_DEPTH_MIN:.2f}) — {title}\n")
        L.append("| model | layer | intent unique ρ² | outcome unique ρ² | outcome/intent | "
                 "outcome > intent | both clear null q95 (P3) | lexical unique ρ² | P4 |")
        L.append("|---|---:|---:|---:|---:|---|---|---:|---|")
        for m in models:
            for x in _all_layer_rows(rows, m, variant, late_only=True):
                ratio = (x["outcome"] / x["intent"]) if abs(x["intent"]) > 1e-9 else float("inf")
                ratio_s = "—" if not np.isfinite(ratio) else f"{ratio:.1f}×"
                L.append(f"| `{m}` | {x['layer']} | {x['intent']:+.4f} | {x['outcome']:+.4f} | "
                         f"{ratio_s} | {'yes' if x['outcome_gt_intent'] else '**no**'} | "
                         f"{'yes' if x['both_above_null'] else '**no**'} | "
                         f"{x['lexical']:+.4f} | "
                         f"{'**excluded**' if x['lex_dom'] else 'ok'} |")
        L.append("")

    L.append("## At each model's peak-intent layer (`lexical_resid` variant)\n")
    L.append("| model | peak layer | intent probe acc | intent unique ρ² | outcome unique ρ² | "
             "outcome > intent |")
    L.append("|---|---:|---:|---:|---:|---|")
    for m in models:
        mr = [r for r in rows if r["model"] == m and r["model_variant"] == "lexical_resid"
              and r["is_peak"] in (True, "True")]
        if not mr:
            continue
        d = {r["predictor"]: r for r in mr}
        acc = d["intent"].get("intent_probe_acc")
        acc_s = f"{acc:.3f}" if isinstance(acc, float) and acc == acc else "—"
        L.append(f"| `{m}` | {d['intent']['layer']} | {acc_s} | "
                 f"{d['intent']['unique_rho2']:+.4f} | {d['outcome']['unique_rho2']:+.4f} | "
                 f"{'yes' if d['outcome']['unique_rho2'] > d['intent']['unique_rho2'] else '**no**'} |")
    L.append("")
    L.append("The intent probe accuracy column is the point of the experiment: a layer can be "
             "the best layer for *decoding* intent while contributing almost no unique variance "
             "to the geometry, which is the asymmetry P2 tests.\n")

    L.append("## Scenario-nuisance robustness\n")
    _, late_s, _, _ = _verdict_counts(rows, variant="plus_scenario")
    s_pass, _, s_und = _tally(models, late_s)
    if len(s_und) == n_models:
        L.append("The `plus_scenario` variant is also fully P4-excluded, for the same reason "
                 "(lexical and scenario are near-duplicates), so it adds nothing beyond the "
                 "`lexical_resid` variant above.\n")
    else:
        agree = (len(s_pass) >= SIGN_TEST_BAR) == p1
        L.append(f"Adding a scenario-identity RDM as a fifth nuisance regressor gives "
                 f"**{len(s_pass)}/{n_models}** models on P1 (primary: "
                 f"{len(p1_pass)}/{n_models}). The two **"
                 f"{'agree' if agree else 'DISAGREE'}** on the ordering.\n")

    path = os.path.join(outdir, "E1_VARIANCE_PARTITION.md")
    open(path, "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(path, ROOT)}")
    m1_late = _m1_counts(rows, models, "primary", "late")
    m1_peak = _m1_counts(rows, models, "primary", "peak")
    print(f"P1 (pre-registered, gated): "
          f"{'undecidable' if len(p1_und) == n_models else f'{len(p1_pass)}/{n_models}'}; "
          f"M1 late {len(m1_late[0])}/{n_models}, M1 peak {len(m1_peak[0])}/{n_models}")


def make_figure(rows, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})")
        return
    models = sorted({r["model"] for r in rows})
    colors = {"intent": "#1f77b4", "outcome": "#d62728",
              "interaction": "#9467bd", "lexical": "#7f7f7f"}
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.1 * nrow),
                             sharex=True, squeeze=False)
    for i, m in enumerate(models):
        ax = axes[i // ncol][i % ncol]
        mr = [r for r in rows if r["model"] == m and r["model_variant"] == "primary"]
        for pname in PREDICTORS:
            pts = sorted([r for r in mr if r["predictor"] == pname],
                         key=lambda r: r["rel_depth"])
            ax.plot([p["rel_depth"] for p in pts], [p["unique_rho2"] for p in pts],
                    marker="o", ms=3.5, lw=1.6, color=colors[pname], label=pname)
            ax.fill_between([p["rel_depth"] for p in pts],
                            [p["null_ci_lo"] for p in pts],
                            [p["null_ci_hi"] for p in pts],
                            color=colors[pname], alpha=0.10, lw=0)
        peak = next((r["rel_depth"] for r in mr if r["is_peak"] in (True, "True")), None)
        if peak is not None:
            ax.axvline(peak, color="k", ls=":", lw=1.0)
            ax.text(peak, ax.get_ylim()[1], " peak intent", fontsize=6, va="top")
        ax.axvspan(LATE_DEPTH_MIN, 1.0, color="k", alpha=0.04, lw=0)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(m, fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(models), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    for r_ in range(nrow):
        axes[r_][0].set_ylabel("unique ρ² (semipartial)", fontsize=8)
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("relative depth", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle("E1 — unique variance in the model RDM by predictor "
                 "(shaded: permutation null 95% band; grey band: late layers)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(outdir, "rdm_variance_partition_layerwise.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
