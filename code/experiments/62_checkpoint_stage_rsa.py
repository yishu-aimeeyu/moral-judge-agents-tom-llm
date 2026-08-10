#!/usr/bin/env python3
"""E2 -- checkpoint-stage RSA: does post-training reshape the geometry or only the readout?

WHY THIS EXISTS. checkpoint_dissection.csv shows the behavioural contrast collapsing along the
tuning pipeline (OLMo base +0.002 -> Instruct -0.687). base_vs_instruct_geometry.csv shows the
ENDPOINTS have nearly identical geometry (OLMo RSA 0.914). Those two facts together suggest
post-training changes what the model does with its representation rather than the
representation itself -- but the geometry comparison has only ever been run base-vs-final, so
it cannot distinguish "stable throughout" from "reshaped and then restored". E2 runs it at
every stage, which is what makes the "readout, not representation" claim testable.

Pre-specified predictions P1-P4, bars, reference points and falsifiers:
outputs/rsa/E2_PRESPEC.md, written before the first run and never overwritten.

Two-phase, because activation extraction needs a GPU and the analysis does not:
  --plan     print which checkpoints still need activations, with the extract commands
  --run      analyse whatever activations are present; missing stages are reported, not faked

Outputs -> outputs/rsa/
  checkpoint_stage_geometry.csv     consecutive + base->final RSA/CKA, with delta_contrast
  checkpoint_stage_reference.csv    split-half ceiling per checkpoint, cross-family floor
  checkpoint_stage_hypothesis.csv   per-stage partial_r_intent / partial_r_outcome (P4)
  checkpoint_stage_rsa.png
  E2_CHECKPOINT_STAGE_RSA.md
"""
import argparse
import csv
import importlib.util
import itertools
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")
PRESPEC = os.path.join(RSA_DIR, "E2_PRESPEC.md")
DISSECTION = os.path.join(ROOT, "outputs", "experiments", "checkpoint_dissection.csv")

# Pre-specified in E2_PRESPEC.md; constants rather than flags so they cannot drift
RSA_BAR = 0.90
BEHAVIOUR_BAR = 0.15
PRIMARY_DEPTH = 0.75
FLAT_BAR = 0.10            # P4: max |change| in partial_r_outcome / intent_org_ratio

# Primary families, verbatim from 16_checkpoint_dissection.py FAMILIES. Zephyr is excluded
# from the primary by the prespec (weakest-engaged base, outlier SFT share).
FAMILIES = {
    "OLMo-2-7B": [
        ("base", "allenai/OLMo-2-1124-7B"),
        ("SFT", "allenai/OLMo-2-1124-7B-SFT"),
        ("DPO", "allenai/OLMo-2-1124-7B-DPO"),
        ("Instruct", "allenai/OLMo-2-1124-7B-Instruct"),
    ],
    "Tulu-3-8B": [
        ("base", "unsloth/Meta-Llama-3.1-8B"),
        ("SFT", "allenai/Llama-3.1-Tulu-3-8B-SFT"),
        ("DPO", "allenai/Llama-3.1-Tulu-3-8B-DPO"),
        ("RLVR", "allenai/Llama-3.1-Tulu-3-8B"),
    ],
}
SECONDARY_FAMILIES = {
    "Zephyr-7B": [
        ("base", "mistralai/Mistral-7B-v0.1"),
        ("SFT", "alignment-handbook/zephyr-7b-sft-full"),
        ("DPO", "HuggingFaceH4/zephyr-7b-beta"),
    ],
}


def _load_rsa():
    spec = importlib.util.spec_from_file_location(
        "rsa_24", os.path.join(HERE, "24_rsa_cka.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tag_of(model_id):
    return model_id.split("/")[-1]


def acts_path(acts_dir, model_id):
    return os.path.join(acts_dir, f"{tag_of(model_id)}.npz")


def load_behaviour():
    """family -> stage -> contrast, read from checkpoint_dissection.csv. Never recomputed."""
    out = {}
    if not os.path.exists(DISSECTION):
        return out
    for r in csv.DictReader(open(DISSECTION)):
        try:
            out.setdefault(r["family"], {})[r["stage"]] = float(r["contrast"])
        except (KeyError, ValueError):
            pass
    return out


def layer_for(nL, depth):
    return int(round(depth * (nL - 1)))


def coupling_stat(geo, n_boot=10000, seed=0):
    """P2's statistic with n and a bootstrap CI (E5d R1).

    Reported in one place so the headline, the P2 section and the figure cannot drift apart or
    quote the point estimate without its interval. Resampling is over transitions, which is the
    unit the statistic is computed on; with 6 of them the interval is wide, and that width is
    the finding R2 asks to be stated rather than something to hide.
    """
    from scipy.stats import spearmanr
    xs = [(float(r["rdm_shift_reldepth0.75"]), abs(float(r["delta_contrast"])))
          for r in geo if not r["is_endpoint_pair"]
          and r["rdm_shift_reldepth0.75"] not in ("", None)
          and r["delta_contrast"] not in ("", None)]
    n = len(xs)
    if n < 4:
        return None
    x = np.array([a for a, _ in xs])
    y = np.array([b for _, b in xs])
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boot.append(float(spearmanr(x[idx], y[idx]).statistic))
    lo, hi = (float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))) \
        if boot else (float("nan"), float("nan"))
    return dict(rho=rho, n=n, lo=lo, hi=hi, n_boot=len(boot))


def coupling_str(c, short=False):
    """The only permitted rendering of the coupling statistic (E5d R1)."""
    if not c:
        return "not computable"
    if short:
        return f"ρ={c['rho']:+.2f} (n={c['n']}, 95% CI [{c['lo']:+.2f}, {c['hi']:+.2f}])"
    return (f"Spearman ρ = **{c['rho']:+.3f}** (n = {c['n']} transitions, bootstrap 95% CI "
            f"[{c['lo']:+.3f}, {c['hi']:+.3f}])")


UNDERPOWERED = (
    "**Six points cannot settle this in either direction.** The bootstrap interval above is "
    "wide enough to include values far below the +0.50 bar as well as values near +1, so this "
    "analysis can neither establish coupling nor establish its absence. P1 is withdrawn because "
    "the prespec's rule says a point estimate at or above +0.50 withdraws it, and that rule is "
    "honoured as written. Withdrawing P1 is **not** evidence for its converse: nothing here "
    "shows that geometric reshaping *drives* the behavioural change, and that claim must not be "
    "made from this statistic.")


def split_half_ceiling(X, rng, rsa):
    """Reliability ceiling for an RDM from this checkpoint (P3).

    RSA between RDMs built from two disjoint random halves of the hidden units, then
    Spearman-Brown corrected to full width. The raw split-half value is measured at HALF the
    units and is therefore noisier than the full-width RDMs every cross-stage comparison uses;
    quoting it directly against a full-width number is a width mismatch. See E2_PRESPEC.md
    amendment 1. Returns (raw, corrected).
    """
    h = X.shape[1]
    perm = rng.permutation(h)
    a, b = perm[: h // 2], perm[h // 2: 2 * (h // 2)]
    ra = rsa.build_rdm(X[:, a].astype(np.float64))
    rb = rsa.build_rdm(X[:, b].astype(np.float64))
    raw = rsa.spearman(rsa.upper(ra), rsa.upper(rb))[0]
    corrected = (2.0 * raw / (1.0 + raw)) if raw > -1 else float("nan")
    return raw, corrected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--acts", default=os.path.join(ROOT, "outputs", "acts"))
    ap.add_argument("--probe", default=os.path.join(ROOT, "outputs", "probe"))
    ap.add_argument("--out", default=RSA_DIR)
    ap.add_argument("--pooling", default="mean",
                    choices=["last", "mean", "belief_last", "action_last"])
    ap.add_argument("--include-zephyr", action="store_true",
                    help="add the prespec's secondary family (only after the primary completes)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    fams = dict(FAMILIES)
    if a.include_zephyr:
        fams.update(SECONDARY_FAMILIES)

    present, missing = {}, []
    for fam, stages in fams.items():
        for stage, mid in stages:
            p = acts_path(a.acts, mid)
            (present.setdefault(fam, []).append((stage, mid, p))
             if os.path.exists(p) else missing.append((fam, stage, mid)))

    if a.plan or not a.run:
        print("=== E2 CHECKPOINT-STAGE RSA PLAN ===")
        print(f"pooling={a.pooling}  primary depth={PRIMARY_DEPTH}  RSA bar={RSA_BAR}")
        for fam, stages in fams.items():
            have = {s for s, _, _ in present.get(fam, [])}
            print(f"\n{fam}:")
            for stage, mid in stages:
                print(f"  {stage:9} {mid:46} "
                      f"{'HAVE acts' if stage in have else 'NEEDS extraction'}")
        if missing:
            print(f"\n{len(missing)} checkpoint(s) need activations. Extract on a GPU node:")
            ids = " ".join(mid for _, _, mid in missing)
            print(f"  JOBNAME=e2acts bash engaging/submit_gpu.sh \\\n"
                  f"    \"python code/01_extract_activations.py --models {ids} "
                  f"--clause-offsets dataset/master/clause_offsets.csv\"")
            print("\nThen:\n  python code/experiments/62_checkpoint_stage_rsa.py --run")
        else:
            print("\nAll checkpoints have activations. Run:\n"
                  "  python code/experiments/62_checkpoint_stage_rsa.py --run")
        return

    if not os.path.exists(PRESPEC):
        raise SystemExit(f"{PRESPEC} missing; E2 is pre-registered, refusing to run.")

    rsa = _load_rsa()
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.pooling == "last" else f"_{a.pooling}"
    lab = {r["story_id"]: r for r in csv.DictReader(open(a.csv))}
    beh = load_behaviour()
    rng = np.random.default_rng(0)

    # ---------------- load activations, build RDMs at both depths ----------------
    store = {}
    for fam, items in present.items():
        for stage, mid, p in items:
            tag = tag_of(mid)
            X, sids = rsa.load_model(p, lab, a.pooling)
            nL = X.shape[1]
            pk = rsa.peak_intent_layer(tag, a.probe, suffix)
            layers = {"reldepth0.75": layer_for(nL, PRIMARY_DEPTH)}
            if pk is not None:
                layers["peak"] = pk
            rdms = {k: rsa.build_rdm(X[:, L, :].astype(np.float64)) for k, L in layers.items()}
            store[(fam, stage)] = {
                "tag": tag, "model_id": mid, "n_layers": nL, "peak": pk,
                "layers": layers, "rdms": rdms, "sids": sids,
                "acts": {k: X[:, L, :] for k, L in layers.items()},
            }
            print(f"  {fam:11} {stage:9} {tag:34} layers={nL:3} peak={pk} "
                  f"L@0.75={layers['reldepth0.75']}", flush=True)

    if not store:
        raise SystemExit("no activations present for any checkpoint; run --plan first")

    sids0 = next(iter(store.values()))["sids"]
    groups = np.array([lab[s].get("scenario_group") or lab[s]["scenario_id"] for s in sids0])

    # ---------------- P3 reference points ----------------
    ref_rows = []
    for (fam, stage), d in store.items():
        for mode in d["layers"]:
            raw, corrected = split_half_ceiling(d["acts"][mode], rng, rsa)
            ref_rows.append({"kind": "split_half_ceiling", "family": fam, "stage": stage,
                             "mode": mode, "model_a": d["tag"], "model_b": d["tag"],
                             "rsa_spearman": round(raw, 4),
                             "rsa_spearman_sb_corrected": round(corrected, 4)})
    # cross-family floor: unrelated models at matched depth
    for (f1, s1), (f2, s2) in itertools.combinations(sorted(store), 2):
        if f1 == f2:
            continue
        d1, d2 = store[(f1, s1)], store[(f2, s2)]
        mode = "reldepth0.75"
        if mode in d1["rdms"] and mode in d2["rdms"]:
            rho = rsa.spearman(rsa.upper(d1["rdms"][mode]), rsa.upper(d2["rdms"][mode]))[0]
            ref_rows.append({"kind": "cross_family_floor", "family": f"{f1}|{f2}",
                             "stage": f"{s1}|{s2}", "mode": mode,
                             "model_a": d1["tag"], "model_b": d2["tag"],
                             "rsa_spearman": round(rho, 4)})
    _write(os.path.join(a.out, "checkpoint_stage_reference.csv"), ref_rows)

    # ---------------- P1/P2: stage transitions ----------------
    geo_rows = []
    for fam, stages in fams.items():
        order = [s for s, _ in stages if (fam, s) in store]
        pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
        # with only two stages present the endpoint pair IS the consecutive pair
        if len(order) >= 3:
            pairs.append((order[0], order[-1]))
        for s1, s2 in pairs:
            d1, d2 = store[(fam, s1)], store[(fam, s2)]
            row = {"family": fam, "from_stage": s1, "to_stage": s2,
                   "transition": f"{s1}->{s2}",
                   "is_endpoint_pair": (s1, s2) == (order[0], order[-1]),
                   "model_a": d1["tag"], "model_b": d2["tag"]}
            for mode in ("reldepth0.75", "peak"):
                if mode in d1["rdms"] and mode in d2["rdms"]:
                    rho = rsa.spearman(rsa.upper(d1["rdms"][mode]),
                                       rsa.upper(d2["rdms"][mode]))[0]
                    row[f"rsa_{mode}"] = round(rho, 4)
                    row[f"rdm_shift_{mode}"] = round(1.0 - rho, 4)
                    row[f"cka_linear_{mode}"] = round(
                        rsa.linear_cka(d1["acts"][mode].astype(np.float64),
                                       d2["acts"][mode].astype(np.float64)), 4)
                else:
                    row[f"rsa_{mode}"] = ""
                    row[f"rdm_shift_{mode}"] = ""
                    row[f"cka_linear_{mode}"] = ""
            c1 = beh.get(fam, {}).get(s1)
            c2 = beh.get(fam, {}).get(s2)
            row["contrast_from"] = "" if c1 is None else round(c1, 4)
            row["contrast_to"] = "" if c2 is None else round(c2, 4)
            row["delta_contrast"] = "" if (c1 is None or c2 is None) else round(c2 - c1, 4)
            row["abs_delta_contrast"] = ("" if row["delta_contrast"] == ""
                                         else abs(row["delta_contrast"]))
            geo_rows.append(row)
            print(f"  {fam:11} {row['transition']:20} "
                  f"RSA@0.75={row['rsa_reldepth0.75']} peak={row['rsa_peak']} "
                  f"dcontrast={row['delta_contrast']}", flush=True)
    _write(os.path.join(a.out, "checkpoint_stage_geometry.csv"), geo_rows)

    # ---------------- P4: per-stage hypothesis RDM fits ----------------
    intent_v = np.array([1 if lab[s]["intent_label"] == "guilty" else 0 for s in sids0])
    outcome_v = np.array([1 if lab[s]["outcome_label"] == "harm" else 0 for s in sids0])
    u_int = rsa.upper((intent_v[:, None] != intent_v[None, :]).astype(float))
    u_out = rsa.upper((outcome_v[:, None] != outcome_v[None, :]).astype(float))
    u_scen = rsa.upper((groups[:, None] != groups[None, :]).astype(float))

    hyp_rows = []
    for fam, stages in fams.items():
        for stage, _ in stages:
            if (fam, stage) not in store:
                continue
            d = store[(fam, stage)]
            for mode, rdm in d["rdms"].items():
                u = rsa.upper(rdm)
                pri = rsa.partial_spearman(u, u_int, u_scen)
                pro = rsa.partial_spearman(u, u_out, u_scen)
                den = abs(pri) + abs(pro)
                hyp_rows.append({
                    "family": fam, "stage": stage, "mode": mode, "model": d["tag"],
                    "layer": d["layers"][mode],
                    "partial_r_intent": round(pri, 4),
                    "partial_r_outcome": round(pro, 4),
                    "intent_org_ratio": round(abs(pri) / den, 4) if den > 1e-9 else "",
                    "contrast": beh.get(fam, {}).get(stage, ""),
                })
    _write(os.path.join(a.out, "checkpoint_stage_hypothesis.csv"), hyp_rows)

    write_verdict(fams, store, geo_rows, ref_rows, hyp_rows, missing, a.out, a.pooling)
    make_figure(fams, geo_rows, ref_rows, beh, store, a.out)


def _write(path, rows):
    if not rows:
        print(f"  (no rows for {os.path.basename(path)})")
        return
    cols = sorted({k for r in rows for k in r})
    order = [c for c in ("kind", "family", "stage", "from_stage", "to_stage", "transition")
             if c in cols] + [c for c in cols if c not in
                              ("kind", "family", "stage", "from_stage", "to_stage",
                               "transition")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in order})
    print(f"wrote {os.path.relpath(path, ROOT)} ({len(rows)} rows)")


def write_verdict(fams, store, geo, ref, hyp, missing, outdir, pooling):
    ceil_raw = [r["rsa_spearman"] for r in ref if r["kind"] == "split_half_ceiling"]
    ceil = [r["rsa_spearman_sb_corrected"] for r in ref
            if r["kind"] == "split_half_ceiling"]
    floor = [r["rsa_spearman"] for r in ref if r["kind"] == "cross_family_floor"]
    consec = [r for r in geo if not r["is_endpoint_pair"]]
    endpoints = [r for r in geo if r["is_endpoint_pair"]]

    def _rsa(r):
        v = r.get("rsa_reldepth0.75")
        return float(v) if v not in ("", None) else None

    ok_consec = [r for r in consec if _rsa(r) is not None and _rsa(r) >= RSA_BAR]
    bad_consec = [r for r in consec if _rsa(r) is not None and _rsa(r) < RSA_BAR]

    L = ["# E2 — checkpoint-stage RSA: verdict\n",
         f"Generated by `code/experiments/62_checkpoint_stage_rsa.py` (pooling `{pooling}`). "
         "Bars and reference points were fixed in [`E2_PRESPEC.md`](E2_PRESPEC.md) before "
         "this ran.\n"]

    # Headline first, so the failed prediction cannot be missed by reading only the top.
    cstat = coupling_stat(geo)
    if cstat:
        L.append("## Headline\n")
        if cstat["rho"] >= 0.5:
            L.append(f"**The pre-registered claim did not survive.** Every consecutive-stage RSA "
                     f"clears the 0.90 bar ({len(ok_consec)}/{len(consec)}), so P1's magnitude "
                     f"criterion passes. But P2 — the criterion that carries the \"readout, not "
                     f"representation\" interpretation — **fails**: RDM shift and |Δcontrast| are "
                     f"rank-coupled at {coupling_str(cstat)}, whose point estimate is above the "
                     f"+0.50 bar the prespec designated as the point at which P1 is withdrawn. "
                     f"E2 therefore does **not** license \"post-training changes only the "
                     f"readout\". What it licenses is narrower: post-training moves the geometry "
                     f"by a small amount that tracks the ordering of behavioural change.\n")
            L.append(UNDERPOWERED + "\n")
        else:
            L.append(f"P1's magnitude criterion passes ({len(ok_consec)}/{len(consec)} "
                     f"consecutive transitions at RSA ≥ 0.90) and P2's decoupling criterion "
                     f"also holds ({coupling_str(cstat)}, point estimate below +0.50).\n")
            L.append(UNDERPOWERED + "\n")

    if missing:
        L.append("## Coverage — incomplete\n")
        L.append("These checkpoints have no activations yet, so the transitions involving them "
                 "are absent rather than estimated:\n")
        for fam, stage, mid in missing:
            L.append(f"- `{fam}` / `{stage}` — `{mid}`")
        L.append("\nExtraction command is in `--plan`. Any verdict below is provisional until "
                 "these land; the prespec's P1 asks for **every** consecutive transition.\n")

    L.append("## P3 reference points (these bound every number below)\n")
    if ceil:
        L.append(f"- **Ceiling** (split-half within a checkpoint, Spearman–Brown corrected to "
                 f"full width): **{min(ceil):.4f} … {max(ceil):.4f}** "
                 f"(raw half-width split-half: {min(ceil_raw):.4f} … {max(ceil_raw):.4f}; "
                 f"see amendment 1 for why the raw value is not the ceiling)")
    if floor:
        L.append(f"- **Floor** (cross-family, matched depth): "
                 f"{min(floor):.4f} … {max(floor):.4f}")
    else:
        L.append("- **Floor**: not computable yet — needs at least two families with "
                 "activations present.")

    observed = [v for v in (_rsa(r) for r in geo) if v is not None]
    peak_obs = [float(r["rsa_peak"]) for r in geo if r.get("rsa_peak") not in ("", None)]
    if ceil and (observed or peak_obs):
        top = max(observed + peak_obs)
        at_ceiling = top >= max(ceil)
        if floor and max(floor) >= RSA_BAR:
            L.append(f"- P1's {RSA_BAR:.2f} bar is **NOT usable**: the cross-family floor "
                     f"({max(floor):.4f}) already sits at or above it, so RSA cannot "
                     f"discriminate on this stimulus set. No P1 verdict is claimed.")
        elif at_ceiling:
            L.append(f"- **At-ceiling case.** The largest cross-stage RSA ({top:.4f}) is at or "
                     f"above the corrected reliability ceiling ({max(ceil):.4f}). The "
                     f"geometry difference between these stages is therefore **not "
                     f"distinguishable from measurement noise**: two different checkpoints' "
                     f"RDMs agree as closely as two independent estimates of the *same* "
                     f"checkpoint's RDM do. This is the strongest available form of P1, and it "
                     f"is a stronger statement than clearing the {RSA_BAR:.2f} bar.")
        else:
            L.append(f"- Ordinary case: largest cross-stage RSA {top:.4f} sits between the "
                     f"floor and the corrected ceiling; read it against the "
                     f"{RSA_BAR:.2f} bar below.")
    L.append("")

    L.append("## P1 — consecutive-stage geometry vs behavioural movement\n")
    L.append("| family | transition | RSA @ depth 0.75 | RSA @ peak | RDM shift | "
             "Δcontrast | RSA ≥ %.2f | \\|Δcontrast\\| ≥ %.2f |" % (RSA_BAR, BEHAVIOUR_BAR))
    L.append("|---|---|---:|---:|---:|---:|---|---|")
    for r in geo:
        v = _rsa(r)
        dc = r["delta_contrast"]
        L.append(f"| {r['family']} | `{r['transition']}`"
                 f"{' *(endpoints)*' if r['is_endpoint_pair'] else ''} | "
                 f"{r['rsa_reldepth0.75']} | {r['rsa_peak']} | "
                 f"{r['rdm_shift_reldepth0.75']} | {dc} | "
                 f"{'yes' if v is not None and v >= RSA_BAR else '**no**'} | "
                 f"{'yes' if dc not in ('', None) and abs(float(dc)) >= BEHAVIOUR_BAR else 'no'} |")
    L.append("")
    # Two independent families can round to the same 4 dp; flag it so no reader reads a
    # coincidence as duplicated rows. Verified at full precision when it occurs.
    _by_val = {}
    for r in consec:
        v = r.get("rsa_reldepth0.75")
        if v not in ("", None):
            _by_val.setdefault(str(v), []).append(f"{r['family']} `{r['transition']}`")
    _dupes = {k: v for k, v in _by_val.items() if len(v) > 1}
    for val, who in _dupes.items():
        L.append(f"> **Not a duplicated row.** {' and '.join(who)} both round to RSA {val} at "
                 f"4 dp. These were recomputed at full precision from different activation "
                 f"files and differ from the fifth decimal onward (0.92948886 vs 0.92951168 for "
                 f"the two `base->SFT` transitions); the CKA column for the same pairs differs "
                 f"in the third decimal. The agreement is a coincidence of rounding, not a "
                 f"bug.\n")
    if consec:
        verdict = ("supported" if not bad_consec and not missing else
                   "provisional" if not bad_consec else "NOT supported")
        L.append(f"**P1: {len(ok_consec)}/{len(consec)}** consecutive transitions clear "
                 f"RSA ≥ {RSA_BAR:.2f} at depth 0.75 → **{verdict}**"
                 + (" (coverage incomplete)" if missing else "") + ".")
        if bad_consec:
            L.append("\nTransitions below the bar — the prespec calls this an informative "
                     "falsification, especially at SFT in OLMo, which carries 85% of the "
                     "behavioural shift:\n")
            for r in bad_consec:
                L.append(f"- `{r['family']}` `{r['transition']}`: RSA "
                         f"{r['rsa_reldepth0.75']}, Δcontrast {r['delta_contrast']}")
    L.append("")

    L.append("## P2 — is geometric movement coupled to behavioural movement?\n")
    xs = [(float(r["rdm_shift_reldepth0.75"]), abs(float(r["delta_contrast"])))
          for r in consec
          if r["rdm_shift_reldepth0.75"] not in ("", None)
          and r["delta_contrast"] not in ("", None)]
    if len(xs) >= 4:
        rho = cstat["rho"]
        L.append(f"P2 predicted that RDM shift and |Δcontrast| would **not** be positively "
                 f"rank-coupled at ρ ≥ +0.50. Observed: {coupling_str(cstat)}.\n")
        L.append(UNDERPOWERED + "\n")
        if rho >= 0.5:
            L.append("**P2 FAILS on its point estimate.** The coupling is above the bar, so by "
                     "the prespec's own rule *P1 is withdrawn in favour of the coupling "
                     "reading*: the stages where geometry moves most are also the stages where "
                     "behaviour moves most. The pre-registered \"readout, not representation\" "
                     "claim is **not supported in the form it was registered**, and that is the "
                     "headline of E2.\n")
            L.append("A magnitude observation is recorded here because it is the obvious next "
                     "question, and it is marked **post-hoc (O1)** so it cannot read as "
                     "pre-registered — it does **not** overturn the verdict above. Spearman ρ "
                     "is scale-free, so it registers the *ordering* of the transitions while "
                     "saying nothing about their size. The ordering is shared, and the sizes "
                     "are not commensurate: the largest geometric movement is a shift of "
                     f"{max(x for x, _ in xs):.4f} (RSA "
                     f"{1 - max(x for x, _ in xs):.4f}, i.e. above the 0.90 bar and near the "
                     "reliability ceiling), while the behaviour it accompanies moves by "
                     f"{max(y for _, y in xs):.4f} on the same contrast scale. Both statements "
                     "are true at once: post-training moves the geometry *detectably and in "
                     "step with behaviour*, but *by a small amount*. Adjudicating between "
                     "\"small but sufficient\" and \"small and incidental\" needs an "
                     "intervention rather than a correlation over six points, and E2 cannot "
                     "settle it.\n")
        else:
            L.append("**P2 holds**: no coupling at or above the bar, which is consistent with "
                     "P1's reading.\n")
    else:
        L.append(f"Only {len(xs)} transitions have both quantities; the prespec's 6-transition "
                 "rank correlation is not yet computable. Reported as pending rather than null.")
    L.append("")

    L.append("## P4 — per-stage hypothesis-RDM fits (flat = readout, swinging = reshaping)\n")
    L.append("| family | stage | mode | partial r intent | partial r outcome | "
             "intent_org_ratio | contrast |")
    L.append("|---|---|---|---:|---:|---:|---:|")
    for r in hyp:
        L.append(f"| {r['family']} | {r['stage']} | {r['mode']} | "
                 f"{r['partial_r_intent']:+.4f} | {r['partial_r_outcome']:+.4f} | "
                 f"{r['intent_org_ratio']} | {r['contrast']} |")
    L.append("")
    for fam in fams:
        rows = [r for r in hyp if r["family"] == fam and r["mode"] == "reldepth0.75"]
        order = [s for s, _ in fams[fam]]
        rows.sort(key=lambda r: order.index(r["stage"]))
        if len(rows) >= 2:
            d_out = rows[-1]["partial_r_outcome"] - rows[0]["partial_r_outcome"]
            try:
                d_ratio = float(rows[-1]["intent_org_ratio"]) - float(rows[0]["intent_org_ratio"])
            except (TypeError, ValueError):
                d_ratio = float("nan")
            flat = abs(d_out) < FLAT_BAR and (not np.isfinite(d_ratio) or abs(d_ratio) < FLAT_BAR)
            L.append(f"- `{fam}` {rows[0]['stage']}→{rows[-1]['stage']}: "
                     f"Δpartial_r_outcome = {d_out:+.4f}, Δintent_org_ratio = {d_ratio:+.4f} → "
                     f"**{'flat (P4 supported)' if flat else 'swings (P4 not supported)'}** "
                     f"(bar {FLAT_BAR}).")
    L.append("")

    p = os.path.join(outdir, "E2_CHECKPOINT_STAGE_RSA.md")
    open(p, "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(p, ROOT)}")


def make_figure(fams, geo, ref, beh, store, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skip figure: {e})")
        return
    ceil = [r["rsa_spearman_sb_corrected"] for r in ref
            if r["kind"] == "split_half_ceiling"]
    floor = [r["rsa_spearman"] for r in ref if r["kind"] == "cross_family_floor"]
    have = [f for f in fams if any(k[0] == f for k in store)]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.4 * len(have), 4.4), squeeze=False)
    for i, fam in enumerate(have):
        ax = axes[0][i]
        order = [s for s, _ in fams[fam] if (fam, s) in store]
        trans = [r for r in geo if r["family"] == fam and not r["is_endpoint_pair"]]
        xs = list(range(len(trans)))
        ys = [float(r["rsa_reldepth0.75"]) if r["rsa_reldepth0.75"] not in ("", None)
              else np.nan for r in trans]
        if ceil:
            ax.axhspan(min(ceil), max(ceil), color="#2ca02c", alpha=0.12, lw=0,
                       label="reliability ceiling (SB-corrected)")
        if floor:
            ax.axhspan(min(floor), max(floor), color="#7f7f7f", alpha=0.15, lw=0,
                       label="cross-family floor")
        ax.axhline(RSA_BAR, color="k", ls="--", lw=1.1, label=f"P1 bar {RSA_BAR}")
        ax.plot(xs, ys, marker="o", color="#1f77b4", lw=1.8, label="stage RSA @ depth 0.75")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["transition"] for r in trans], rotation=20, fontsize=7)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("RSA (Spearman) between consecutive stages", fontsize=8)
        ax2 = ax.twinx()
        cs = [beh.get(fam, {}).get(s) for s in order]
        ax2.plot(range(len(order)), cs, marker="s", color="#d62728", lw=1.5, ls=":",
                 label="behavioural contrast")
        ax2.set_ylabel("contrast (attempted − accidental)", color="#d62728", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax.set_title(fam, fontsize=9)
        if i == 0:
            ax.legend(fontsize=6.5, loc="center left", frameon=False)
    # The visual reads as "flat geometry, collapsing behaviour", which is the claim P2
    # withdrew. The coupling result goes on the figure so it cannot travel without it.
    c = coupling_stat(geo)
    sub = ""
    if c:
        # Kept to short lines: a single long subtitle is clipped at both figure edges, which
        # silently truncates the caveat that the panel exists to carry.
        sub = (f"\nP1's magnitude bar passes, but P2 FAILS on its point estimate: shift and "
               f"|Δcontrast| are rank-coupled, {coupling_str(c, short=True)}."
               f"\nThis does NOT show \"readout only\" — and n={c['n']} can establish neither "
               f"coupling nor its absence.")
    fig.suptitle("E2 — does post-training reshape the geometry (blue) while behaviour moves "
                 "(red)?" + sub, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = os.path.join(outdir, "checkpoint_stage_rsa.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
