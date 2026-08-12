#!/usr/bin/env python3
"""E7b — the symmetric residual control. Pre-registered in E7_PRESPEC.md, tag `prespec-e7`.

E6b removed outcome and intent from the nuisance directions and found the erasure inert. E7b is the
mirror image: it removes scenario identity from the OUTCOME direction and asks whether W8A's
attempted-accidental recovery survives the 0.0988 floor.

The question is live because E6a-live found the outcome direction places 0.44-0.66 of its variance
inside the k=10 scenario subspace under chat templating, which leaves W8A's result consistent with
two mechanisms: (A) the recovery is carried by the outcome-specific component and the overlap is
incidental, or (B) it is carried by the scenario-aligned component, in which case "erasing outcome"
is partly erasing scenario identity.

Nothing here is chosen at run time. The site, the estimator, k, every threshold, the arms and the
decision table are all fixed in E7_PRESPEC.md and read from these constants.

Usage:
  python code/experiments/71_e7b_residual_control.py --dry-run
  python code/experiments/71_e7b_residual_control.py --run
  python code/experiments/71_e7b_residual_control.py --report
"""
import argparse
import csv
import importlib.util
import os
import sys
from collections import defaultdict

import numpy as np

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402

OUT_DIR = os.path.join(tc.ROOT, "outputs", "experiments")
PRESPEC = os.path.join(OUT_DIR, "E7_PRESPEC.md")

# ---- all inherited (E7_PRESPEC.md, "Thresholds — all inherited, none new") ----
J4_EFFECT_FLOOR = 0.0988
PPL_RATIO_MAX = 1.5
CELL_SD_RETENTION = 0.50
V1_TOL = 0.05
OUTCOME_CHANCE = 0.5168
SCENARIO_K = 10
SCALES = [0.25, 0.50, 0.75, 1.00]
RANDOM_SEEDS = [0, 1, 2, 3, 4]

W3_LAYERS = {
    "allenai/OLMo-2-1124-7B-Instruct": 16,
    "Qwen/Qwen2.5-7B-Instruct": 19,
    "allenai/Llama-3.1-Tulu-3-8B": 17,
}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def scenario_basis(X, groups, k=SCENARIO_K):
    """Orthonormal basis of the k-dimensional scenario subspace (E6a's construction, k inherited)."""
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    M = np.stack([X[v].mean(0) for v in idx.values()])
    M = M - M.mean(0, keepdims=True)
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    return Vt[:k]


def split_by_subspace(w, B):
    """Split a unit direction into its parts outside and inside the subspace spanned by B.

    Returns both parts renormalised to unit length, the surviving fraction ||w_perp||/||w||
    (E7_PRESPEC 4), and the fraction of w's variance lying inside the subspace (the E6a-live
    quantity, so a discrepancy against 0.44-0.66 shows up in this run).
    """
    w = w / np.linalg.norm(w)
    inside = B.T @ (B @ w)
    perp = w - inside
    surviving, n_in = float(np.linalg.norm(perp)), float(np.linalg.norm(inside))
    return (perp / max(surviving, 1e-30), inside / max(n_in, 1e-30),
            surviving, n_in ** 2)


def run_one(model_id, rows, a, w3, e6b):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    beh = w3._load_behavioral()
    tag = model_id.split("/")[-1]
    L = W3_LAYERS[model_id]
    print(f"\n{'=' * 70}\n=== E7b {tag}: erase at L{L} ===\n{'=' * 70}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, output_hidden_states=True,
        device_map="auto", trust_remote_code=True)
    model.eval()
    dig = beh.HFBackend._digit_token_ids(w3._DigitShim(tok, model_id))

    def fmt(row):
        prompt, s_min, s_max = beh.build_prompt(row["text"], a.template, row["source"])
        text = (tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
                if tok.chat_template else prompt)
        return text, s_min, s_max

    prepared = [(r,) + fmt(r) for r in rows]

    feats, labels_o, labels_i, groups = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(prepared), a.batch_size):
            chunk = prepared[i:i + a.batch_size]
            enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True).to(model.device)
            feats.append(model(**enc).hidden_states[L][:, -1, :].float().cpu().numpy())
            for (row, _t, _lo, _hi) in chunk:
                labels_o.append(int(row["condition"] in w3.HARM))
                labels_i.append(int(row["condition"] in w3.GUILTY))
                groups.append(tc.scenario_group_of(row["story_id"]))
    X = np.concatenate(feats).astype(np.float64)
    yo, yi = np.array(labels_o), np.array(labels_i)

    # ---------------- the construction ----------------
    w_out = e6b.meandiff(X, yo)
    B = scenario_basis(X, groups)
    w_perp, w_inside, surv, var_in = split_by_subspace(w_out, B)
    print(f"  outcome direction: {var_in:.4f} of its variance lies inside the k={SCENARIO_K} "
          f"scenario subspace (E6a-live reported 0.44-0.66)", flush=True)
    print(f"  surviving fraction after removing scenario: {surv:.4f}", flush=True)

    dirs = {"outcome_perp_scenario": w_perp, "outcome_full_dirmatched": w_out,
            "outcome_in_scenario": w_inside}
    for s in RANDOM_SEEDS:
        r = np.random.default_rng(s).standard_normal(X.shape[1])
        dirs[f"random_matched_seed{s}"] = r / np.linalg.norm(r)

    # Displacement is exactly linear in s for a rank-1 directional projection, so each arm's
    # matched scale is closed-form. The target is the SMALLEST s=1 displacement across arms, so
    # every arm is scaled DOWN to it and none is ever scaled up past its own s=1 (E7_PRESPEC 6).
    unit_disp = {}
    for name, d in dirs.items():
        fn, _ = e6b.rank1_dir_eraser(X, d, 1.0)
        unit_disp[name] = float(np.mean(np.linalg.norm(X - fn(X), axis=1)
                                        / np.linalg.norm(X, axis=1)))
    d_target = min(unit_disp.values())
    print(f"  s=1 displacements: "
          + ", ".join(f"{k.replace('random_matched_seed', 'rand')}={v:.4f}"
                      for k, v in unit_disp.items()), flush=True)
    print(f"  matched displacement target = {d_target:.4f} (the smallest, so nothing scales up)",
          flush=True)

    # ---------------- hook, scoring, ppl (identical machinery to W8A/E6b) ----------------
    state = {"block": max(0, L - 1), "fn": None, "gamma": 0.0}

    def make_hook(idx):
        def hook(_m, _inp, out):
            if state["fn"] is None or state["block"] != idx:
                return out
            h = out[0] if isinstance(out, tuple) else out
            g = state["gamma"]
            h2 = ((1.0 - g) * h.to(torch.float32) + g * state["fn"](h)).to(h.dtype)
            return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
        return hook

    for _i, _b in enumerate(model.model.layers):
        _b.register_forward_hook(make_hook(_i))

    w8a = _load("w8a", "63_w8a_amnesic_outcome.py")

    def set_erasure(fn_np, gamma):
        state["fn"] = (None if fn_np is None
                       else w8a.torch_eraser(fn_np, model.device, torch.float32))
        state["gamma"] = gamma

    def score():
        by_group = defaultdict(dict)
        with torch.no_grad():
            for i in range(0, len(prepared), a.batch_size):
                chunk = prepared[i:i + a.batch_size]
                enc = tok([c[1] for c in chunk], return_tensors="pt",
                          padding=True).to(model.device)
                logits = model(**enc).logits[:, -1, :].float()
                for (row, _t, s_min, s_max), lg in zip(chunk, logits):
                    vals = [d for d in range(int(s_min), int(s_max) + 1) if d in dig]
                    lp = torch.tensor([lg[dig[d]].item() for d in vals])
                    p = torch.softmax(lp, 0).tolist()
                    ev = sum(p[j] * vals[j] for j in range(len(vals)))
                    by_group[tc.scenario_group_of(row["story_id"])][row["condition"]] = \
                        (ev - s_min) / (s_max - s_min)
        cells = {c: float(np.mean([g[c] for g in by_group.values() if c in g]))
                 for c in w3.CELLS if any(c in g for g in by_group.values())}
        diffs = [g["attempted"] - g["accidental"] for g in by_group.values()
                 if "attempted" in g and "accidental" in g]
        return (float(np.mean(diffs)) if diffs else float("nan")), cells

    def ppl():
        vals = []
        with torch.no_grad():
            for t in w3.PPL_TEXTS:
                enc = tok(t, return_tensors="pt").to(model.device)
                vals.append(float(torch.exp(model(**enc, labels=enc["input_ids"]).loss)))
        return float(np.mean(vals))

    def cell_stats(cells):
        vs = [cells[c] for c in w3.CELLS if c in cells]
        b_int = ((cells.get("attempted", np.nan) + cells.get("intentional", np.nan))
                 - (cells.get("neutral", np.nan) + cells.get("accidental", np.nan))) / 2.0
        return (float(np.std(vs)) if vs else float("nan")), float(b_int)

    set_erasure(None, 0.0)
    base_c, base_cells = score()
    base_ppl = ppl()
    base_sd, base_bint = cell_stats(base_cells)
    print(f"  baseline contrast={base_c:+.4f} cellSD={base_sd:.4f} ppl={base_ppl:.3f}", flush=True)

    out_rows = []

    def emit(arm, kind, direction, scale, note):
        fn, _ = e6b.rank1_dir_eraser(X, direction, scale)
        set_erasure(fn, 1.0)
        c, cells = score()
        p = ppl()
        sd, b_int = cell_stats(cells)
        Xg = fn(X)
        disp = float(np.mean(np.linalg.norm(X - Xg, axis=1) / np.linalg.norm(X, axis=1)))
        # Reported, not gated: a rank-1 directional perturbation is not a full erasure and cannot
        # drive outcome to chance (E7_PRESPEC amendment 1).
        ins = w8a.insample_acc(Xg, yo)
        row = dict(model=tag, layer=L, arm=arm, kind=kind, scale=round(scale, 6),
                   contrast=round(c, 4), dcontrast=round(c - base_c, 4),
                   baseline_contrast=round(base_c, 4),
                   abs_dcontrast=round(abs(c - base_c), 4),
                   above_j4_floor=bool(abs(c - base_c) >= J4_EFFECT_FLOOR),
                   displacement_frac=round(disp, 4),
                   cell_sd=round(sd, 4),
                   cell_sd_retention=round(sd / base_sd, 4) if base_sd else "",
                   cell_sd_pass=bool(base_sd and (sd / base_sd) >= CELL_SD_RETENTION),
                   ppl=round(p, 3), ppl_ratio=round(p / base_ppl, 3),
                   coherence_pass=bool(p / base_ppl <= PPL_RATIO_MAX),
                   b_intent=round(b_int, 4), b_intent_baseline=round(base_bint, 4),
                   insample_outcome_acc=round(ins, 4),
                   outcome_var_in_scenario_subspace=round(var_in, 4),
                   surviving_fraction=round(surv, 4), scenario_k=SCENARIO_K,
                   matched_displacement_target=round(d_target, 4), note=note)
        out_rows.append(row)
        print(f"    {arm:28} s={scale:<7.4f} |dh|={disp:.4f} dC={c - base_c:+.4f} "
              f"{'>=FLOOR' if row['above_j4_floor'] else '<floor '} "
              f"ppl={p / base_ppl:.3f}{'' if row['coherence_pass'] else ' INCOHERENT'}",
              flush=True)
        return row

    # dose grid on the primary arm
    for s in SCALES:
        emit(f"outcome_perp_scenario_s{s:g}", "primary_dose", w_perp, s,
             "the test: outcome with scenario identity removed")
    # matched-displacement comparison across arm types
    for name, d in dirs.items():
        s = min(d_target / unit_disp[name], 1.0)
        kind = ("primary_matched" if name == "outcome_perp_scenario" else
                "positive_control" if name == "outcome_full_dirmatched" else
                "complement" if name == "outcome_in_scenario" else "null")
        emit(f"{name}_matched", kind, d, s,
             f"matched to displacement {d_target:.4f}")

    path = os.path.join(OUT_DIR, f"e7b_residual_control_{tag}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  wrote {os.path.relpath(path, tc.ROOT)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out_rows


# ------------------------------------------------------------------ report
def _read(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def report():
    """Apply the pre-registered decision table. No thresholds are chosen here."""
    per = {}
    for model_id in W3_LAYERS:
        tag = model_id.split("/")[-1]
        rows = _read(os.path.join(OUT_DIR, f"e7b_residual_control_{tag}.csv"))
        if not rows:
            continue
        by = {r["arm"]: r for r in rows}
        pos = by.get("outcome_full_dirmatched_matched")
        test = by.get("outcome_perp_scenario_matched")
        comp = by.get("outcome_in_scenario_matched")
        rands = [r for k, r in by.items() if k.startswith("random_matched_seed")]
        # The pre-registered coherence substitution: the largest dose in the grid that keeps the
        # perplexity ratio within 1.5.
        doses = sorted([r for r in rows if r["kind"] == "primary_dose"],
                       key=lambda r: float(r["scale"]))
        coherent = [r for r in doses if r["coherence_pass"] == "True"]
        chosen_dose = coherent[-1] if coherent else None
        per[tag] = dict(
            pos=pos, test=test, comp=comp, rands=rands, dose=chosen_dose,
            pos_ok=bool(pos and pos["above_j4_floor"] == "True"),
            test_ok=bool(test and test["above_j4_floor"] == "True"),
            comp_ok=bool(comp and comp["above_j4_floor"] == "True"),
            rand_violation=any(r["above_j4_floor"] == "True" for r in rands))
    if not per:
        raise SystemExit("no E7b CSVs found; run --run first")

    # The table is applied per model first: a model whose positive control fails, or whose random
    # null clears the floor, is void and contributes to neither side of the (A)/(B) count.
    n = len(per)
    n_pos = sum(v["pos_ok"] for v in per.values())
    valid = [v for v in per.values() if v["pos_ok"] and not v["rand_violation"]]
    n_test = sum(v["test_ok"] for v in valid)
    if not valid:
        row, verdict = "void", "**The test is VOID, not negative.**"
    elif n_test == len(valid):
        row, verdict = "A", "**(A) The recovery is carried by the outcome-specific component.**"
    elif n_test == 0:
        row, verdict = "B", "**(B) The recovery requires the scenario-aligned component.**"
    else:
        row, verdict = "split", "**Model-dependent: reported as a split, with no pooled claim.**"

    L = ["# E7b — the symmetric residual control: verdict\n",
         "Pre-registered in `E7_PRESPEC.md` (tag `prespec-e7`) before any compute; generated by "
         "`code/experiments/71_e7b_residual_control.py --report`. The decision table below was "
         "fixed in advance and is applied, not chosen.\n",
         "## Headline\n", f"{verdict} Selected row: `{row}`. "
         f"Positive control clears the {J4_EFFECT_FLOOR} floor in {n_pos} of {n} models; among the "
         f"{len(valid)} models where the test is valid, the test arm clears it in {n_test}.\n"]
    L.append("## The construction, as measured\n")
    L.append("| model | outcome variance inside the k=10 scenario subspace | surviving fraction "
             "after removing scenario | matched displacement |")
    L.append("|---|---:|---:|---:|")
    for tag, v in per.items():
        t = v["test"] or v["dose"]
        L.append(f"| `{tag}` | {t['outcome_var_in_scenario_subspace']} | "
                 f"{t['surviving_fraction']} | {t['matched_displacement_target']} |")
    L.append("")
    L.append("E6a-live reported 0.44–0.66 for the variance fraction; the values above are that "
             "same quantity recomputed in this run, so a discrepancy would show here.\n")
    L.append("## Arms at the matched displacement\n")
    L.append("| model | arm | role | Δcontrast | ≥ 0.0988 | cell-SD retention | ppl ratio |")
    L.append("|---|---|---|---:|:--:|---:|---:|")
    for tag, v in per.items():
        for key, label in (("test", "the test"), ("pos", "positive control"),
                           ("comp", "complement")):
            r = v[key]
            if not r:
                continue
            L.append(f"| `{tag}` | `{r['arm']}` | {label} | {r['dcontrast']} | "
                     f"{'YES' if r['above_j4_floor'] == 'True' else 'no'} | "
                     f"{r['cell_sd_retention']} | {r['ppl_ratio']} |")
        if v["rands"]:
            worst = max(v["rands"], key=lambda r: abs(float(r["dcontrast"])))
            L.append(f"| `{tag}` | `random_matched` (worst of {len(v['rands'])} seeds) | null | "
                     f"{worst['dcontrast']} | "
                     f"{'YES — VOIDS THIS MODEL' if worst['above_j4_floor'] == 'True' else 'no'} | "
                     f"{worst['cell_sd_retention']} | {worst['ppl_ratio']} |")
    L.append("")
    L.append("## Dose grid on the test arm\n")
    L.append("| model | s | Δcontrast | ≥ 0.0988 | ppl ratio | coherent |")
    L.append("|---|---:|---:|:--:|---:|:--:|")
    for tag in per:
        for r in sorted(_read(os.path.join(OUT_DIR, f"e7b_residual_control_{tag}.csv")),
                        key=lambda r: (r["kind"] != "primary_dose", float(r["scale"]))):
            if r["kind"] != "primary_dose":
                continue
            L.append(f"| `{tag}` | {float(r['scale']):.2f} | {r['dcontrast']} | "
                     f"{'YES' if r['above_j4_floor'] == 'True' else 'no'} | {r['ppl_ratio']} | "
                     f"{'yes' if r['coherence_pass'] == 'True' else 'NO'} |")
    L.append("")
    L.append("## Reading this against the pre-registered table\n")
    if row == "void":
        L.append("The positive control does not clear the floor at the matched displacement in any "
                 "model, so the matched dose is too small to move behaviour at all. E7_PRESPEC "
                 "wrote this row down in advance precisely so that this pattern is reported as "
                 "**no information about (A) vs (B)**, rather than as evidence for (A).\n")
    elif row == "A":
        L.append("The positive control clears the floor and so does the test arm, in every model. "
                 "The recovery survives removing the scenario-aligned component of the outcome "
                 "direction, so the overlap E6a-live measured is incidental to the effect and "
                 "W8A's framing stands as written.\n")
    elif row == "B":
        L.append("The positive control clears the floor and the test arm does not, in any model. "
                 "The recovery requires the scenario-aligned component: W8A's intervention is not "
                 "outcome-specific in the sense the framing implies, and this bound belongs in "
                 "LIMITATIONS.md and the abstract.\n")
    else:
        L.append("The result is split across models, so no pooled claim is made. The mechanism is "
                 "not established in the models whose test arm fails.\n")
    L.append("Note on V1a. The inherited in-sample-accuracy gate is reported per arm "
             "(`insample_outcome_acc`) and is not used as a gate here: every E7b arm is a rank-1 "
             "directional perturbation at a matched norm, which is not a full erasure and cannot "
             "drive outcome decoding to chance. See E7_PRESPEC amendment 1, appended before "
             "compute.\n")
    p = os.path.join(OUT_DIR, "E7B_RESIDUAL_CONTROL.md")
    open(p, "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(p, tc.ROOT)}")
    print(f"  row={row}; positive control {n_pos}/{n}, test arm {n_test}/{n}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(W3_LAYERS))
    ap.add_argument("--template", default="human_verbatim")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(PRESPEC):
        raise SystemExit(f"{PRESPEC} missing; E7b is pre-registered, refusing to run.")
    if a.report:
        raise SystemExit(report())

    w3 = _load("w3", "48_w3_causal_steering.py")
    e6b = _load("e6b", "67_e6b_orthogonalized.py")
    rows = w3.load_dataset()
    print(f"E7b: {len(rows)} items, {len(set(r['scenario_group'] for r in rows))} groups; "
          f"k={SCENARIO_K}, floor={J4_EFFECT_FLOOR}")
    if a.dry_run or not a.run:
        print(f"Plan per model: {len(SCALES)} dose arms + 3 matched arms + "
              f"{len(RANDOM_SEEDS)} matched random nulls = {len(SCALES) + 3 + len(RANDOM_SEEDS)}.")
        return
    # Runs from a pristine checkout of HEAD (see E7A_REPRODUCTION.md), so this passes without an
    # override and the numbers are attributable to a commit.
    _load("guard", "_run_guard.py").require_clean_tree(
        "71_e7b_residual_control.py", extra_deps=("code/03_behavioral.py",))
    for mid in a.models:
        run_one(mid, rows, a, w3, e6b)
    report()


if __name__ == "__main__":
    main()
