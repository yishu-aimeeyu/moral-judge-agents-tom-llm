#!/usr/bin/env python3
"""E8g verdict -- does OLMo-2-7B show Pythia's position-by-factor dissociation? [CPU]

Applies the rule fixed in E8_PRESPEC.md amendment 3.1 before any OLMo beta existed: at the
last stage-1 revision, at each of the five depths, the target factor (outcome at `last`,
intent at `belief_last`) clears its band and the other factor does not; >= 3 of 5 depths per
position. The band test is the Pythia verdict's own `clears_band()`, imported, not copied.

The shape fits (`checkpoint_shape_fits_OLMo-2-1124-7B.csv`, from 75 at n = 8) and the
post-training stages are reported descriptively and do not enter the verdict (amendment 3).

Output -> outputs/rsa/E8G_OLMO_REPLICATION.md
"""
import argparse
import csv
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")
MODEL = "OLMo-2-1124-7B"
DEPTHS = ["0.2", "0.4", "0.6", "0.8", "1.0"]
TARGET = {"last": "outcome", "belief_last": "intent"}
OTHER = {"outcome": "intent", "intent": "outcome"}
MIN_DEPTHS = 3


def _load(fname, modname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V = _load("76_e8_verdict.py", "e8_76")


def pattern_table(rows, step):
    at = {(r["position"], r["rel_depth_requested"], r["predictor"]): r for r in rows
          if r["spec"] == "primary" and int(r["step"]) == step}
    out = {}
    for pos, tgt in TARGET.items():
        cells = []
        for d in DEPTHS:
            t, o = at[(pos, d, tgt)], at[(pos, d, OTHER[tgt])]
            ct, co = V.clears_band(t), V.clears_band(o)
            cells.append(dict(depth=d, target=tgt, b_t=float(t["beta"]), clr_t=ct,
                              ci_t=(float(t["cluster_ci_lo"]), float(t["cluster_ci_hi"])),
                              q_t=float(t["null_q975_abs"]),
                              b_o=float(o["beta"]), clr_o=co, holds=ct and not co))
        out[pos] = cells
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    guard = _load("_run_guard.py", "run_guard")
    guard.require_clean_tree(os.path.basename(__file__),
                             extra_deps=("code/experiments/76_e8_verdict.py",),
                             allow_dirty=a.allow_dirty)

    rows = list(csv.DictReader(open(os.path.join(RSA_DIR, f"checkpoint_betas_{MODEL}.csv"))))
    post = list(csv.DictReader(open(os.path.join(RSA_DIR,
                                                 f"checkpoint_betas_{MODEL}_poststages.csv"))))
    shp_path = os.path.join(RSA_DIR, f"checkpoint_shape_fits_{MODEL}.csv")
    shapes = list(csv.DictReader(open(shp_path))) if os.path.exists(shp_path) else []
    steps = sorted({int(r["step"]) for r in rows})
    final = steps[-1]
    tab = pattern_table(rows, final)
    n_hold = {pos: sum(c["holds"] for c in cells) for pos, cells in tab.items()}
    rep = {pos: n >= MIN_DEPTHS for pos, n in n_hold.items()}
    verdict = ("REPLICATES" if all(rep.values()) else
               "PARTIAL" if any(rep.values()) else "DOES NOT REPLICATE")

    # descriptive: first stage-1 revision from which the target clears at 0.8 and stays clear
    emerg = {}
    for pos, tgt in TARGET.items():
        s = [r for r in rows if r["spec"] == "primary" and r["position"] == pos
             and r["rel_depth_requested"] == "0.8" and r["predictor"] == tgt]
        emerg[pos] = V.emergence_step(s)

    L = [f"# E8g — OLMo-2-7B: does the Pythia dissociation replicate?", "",
         f"**{verdict}** under the rule in `E8_PRESPEC.md` amendment 3.1, fixed before any "
         f"OLMo β was computed: read at the last stage-1 revision (step {final:,}), the "
         f"pattern holds at **{n_hold['belief_last']} of 5** depths at `belief_last` "
         f"(intent clears, outcome does not) and **{n_hold['last']} of 5** at `last` "
         f"(outcome clears, intent does not); ≥ {MIN_DEPTHS} of 5 needed per position. "
         f"Pythia-1.4B under the same rule: 5 of 5 and 4 of 5.", "",
         f"![joined trajectory](checkpoint_trajectory_{MODEL}_joined.png)", "",
         f"Eight stage-1 revisions ({steps[0]:,} … {final:,}); cluster-bootstrap CIs; "
         f"1000 permutations per cell. **No step-0 gate**: OLMo-2 publishes no random-init "
         f"checkpoint, so nothing here was checked against an untrained network of this "
         f"family; Pythia's passing gate covers only the RDM construction and predictors the "
         f"two share.", "",
         "## The rule, cell by cell (primary spec, last stage-1 revision)", "",
         "| position | depth | target | β target | cluster 95% CI | null q97.5 |β| | target clears | "
         "β other | other clears | pattern holds |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for pos in ("belief_last", "last"):
        for c in tab[pos]:
            L.append(f"| `{pos}` | {c['depth']} | {c['target']} | {c['b_t']:+.3f} | "
                     f"[{c['ci_t'][0]:+.3f}, {c['ci_t'][1]:+.3f}] | {c['q_t']:.3f} | "
                     f"{'yes' if c['clr_t'] else 'no'} | {c['b_o']:+.3f} | "
                     f"{'yes' if c['clr_o'] else 'no'} | **{'yes' if c['holds'] else 'no'}** |")

    L += ["", "## Across training (descriptive)", "",
          "At depth 0.8, the first stage-1 revision from which the target factor clears its "
          "band and stays clear through step 928,646:", ""]
    for pos, tgt in TARGET.items():
        e = emerg[pos]
        L.append(f"- `{pos}`, {tgt}: " + (f"step {e:,}" if e is not None else
                                           "never stays clear"))
    L += ["", f"The earliest OLMo revision is step {steps[0]:,}. Pythia's outcome structure "
          "at `last` was already moving off zero by step 512, so an OLMo series that is "
          "clear from its first point says the structure is present early, not when it "
          "starts. Step counts are also not comparable across families at face value: "
          "OLMo-2's stage-1 batch is ~4M tokens per step and Pythia's ~2M.", ""]

    if shapes:
        L += ["Shape fits at n = 8 (amendment 3: reported, not scored):", "",
              "| position | depth | predictor | ΔAICc (piecewise − sigmoid) | R² sigmoid | "
              "75's label |", "|---|---|---|---|---|---|"]
        for s in shapes:
            if s["spec"] != "primary" or s["rel_depth_requested"] != "0.8":
                continue
            L.append(f"| `{s['position']}` | 0.8 | {s['predictor']} | {s['delta_aicc']} | "
                     f"{s['r2_sigmoid']} | {s['verdict']} |")
        L.append("")

    L += ["## Post-training stages (descriptive; same regression, same depths)", "",
          "| stage | position | depth | β intent | β outcome |", "|---|---|---|---|---|"]
    pi = {(r["stage"], r["position"], str(float(r["rel_depth_requested"])), r["predictor"]): r
          for r in post if r["spec"] == "primary"}
    for st in ("base", "SFT", "DPO", "Instruct"):
        for pos in ("belief_last", "last"):
            bi, bo = pi[(st, pos, "0.8", "intent")], pi[(st, pos, "0.8", "outcome")]
            L.append(f"| {st} | `{pos}` | 0.8 | {float(bi['beta']):+.3f} | "
                     f"{float(bo['beta']):+.3f} |")
    L += ["", "Two seams on the joined axis: the stage activations were extracted at "
          "float16 and the stage-1 revisions at float32; and `base` includes stage-2 "
          "annealing, which E8g did not sample, so the last stage-1 point and `base` are "
          "different weights joined by a dotted line.", "",
          "## Limits", "",
          "- Eight revisions, by scope decision (amendment 3). The full ~20-revision sweep "
          "would sharpen the timing, not the verdict, which is read at one revision.",
          "- No step-0 gate for this family.",
          "- The behavioural panel shows post-training stages only, from "
          "`checkpoint_dissection.csv`; OLMo stage-1 revisions were not behaviourally "
          "scored, and OLMo base itself is below the engagement floor.", ""]
    p = os.path.join(RSA_DIR, "E8G_OLMO_REPLICATION.md")
    open(p, "w").write("\n".join(L))
    print(f"verdict: {verdict}  belief_last {n_hold['belief_last']}/5, last {n_hold['last']}/5")
    print(f"-> {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
