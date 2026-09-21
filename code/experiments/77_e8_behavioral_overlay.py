#!/usr/bin/env python3
"""E8f -- behavioural 2x2 scoring along the same checkpoint trajectory. [GPU]

SECONDARY PANEL. This cannot falsify P1-P4 and is not allowed to. The two series are not
the same series and the figure does not pretend otherwise:

  representational beta   exists from step 0 -- it needs only activations, and a randomly
                          initialised network has activations
  behavioural b_intent    exists only from the first checkpoint that produces usable
                          ratings -- an untrained network cannot rate anything

So `b_intent` and `b_outcome` are plotted from the **first gradeable checkpoint** onward,
that checkpoint is marked, and the region before it is left blank. Not interpolated, not
extrapolated, not drawn at zero. Conflating the two would imply judgment was measured in an
untrained network, which it was not.

Scoring reuses `score_checkpoint()` from `16_checkpoint_dissection.py`, which is the same
logprob-EV path that produced `checkpoint_dissection.csv` for the post-training stages. That
matters for E8g: the pretraining trajectory and the post-training stages have to be on one
axis, and they can only be joined if they were scored the same way.

A NOTE ON "PARSEABLE", because the prespec's wording assumes a scoring mode this does not
use. E8_PRESPEC.md section 7 makes a checkpoint gradeable on (1) >=80% parseable ratings and
(2) rating_std >= 0.2191. Under logprob-EV scoring there is nothing to parse: the rating is
an expectation over the digit-token distribution, so criterion (1) is satisfied by
construction at every checkpoint including a random one. Only criterion (2) binds, and the
`degenerate` flag is recorded beside it. This is disclosed rather than quietly reinterpreted.

Outputs -> outputs/rsa/checkpoint_behavioral.csv
"""
import argparse
import csv
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(ROOT, "code")
RSA_DIR = os.path.join(ROOT, "outputs", "rsa")

ENGAGEMENT_FLOOR = 0.2191      # outputs/stats/FLOOR_DERIVATION.md
PARSE_FRAC_FLOOR = 0.80        # E8_PRESPEC.md section 7; vacuous under logprob-EV, see above


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="pythia-1.4b")
    ap.add_argument("--out", default=os.path.join(RSA_DIR, "checkpoint_behavioral.csv"))
    # human_verbatim is 16_checkpoint_dissection.py's own default, and therefore what
    # checkpoint_dissection.csv was scored with. E8g joins the pretraining trajectory to
    # those post-training stages on one axis, which is only legitimate if both were scored
    # under the same template.
    ap.add_argument("--templates", nargs="*", default=["human_verbatim"])
    ap.add_argument("--revisions", nargs="*", default=None)
    ap.add_argument("--no-evict", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, CODE)
    ext = _load(os.path.join(HERE, "73_e8_checkpoint_extract.py"), "extract_73")
    dis = _load(os.path.join(HERE, "16_checkpoint_dissection.py"), "dissect_16")
    beh = _load(os.path.join(CODE, "03_behavioral.py"), "beh_03")
    reg11 = _load(os.path.join(CODE, "11_interaction_regression.py"), "reg_11")
    acts01 = _load(os.path.join(CODE, "01_extract_activations.py"), "extract_01")

    model_name, revisions = ext.SUITES[a.suite]
    if a.revisions:
        revisions = a.revisions
    templates = list(a.templates)
    rows = dis.load_dataset()

    done = {}
    if os.path.exists(a.out):
        done = {r["revision"]: r for r in csv.DictReader(open(a.out))}
    todo = [r for r in revisions if r not in done]

    if a.dry_run:
        print("=== E8f BEHAVIOURAL OVERLAY PLAN (dry-run) ===")
        print(f"model      : {model_name}")
        print(f"revisions  : {len(revisions)}  ({len(done)} already scored, {len(todo)} to do)")
        print(f"items      : {len(rows)}   templates: {templates}")
        print(f"scoring    : logprob-EV via 16_checkpoint_dissection.score_checkpoint")
        print(f"gradeable  : rating_std >= {ENGAGEMENT_FLOOR} "
              f"(parse-rate criterion is vacuous under logprob-EV; see the docstring)")
        print(f"out        : {a.out}")
        print(f"\nLaunch:\n  JOBNAME=e8f PART=mit_preemptable TIME=1-00:00:00 "
              f"bash engaging/submit_gpu.sh \\\n"
              f"    \"python code/experiments/77_e8_behavioral_overlay.py --suite {a.suite}\"")
        return

    guard = _load(os.path.join(HERE, "_run_guard.py"), "run_guard")
    guard.require_clean_tree(os.path.basename(__file__),
                             extra_deps=("code/03_behavioral.py",
                                         "code/experiments/16_checkpoint_dissection.py"),
                             allow_dirty=a.allow_dirty)

    # score_checkpoint() builds its own HFBackend from a model id, so the revision is passed
    # by binding it onto the constructor rather than by copying the scoring loop here.
    real_backend = beh.HFBackend
    current = {"revision": None}

    class RevisionBackend(real_backend):
        def __init__(self, model_id, scoring, **kw):
            super().__init__(model_id, scoring, revision=current["revision"], **kw)

    beh.HFBackend = RevisionBackend

    out_rows = list(done.values())
    print(f"=== E8f: {len(todo)} revision(s) to score ===", flush=True)
    for i, rev in enumerate(todo, 1):
        current["revision"] = rev
        step = ext.revision_step(rev)
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {model_name}@{rev} (step {step})", flush=True)
        met = dis.score_checkpoint(beh, reg11, model_name, rows, templates)
        gradeable = bool(met["rating_std"] >= ENGAGEMENT_FLOOR)
        rec = {"model": model_name, "revision": rev, "step": step,
               "n_items": met["n_items"], "n_scenario_groups": met["n_scenario_groups"],
               "contrast": round(met["contrast"], 5),
               "b_intent": round(met["b_intent"], 5),
               "b_outcome": round(met["b_outcome"], 5),
               "b_interaction": round(met["b_interaction"], 5),
               "rating_std": round(met["rating_std"], 5),
               "degenerate": met["degenerate"],
               "engagement_floor": ENGAGEMENT_FLOOR,
               "gradeable": gradeable,
               "parse_frac": 1.0,
               "parse_criterion": "vacuous under logprob-EV",
               "templates": "|".join(templates),
               "seconds": round(time.time() - t0, 1)}
        out_rows.append(rec)
        out_rows.sort(key=lambda r: int(r["step"]))
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rec.keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"  rating_std={met['rating_std']:.4f} "
              f"{'GRADEABLE' if gradeable else 'below the engagement floor'}  "
              f"b_intent={met['b_intent']:+.4f} b_outcome={met['b_outcome']:+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not a.no_evict:
            acts01.evict_from_cache(model_name, rev)

    grad = [r for r in out_rows if str(r["gradeable"]).lower() in ("true", "1")]
    print(f"\n=== {len(grad)} of {len(out_rows)} checkpoints are gradeable "
          f"(rating_std >= {ENGAGEMENT_FLOOR}) ===")
    if grad:
        first = min(grad, key=lambda r: int(r["step"]))
        print(f"first gradeable checkpoint: {first['revision']} (step {first['step']}), "
              f"rating_std={first['rating_std']}")
    else:
        print("NO checkpoint clears the engagement floor. That is a reportable outcome, "
              "not a failed run: the prespec expected many and possibly all pretraining "
              "checkpoints to fail it, and the behavioural panel is then simply empty.")
    print(f"-> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
