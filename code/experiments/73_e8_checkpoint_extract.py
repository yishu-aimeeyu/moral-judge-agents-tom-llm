#!/usr/bin/env python3
"""E8b -- walk a pretraining checkpoint trajectory and extract activations at each step. [GPU]

This is a driver, not a second extractor. It loops the pre-registered revision list and calls
`extract_for_model()` from `code/01_extract_activations.py` with `--revision`, so the
checkpoint activations are produced by exactly the code path that produced every other
activation file in the project. Forking the extractor would have made the step-0 gate a test
of a new script rather than of the pipeline.

THREE THINGS THIS HANDLES THAT A SHELL LOOP WOULD NOT.

  Resumability. 26 revisions of a 1.4B model is a multi-hour job and the partition that has
  the time limit for it is preemptable. Every revision whose .npz already exists is skipped,
  so a resubmit continues rather than restarting.

  Disk. Each revision's weights are evicted from the HF cache as soon as its activations are
  written. Pythia-1.4B is 2.93 GB per revision and 26 of them is ~76 GB against 124 GB of
  home headroom; OLMo-2-7B is ~14 GB per revision and ~280 GB in total. Without eviction the
  sweep does not fit. See E8_PRESPEC.md section 9.

  dtype. float32 by default, against the project's usual float16. The step-0 gate is a
  claim about a random-init network being indistinguishable from its own null, and it is not
  worth defending that claim against the objection that it was measured in half precision.

Revisions, layers and positions are all fixed in outputs/experiments/E8_PRESPEC.md section 5
and are duplicated here as constants rather than passed as flags, so that a sweep cannot
quietly be run over a different set than the one pre-registered.

Outputs -> outputs/acts_ckpt/<model>@<revision>.npz  (gitignored)
           outputs/acts_ckpt/extraction_log.csv
"""
import argparse
import csv
import importlib.util
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ACTS_DIR = os.path.join(ROOT, "outputs", "acts_ckpt")

# E8_PRESPEC.md section 5. Log-spaced through the first 512 steps, where Pythia's dynamics
# are fastest, then thinning out across the remaining 142k.
PYTHIA_REVISIONS = [
    "step0", "step1", "step2", "step4", "step8", "step16", "step32", "step64",
    "step128", "step256", "step512",
    "step1000", "step2000", "step3000", "step4000", "step8000", "step16000", "step24000",
    "step32000", "step48000", "step64000", "step80000", "step96000", "step112000",
    "step128000", "step143000",
]

# E8g. Eight log-spaced stage-1 revisions of the 928 available, chosen by nearest step to a
# logspace grid over the full pretraining range (150 .. 928,646).
#
# NOTE, and it is a real limitation rather than a detail: OLMo-2 publishes no random-init
# checkpoint, so **the E8c step-0 gate cannot be re-run on this family**. The gate is a test
# of the RDM construction, and the construction is shared, so Pythia's passing gate covers
# the pipeline -- but it does not cover anything OLMo-specific, such as its tokenizer's
# treatment of the manipulated clause. The earliest available revision is step 150, which is
# already past the point where Pythia's outcome structure begins to move.
OLMO_REVISIONS = [
    # amendment 3 (first eight) + 3.2 (nearest published branch to 22 log-spaced targets)
    "stage1-step150-tokens1B",
    "stage1-step600-tokens3B",
    "stage1-step850-tokens4B",
    "stage1-step1000-tokens5B",
    "stage1-step2000-tokens9B",
    "stage1-step3000-tokens13B",
    "stage1-step4000-tokens17B",
    "stage1-step6000-tokens26B",
    "stage1-step10000-tokens42B",
    "stage1-step15000-tokens63B",
    "stage1-step22000-tokens93B",
    "stage1-step33000-tokens139B",
    "stage1-step51000-tokens214B",
    "stage1-step77000-tokens323B",
    "stage1-step116000-tokens487B",
    "stage1-step176000-tokens739B",
    "stage1-step267000-tokens1120B",
    "stage1-step404000-tokens1695B",
    "stage1-step613000-tokens2572B",
    "stage1-step928646-tokens3896B",
]
LAYERS_REL = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SUITES = {
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", PYTHIA_REVISIONS),
    "olmo2-7b": ("allenai/OLMo-2-1124-7B", OLMO_REVISIONS),
}


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def revision_step(rev):
    """Training step as an integer, for the x-axis.

    `step1000` -> 1000, and `stage1-step22000-tokens93B` -> 22000. OLMo's revision names
    carry three numbers, so a naive digit-scrape would return 12200093 for that one.
    """
    m = re.search(r"step(\d+)", rev)
    return int(m.group(1)) if m else -1


def append_log(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="pythia-1.4b", choices=sorted(SUITES))
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master",
                                                  "moral_2x2_master.csv"))
    ap.add_argument("--clause-offsets",
                    default=os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    ap.add_argument("--out", default=ACTS_DIR)
    ap.add_argument("--dtype", default="float32",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--revisions", nargs="*", default=None,
                    help="override the pre-registered list (for a single-revision smoke test; "
                         "a result run uses the full list)")
    ap.add_argument("--no-evict", action="store_true",
                    help="keep weights cached; only safe for a one-revision test")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()

    model_name, revisions = SUITES[a.suite]
    if a.revisions:
        revisions = a.revisions

    ex = _load(os.path.join(ROOT, "code", "01_extract_activations.py"), "extract_01")
    rows = ex.load_stimuli(a.csv)
    clause = ex.load_clause_offsets(a.clause_offsets)

    todo, done = [], []
    for rev in revisions:
        p = os.path.join(a.out, f"{ex.model_tag(model_name, rev)}.npz")
        (done if os.path.exists(p) else todo).append(rev)

    if a.dry_run:
        print("=== E8b CHECKPOINT EXTRACTION PLAN (dry-run) ===")
        print(f"model      : {model_name}")
        print(f"revisions  : {len(revisions)}  "
              f"({revisions[0]} .. {revisions[-1]})")
        print(f"already on disk: {len(done)}   to extract: {len(todo)}")
        print(f"items      : {len(rows)}   clause offsets: {len(clause)}")
        print(f"layers-rel : {LAYERS_REL}   positions: last, mean, belief_last, action_last")
        print(f"dtype      : {a.dtype}   evict after each: {not a.no_evict}")
        print(f"out        : {a.out}")
        print(f"\nLaunch:\n  JOBNAME=e8b PART=mit_preemptable TIME=1-00:00:00 "
              f"bash engaging/submit_gpu.sh \\\n"
              f"    \"python code/experiments/73_e8_checkpoint_extract.py --suite {a.suite}\"")
        return

    guard = _load(os.path.join(HERE, "_run_guard.py"), "run_guard")
    guard.require_clean_tree(os.path.basename(__file__),
                             extra_deps=("code/01_extract_activations.py",),
                             allow_dirty=a.allow_dirty)

    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "extraction_log.csv")
    print(f"\n=== E8b: {model_name}, {len(todo)} revision(s) to extract "
          f"({len(done)} already on disk) ===", flush=True)

    for i, rev in enumerate(todo, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(todo)}] {model_name}@{rev}", flush=True)
        ex.extract_for_model(model_name, rows, a.out, batch_size=a.batch_size,
                             clause_offsets=clause, revision=rev,
                             layers_rel=LAYERS_REL, dtype=a.dtype,
                             evict=not a.no_evict)
        dt = time.time() - t0
        p = os.path.join(a.out, f"{ex.model_tag(model_name, rev)}.npz")
        append_log(log, {"model": model_name, "revision": rev,
                         "step": revision_step(rev),
                         "seconds": round(dt, 1),
                         "mb": round(os.path.getsize(p) / 1e6, 1),
                         "dtype": a.dtype,
                         "layers_rel": " ".join(str(x) for x in LAYERS_REL)})
        print(f"  done in {dt:.0f}s", flush=True)

    print(f"\n-> {a.out}\n-> {os.path.relpath(log, ROOT)}")


if __name__ == "__main__":
    main()
