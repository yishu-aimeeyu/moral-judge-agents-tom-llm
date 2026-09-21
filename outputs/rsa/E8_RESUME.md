# E8 — where this stopped, 2026-09-21 ~17:00

Paused mid-run. Two Slurm jobs were still going and they are disconnect-safe (`sbatch`, not
a login shell), so logging out does not kill them.

## Done and committed

| stage | state | artifact |
|---|---|---|
| prespec | committed, tagged `prespec-e8`, + 2 append-only amendments | `outputs/experiments/E8_PRESPEC.md` |
| **E8a** predictors | done | `outputs/rsa/predictors/` |
| **E8b** Pythia extraction | done, 26/26 revisions | `outputs/acts_ckpt/pythia-1.4b@*.npz` (gitignored) |
| **E8c** step-0 gate | **PASS** | `outputs/rsa/E8_STEP0_GATE.md` |
| **E8d** β trajectories | done, 3380 rows | `outputs/rsa/checkpoint_betas.csv` + `.png` |
| **E8e** shape test | done, smooth wins 10/10 | `outputs/rsa/checkpoint_shape_fits.csv` |
| verdict | written | `outputs/rsa/E8_CHECKPOINT_EMERGENCE.md` + `checkpoint_emergence_by_position.png` |

`make check` passes (prespec append-only, prespec tagged, regeneration, headers).

## The result so far, in one line

Outcome structure and intent structure both emerge smoothly around step 1,000–3,000, but at
**different read positions**, and each is invisible at the other's: at the end of the story
β_outcome reaches +0.49 while intent never leaves its null; at the end of the belief clause
β_intent reaches +0.10 while outcome never leaves its null. P1 and P4 were both falsified
and are reported as they fell. Details in `E8_CHECKPOINT_EMERGENCE.md`.

## In flight when this stopped

| job | id | what | progress at pause |
|---|---|---|---|
| `e8f` | 23417911 | Pythia behavioural overlay, 26 revisions | 11/26 |
| `e8g` | 23417374 | OLMo-2-7B activation extraction, 8 revisions | 1/8 |

Both write incrementally and both skip completed work on resubmit, so a preemption is
recoverable by resubmitting the same command.

```
squeue -u $USER
tail -f outputs/logs/gpu_23417911.log     # e8f
tail -f outputs/logs/gpu_23417374.log     # e8g
```

**E8f is heading for an all-blank panel, which the prespec anticipated.** Every Pythia
checkpoint scored so far sits at `rating_std` 0.02–0.08, far below the 0.2191 engagement
floor, and `b_intent` / `b_outcome` are ~0.00 to −0.02. If that holds through step 143,000
there is no first gradeable checkpoint and the behavioural panel is empty — which section 7
pre-registered as a reportable outcome rather than a failed run.

## Next steps, in order

1. **Check the two jobs finished.** `outputs/rsa/checkpoint_behavioral.csv` should have 26
   rows; `outputs/acts_ckpt/` should have 8 `OLMo-2-1124-7B@stage1-*.npz`.
2. **E8f panel + write-up.** Mark the first gradeable checkpoint, or state plainly that
   there is none. Do not interpolate the pre-gradeable region.
3. **E8g analysis.** `74_e8_checkpoint_betas.py` currently globs all of `outputs/acts_ckpt/`
   and requires an `@step0` file, so it needs two changes before it can run on OLMo:
   a `--model` filter, and an explicit flag for "this family publishes no random-init
   checkpoint so the E8c gate cannot be re-run" — recorded, not silently skipped. Output
   filenames need a per-model suffix so Pythia's CSVs are not overwritten.
4. **Join to `checkpoint_dissection.csv`.** The post-training stages were scored with
   `--templates human_verbatim`, and E8f uses the same template for exactly this reason, so
   the pretraining trajectory and base → SFT → DPO → Instruct can share one x-axis.
5. **E8g caveat to carry.** OLMo-2 publishes no random-init checkpoint, and its earliest
   available revision (step 150) is already past where Pythia's outcome structure starts to
   move. So E8g can replicate the *shape* but cannot repeat the step-0 control.

## Open items not part of E8

- `outputs/BEYOND_BEHAVIOR_MASTER.md` still links to `github.com/MisssAimeee/...`; the
  remotes were renamed to `yishu-aimeeyu` but in-document URLs were not swept.
