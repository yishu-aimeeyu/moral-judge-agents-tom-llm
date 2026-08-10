# E2 — checkpoint-stage RSA: does post-training reshape the geometry or only the readout?

Written before any stage-level RDM existed. Recorded verbatim by
`code/experiments/62_checkpoint_stage_rsa.py` to `outputs/rsa/E2_PRESPEC.md` on the first
`--run`, and never overwritten.

**Why this exists.** `checkpoint_dissection.csv` shows the behavioural contrast collapsing
along the tuning pipeline (OLMo base +0.002 → Instruct −0.687, a 0.689 swing; Tulu base
+0.007 → RLVR −0.398). `base_vs_instruct_geometry.csv` shows the *endpoints* of those
pipelines have nearly identical geometry (OLMo RSA 0.914; Qwen 0.880–0.979). Put together
those two facts suggest post-training changes what the model *does with* its representation
rather than the representation itself — but the geometry comparison has only ever been run
base-vs-final, so it cannot say whether the geometry was stable *throughout* or was reshaped
and then restored. E2 extends the comparison to every stage, which is what makes the
"readout, not representation" claim testable rather than suggestive.

## Scope, fixed here

Two families with published intermediate checkpoints and a non-degenerate base, taken
verbatim from `FAMILIES` in `code/experiments/16_checkpoint_dissection.py`:

| family | stages | base→final Δcontrast |
|---|---|---|
| OLMo-2-7B | `allenai/OLMo-2-1124-7B` → `-SFT` → `-DPO` → `-Instruct` | −0.689 |
| Tulu-3-8B | `unsloth/Meta-Llama-3.1-8B` → `Llama-3.1-Tulu-3-8B-SFT` → `-DPO` → `Llama-3.1-Tulu-3-8B` | −0.405 |

Zephyr-7B is **excluded from the primary** and run only if the two primary families complete:
its base is the weakest-engaged of the three (`rating_std` 0.0285) and its SFT share (27%) is
the outlier, so including it would mix the cleanest test with the noisiest case. Stated now.

Activations are extracted with `code/01_extract_activations.py` (unchanged, `--models` per
checkpoint, with `--clause-offsets` so `belief_last` is available), and RDMs are built with
`build_rdm()` from `24_rsa_cka.py` — same 1 − Pearson metric, same 298 stimuli, same `mean`
pooling. Nothing about the RDM construction changes between E2 and the existing RSA outputs,
so the numbers are directly comparable to `base_vs_instruct_geometry.csv`.

## The pre-specified predictions

```
=========================
Written before any stage-level RDM existed. Recorded verbatim to outputs/rsa/E2_PRESPEC.md
on first --run, and never overwritten.

  P1 (readout, not representation). Every consecutive-stage RDM similarity, and the
      base-to-final RDM similarity, is HIGH while behaviour moves a lot. Bars fixed now:
      RSA Spearman >= 0.90 at matched relative depth 0.75 and at the peak-intent layer, for
      every consecutive stage pair in both families, while |Δcontrast| from
      checkpoint_dissection.csv is >= 0.15 across that same pair or pipeline. 0.90 is chosen
      because it is the value the existing base-vs-final comparison already reports for OLMo
      (0.9141), so P1 asks the intermediate stages to be no less similar than the endpoints
      already are.
  P2 (localisation mismatch). If the geometry DOES move, the stage where it moves most is
      NOT the stage where behaviour moves most. Operationalised as: Spearman rank
      correlation, across the stage transitions of both families pooled (6 transitions),
      between per-transition RDM shift (1 - RSA) and per-transition |Δcontrast|, is NOT
      positive at rho >= +0.5. A positive coupling at or above +0.5 supports the opposite
      reading -- that behavioural change is carried by geometric change -- and P1 is
      withdrawn in favour of it.
  P3 (bounded interpretation). Every cross-stage RSA value is reported against two
      pre-computed reference points measured in the same run, and no claim is made from a
      raw RSA number alone:
        CEILING  = split-half reliability of the RDM within a single checkpoint, built from
                   two disjoint random halves of the hidden units. This is how similar an
                   RDM can be to itself given finite width and 298 stimuli.
        FLOOR    = cross-family RSA at matched depth (OLMo stage vs Tulu stage), i.e. how
                   similar two unrelated 7-8B models already look on these stimuli.
      P1's 0.90 bar is only meaningful if FLOOR < 0.90 <= CEILING. If the floor is itself
      >= 0.90, RSA cannot discriminate on this stimulus set and E2 reports that instead of a
      verdict.
  P4 (the hypothesis RDM check, per stage). Alongside raw similarity, the intent and outcome
      hypothesis-RDM fits from 24_rsa_cka.py R4 are recomputed per stage. Under the readout
      reading these should be roughly FLAT across stages while b_outcome/b_intent grows
      3.0-7.7x behaviourally. Bar for "flat": the change in scenario-partialled
      partial_r_outcome from base to final is smaller in absolute value than 0.10, and the
      intent_org_ratio does not move by more than 0.10. A large swing here is geometric
      reorganisation of exactly the factors under study, and it falsifies P1 more directly
      than the raw RSA number does.

  FALSIFICATION. P1 fails if any consecutive-stage RSA drops below 0.90 at both reported
  depths, and it fails informatively: a drop concentrated at the SFT step in OLMo, the step
  that carries 85% of the behavioural shift, would mean post-training DOES reshape the
  geometry and the "readout only" claim must be withdrawn from the paper rather than
  qualified.

  NON-TRIVIALITY. High RSA between stages is not automatic. Weights change by a full
  optimisation stage between these checkpoints, and RSA here is computed on rank-ordered
  pairwise dissimilarities over 44,253 stimulus pairs, which is not preserved by generic
  weight drift. The floor in P3 is what makes this concrete: if unrelated models already sit
  at 0.90, the measure is uninformative and nothing is claimed.
========================================================================================
```

## Reporting rules, fixed here

- Depths: matched relative depth 0.75 (primary, matches the `reldepth0.75` mode already in
  `model_similarity.csv`) and each checkpoint's own peak-intent layer (secondary). Both are
  reported for every pair; neither is selected after the fact.
- The peak-intent layer is re-derived per checkpoint from its own probe CSV, not inherited
  from the base, so a stage that moves its peak is visible rather than silently re-indexed.
- Behavioural Δcontrast is read from `outputs/experiments/checkpoint_dissection.csv` and is
  **not** recomputed here, so E2 cannot change a behavioural number.
- Where a checkpoint's probe CSV does not exist yet, the peak-layer row is written as empty
  rather than back-filled from the family's other stages.

## Outputs

| path | contents |
|---|---|
| `outputs/rsa/checkpoint_stage_geometry.csv` | family × transition: `rsa_spearman`, `cka_linear`, `cka_rbf` at both depths, `delta_contrast` from checkpoint dissection, `rdm_shift = 1 − rsa` |
| `outputs/rsa/checkpoint_stage_reference.csv` | split-half ceiling per checkpoint, cross-family floor per depth |
| `outputs/rsa/checkpoint_stage_hypothesis.csv` | per-stage `partial_r_intent`, `partial_r_outcome`, `intent_org_ratio` (P4) |
| `outputs/rsa/checkpoint_stage_rsa.png` | per family: RSA across stages on one axis, behavioural contrast on the twin axis, ceiling and floor bands shaded |
| `outputs/rsa/E2_CHECKPOINT_STAGE_RSA.md` | verdict against P1–P4 |

`base_vs_instruct_geometry.csv` is **extended, not rewritten**: E2 writes new files and the
existing endpoint CSV is left byte-identical so the earlier numbers stay auditable.

Raw activations for the intermediate checkpoints stay in `outputs/acts/` and stay gitignored;
only the CSVs, the figure, and the verdict MD are tracked. Nothing under
`outputs/figures_final/_pending_rescore/` is read or written. No new file is named
`rsa_convergence_*`. No figure is copied into `presentation_figures/` until the P1–P4 verdict
is written.
