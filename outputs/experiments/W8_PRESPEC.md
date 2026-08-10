# W8 (E3) — amnesic outcome ablation: pre-specified predictions

Written before any erasure run. Recorded verbatim by
`code/experiments/63_w8_amnesic_outcome.py` to `outputs/experiments/W8_PRESPEC.md` on the
first `--run`, and never overwritten, so results cannot be retrofitted to the hypothesis.

**Why this exists, and why it is not W3 again.** W3 *added* intent signal and the judgment
did not move (probe-weight intent direction: |Δcontrast| ≤ 0.015 against an outcome control
of 0.232–0.259, `W3_STEERING_SUMMARY.md`). That rules in "the readable intent direction is
not causally wired", but it cannot rule out "intent is wired and actively overridden by a
dominant outcome signal" — adding to a suppressed channel need not show up if the suppressor
is still running. W8 is the mirror intervention: *remove* the outcome signal and see whether
intent-based judgment appears. It is the one manipulation that distinguishes
latent-but-suppressed from never-wired.

**Structure is deliberately parallel to W3**: same models, same layers, same coherence
gating, same manipulation-check discipline, same sensitivity-in-control-units reporting. Any
asymmetry in how the two are analysed would make the comparison uninterpretable.

## Scope, fixed here

| model | erasure layer | source of that layer | unsteered contrast (W3) |
|---|---|---|---|
| `allenai/OLMo-2-1124-7B-Instruct` | L16 | peak-intent layer used by W3 (`W3_STEERING_OLMo-2-1124-7B-Instruct.md`) | −0.4452 |
| `Qwen2.5-7B-Instruct` | L19 | peak-intent layer used by W3 | −0.2667 |

The layer is inherited from W3 rather than re-selected, so W8 cannot gain an effect by
choosing a friendlier depth. A secondary sweep at the two next-deeper sampled depths is run
only after the primary verdict is written, and is labelled exploratory.

**Method.** LEACE (concept-erasure, affine guardedness) applied to the residual stream at the
erasure layer, at every token position, on every forward pass. If `concept_erasure` is not
importable in the run environment, the fallback is iterative mean-projection (repeated
removal of the class mean-difference direction with re-fitting, INLP-style) to the same
acceptance gates. Which method ran is recorded in every output row; the gates below are
method-agnostic and are what licence the interpretation, not the choice of eraser.

**Scoring** is the fixed logprob-EV digit scoring from `03_behavioral.HFBackend`, pooled
within `scenario_group`, exactly as W3 — so a W8 contrast and a W3 contrast are the same
quantity.

## The pre-specified predictions

```
=========================
Written before any erasure run. Recorded verbatim to outputs/experiments/W8_PRESPEC.md on
first --run, and never overwritten.

  V-GATES (must all pass before ANY behavioural number is interpreted).
    V1 (erasure worked). Outcome decoding at the erasure layer falls to chance, defined in
        advance as within 0.05 of the majority-class rate for this dataset (154 harm / 144
        no_harm -> chance = 0.5168), using probes fitted on UNERASED activations under the
        existing grouped-CV protocol.
    V2 (collateral damage bounded). Intent decoding at the same layer retains at least 90%
        of its above-chance accuracy: (acc_erased - 0.50) / (acc_unerased - 0.50) >= 0.90,
        with intent chance = 0.50 (149/149). If erasing outcome also destroys intent
        decodability, the manipulation is not selective and no conclusion about intent-use
        can be drawn from it.
    V3 (coherence). The four W3 bounds, unchanged and for the same reasons: perplexity ratio
        <= 1.5x baseline, refusal-rate increase <= 0.10 absolute, degenerate-generation
        fraction <= 0.10, answer-rate drop <= 0.10.
    V4 (instrument check, known answer). Layers BELOW the erasure site must read exactly
        unchanged. Any deviation means the hook is mis-wired and the run is void.

  P1 (recovery, the primary). With V1-V4 passed, the contrast (attempted - accidental) moves
      POSITIVE by at least the J4 effect floor of 0.0988 (floor_derivation.csv, q95 of the
      within-scenario-group permutation null), AND the post-erasure b_intent is not lower
      than the pre-erasure b_intent by more than that same 0.0988. Both halves are required.
      The second half is what makes P1 a test of intent-use rather than of signal removal;
      see P2.
  P2 (collapse discriminator -- the confound that would otherwise fake P1). The headline
      contrast is a DIAGONAL: attempted differs from accidental on BOTH factors (intent
      present/no harm vs intent absent/harm). Erasing outcome therefore moves the contrast
      toward zero mechanically, by roughly +0.45 in OLMo and +0.27 in Qwen, even if intent
      is used exactly as little as before. Raw movement past 0.0988 is consequently NOT
      evidence of recovery, and this prediction is registered so that it cannot be reported
      as if it were. The three outcomes are defined now:
        RECOVERY  : accidental falls, attempted does not fall, b_intent preserved per P1,
                    and the spread of the four cell means (SD across neutral / accidental /
                    attempted / intentional) retains at least 50% of its pre-erasure value.
                    Reading: intent was latent and suppressed by the outcome signal.
        COLLAPSE  : all four cell means converge -- cell-mean SD falls by more than 50% --
                    and b_intent falls toward zero. Contrast rises toward 0 with no
                    ordering. Reading: outcome removal, not intent recovery; the judgment
                    had nothing else to run on, which is itself the "never wired" result.
        NO CHANGE : |Δcontrast| < 0.0988. Reading: the judgment does not depend on the
                    erased outcome direction at this depth.
      A result is reported under exactly one of these three labels, chosen by the numbers
      above and not by inspection.
  P3 (specificity). Erasing the INTENT direction instead, to the same gates, moves the
      contrast LESS than erasing outcome does. Under "intent is unused" the intent-erasure
      arm should be near-inert; a large intent-erasure effect would mean intent is in fact
      load-bearing and the whole "represented but unused" reading is wrong.
  P4 (null control and dose). A rank-matched RANDOM-direction erasure is run at every dose
      and must not reproduce the outcome-erasure effect. Dose is the erasure fraction
      gamma in {0.25, 0.50, 0.75, 1.00} interpolating from unerased to fully erased; under
      P1 the contrast movement is monotone in gamma. A non-monotone effect that appears only
      at gamma = 1.00 is reported as damage, not dose-response.

  FALSIFICATION AND SYMMETRY. P1 fails if the movement does not clear 0.0988, or if it
  clears it under the COLLAPSE signature rather than the RECOVERY signature. Both failures
  are reportable and neither weakens the paper: RECOVERY says intent is present and
  suppressed, COLLAPSE and NO CHANGE say the judgment was never wired to intent. What is NOT
  reportable is a raw contrast movement quoted without the P2 label attached.

  SENSITIVITY IN CONTROL UNITS (mirrors W3 M3). The summary must state the largest
  gate-passing |Δcontrast| for the outcome erasure, for the intent erasure, and for the
  random erasure, plus the outcome/random ratio, so that a null carries an explicit
  detectable-effect-size statement instead of leaving "underpowered" open.

  NON-TRIVIALITY. All four cell means are recorded at every dose for every arm, because a
  uniform shift in all four cells leaves their DIFFERENCE unchanged and any contrast change
  accompanied by all four cells moving together is compression against the scale ends rather
  than a change in intent-weighting. This is the same caveat W3 carries and it applies here
  with the sign reversed.
========================================================================================
```

## Calibration, fixed here

- **Calibration pass first**, before any behavioural readout, exactly as W3: for each model
  and each arm, the erasure is applied at increasing `gamma` and V1–V3 are measured. The
  gate-passing `gamma` range is written to disk *before* contrasts are read out of it, so the
  coherent band cannot be redrawn around a result.
- **Probes are never fitted on erased activations.** V1/V2 use probes fitted on unerased
  activations under grouped CV (`scenario_group` folds), scoring erased activations
  out-of-fold, matching the W3 manipulation-check protocol.
- **Erasure is fitted on train folds only** for the reported V1/V2 numbers, so "outcome
  decoding is at chance" is not an artifact of erasing using the labels it is then tested on.
  The behavioural arms use the full-data eraser; both are recorded.
- Baseline contrast is re-measured in the same session as an unerased arm rather than copied
  from W3, and it must reproduce the W3 value within 0.01 or the run is flagged.

## Outputs

| path | contents |
|---|---|
| `outputs/experiments/w8_amnesic_<tag>.csv` | arm × gamma: contrast, Δcontrast, four cell means, cell-mean SD, b_intent, b_outcome, coherence fields, gate flags, `p2_label` |
| `outputs/experiments/w8_amnesic_verification_<tag>.csv` | V1/V2 per layer/target: acc unerased → erased, above-chance retention, margin shift in SD; V4 upstream rows |
| `outputs/experiments/w8_amnesic_calibration_<tag>.csv` | gate-passing gamma band, written before behavioural readout |
| `outputs/experiments/w8_amnesic_<tag>.png` | dose-response with gate-passing band shaded, four cell means panel |
| `outputs/experiments/w8_generations_<tag>.txt` | ~20 generations per dose for the manual read required by V3 |
| `outputs/experiments/W8_AMNESIC.md` | cross-model verdict against V1–V4 and P1–P4, with the P2 label stated in the headline |

W3 outputs are not modified. Nothing under `outputs/figures_final/_pending_rescore/` is read
or written. Activations stay gitignored. No W8 figure is copied into `presentation_figures/`
until `W8_AMNESIC.md` states the verdict against these thresholds.
