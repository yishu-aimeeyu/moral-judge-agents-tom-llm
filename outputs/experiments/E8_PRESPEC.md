# E8 — representational emergence across training checkpoints

**Written 2026-09-21, before any checkpoint was downloaded and before any activation was
extracted.** Committed on its own commit and tagged `prespec-e8`. Append-only: amendments go
at the bottom with a date, and nothing above this line is ever rewritten.

E8 runs the RSA machinery along the *training* axis instead of the layer axis (E1) or the
post-training-stage axis (E2). For each pretraining checkpoint it regresses the observed
298×298 RDM on five design predictors and asks whether outcome structure and intent structure
appear at different times and at different strengths.

---

## 0. Why this needs a prespec at all

Two things here are easy to get wrong after the fact, and both are decided below in advance.

**The shape of the curve.** "Did it click, or did it grow?" is a question where the analyst's
freedom to pick a cell, a metric, and a model class determines the answer. Apparent sharp
emergence in LLMs is frequently an artifact of a thresholded measure (Schaeffer et al., *Are
Emergent Abilities of Large Language Models a Mirage?*). β is continuous and unthresholded, so
E8 is better placed than accuracy-based emergence work to answer honestly — but only if the
comparison, the cell, and the "indistinguishable" band are fixed before the curve is seen.

**The step-0 gate.** A gate that the analyst can soften after seeing it fail is not a gate.
The exact rule, the exact percentile, and the consequence of failure are written below.

---

## 1. The pre-specified predictions

```
=========================================================================================
Written before any E8 compute. Never overwritten.

  P0 (STEP-0 GATE, BLOCKING — this is not a prediction, it is a validity check).
      At `step0` (random initialisation) the standardised beta for INTENT and the
      standardised beta for OUTCOME are each inside their own within-scenario permutation
      null, in every (layer x position) cell, at the Bonferroni-corrected two-sided
      threshold defined in section 4. An untrained network cannot have learned the moral
      belief x outcome factors; if it scores on them, the RDM is reading tokenisation,
      sequence length, or some other input-side artifact through the distance metric.
      CONSEQUENCE OF FAILURE: halt. E8d is not run, no trajectory is reported, and the
      failure is written up as a measurement defect. There is no version of E8 in which a
      failed P0 is reported as an early-emergence finding.

  P1 (outcome structure emerges, and emerges earlier than intent structure). Over the
      Pythia-1.4B trajectory, the standardised beta for OUTCOME leaves its step-0 null band
      at an earlier training step than the standardised beta for INTENT does, at the
      primary cell defined in section 5. "Leaves the band" = the bootstrap 95% CI for that
      beta excludes the step-0 null band's upper edge, at that checkpoint and at every
      later checkpoint in the trajectory (a single isolated excursion does not count as
      emergence; the crossing must hold).

  P2 (the gap widens, or at least does not close). The OUTCOME SHARE defined in section 3
      is non-decreasing across the second half of the trajectory (steps >= 1000), tested as
      a positive Spearman rho between log(step) and the share over those checkpoints, with
      a bootstrap 95% CI excluding zero.

  P3 (anything quoted is above its own null). Any beta quoted in support of P1 or P2
      exceeds the 97.5th percentile of its own within-scenario permutation null at that
      checkpoint, layer and position. A beta inside its null is reported as
      indistinguishable from zero and cannot carry a comparison in either direction,
      including a ratio.

  P4 (shape). For beta_outcome(t), the piecewise-linear model with a free changepoint beats
      the sigmoid in log(step) by AICc, and the bootstrap 95% CI on the changepoint location
      spans less than half the fitted log-step range. Both halves are required: a changepoint
      whose CI covers most of the x-axis is an unlocated changepoint and is reported as such,
      not as a breakpoint.

  FALSIFICATION.
    P1 fails if intent leaves its band at the same checkpoint as outcome or earlier, or if
      neither leaves its band anywhere on the trajectory. "Neither emerges" is a real and
      reportable outcome: it would mean 298 items and six layers cannot resolve the factors
      in a 1.4B model, and it would put a power ceiling on the whole E8 design rather than
      on Pythia.
    P2 fails if the Spearman CI includes zero (flat) or lies below zero (gap closing).
    P4 fails toward "smooth" if the sigmoid wins by AICc, and toward "undecided" if
      |dAICc| < 2. A smooth curve is a result and is reported as plainly as a breakpoint
      would be. The write-up does not prefer the more dramatic shape, and the phrase
      "phase transition" is not used unless P4 is supported in both halves.

  NON-TRIVIALITY. A trajectory that only reproduces "later checkpoints have more structure
      of every kind" says nothing specific. The lexical and scenario betas are therefore
      reported on the same axes as intent and outcome. If all five predictors rise together
      in lockstep, the finding is "representations sharpen", not "moral factors emerge", and
      it is written up that way.
=========================================================================================
```

---

## 2. Predictors, fixed here

All five are built over the same 298 items in the same order, from
`dataset/master/moral_2x2_master.csv`, sorted `belief -> outcome -> scenario_id` so the block
structure is legible in the rendered heatmaps.

| predictor | `D[i,j]` | role |
|---|---|---|
| `intent` | `1` if `intent_label` differs, else `0` | factor of interest |
| `outcome` | `1` if `outcome_label` differs, else `0` | factor of interest |
| `interaction` | `abs(I_i*O_i - I_j*O_j)`, the treatment-coded product term | factor of interest |
| `scenario` | `1` if `scenario_group` differs, else `0` | **covariate** |
| `lexical` | cosine distance on `tfidf_word_1_2` | **covariate** |

`lexical` reuses `FEATURE_SETS["tfidf_word_1_2"]` from `21_surface_baseline.py` via
`lexical_rdm()` in `61_rdm_variance_partition.py` — the same vectoriser settings as E1, not a
re-implementation. `scenario` uses `scenario_group` where present and falls back to
`scenario_id`, matching `24_rsa_cka.py`.

### 2.1 Two lessons from E1 and E6a that are binding here

**Lexical is a covariate and is never a gate.** E1's P4 gated on the lexical predictor and
excluded 36 of 36 model×layer cells, making the whole analysis undecidable. The reason is a
property of the stimuli, not of any model: within a scenario group the four conditions differ
by exactly the clause that realises the factor, so a word-overlap predictor necessarily
outranks the binary predictor expressing the same manipulation. E8 keeps `lexical` in the
design and reports its beta, and no E8 decision rule reads it. The same applies to `scenario`.

**Scenario is not an innocent nuisance.** E6a-live measured 0.44–0.66 of the outcome
direction's variance sitting in the scenario subspace under chat templating. Dropping scenario
from the design would let that variance land on `outcome` and inflate it.

**Consequence for the headline quantity.** Because both factors are residualised against the
same covariates, their *absolute* betas are both deflated by the collinearity, but their
*relative* size is not. The primary quantity is therefore relative (section 3), and absolute
betas are reported as supporting detail.

---

## 3. Estimator, and the primary quantity

**Regression.** Lower triangle of the observed RDM (43,053 pairs), rank-transformed then
z-scored; each predictor column likewise rank-transformed then z-scored; OLS with intercept on
all five predictors jointly. Coefficients are therefore standardised betas and are comparable
across checkpoints, layers, positions and models. Rank-then-z matches E1 so E8 betas can be
read beside E1's partition. Variance inflation factors are reported per predictor per cell;
a VIF above 10 is flagged in the CSV and **does not gate** anything.

**Primary quantity: the outcome share.**

```
S = |beta_outcome| / (|beta_outcome| + |beta_intent|)          in [0, 1]
```

The design document asks for the `beta_outcome : beta_intent` ratio as the primary quantity.
`S` *is* that ratio, mapped monotonically onto a bounded scale: `S = R/(1+R)` for
`R = |beta_outcome|/|beta_intent|`. The bounded form is used because the raw ratio is exactly
0/0 at step 0 and near-singular through early training, where `beta_intent` is expected to be
inside its null — a quantity that explodes precisely where the interesting dynamics are cannot
be the headline. `S = 0.5` means the two factors organise the geometry equally; `S -> 1` means
outcome dominates. The raw ratio `R` is reported in the same CSV, blanked in any cell where
`beta_intent` is inside its permutation null, because a ratio with a null denominator is not a
measurement. This substitution is a change of scale, not of threshold, and it is recorded here
before any E8 compute.

Absolute values are also reported, with the caveat above attached, because "outcome dominates"
is uninformative if both betas are 0.01.

**Permutation null.** Stimulus identity permuted *within scenario group* using
`permute_within_groups()` from `24_rsa_cka.py`, the observed RDM rebuilt under the permutation,
and the full five-predictor regression refitted. This is the project-wide convention and is the
conservative choice: it preserves scenario membership and most lexical similarity, so it tests
the 2×2 factors specifically rather than testing "is there any structure at all".
`n_perm = 2000` at step 0 (the gate needs resolution below the Bonferroni alpha),
`n_perm = 1000` at every other checkpoint. Seeded `rng(0)`.

**Bootstrap CIs.** 1000 resamples of *items* with replacement (not of pairs — pairs are not
independent), rebuilding the lower triangle each time. Percentile CIs at [2.5, 97.5].

---

## 4. The step-0 gate, stated exactly (P0)

At `step0`, for each of the `n_cells` (layer × position) combinations, and separately for
`intent` and for `outcome`:

> **Pass** iff `|beta_observed|` is at or below the `100*(1 - 0.05/(2*n_cells))`-th percentile
> of the absolute values of that predictor's own permutation null in that cell.

The Bonferroni correction over cells is part of the rule and is fixed here rather than chosen
later. Without it the gate is miscalibrated in the wrong direction: with the planned 6 layers ×
2 positions = 12 cells, an uncorrected two-sided 95% per-cell rule would be expected to produce
about 0.6 spurious failures per predictor under a perfectly clean pipeline, so "all 12 cells
pass" would fail roughly half the time by chance alone and the gate would mostly be measuring
its own multiplicity. The uncorrected per-cell exceedance count is reported alongside, together
with its expected value under the null, so both readings are visible.

**Expected at step 0, and not a failure.** `beta_lexical` and `beta_scenario` are expected to
be *large* at random initialisation. A random network is close to a random projection, and
random projections approximately preserve input similarity, so token-overlap structure survives
to the hidden states. This is stated in advance so that a large step-0 lexical beta is not
later misread as contamination. The gate is on `intent` and `outcome` only, and the
within-scenario permutation is what makes that separation meaningful.

**On failure.** Halt. Write `outputs/rsa/E8_STEP0_GATE.md` naming the failing cells and the
margin, do not run E8d, and diagnose before anything else. The two leading suspects, in the
order they will be checked, are (i) sequence-length leakage — the four conditions in a group
differ in length, and last-token pooling over a length-correlated activation norm can encode
that — and (ii) tokenisation of the manipulated clause. Diagnostics for both are run only if
the gate fails, so that a passing gate is not followed by post-hoc exploration.

---

## 5. Sampling plan, and the primary cell

**Model, first pass:** `EleutherAI/pythia-1.4b`.

**Revisions (30), fixed here:** `step0`, `step1`, `step2`, `step4`, `step8`, `step16`,
`step32`, `step64`, `step128`, `step256`, `step512`, `step1000`, `step2000`, `step3000`,
`step4000`, `step8000`, `step16000`, `step24000`, `step32000`, `step48000`, `step64000`,
`step80000`, `step96000`, `step112000`, `step128000`, `step143000`. (26 as listed; the list
in the file is authoritative and any addition is an amendment.)

**Positions:** `belief_last` and the default last token, both. The divergence between them is
part of the result, not a robustness check to be reported only if it agrees.

**Layers:** relative depths `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}` of the 24-layer model, i.e. a
6-layer subset rather than all layers. Subsetting is for cost; the subset is fixed here so it
cannot be widened after seeing which depth looks best.

**Primary cell, fixed before any curve is seen:** relative depth **0.8**, position
**`belief_last`**. Rationale: E1 and E2 both read the late-but-not-final band, and
`belief_last` is the position at which intent is representable *before* the harm is mentioned
in the text, which is the position where the intent signal has any chance at all. The headline
numbers in P1, P2 and P4 come from this cell.

**Secondary reporting, to prevent the primary cell from being load-bearing by itself:** the
same fits are run at all 12 cells and the *distribution* of verdicts is reported (how many
cells prefer piecewise, how many sigmoid, how many are within the indistinguishable band). If
the primary cell disagrees with the majority of cells, that disagreement is the headline.

---

## 6. Shape test (P4), stated exactly

Fitted to `beta_outcome(t)` and to `beta_intent(t)` separately, on `x = log10(step)` over
**steps >= 1**. `step0` is excluded from the fit — `log10(0)` is undefined and step 0 is the
gate anchor, not a trajectory point — and is drawn on the figure as a separate marker with its
null band.

- **Smooth:** `y = b + L / (1 + exp(-k*(x - x0)))`. 4 free parameters.
- **Breakpoint:** `y = a + s1*(x - c)` for `x <= c`, `a + s2*(x - c)` for `x > c`. 4 free
  parameters, continuous at `c`.

Equal parameter counts, so the comparison is driven by residual sum of squares; AICc is used
anyway because `n` is about 25 and the small-sample correction is not negligible. **Decision
bar: `|dAICc| >= 2` to prefer a model; below 2 the shapes are reported as indistinguishable**
and neither "gradual" nor "sudden" is claimed. Changepoint CI: 1000 bootstrap resamples over
checkpoints with replacement, refitting `c` each time, percentile [2.5, 97.5]. A CI spanning
more than half the fitted log-step range is reported as **unlocated**.

---

## 7. Behavioral overlay (E8f) — secondary panel, gaps expected

The representational series exists from step 0 because it needs only activations. The
behavioral series exists only from the first checkpoint that emits parseable ratings. These are
not the same series and the figure will not imply they are.

A checkpoint is **gradeable** iff both hold:

1. at least 80% of items yield a parseable in-range numeric rating, and
2. `rating_std >= 0.2191`, the engagement floor derived in `outputs/stats/FLOOR_DERIVATION.md`.

`b_intent` and `b_outcome` are plotted from the first gradeable checkpoint onward, that
checkpoint is marked on the axis, and the region before it is left **blank** — not
interpolated, not extrapolated, not drawn at zero. Many or all pretraining checkpoints failing
this is an expected outcome and is itself reportable: base models already sit near the
engagement floor in the existing ladder. The behavioral panel cannot falsify P1–P4.

---

## 8. Outputs and naming

| path | contents |
|---|---|
| `outputs/rsa/predictors/*.npy`, `*.png` | the five predictor matrices and their heatmaps |
| `outputs/rsa/predictors/predictor_correlations.csv` | pairwise Spearman between predictors |
| `outputs/rsa/checkpoint_betas.csv` | model × revision × layer × position × predictor: standardised beta, null mean/q97.5/CI, bootstrap CI, VIF, `outcome_share`, raw ratio (blank when the denominator is inside its null) |
| `outputs/rsa/checkpoint_betas.png` | x = log training step, y = standardised beta, one line per predictor, step-0 null band shaded |
| `outputs/rsa/checkpoint_shape_fits.csv` | per cell: AICc for both models, dAICc, changepoint and its bootstrap CI, verdict |
| `outputs/rsa/E8_STEP0_GATE.md` | the P0 verdict, written whether it passes or fails |
| `outputs/rsa/E8_CHECKPOINT_EMERGENCE.md` | the verdict against P1–P4 |

Naming follows the `checkpoint_*` / `rsa_vs_behavior` families. **No new file is named
`rsa_convergence_*`**; that prefix is reserved for the superseded model-level scatter. Nothing
under `outputs/figures_final/_pending_rescore/` is read or written. Raw activations and cached
`.npy` RDMs stay gitignored; every number quoted in a write-up is read back from a committed
CSV. No figure is copied into `presentation_figures/` before the P1–P4 verdict exists.

---

## 9. Cost and disk, checked before starting

Checked 2026-09-21 against the live HF API and the actual quota, because the design document's
estimate covers activations only and omits the weights.

- `pythia-1.4b` is **2.93 GB per revision**. 26 revisions is ~76 GB of transfer. Home quota is
  195 GB soft / 200 GB hard with 70 GB already used, i.e. ~124 GB of headroom, so the full set
  does **not** comfortably coexist with the rest of the project.
- **Therefore the extraction evicts each revision's snapshot from the HF cache immediately
  after its activations are written.** Peak disk is one revision (~3 GB) plus outputs. This is
  a property of the runner, not an operational nicety, and it is stated here so a later reader
  knows the cache churn was deliberate.
- Extracted activations, 6 layers × 2 positions × 298 items × 2048 hidden, float32:
  ~30 MB per revision, ~0.8 GB for the set. Kept on disk, gitignored.
- `allenai/OLMo-2-1124-7B` (E8g) is ~14 GB per revision and ~20 revisions, so ~280 GB of
  transfer. **That does not fit in the home quota even transiently without eviction, and does
  not fit at all if retained.** E8g therefore runs with `HF_HOME` pointed at group storage
  `/orcd/data/jhm/001` (73 TB available, writable) and with the same per-revision eviction.
  E8g does not start until the Pythia trajectory is in hand.
- GPU: ~1–2 GPU-hours for the Pythia set. No API spend.

---

## 10. Scope limits

E8 is one model family on the first pass, one stimulus set, and 298 items. It cannot establish
that anything generalises across architectures; that is what E8g is for, and two families is
still two. Nothing in E8 licenses a causal claim — E7's erasure results are the causal arm, and
E8 is descriptive geometry over training time. The prior art (false-belief emergence tracked
across Pythia and OLMo checkpoints) tracks behavioral accuracy; E8's difference is that it
tracks representational geometry for the moral belief × outcome factors, and that the canonical
Young & Saxe 2×2 has not been run along a pretraining trajectory. That is a difference in
measurement, not a claim of priority over the trajectory method itself, and the write-up cites
the prior work rather than presenting the axis as new.

---

## Amendment 1, 2026-09-21 — what E8a measured about the covariates, before any checkpoint ran

E8a is a property of the stimuli, not of any model, so running it before the checkpoints is
not peeking. It produced two things that change what section 3 is allowed to say. Both are
recorded here before a single revision was downloaded.

**Correction.** Section 3 says the lower triangle has 43,053 pairs. It has **44,253**
(`C(298,2)`). Arithmetic slip, no consequence for any rule.

**The ratio's stated justification does not apply to this specification, and the reason it
does not is worth keeping.** Section 2.1 argues the outcome share is robust "because both
factors are residualised against the same covariates" and so are "both deflated by the
collinearity". Measured in the pre-registered all-pairs design, there is almost no
collinearity to deflate anything: variance inflation factors are `intent` 1.087, `outcome`
1.066, `interaction` 1.155, `scenario` 1.060, `lexical` 1.062. E1's deflation problem is not
inherited by E8's specification. The outcome share **remains the primary quantity** on the
other ground given in section 3, which is untouched: the raw ratio is 0/0 at step 0 and
near-singular through early training. No threshold moves.

**The asymmetry that is real, and where it lives.** Marginally, `lexical` is uncorrelated with
both factors (`intent` −0.0008, `outcome` +0.0215). Within a scenario group it is not, and it
is not symmetric between them:

| pair | all 44,253 pairs | within-scenario (821 pairs) |
|---|---:|---:|
| `lexical` × `outcome` | +0.0215 | **+0.8165** |
| `lexical` × `intent` | −0.0008 | +0.1297 |
| `intent` × `outcome` | −0.0034 | −0.2190 |

Within scenario, `lexical` and `outcome` are close to the same predictor (VIF 4.32 and 4.43),
while `lexical` and `intent` are not (VIF 1.52). The cause is in the stimulus construction: the
outcome manipulation rewrites the end of the story, the belief manipulation alters a mental
state phrase, and the first is the larger lexical edit. So "residualise both factors against
the same covariate" is *not* the same as "treat both factors identically" — it would deflate
outcome far more than intent if the within-scenario pairs drove the fit.

They do not drive it: 821 pairs is 1.86% of the design, which is why the all-pairs VIFs are
flat. But those are the only pairs carrying a factor contrast with everything else held
constant, so the point cannot simply be dropped either.

**Pre-registered consequence (added now, decided before any checkpoint compute):**

1. Every cell is fit **twice**, once with `lexical` in the design and once without, and both
   outcome shares go in `checkpoint_betas.csv` under a `spec` column (`primary` /
   `no_lexical`). If the two disagree about P1's ordering or about P2's sign, **the
   disagreement is the reported result** and P1/P2 are recorded as not established. This is
   not a choice between two numbers made after seeing them; it is a joint condition fixed here.
2. A **within-scenario-only** specification (821 pairs, `scenario` dropped as it is constant)
   is run and reported as a third `spec` (`within_scenario`). It is expected to be undecidable
   between `outcome` and `lexical` at VIF ~4.4, which is exactly E1's trap. It is reported for
   completeness and **nothing gates on it** — including, explicitly, that it may not be used to
   argue the all-pairs result away, nor to support it.
3. Observed VIFs are written per cell per spec, as already required by section 3.

---

## Amendment 2, 2026-09-21 — two defects found by the step-0 gate run, before E8d

E8c was run and **passed** (`outputs/rsa/E8_STEP0_GATE.md`): all 24 intent/outcome
predictor-cells sit inside their null, with 0 exceeding even the *uncorrected* 95% band
against 1.2 expected by chance. Reading that output turned up two things that have to be
fixed before E8d, and both are recorded here with the gate verdict they would change (none).

### 2.1 Relative depth 0.0 is a token-identity indicator, not a representation

At a single-token read position the layer-0 activation is a token-embedding lookup. Measured:
**every one of the 298 vignettes ends in `.` (295) or `).` (3)**, so the layer-0 activation
takes exactly **2 distinct values across 298 items**, at `last` and at `belief_last`, at
`step0` and at `step143000` alike. The resulting RDM has two distinct entries, the regression
on it is a fit to "do these two items end in the same token", and its permutation null is
atomic — which is why every predictor in that cell reported the identical `p_perm = 0.5098`.

This is structural and checkpoint-invariant, so it is not something later checkpoints outgrow.

**Consequence: relative depth 0.0 is dropped from every E8 analysis.** The layer set becomes
{0.2, 0.4, 0.6, 0.8, 1.0}, and the gate's cell count falls from 12 to 10.

Dropping cells from a Bonferroni family **tightens** the gate rather than loosening it
(α/2n goes from 0.05/24 = 0.00208 to 0.05/20 = 0.00250). The gate verdict is unchanged: the
smallest intent-or-outcome `p_perm` at any retained cell is 0.0300, an order of magnitude
above the tightened threshold. The exclusion is therefore not rescuing a failing gate, and
the already-written `E8_STEP0_GATE.md` is kept as the record of the 12-cell run.

Activations at depth 0.0 are still extracted; they are excluded at analysis time, so the
decision is visible in the code rather than hidden in the sampling.

### 2.2 The 298 items are 204 distinct texts, and the bootstrap assumed otherwise

**94 of the 298 rows are exact word-for-word reprints** of another row: all 24 YS2009
scenarios are reprints of YS2008 scenarios under different `scenario_id`s. This is a known
and already-repaired defect — `dataset/master/CONTAMINATION_REPAIR.md` §6 found it and
collapsed the reprints into a shared `scenario_group`, and the rows were deliberately
retained rather than dropped.

**E8 inherits that repair correctly** and this was verified rather than assumed: all 94
duplicate pairs fall inside a single `scenario_group` (0 cross-group duplicates), and E8 uses
`scenario_group` both for the `scenario` predictor and for the within-scenario permutation.
So the duplicates are same-scenario, same-condition, zero-distance pairs and they are not
contradicting any predictor.

Two things do follow.

**Amendment 1's asymmetry is not an artifact of the duplicates, and is slightly understated
by them.** Excluding the 94 duplicate pairs from the 821 within-scenario pairs:

| pair | within-scenario (821) | excluding duplicate texts (727) |
|---|---:|---:|
| `lexical` × `outcome` | +0.8165 | +0.7712 |
| `lexical` × `intent` | +0.1297 | **−0.1445** |
| `intent` × `outcome` | −0.2190 | −0.4944 |

`lexical` remains near-collinear with `outcome` and is, if anything, *anti*-correlated with
`intent` once the reprints are removed. Amendment 1's sensitivity requirement stands.

**The bootstrap's independence assumption does not hold.** Section 3 pre-registered a
bootstrap that resamples items with replacement. With 94 exact reprints the effective number
of distinct texts is 204, and separately the 4–6 conditions within a scenario are not
independent of each other either, so an item bootstrap produces CIs that are too narrow.

**Pre-registered consequence, decided now:** both bootstraps are computed and both are
reported, as `boot_ci_*` (the pre-registered item bootstrap) and `cluster_ci_*` (resampling
the 53 `scenario_group` clusters with replacement). The item bootstrap remains the
pre-registered primary so the original commitment is still readable. **Where the two disagree
about whether a β clears its null band — that is, about P1 — the cluster bootstrap governs,
because it is the one whose independence assumption is true of this dataset.** That precedence
is fixed here, before the trajectory exists, rather than chosen once it does.
