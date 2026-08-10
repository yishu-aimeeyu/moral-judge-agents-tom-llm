# E1 — layerwise hypothesis-RDM variance partitioning: pre-specified predictions

Written before any variance-partition result existed. Recorded verbatim by
`code/experiments/61_rdm_variance_partition.py` to `outputs/rsa/E1_PRESPEC.md` on the first
`--run`, and never overwritten, so results cannot be retrofitted to the hypothesis.

**Why this exists.** The RSA convergence null (r = +0.098, n = 28,
`outputs/rsa/convergence_test.json`) says behaviourally similar models do not have similar
geometry. `hypothesis_rdm.csv` already reports scenario-partialled Spearman correlations
between each model RDM and an intent RDM and an outcome RDM separately, but those two
correlations are not competed against each other, and neither is competed against surface
lexical structure. A model whose geometry is organised entirely by word overlap would score
well on both. E1 replaces two marginal correlations with one variance partition, so
"the geometry is organised by outcome, not intent" becomes a number with a null attached.

## Predictors (fixed here, before fitting)

All predictors are 298×298 dissimilarity matrices reduced to their upper triangle
(44,253 pairs) and rank-transformed before regression, so the whole analysis is Spearman-
framework and matches `partial_spearman` in `24_rsa_cka.py`.

| predictor | definition | source |
|---|---|---|
| `intent` | `1` if the pair differs on `intent_label` (guilty vs innocent), else `0` | `dataset/master/moral_2x2_master.csv` |
| `outcome` | `1` if the pair differs on `outcome_label` (harm vs no_harm), else `0` | same |
| `interaction` | `\|I_i·O_i − I_j·O_j\|` with `I`/`O` the 0/1 intent and outcome indicators — the treatment-coded intent×outcome product term, i.e. "intentional vs the other three cells" | same |
| `lexical` | cosine distance between `tfidf_word_1_2` vectors of the story texts | `FEATURE_SETS["tfidf_word_1_2"]` in `code/experiments/21_surface_baseline.py`, identical vectoriser settings |

`intent` is the belief factor: in this dataset the intent manipulation *is* the belief
manipulation (guilty = believed the act would cause harm), which is why the intent probe is
run at `belief_last` elsewhere in the project.

**Scenario is not a predictor of interest and not in the primary model.** The primary model
is the four predictors above, as specified. Because the 53 scenario groups are the dominant
source of both lexical and geometric structure, a secondary model adding a scenario-identity
RDM as a fifth nuisance regressor is fitted in the same pass and reported alongside. If the
two models disagree about the *ordering* of intent vs outcome unique variance, the primary
result is reported as scenario-confounded and P1 is not claimed. This is stated now, before
either model has been fitted.

## The pre-specified predictions

```
=========================
Written before any variance-partition run. Recorded verbatim to outputs/rsa/E1_PRESPEC.md
on first --run, and never overwritten.

  P1 (outcome dominates the geometry late). At LATE layers, defined in advance as
      relative depth >= 0.75 including the final layer, the unique variance of the
      OUTCOME predictor exceeds the unique variance of the INTENT predictor. Bar: this
      holds in at least 7 of the 8 models with cached activations (one-sided sign test
      p <= 0.05 at 7/8; 6/8 is p = 0.145 and does NOT clear the bar).
  P2 (the dissociation survives at the intent probe's own best layer). The same ordering
      (outcome unique > intent unique) holds AT each model's peak-intent layer, the layer
      selected by `peak_intent_layer()` from outputs/probe/. This is the sharp version of
      the claim: the layer chosen because intent is maximally decodable there is still a
      layer whose geometry is organised by outcome. Bar: at least 7 of 8 models.
  P3 (both effects are real, not noise). Any unique variance quoted in support of P1 or P2
      exceeds the 95th percentile of its own within-scenario permutation null at that
      layer. A predictor whose unique variance sits inside its null is reported as
      indistinguishable from zero and cannot carry a comparison in either direction.
  P4 (surface control). The LEXICAL predictor's unique variance does not exceed BOTH the
      intent and the outcome unique variance at the layers used for P1/P2. If it does, the
      model RDM at that layer is primarily surface-organised, the layer is marked
      uninformative, and it is excluded from the P1/P2 counts with the exclusion reported.

  FALSIFICATION. P1 fails if intent unique variance is greater than or equal to outcome
  unique variance at late layers in 3 or more of the 8 models, or if the two are
  statistically indistinguishable (overlapping permutation CIs) in a majority of models.
  Either outcome is reportable and would matter: it would mean the RSA convergence null
  cannot be explained by "the geometry encodes outcome and the behaviour follows the
  geometry", and the null would need a different account.

  NON-TRIVIALITY. Unique variance is not a redescription of decodability. A predictor can
  be highly decodable by a supervised probe while contributing near-zero unique variance to
  the unsupervised geometry, which is exactly the intent/outcome asymmetry under test. The
  intent probe accuracy at each layer is therefore recorded in the same CSV so the two
  quantities can be read side by side and the comparison cannot be lost.
========================================================================================
```

## Estimator and null, fixed here

- **Unique variance** = semipartial ρ²: `R²(full model) − R²(model with that predictor
  dropped)`, on rank-transformed vectors, OLS. Reported per model per layer per predictor.
- **Total** `R²_full` is reported in the same row, so unique variances can be read as a
  share of explained geometry rather than in isolation.
- **Permutation null** (P3): stimulus identity is permuted *within scenario group* using
  `permute_within_groups()` from `24_rsa_cka.py`, the model RDM is rebuilt under the
  permutation, and the whole partition is refitted. `n_perm = 1000`, seeded `rng(0)`.
  Within-scenario permutation is the project-wide convention and is the conservative choice
  here: it preserves scenario structure and lexical similarity, so it tests the 2×2 factors
  specifically rather than testing "is there any structure at all".
- **Reported interval** is the permutation null's [2.5, 97.5] percentiles plus the observed
  value; the decision rule in P3 uses the 95th percentile one-sided.
- Layers: the same set `24_rsa_cka.py` already caches — relative depths
  {0.0, 0.25, 0.5, 0.75, 1.0} plus each model's peak-intent layer.
- Pooling: `mean` for geometry, matching the existing RSA outputs. `belief_last` is run as a
  secondary pooling only if the primary completes; it is not part of P1–P4.

## Outputs

| path | contents |
|---|---|
| `outputs/rsa/rdm_variance_partition.csv` | model × layer × predictor: unique ρ², `R²_full`, permutation null mean/q95/CI, `is_peak`, `rel_depth`, intent probe accuracy at that layer, `model_variant` (`primary` / `plus_scenario`) |
| `outputs/rsa/rdm_variance_partition_layerwise.png` | unique variance vs relative depth, one panel per model, four predictor lines, peak-intent layer marked |
| `outputs/rsa/E1_VARIANCE_PARTITION.md` | verdict against P1–P4 |

New outputs use the `rdm_variance_partition` / `rsa_vs_behavior` naming family. No new file
is named `rsa_convergence_*`; that prefix is reserved for the superseded model-level scatter.
Nothing under `outputs/figures_final/_pending_rescore/` is read or written. Raw activations
and cached `.npy` RDMs stay gitignored. No figure is copied into `presentation_figures/`
until the P1–P4 verdict is written.

---

## Amendment 1, 2026-08-09 — P4's surface control is collinear with scenario identity

A 10-permutation smoke run (not a result run; written up here rather than discarded) showed
that **P4 as originally written gates out every layer in every model**, which is the same
failure mode W3's amendment 1 fixed: an instrument that excludes the entire design space is
broken, not informative. The cause is specific and was foreseeable:

`lexical` is cosine distance on TF-IDF over the story texts, and the four conditions within a
scenario group are near-identical strings differing by a clause. So the lexical RDM is very
nearly the **scenario-identity RDM**, and it absorbs the scenario structure that dominates
every model RDM (lexical unique ρ² 0.04–0.47, versus 0.0000–0.0035 for intent and outcome).
`lexical` therefore exceeds both intent and outcome everywhere, and P4's exclusion rule fires
at every layer. The rest of the project already treats scenario as a nuisance to be partialled
out for exactly this reason (`partial_spearman` in `24_rsa_cka.py` R4).

**Disclosure.** This amendment was written after the smoke run, so the intent-vs-outcome point
estimates at 10 permutations had been seen. No threshold is changed by it: P1's 7/8 bar, P2,
P3's null-q95 rule, and P4's comparison rule are all unchanged, and the primary
four-predictor verdict is still reported exactly as pre-registered, including its P4
exclusions.

**What is added** is a third predictor variant, so that P4's *surface* control tests surface
structure rather than scenario identity:

| variant | predictors | role |
|---|---|---|
| `primary` | intent, outcome, interaction, lexical | as pre-registered; reported first, P4 exclusions reported |
| `plus_scenario` | + scenario identity | as pre-registered secondary |
| `lexical_resid` | intent, outcome, interaction, scenario, **lexical residualised on scenario** | added here; P4 is evaluated against residual surface structure |

**Reporting rule, fixed now.** When P4 excludes every layer in the `primary` variant, the P1
and P2 counts in that variant are reported as **undecidable**, not as "not supported". A count
of 0/8 arising from an empty comparison set is not evidence about the direction of the effect,
and conflating the two would be the reverse of the error the prespec exists to prevent. The
directional verdict is then taken from `lexical_resid`, labelled as amended rather than
pre-registered.

## Amendment 2, 2026-08-09 — P4's exclusion rule is redundant with the estimator (M1 added)

The `lexical_resid` variant added in amendment 1 **also** excludes every layer: residualising
lexical distance on scenario identity does not shrink it below intent and outcome. The reason
is structural, and it is a property of every vignette design of this kind, not of these models:
**the intent and outcome manipulations are realised lexically.** Within a scenario group the
four conditions differ by the clause that makes the act intended or the harm occur. A
continuous, fine-grained word-overlap predictor will therefore always capture more unique
variance than the binary factor it expresses. No residualisation fixes that, because there is
no version of these stimuli in which intent is manipulated without changing words.

**The rule was therefore mis-designed, and it is important to say why rather than to relax it
quietly.** Unique variance here is *semipartial*: intent's unique ρ² is already computed with
`lexical` in the model, and so is outcome's. Both quantities are already adjusted for surface
overlap. The comparison P1 and P2 actually make — intent's unique variance against outcome's —
is thus lexical-adjusted by construction, and the size of `lexical` itself does not threaten
it. P4's gate does not protect the comparison from a confound; it only prevents the comparison
from being made at all.

**What is added, and what is not.** Following the W3 precedent for post-hoc additions, the
gate-free comparison is numbered **M1**, not P5, so that it never reads as pre-registered:

> **M1 (lexical-adjusted ordering).** Across late layers and at the peak-intent layer, the
> ordering `outcome unique ρ² > intent unique ρ²` is evaluated **without** the P4 exclusion,
> with P3's permutation requirement still enforced: both predictors must clear the 95th
> percentile of their own within-scenario null at that layer. The bar is unchanged at
> **7 of 8 models**. The lexical unique variance is reported in the same table so a reader can
> see what was set aside.

**Disclosure.** Written after the 10-permutation smoke run, so the intent-vs-outcome point
estimates and their ratios had been seen (outcome/intent ratios in the range ~4–18× at late
layers). No bar was changed: 7/8 for the count, q95 for the null. What changed is that a gate
whose logic does not hold is reported as a finding about the design rather than applied as an
exclusion. The pre-registered gated verdict is still published above it, and both appear in
`E1_VARIANCE_PARTITION.md`.
