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
