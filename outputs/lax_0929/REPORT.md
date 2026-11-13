# Lax 9/29 work order — report

All paths are relative to `tom_project/`. Scripts live in `code/experiments/`, and every number below has a `.csv` in this folder (`outputs/lax_0929/`).

**Shared conventions**
- Cross-validation and bootstraps are grouped by `scenario_group` (n = 53).
- Permutation nulls shuffle labels within scenarios, with p = (k + 1)/(n + 1). With 1000 permutations the floor is p = 0.001.
- TF-IDF surface baselines come from `21_surface_baseline.FEATURE_SETS` (word 1–2-grams and char_wb 3–5) and use the same text span and the same splits as the probe.
- The probe protocol is the one in `02_probe.py`: StandardScaler, then row-space projection, then LogisticRegression(C = 1), with GroupKFold(5).
- The headline behavioural contrast is attempted − accidental (YS2008 + YS2009) on endpoint-normalized ratings in [0, 1].

**Commits** (local `main`; see "Not done / blocked" about the push):

| script | commit | one-line result |
|---|---|---|
| 86 MOR4 trials | `e29f2f7` | Tidy trial table built. 13 of 18 subjects responded; 39.6 % of trials have no response; `story_id` is unmapped. |
| 87 T1 | `e5d02cc` | The model ordering is stable across 4 normalizations (ρ ≥ 0.94). Instruct < base in every scheme. Items cluster by outcome. |
| 88 T2 + T3 | `5db8af8` | Harm cells have about 2× the across-model variance; length predicts nothing. MOR4 within-condition split-half is 0.40. |
| 89 T4 | `8c56290` | Outcome is decodable at the last belief token (0.70–0.87, TF-IDF 0.63). It drops to exactly 0.5 without the foreshadow sentence. |
| 90 + 91 T5 | `cab8b81` | Intent decoding survives construct-held-out templates. Instruct models generalize better across templates than base models. |
| 92 T6 | `d26a098`, fix `e499adf` | Intent rises in the belief clause and decays afterwards. Outcome is carried by the foreshadow and then by the outcome clause. |
| 93 + `code/submit_lax_t7.sh` T7 | `77a7589` | Instruct models have stronger story-free priors and much larger context sensitivity. Condition × class is the largest unique term. |
| 94 T5/T6 tables + figures | `bb64669` | Summary CSVs and figures for T5 and T6. |

Reused without modification: `02_probe.py`, `03_behavioral.py`, `21_surface_baseline.py`, `81_extract_token_trajectories.py`, `82_temporal_decoding.py`, `83_reliability_normalization.py`, `stage_landmarks.py`, and `dataset/master/clause_offsets.csv`.

`14_prompt_invariance_decomposition.py` only computes per-model contrast stability, and `33_prompt_factorial_analysis.py` only has a wording × construct ANOVA. T7's variance decomposition is therefore new code in script 93.

---

## T1 — Normalization audit and clustering

**Question.** Do the model ordering and the base-vs-instruct conclusion depend on how ratings on different native scales (1–3, 1–4, 1–7) are normalized? And do items cluster by source or scale, or by design factors?

**What was run.** `87_lax_normalization_audit.py` (`e5d02cc`), with data loading reused from 83 and 03.
- Scale audit: every prompt's scale, anchors and direction, parsed from `03.build_prompt`, plus the scales the humans actually saw.
- Four normalizations:
  - `endpoint`: (x − min)/(max − min) of the native scale.
  - `global_mm`: min-max per model over all its ratings.
  - `z_mts`: z-score within model × template × source.
  - `rank_mts`: rank within model × template × source.
- Per-model attempted − accidental contrast, with a 2000-resample scenario bootstrap; 9 base/instruct pairs.
- Clustering: Ward linkage and MDS on correlation distance over items. Silhouette and adjusted Rand index (ARI) against source, native scale, condition, intent and outcome, using three feature sets.

**Result.**
- **Scale audit** (`scale_audit.csv`):
  - The YS2008 human scale (1 = Forbidden … 3 = Permissible) is reversed relative to our model `human_verbatim` prompt (higher = more impermissible). The 2007 adult anchor is also reversed.
  - YS2009 (1–4 blame) and YS2011 (1–7 wrong) run in the same direction as the model prompts.
  - All 6 factorial templates use 1–7, higher = more condemnation.
- **Ordering** (`normalization_ordering.csv`, 34 rater entries). Spearman ρ of the model ordering against `endpoint`:
  - `global_mm` 0.996, with no sign flips.
  - `z_mts` 0.945, with 1 sign flip (OLMo-2-7B base).
  - `rank_mts` 0.946, with 2 sign flips (OLMo-2-7B base and gemini-2.5-flash).
- **Base vs instruct** (`normalization_base_vs_instruct.csv`). Mean of 11 open instruct models minus mean of 9 base models:

  | scheme | instruct − base | 95 % CI |
  |---|---|---|
  | `endpoint` | −0.319 | [−0.345, −0.291] |
  | `global_mm` | −0.309 | [−0.334, −0.281] |
  | `z_mts` | −0.898 SD | [−1.007, −0.786] |
  | `rank_mts` | −0.259 | [−0.293, −0.225] |

  - Under `endpoint` and `global_mm`, all 9 pairs are negative with CIs that exclude 0.
  - Under `z_mts`, Qwen2.5-7B's CI includes 0, and Qwen2.5-14B flips sign (+0.19 [0.03, 0.37]).
  - Under `rank_mts`, Qwen2.5-14B also flips (+0.05 [0.005, 0.10]).
  - Per-model contrasts are in `normalization_robustness.csv`.
- **Clustering** (`clustering_silhouette.csv`):

  | feature set | outcome silhouette | outcome ARI | condition ARI | source silhouette | source ARI | intent silhouette | intent ARI |
  |---|---|---|---|---|---|---|---|
  | all 7 templates | 0.30 | 0.88 | 0.39 | 0.01 | 0.07 | 0.01 | ≈ 0 |
  | factorial templates 1–7 only | 0.31 | 0.88 | 0.39 | −0.06 | −0.01 | 0.01 | ≈ 0 |
  | `human_verbatim` only | 0.25 | 0.75 | 0.27 | 0.21 | 0.22 | 0.01 | ≈ 0 |

  - Source clustering appears only in the `human_verbatim` feature set, which is the one where source and native scale are confounded (source silhouette 0.21, ARI 0.22). Source and native scale give identical numbers because each source has its own scale.
  - Intent does not organize the item space in any feature set.

**Figures.** `fig_T1_normalization.{png,pdf}`, `fig_T1_clustering.{png,pdf}`; MDS coordinates are in `clustering_mds_*.csv`.

**Caveats.**
- Closed models are included in the ordering but have no base partner.
- The Qwen chat-off reruns appear as separate "[chat off]" rows and are excluded from the class means.
- UMAP is not installed in `.venv`, so only MDS was used.

## T1.5 — MOR4 human anchor

**Question.** Can the MOR4 scanner behaviour (YS2008 design, 213 files in `ToM Project/MOR4_behavioural`) serve as a per-item human anchor?

**What was run.** `86_mor4_trials.py` (`e29f2f7`).
- Reads `SAX_MOR4_*.MOR4.[0-9].mat`.
- Recovers the condition from the SPM onsets in `SPMinputs` and asserts that it matches `theOrder` (verified for all sessions checked).
- Maps keys 85/86/87 to 50/51/52.
- Writes one row per trial.

**Result.**
- `mor4_trials.csv`: 18 subjects, up to 6 runs × 8 trials each.
- Responses: 5 subjects (01, 13, 14, 15, 16) have no responses at all, leaving 13 responding subjects. Overall missingness is 39.6 % (`mor4_missingness.csv`).
- Key distribution (`mor4_key_distribution.csv`):
  - 314 trials with no key.
  - 50/51/52: 209/106/108 presses.
  - Alternate keys 85/86/87: 27/10/16 presses.
  - 2 stray presses of 49.
- So only **3** response buttons were used, which matches the YS2008 appendix 1–3 scale rather than the 1–4 implied by the "key − 48" mapping in the work order.
- Condition means on the key − 48 scale (`mor4_condition_means.csv`; per-subject means in `_by_subject.csv`):

  | foreshadow | BT_OT | BT_OG | BG_OT | BG_OG |
  |---|---|---|---|---|
  | BF | 3.35 | 3.04 | 2.44 | 2.03 |
  | OF | 3.55 | 3.15 | 2.45 | 2.11 |

  - The intent effect (T vs G belief) is about 0.9–1.1 key units; the outcome effect is about 0.3–0.4.
  - Attempted (BT_OG) − accidental (BG_OT) is +0.60 (BF) and +0.70 (OF). If the three keys are taken as a 3-point scale, this is about +0.30 and +0.35 in endpoint units.

**Stopped here, per the work order.** No decoder from `theStories` 1–48 to scenario text was found. Places searched:
- MOR4_2 and MOR4_3.
- JKH_scripts (`DIS_MOR4_MOR5_eval.m` uses only aggregate accidental/intentional harm).
- GSS, saxelab `ItemAnal`, and `FMRI_Experiments.zip`.
- A filesystem-wide find over mindhive.

There are no SPM.mat or stimulus files. The tidy table is keyed by story number only (`story_id` left empty).

**File needed:** the MOR4 experiment script or stimulus list that maps story numbers 1–48 to YS2008 scenarios, and that defines T/G. The reading T = negative and G = neutral is inferred from the data: BG_OG is the lowest cell, with SD 0.26.

**Caveats.**
- The direction "higher key = more condemnation" is inferred from the cell means. It is not documented anywhere.
- The YS2008 paper's scale is reversed (1 = Forbidden). It is unknown which of the two applies to the scanner button box.
- Spearman-based analyses (T3) are unaffected by either issue.

## T2 — Rating variance per story

**Question.** Which stories and conditions produce the most disagreement across templates and models, and does story length explain it?

**What was run.** `88_lax_variance_reliability.py` (`5db8af8`).
- Variance and IQR per story of endpoint-normalized ratings across 7 templates (within model, then averaged), across 20 open models, and across everything.
- Clause lengths from `clause_offsets.csv`.
- OLS with scenario-clustered SEs: variance ~ source + condition + word_count + belief/action/outcome words.

**Result.** Mean variance per condition (`story_variance_by_source_condition.csv`):

| source | condition | across models | across templates |
|---|---|---|---|
| YS2008 | accidental | 0.087 | 0.017 |
| YS2008 | intentional | 0.087 | 0.019 |
| YS2008 | attempted | 0.043 | 0.016 |
| YS2008 | neutral | 0.039 | 0.009 |
| YS2009 | accidental | 0.082 | 0.014 |
| YS2009 | intentional | 0.081 | 0.017 |
| YS2009 | attempted | 0.039 | 0.013 |
| YS2009 | neutral | 0.036 | 0.008 |

- The harm-outcome cells (accidental, intentional) have about **2×** the across-model variance of the no-harm cells.
- Across-template variance is about 5× smaller than across-model variance.
- Regression (`story_variance_regression.csv`), across models (R² = 0.73):
  - accidental +0.048 and intentional +0.048 vs neutral, both p < 1e-4.
  - attempted +0.0035, p = 0.058.
  - None of the length terms is significant (all p > 0.2).
- Regression across templates (R² = 0.37): all three non-neutral conditions are > neutral, p < 1e-4, and length is n.s. (all p > 0.1).
- The 15 highest- and lowest-variance stories are in `story_variance_top_bottom15.csv`.

**Figures.** `fig_T2_story_variance.{png,pdf}`.

**Caveats.**
- Closed models are excluded, because there are too few to estimate per-story variance.
- **The planned human-vs-model variance correlation was not run**, because MOR4 has no `story_id` mapping (see T1.5).

## T3 — Split-half reliability

**Question.** How reliable are item-level ratings per source, for templates, models, sampling runs and humans?

**What was run.** `88_lax_variance_reliability.py` (`5db8af8`).
- Spearman split-half with Spearman–Brown correction, 1000 random splits. The table reports the median and the 2.5–97.5 % range.
- "Within-condition" removes condition means first, so it measures item reliability beyond the 2×2 design.
- Rater types:
  - Templates within each model.
  - Base models (9) and instruct models (11), z-scored within model × template × source.
  - Closed-model sampling runs (`behavior_v3`).
  - MOR4 subjects.

**Result** (`split_half_summary.csv`, `split_half_reliability.csv`; per-item SD of the half-difference in `split_half_per_item_sd.csv`):

| raters | YS2008 raw | YS2008 within-condition | YS2009 within-condition | YS2011 within-condition |
|---|---|---|---|---|
| templates (median over 20 models) | 0.94 | 0.89 | 0.92 | 0.93 |
| instruct models (11) | 0.97 [0.96, 0.98] | 0.83 [0.76, 0.88] | 0.84 [0.77, 0.90] | 0.93 [0.83, 0.99] |
| base models (9) | 0.59 [0.29, 0.68] | 0.36 [0.14, 0.52] | 0.59 [0.41, 0.72] | 0.64 [−0.23, 0.94] |
| closed sampling runs | 1.00 | 1.00 (min 0.55) | 1.00 | 1.00 |
| MOR4 humans (13 subjects, 48 story numbers) | 0.22 [−0.31, 0.56] | **0.40 [0.10, 0.61]** | — | — |

- MOR4 on story × condition cells (225 cells): 0.69 [0.59, 0.80].
- The closed sampling runs are deterministic: the samples are identical for almost all items, so their "reliability" of 1.0 is not informative.

**Per-item human data does not exist for YS2009 or YS2011.** Only paper-level condition means exist, so no human reliability is claimed for those sources. MOR4 is a scanner replication of the YS2008 design and is shown only on the YS2008 row.

**Figures.** `fig_T3_split_half.{png,pdf}`; the MOR4 reference line is on the YS2008 row only.

**Caveats.**
- MOR4 items are story *numbers*, not mapped scenarios. Each subject saw each story in a counterbalanced condition, which is why the within-condition residualization matters.
- Thirteen subjects with heavy missingness means each half has about 6 raters.

## T4 — Outcome decoded from the intention stage

**Question.** Is the outcome already decodable before the outcome clause, at the end of the belief (intention) stage? Is that beyond surface text?

**What was run.** `89_lax_outcome_from_intent_stage.py` (`8c56290`).
- **Part A.** Existing `outputs/acts/*.npz` for 14 models, using the `belief_last` and `action_last` poolings. The 02 clause mask leaves 290 stories.
  - All layers were scanned; reported layers are the fixed depth round(0.65·(nL − 1)) and the peak.
  - 1000 permutations; span-matched TF-IDF.
- **Part B.** Token trajectories (`*__story__chatoff`, from 81) for 3 base/instruct pairs at the last belief token, for the original text versus a **no_foreshadow** variant (288 YS2008/YS2009 stories). All saved depths were used.

**Result.**
- **Part A** at the fixed layer (`outcome_from_intent_stage_summary.csv`; per-layer results in `_layers.csv`; per-item results in `_items.csv`).
  - At `belief_last`, outcome accuracy is **0.70–0.87** across the 14 models: Qwen2.5-0.5B 0.70; OLMo-2-7B 0.85; Qwen2.5-7B 0.87.
  - The TF-IDF baselines are word 0.631 and char 0.572.
  - The gap over the best TF-IDF baseline is +0.07 to +0.24, and p = 0.001 for all 14 models.
  - At `action_last`, accuracy is 0.62–0.83.
  - Intent accuracy at the same layer and token is 0.85–0.96.
- **Part B** (`outcome_from_intent_stage_noforeshadow.csv`).
  - **Original text:** peak outcome accuracy is 0.71 / 0.80 (Llama-3.1-8B base / instruct, L16), 0.77 / 0.81 (OLMo-2-7B, L16) and 0.73 / 0.73 (Qwen2.5-7B, L22 / L14). TF-IDF on the same text is word 0.60 and char 0.59; p = 0.001.
  - **No foreshadow:** accuracy is exactly 0.50 (OLMo base reaches 0.503 at two layers). The prefix up to the end of the belief is **byte-identical** between outcome partners in 100 % of items, so the result is 0.5 by construction. Permutations were skipped and p is set to 1 (structural).

**Diagnosis.** In every YS2008/YS2009 story, the only outcome information before the end of the belief is the foreshadow sentence. Outcome decodability at the intention stage is therefore a readout of the foreshadow text, not inference of the outcome. The probe does beat TF-IDF on that same text by 0.1–0.2.

**Figures.** `fig_T4_outcome_from_intent_stage.{png,pdf}`.

**Caveats.**
- Part B uses chat-off trajectories for the instruct models too, which is how they were extracted in 81.
- The trajectory depths are a fixed saved subset, not every layer.

## T5 — Intent decoding with held-out templates

**Question.** Does a probe for intent trained on one prompt template generalize to a different template? Is that better than surface text?

**What was run.**
- `90_extract_template_last_token.py` (`cab8b81`): last prompt token, every layer, for each of the 7 factorial templates. Models: Qwen2.5-7B ± Instruct and Llama-3.1-8B ± Instruct. Chat formatting is automatic: on for both Qwen models and for Llama-Instruct, off for Llama base, which has no chat template.
- `91_lax_template_heldout_decoding.py` (`cab8b81`). Every split is crossed with GroupKFold by scenario:
  - **within**: train and test on the same template.
  - **pair**: train on template A, test on template B, for all 7 × 7 pairs.
  - **construct**: hold out one construct (blame, wrong or punish), with `human_verbatim` always in training.
- 1000 permutations at the fixed 0.65 depth and at the peak construct layer; TF-IDF on the full prompt with the same splits.
- Summaries and figures from `94_lax_t5_t6_figures.py` (`bb64669`).

**Result** (`template_heldout_summary.csv`, `template_heldout_perm.csv`, `template_heldout_layers.csv`; per-model files in `T5/`):

| model | layer | within | pair | construct | within − construct (perm p) | peak construct (layer) |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 18 | 0.647 | 0.568 | 0.635 | 0.012 (p = 0.15) | 0.755 (L22) |
| Qwen2.5-7B-Instruct | 18 | 0.750 | 0.580 | 0.680 | 0.070 (p = 0.001) | 0.825 (L21) |
| Llama-3.1-8B | 21 | 0.698 | 0.560 | 0.601 | 0.097 (p = 0.001) | 0.639 (L17) |
| Llama-3.1-8B-Instruct | 21 | 0.859 | 0.706 | 0.801 | 0.058 (p = 0.001) | 0.814 (L27) |

- Every within, pair and construct accuracy has p = 0.001; the null mean is 0.50 and the null 95th percentile is ≤ 0.53.
- TF-IDF on the same splits:
  - Word: within 0.602, pair 0.596, construct 0.600.
  - Char: within 0.555, pair 0.550, construct 0.543.
- **Construct-held-out decoding beats TF-IDF for all four models.** At peak layers the margin is +0.15 to +0.23 for Qwen and Llama-Instruct, and +0.04 for Llama base.
- **Single-template pair transfer is weak for base models.** Base pair accuracy is 0.56–0.57, which is *below* the word TF-IDF baseline of 0.60 at the fixed layer. Instruct models reach 0.58–0.75.
- **Generalization gap.** At the fixed layer, within − construct is 0.06–0.10 except for Qwen base (0.01, n.s.). Within − pair is larger, about 0.10–0.20 at depth ≥ 0.6 (see figure).

**Figures.** `fig_T5_template_heldout.{png,pdf}` shows accuracy and gap versus relative depth. `fig_T5_pair_matrices.{png,pdf}` shows the 7 × 7 train → test matrices at the best layer.

**Caveats.**
- Only 2 base/instruct pairs were run (one GPU per user). OLMo was not extracted for T5.
- The "best layer" for Llama (L16) was assumed, because there is no 02 probe run for Llama to choose it from. The fixed-depth numbers above do not depend on that choice.
- The TF-IDF features are the full prompt, so they include template wording; that wording is constant within a template and cannot predict intent.

## T6 — Decoding over time, temporal generalization (TG) and RDM

**Question.** When during the story do intent and outcome become decodable? Do the codes generalize across time? How does the 4-condition representational geometry evolve?

**What was run.** `92_lax_time_decoding_clauses.py` (`d26a098`, RDM fix `e499adf`), with figures and tables from `94_lax_t5_t6_figures.py` (`bb64669`).
- Reuses the 81 token trajectories, so no new activations were needed. Models: Qwen2.5-7B, OLMo-2-7B and Llama-3.1-8B, each ± Instruct.
- Positions: 10 bins per stage over background, belief, action and outcome, taken from `clause_offsets.csv` (40 bins). The 02 mask is applied, and stories with non-monotone stage ends are dropped (`time_decoding_n.csv`).
- At each saved layer:
  - GroupKFold decoding curve, with a 95 % band from 100 within-scenario shuffles.
  - TF-IDF on the same text prefix.
  - 40 × 40 TG matrix (`82_temporal_decoding.temporal_generalization`).
  - Pre-registered hypothesis statistics, with a 1000-resample scenario bootstrap.
  - RSA: the 4 × 4 condition RDM regressed on intent, outcome and an intent ∧ outcome conjunction model RDM.

**Result.** Best layers: Qwen 18, OLMo 16, Llama 16. Values are from `time_decoding_hypotheses_bestlayer.csv`; all CIs exclude 0.

| statistic | Qwen-7B base / instruct | OLMo-7B base / instruct | Llama-8B base / instruct | TF-IDF |
|---|---|---|---|---|
| intent rise across the belief clause | 0.44 / 0.43 | 0.40 / 0.42 | 0.37 / 0.38 | 0.13 |
| intent, late − belief end | −0.20 / −0.21 | −0.16 / −0.17 | −0.18 / −0.18 | 0.00 |
| intent slope late (per bin) | −0.013 / −0.012 | −0.011 / −0.011 | −0.015 / −0.012 | 0.001 |
| outcome slope pre-outcome (per bin) | −0.010 / −0.006 | −0.007 / −0.005 | −0.006 / −0.004 | −0.004 |
| outcome jump at the outcome clause | 0.31 / 0.29 | 0.20 / 0.14 | 0.20 / 0.16 | 0.04 |

- **Intent** rises from chance (0.5) to 0.95–0.98 by the end of the belief clause. It then decays to about 0.6–0.75 through the action and outcome clauses, and recovers to 0.84–0.86 at the final token. Throughout the belief stage it sits well above the shuffle band and the TF-IDF curve (TF-IDF ≤ 0.65).
- **Outcome** is already about 0.85–0.9 at the foreshadow, at the end of the background (consistent with T4). It drifts down through the belief and action stages, then jumps to about 1.0 in the outcome clause.
- Base and instruct curves are nearly indistinguishable for all three families.
- **TG** (`fig_T6_temporal_generalization`): the intent code is a block over the belief and early action bins, with partial off-diagonal generalization. The outcome code trained in the foreshadow or background transfers only weakly to the outcome clause; there is a separate late block.
- **RDM betas** (`time_decoding_rsa.csv`; stage means at the best layers, after the fix):

  | stage | intent β | outcome β | interaction β |
  |---|---|---|---|
  | background | 0.08–0.16 | 1.46–1.57 | −0.00 to 0.08 |
  | belief | 1.57–1.69 | 0.47–0.54 | −0.06 to −0.03 |
  | action | 1.54–1.95 | 0.69–0.82 | −0.10 to 0.02 |
  | outcome | 0.25–0.37 | 1.86–1.92 | −0.12 to −0.08 |

  The conjunction (interaction) term stays near 0 throughout: the condition geometry is additive in intent and outcome.

**Figures.** `fig_T6_time_decoding.{png,pdf}` (curves with null bands, TF-IDF, and RDM betas over time) and `fig_T6_temporal_generalization.{png,pdf}`. Per-model files are in `T6/`.

**Caveats.**
- The first T6 run coded the interaction RDM as ±1 "differs". Together with the intent and outcome RDMs this sums to a constant over the 6 condition pairs, so the betas were not identifiable. It was recoded as a 0/1 conjunction (verified rank 3), the old outputs were deleted, and all 6 models were rerun (`e499adf`). Only the corrected numbers are reported.
- 100 shuffles per point makes the null band coarse. The hypothesis statistics use 1000 bootstraps.
- Trajectories are chat-off for all models.
- The intent decay in the RDM (belief β about 1.6, outcome-stage β about 0.3) is stronger than the decay in decoding accuracy. These measure different things: linear separability versus mean-pattern distance.

## T7 — Priors, prompt ablations and variance decomposition

**Question.** What does a model say with no story, or with partial stories (its prior)? How much does the story move it (context sensitivity)? How is rating variance split among scenario, condition, template, scale, family and base/instruct?

**What was run.** `93_lax_priors_ablation.py` and `code/submit_lax_t7.sh` (`77a7589`).
- `--build` writes 4 ablated stimulus sets. Cut points come from `stage_landmarks`; `stimuli_audit.csv` lists every cut.
  - `no_story`: "<Agent>." only (298 stories).
  - `background_only`: text before the foreshadow (298).
  - `belief_removed`: belief sentence(s) deleted (298).
  - `outcome_removed`: text up to the outcome sentence (288; YS2011 has no outcome partner).
- Scoring: `03_behavioral.py --backend hf --scoring logprob` on the 7 factorial templates, for Qwen2.5-7B ± Instruct and Llama-3.1-8B ± Instruct.
- `--analyze` computes:
  - Prior strength = |rating − 0.5|.
  - Context sensitivity = full − ablated.
  - Ablated attempted − accidental contrasts.
  - Instruct − base differences with a scenario bootstrap.
  - Type I sums of squares plus the unique R² drop for each term.

**Result.**
- **Priors and context sensitivity** (`ablation_summary_by_class.csv`, `ablation_base_vs_instruct.csv`; 4 models, 53 scenarios):

  | ablation | prior strength, base / instruct | instruct − base [95 % CI] | \|context sensitivity\|, base / instruct | instruct − base [95 % CI] |
  |---|---|---|---|---|
  | no_story | 0.17 / 0.23 | +0.056 [0.050, 0.061] | 0.06 / 0.29 | +0.238 [0.219, 0.255] |
  | background_only | 0.17 / 0.24 | +0.059 [0.037, 0.082] | 0.05 / 0.28 | +0.228 [0.201, 0.257] |
  | belief_removed | 0.17 / 0.26 | +0.083 [0.065, 0.101] | 0.02 / 0.13 | +0.112 [0.093, 0.134] |
  | outcome_removed | 0.15 / 0.16 | +0.007 [−0.003, 0.018] | 0.02 / 0.07 | +0.050 [0.043, 0.057] |

  - Base models barely move between no story and the full story: their |context sensitivity| is 0.02–0.06.
  - The per-condition no-story priors are in `ablation_prior_context_by_condition.csv`. Examples on the endpoint scale: Llama-8B-Instruct sits at 0.15–0.16 with no story and rises to 0.49–0.68 with the full story; Qwen-7B base sits at 0.53 with no story and moves by less than 0.09.
- **Ablated contrasts** (attempted − accidental; `ablation_contrasts.csv`):
  - `no_story` and `background_only`: exactly 0 for all models, because the text is identical across conditions (structural).
  - `belief_removed`: the instruct contrast roughly doubles in magnitude (Qwen-I −0.24 → −0.45; Llama-I −0.20 → −0.47). Without the belief, only harm differs, and the intent effect is 0.
  - `outcome_removed` (foreshadow kept): the instruct contrast halves (Qwen-I −0.10; Llama-I −0.11). The harm effect is still 0.15, carried by the foreshadow sentence, and the intent effect is 0.04–0.05.
  - Base models stay within ±0.08 in every ablation.
- **Variance decomposition, full stories** (`variance_decomposition_full.csv`; 20 open models × 7 templates × 298 stories; model R² = 0.49). Type I shares in entry order:

  | term | Type I share | unique R² drop |
  |---|---|---|
  | condition | 17.3 % | — |
  | class (base vs instruct) | 10.1 % | — |
  | condition × class | 11.0 % | **8.6 %** |
  | condition × family | 2.4 % | 2.4 % |
  | scenario | 2.3 % | 2.1 % |
  | template | 2.2 % | — |
  | template × class | 1.8 % | 1.8 % |
  | family | 1.3 % | — |
  | condition × template | 0.4 % | 0.4 % |
  | scale | 0.1 % | 0.1 % |
  | residual | 51.1 % | — |

- **With ablations** (`variance_decomposition_with_ablation.csv`; 4 models; R² = 0.38):
  - condition × ablation × class: Type I 7.6 % and unique 7.6 %.
  - ablation × class: 7.4 %.
  - ablation: 6.7 %.
  - condition: 5.7 %.
  - condition × ablation: 5.4 %.
  - template: 1.8 %.
  - scenario: 2.0 %.

**Figures.** `fig_T7_priors.{png,pdf}`.

**Caveats.**
- The ablations ran on 4 models only (2 pairs), to stay within one GPU per user.
- `belief_removed` and `outcome_removed` keep the foreshadow sentence, by design. It is the only outcome cue before the outcome clause; see T4.
- The unique R² drop is 0 for main effects that are nested in included interactions (condition, class, template, family), so read their Type I shares, which depend on entry order.
- Scale has only 3 levels and is confounded with source.

---

## Not done / blocked

1. **The MOR4 story mapping is missing.** No decoder from `theStories` 1–48 to scenario text exists on the node (search list in T1.5).
   - **Needed:** the MOR4 experiment script or stimulus list, including the T/G definitions.
   - **Blocks:** keying the T1.5 table by `story_id`, the T2 correlation between human and model variance, and any per-scenario human-vs-model comparison.
2. **The MOR4 key scale is ambiguous.** The data contain 3 buttons (50–52, or alternates 85–87), which is consistent with YS2008's 1–3 scale. The work order's "key − 48 → 1–4" mapping was implemented as written. The button direction is inferred from the data.
3. **No per-item human data for YS2009 or YS2011,** so T3 claims no human reliability for those sources.
4. **`git push origin main` fails** ("could not read Username": no credentials on the node).
   - All commits listed at the top are on local `main` only.
   - The GPU jobs ran committed code from local `main`, not pushed code.
   - Fetching works. Remote `main` (`c0d7b0a`) is fully contained in local `main`, which is 53 commits ahead (including the Oct-5 robustness commit `00b470c`), so nothing upstream was missed.
   - `outputs/` is gitignored, so this report and the CSVs/figures are not committed.
5. **Closed models** were excluded from T2 (too few for per-story variance). Their sampling runs in T3 are deterministic, so their reliability values are uninformative.
6. **T5 and T7 ran on only 2 base/instruct pairs** (Qwen2.5-7B and Llama-3.1-8B). OLMo was not extracted for T5 or scored for T7.
7. **The Llama best layer in T5/T6 (L16) was assumed,** because there is no 02 probe run for Llama. Fixed-depth results are reported alongside it.
8. **UMAP** is not installed in `.venv`, so T1 clustering uses MDS only.
9. **The first T6 run used a non-identifiable interaction RDM.** It was fixed in `e499adf` and fully rerun; no uncorrected numbers are reported.
