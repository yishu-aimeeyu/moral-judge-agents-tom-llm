# Claim–evidence map

This map links the manuscript's empirical claims to project-local sources. External related-work citations are audited separately in `LITERATURE_AUDIT.md`.

## Corpus and protocols

- **298 post-repair stimulus rows and 53 scenario groups.**  
  Sources: `outputs/paper/APPENDIX_PROVENANCE.md`; `outputs/paper/REPRODUCIBILITY.md`; `outputs/stats/mixed_effects_2x2.csv`.  
  Boundary: 53 groups, not 298 rows, are the effective bootstrap and grouping resolution.

- **Corpus composition: 192 Young–Saxe 2008 rows, 96 Young–Saxe 2009 rows, and ten Young–Saxe 2011 rows.**  
  Source: `outputs/paper/APPENDIX_PROVENANCE.md`.  
  Boundary: the 2011 rows do not form a complete four-cell factorial.

- **Open protocol: 20 models, largest 14B, ten Qwen2.5 variants, deterministic expected-value digit scoring.**  
  Sources: `outputs/master_all_models_digitized_openonly.csv`; `outputs/MENTOR_PACKET.md`; `outputs/paper/REPRODUCIBILITY.md`; `outputs/paper/LIMITATIONS.md`.

- **Closed clean v3 protocol: 298 rows, three prompts, five samples, temperature 1; nine complete models and Gemini-2.5-Pro excluded as partial.**  
  Sources: `outputs/RUN_STATUS.md`; `outputs/agents/stats/contrast_by_model.csv`.  
  Boundary: these local clean-v3 artifacts need archival provenance before submission.

- **Human values are descriptive reference shapes, not item-matched ground truth.**  
  Sources: `dataset/human_reference/ADULT_ANCHOR_MATCH_AUDIT.md`; `dataset/human_reference/methods_child_measure.md`; `outputs/paper/LIMITATIONS.md`.

## Behavioral profile

- **Open roster: 0/20 reproduce the adult-reference cell ordering; 14 are inverted and six tied.**  
  Sources: `outputs/stats/mixed_effects_2x2.csv` (`spec=primary`, `cell_order`); `outputs/BEYOND_BEHAVIOR_MASTER.md`.

- **Open attempted-minus-accidental contrasts and intervals.**  
  Source: `outputs/master_all_models_digitized_openonly.csv`.  
  The generated manuscript table is rebuilt directly from this file.

- **Closed clean v3: four positive intervals, four negative intervals, and one interval crossing zero.**  
  Source: `outputs/agents/stats/contrast_by_model.csv`.  
  Boundary: open and closed protocols are displayed separately and not inferentially pooled.

- **Adult reference cells 0.033, 0.267, 0.933, 0.967 and contrast +0.666.**  
  Sources: `outputs/stats/mixed_effects_2x2.csv` human row; `dataset/human_reference/ADULT_ANCHOR_MATCH_AUDIT.md`.

- **Digitized child wrongness contrasts +0.240, +0.500, and +0.630.**  
  Sources: `dataset/human_reference/human_reference_digitized.csv`; `dataset/human_reference/cushman_naughty_digitized.csv`; `dataset/human_reference/methods_child_measure.md`.  
  Boundary: approximate digitized two-cell proportions on a different measure.

- **All 18 non-degenerate open profiles are at or below the youngest wrongness and punishment references.**  
  Sources: `outputs/master_all_models_digitized_openonly.csv`; `outputs/master_all_models_punish_openonly.csv`.  
  Boundary: descriptive placement, not a developmental-age assignment.

- **Independent booklet anchor: accidental-wrongness blame 0.398, 0.142, and 0.010 across ages 4–5, 6–7, and 8+; first-session sensitivity at most 0.047.**  
  Sources: `dataset/human_reference/children/README.md`; `dataset/human_reference/children/children_child_anchor.csv`; `dataset/human_reference/children/children_child_anchor_first_session_only.csv`.  
  Boundary: booklet intent is intentional harm, so its contrast cannot replace attempted-minus-accidental.

- **Prompt robustness: 5/20 sign flips on common 1–7 prompts; source-scale agreement 0.71 with bias −0.06 and limits [−0.44,+0.33].**  
  Sources: `outputs/analysis/prompt_invariance_decomposition.csv`; `outputs/MENTOR_PACKET.md`.

## Explicit versus applied ToM

- **Six engaged open instruct models score 0.815–0.985 on hard BigToM false belief while their moral contrasts range from −0.370 to −0.646.**  
  Sources: `outputs/tom_benchmarks/tom_vs_contrast.csv`; `outputs/BEYOND_BEHAVIOR_MASTER.md`.

- **ToMi excluded because 82% of the scored slice was non-ToM.**  
  Sources: `outputs/tom_benchmarks/TOMI_SCORING_AUDIT.md`; `outputs/BEYOND_BEHAVIOR_MASTER.md`.

- **Instruct-only, adjusted, and within-family correlations are uninformative.**  
  Sources: `outputs/tom_benchmarks/TOM_VS_CONTRAST.md`; `outputs/BEYOND_BEHAVIOR_MASTER.md`.  
  Boundary: no raw all-model correlation is interpreted.

- **Closed generative BigToM inventory, including partial and failed runs.**  
  Source: `outputs/tom_benchmarks/CLOSED_TOM.md`.  
  Boundary: reported standalone; not pooled with open log-probability scores or post-hoc correlated with clean-v3 moral results.

## Post-training stages

- **SFT shifts: OLMo −0.5879, Tulu −0.2651, Zephyr −0.1487.**  
  Sources: `outputs/experiments/checkpoint_stage_shares.csv`; `outputs/experiments/checkpoint_dissection.csv`.

- **SFT shares 85.4%, 65.4%, and 27.0%; Zephyr DPO share 73.0%.**  
  Sources: `outputs/experiments/CHECKPOINT_STAGE_SHARES.md`; `outputs/experiments/checkpoint_stage_shares.csv`.

- **Outcome/intent coefficient ratio 3.0–7.7 across non-base stages.**  
  Sources: `outputs/experiments/checkpoint_dissection.csv`; `outputs/experiments/checkpoint_dissection_writeup.md`.

## Representations and links to behavior

- **Final-token intent decoding approximately 0.67–0.88 and outcome decoding approximately 0.98–1.00 across eight core checkpoints.**  
  Sources: `outputs/probe/*_probe_action_last.csv`; `outputs/BEYOND_BEHAVIOR_MASTER.md`.

- **Belief-end intent decoding 0.90–0.98; span-matched TF–IDF baseline 0.591.**  
  Sources: `outputs/probe/*_probe_belief_last.csv`; `outputs/probe/surface_baseline.csv`; `outputs/probe/gap_over_surface_span_matched.csv`; `outputs/BEYOND_BEHAVIOR_MASTER.md`.

- **Intent-minus-outcome probe/surface gap +0.0871 in 8/8 models, sign-test p=0.0078.**  
  Source: `outputs/probe/gap_over_surface_span_matched.csv`.  
  Boundary: setup text can contain hazard information, so this is not anticipation of unseen outcomes. Follow-up (`outputs/lax_0929/outcome_from_intent_stage_noforeshadow.csv`) localizes the pre-belief outcome information to the foreshadow sentence.

- **Item-level primary slope +0.0619 [−0.1463,+0.2701], matched slope +0.0291 [−0.1649,+0.2231].**  
  Sources: `outputs/link/ITEM_LEVEL_DISSOCIATION.md`; `outputs/link/item_level_dissociation.csv`.

- **RSA geometry–behavior link r=+0.098 [−0.31,+0.46], n=28 pairs.**  
  Sources: `outputs/rsa/convergence_test.json`; `outputs/MENTOR_PACKET.md`.

- **E1 variance partition excludes 36/36 designated cells under both lexical controls and is undecidable.**  
  Source: `outputs/rsa/E1_VARIANCE_PARTITION.md`.

- **E2 checkpoint-stage RSA rho=+0.870, n=6, interval [+0.200,+1.000], withdrawn for power.**  
  Source: `outputs/rsa/E2_CHECKPOINT_STAGE_RSA.md`.

## Prompting and interventions

- **Probe-weight intent steering changes the moral contrast by at most 0.0128 in OLMo and 0.0155 in Qwen; outcome control changes it by 0.2319 and 0.2592.**  
  Sources: `outputs/experiments/W3_STEERING_SUMMARY.md`; model-specific W3 reports in `outputs/experiments/`.

- **Manipulation check: downstream intent decoding goes from about 0.89 to 0.50 with probe-margin displacement −3.18 SD and +7.21 SD.**  
  Sources: `outputs/experiments/W3_STEERING_OLMo-2-1124-7B-Instruct.md`; `outputs/experiments/W3_STEERING_Qwen2.5-7B-Instruct.md`.

- **Unprompted explanations mention belief/intent in 81.5% of OLMo and 69.1% of Qwen stories; conditional outcome/intent ratios 5.7 and 12.2.**  
  Source: `outputs/experiments/W3_PROSE_RATING.md`.  
  Boundary: generated after the rating and not causal.

- **Prompt curriculum clears its top-level improvement criterion in only 3/6 models; five improvements primarily reduce accidental blame; intent-led movement 2/6.**  
  Sources: `outputs/experiments/W4_CURRICULUM.md`; `outputs/experiments/w4_prompt_curriculum.csv`.

- **Outcome erasure: OLMo +0.3182 with 0.5124 spread retention; Qwen +0.2312 with 0.6697; Tulu +0.3480 with 0.3753 and collapse.**  
  Sources: `outputs/experiments/W8A_AMNESIC.md`; `outputs/experiments/e6_orthogonalized_*_rerun_l40s.csv`.

- **E5a Qwen scenario control +0.1345, separation 1.72× versus required 2×.**  
  Source: `outputs/experiments/W8A_AMNESIC.md`.  
  Boundary: outcome is the most effective tested direction, not the only effective perturbation.

- **E6 orthogonalized nuisances inert with maximum absolute movement 0.0160 and separations 9.9×, 22.3×, 24.6×.**  
  Source: `outputs/experiments/E6_CONTROLS.md`.

- **E7b split: OLMo A-shaped, Qwen void, Tulu pre-registered falsifier with test arm 0.0941 and complement 0.1008.**  
  Source: `outputs/experiments/E7B_RESIDUAL_CONTROL.md`.  
  Boundary: no pooled outcome-specific mechanism.

## Training-time emergence and human-range context

- **Pythia trajectory: 26 checkpoints from step 0 through 143,000; no gradeable checkpoint.**  
  Source: `outputs/rsa/E8_CHECKPOINT_EMERGENCE.md`.

- **OLMo trajectory: 20 stage-1 revisions; no public random initialization; stage-2 annealing unsampled; no gradeable pretraining point; first gradeable point SFT.**  
  Source: `outputs/rsa/E8G_OLMO_REPLICATION.md`.

- **Smooth, read-position-dependent representational development rather than a supported sharp phase transition.**  
  Sources: `outputs/rsa/E8_CHECKPOINT_EMERGENCE.md`; `outputs/rsa/E8G_OLMO_REPLICATION.md`.  
  Boundary: coefficients are standardized RDM regressions, not activations or weights.

- **Cross-cultural placement: nine of ten societies have detectable positive effects; Yasawa is 0.0000 [−0.0265,+0.0227]; 6/6 engaged and 18/18 non-degenerate models lie below the nine detectable-effect societies, while 4/6 and 12/18 lie below the literal minimum.**  
  Source: `outputs/crosscultural/E4_CROSSCULTURAL.md`.  
  Boundary: this uses intentional-minus-accidental at constant harm, not the headline diagonal; “outside the human range” is unsupported.

## Follow-up analyses (scripts 86–94, `outputs/lax_0929/`)

All paths below are under `outputs/lax_0929/` (gitignored; scripts committed locally on `main`, e29f2f7..bb64669, not pushed). Summary prose: `REPORT.md`; every number was checked against the CSV named.

- **Normalization robustness: instruct − base contrast −0.319 [−0.345,−0.291] (endpoint), −0.309 [−0.334,−0.281] (global min–max), −0.898 SD [−1.007,−0.786] (z), −0.259 [−0.293,−0.225] (rank); 9/9, 9/9, 7/9, 8/9 pairs negative with CI excluding 0; Qwen2.5-14B reverses under z (+0.19 [0.03,0.37]) and rank (+0.05 [0.005,0.10]).**  
  Source: `normalization_base_vs_instruct.csv`; per-model: `normalization_robustness.csv`.  
  Boundary: nine pairs; Qwen chat-off reruns excluded from class means. The earlier −0.216 Qwen-only paired shift comes from a different analysis (`outputs/analysis/prompt_invariance_decomposition.csv` / `MENTOR_PACKET.md`).

- **Model ordering stable across normalizations: Spearman ρ 0.996 / 0.945 / 0.946 vs endpoint over 34 rater entries; 0 / 1 / 2 sign flips (OLMo-2-7B base; Gemini 2.5 Flash).**  
  Source: `normalization_ordering.csv`.

- **Scale directions: YS2008 human scale and 2007 adult anchor reversed relative to the model source-verbatim prompt; YS2009, YS2011 and factorial templates same direction.**  
  Source: `scale_audit.csv`.

- **Items cluster by outcome (silhouette 0.30, ARI 0.88), condition partly (ARI 0.39), not intent (0.01, ≈0) or source (0.01, 0.07); source clustering only with human_verbatim (0.21, 0.22). MDS only, no UMAP.**  
  Source: `clustering_silhouette.csv`; coordinates `clustering_mds_*.csv`.

- **Story variance: harm-outcome cells ~2× across-model variance of no-harm cells; across-template variance 2–5× smaller; length terms n.s. (all p>0.2 across models, >0.1 across templates); accidental/intentional +0.048 vs neutral, R²=0.73.**  
  Sources: `story_variance_by_source_condition.csv`; `story_variance_regression.csv`.  
  Boundary: human-vs-model variance comparison not run (no MOR4 story mapping).

- **Split-half reliability (within condition): instruct 0.83 [0.76,0.88] / 0.84 [0.77,0.90] / 0.93 [0.83,0.99]; base 0.36 [0.14,0.52] / 0.59 [0.41,0.72] / 0.64 [−0.23,0.94] (YS2008/2009/2011); template medians 0.89/0.92/0.93; YS2008 raw instruct 0.97 [0.95,0.98], base 0.59 [0.29,0.68].**  
  Sources: `split_half_reliability.csv`; `split_half_summary.csv`.  
  Boundary: closed-model sampling runs identical across samples (`frac_items_identical_across_samples`), so their ≈1.0 values are not reported as a result.

- **MOR4 humans: within-condition split-half 0.39 [0.10,0.61] (REPORT.md rounds to 0.40), raw 0.22 [−0.31,0.56], story×condition cells 0.69 [0.59,0.80].**  
  Source: `split_half_reliability.csv`.  
  Boundary: supplement only; story numbers unmapped.

- **MOR4 trial table: 18 subjects, 5 with no responses, 314/792 trials (39.6%) missing; three response keys used; condition means BF/OF 3.35/3.55, 3.04/3.15, 2.44/2.45, 2.03/2.11.**  
  Sources: `mor4_trials.csv`; `mor4_missingness.csv`; `mor4_key_distribution.csv`; `mor4_condition_means.csv`.  
  Boundary: T/G meaning and button direction inferred from data; 1–3 vs assumed 1–4 scale.

- **Outcome at belief endpoint: 0.70–0.87 at fixed depth 0.65 (14 checkpoints, 290 stories), peak up to 0.92; TF–IDF word 0.631, char 0.572; gap +0.07 to +0.24; all p=0.001; action endpoint 0.62–0.83; intent at same token/depth 0.85–0.96.**  
  Source: `outcome_from_intent_stage_summary.csv` (layers: `outcome_from_intent_stage_layers.csv`).  
  Note: the eight-core-checkpoint peak values (0.793–0.924) match the existing probe table; the existing 0.581 outcome baseline is on all 298 rows (`outputs/probe/surface_baseline.csv`).

- **Foreshadow deletion: prefix byte-identical between outcome partners in 100% of items; outcome accuracy 0.496–0.504 at every saved layer in 6 models; original-text peaks 0.71/0.80 (Llama), 0.77/0.81 (OLMo), 0.73/0.73 (Qwen); same-text TF–IDF 0.60/0.59.**  
  Source: `outcome_from_intent_stage_noforeshadow.csv`.  
  Boundary: chance is structural; no permutation test.

- **Template-/construct-held-out intent decoding: peak construct 0.755 / 0.825 / 0.639 / 0.814 (Qwen base / Qwen-I / Llama base / Llama-I); fixed-layer pair 0.568 / 0.580 / 0.560 / 0.706; TF–IDF word within/pair/construct 0.602/0.596/0.600; all p=0.001 except within−construct gap for Qwen base (p=0.15).**  
  Sources: `template_heldout_summary.csv`; `template_heldout_perm.csv`; `template_heldout_layers.csv`.  
  Boundary: two base/instruct pairs only; Llama best layer assumed.

- **Token-wise decoding: intent 0.95–0.98 at belief end, decline 0.16–0.21, final token 0.81–0.86 (REPORT.md says 0.84–0.86; CSV gives 0.807 Qwen base and 0.818 Llama base); outcome 0.87–0.90 at end of background, 0.99–1.00 in outcome clause; TF–IDF intent ≤0.64.**  
  Sources: `time_decoding_curves.csv`; `time_decoding_hypotheses_bestlayer.csv`; `time_decoding_n.csv` (280 stories).

- **RDM regression over time (corrected coding, e499adf): intent β 0.08–0.16 background, 1.57–1.69 belief; outcome β 1.46–1.57 background, 1.86–1.92 outcome stage; conjunction −0.12 to +0.08.**  
  Source: `time_decoding_rsa.csv` (stage means of per-bin betas at fixed layers).  
  Boundary: only corrected values reported.

- **Base/instruct behavioral contrast differences for the T6 pairs −0.18 to −0.69 (endpoint).**  
  Source: `normalization_base_vs_instruct.csv`.

- **Story ablations (2 pairs): |context sensitivity| no-story base 0.06 vs instruct 0.29, instruct − base +0.238 [0.219, 0.255]; full table of priors and sensitivities.**  
  Sources: `T7/ablation_summary_by_class.csv`; `T7/ablation_base_vs_instruct.csv`; contrasts `T7/ablation_contrasts.csv`.  
  Note: REPORT.md gives the upper CI as 0.26; CSV gives 0.2547.

- **Variance decomposition (20 open models): condition × base/instruct largest unique R² drop 8.6%; Type I shares condition 17.3%, condition × class 11.0%, class 10.1%.**  
  Source: `T7/variance_decomposition_full.csv`.

## Explicitly unavailable

- Authors, affiliations, corresponding-author details, funding, competing interests, ethics determination, acknowledgments, and archival data/code URL: **not specified in experiment data or text**.
- Official 2027 venue deadlines, templates, and dual-submission rules: **not verified**.
- Complete Gemini-2.5-Pro clean behavior, complete reasoning-dose runs, and a model-general outcome-specific mechanism: **insufficient information**.
