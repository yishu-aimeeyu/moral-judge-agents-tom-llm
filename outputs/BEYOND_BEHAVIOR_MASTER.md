# Beyond-behavior master index (RSA, probes, link, W3–W7, ToM)

**Repo:** [MisssAimeee/moral-judge-agents-tom-llm](https://github.com/MisssAimeee/moral-judge-agents-tom-llm)  
**This file on GitHub:** [`outputs/BEYOND_BEHAVIOR_MASTER.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/BEYOND_BEHAVIOR_MASTER.md)  
**Companion one-pager (includes behavior):** [`outputs/MENTOR_PACKET.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/MENTOR_PACKET.md)  
**Figure → CSV → script map:** [`outputs/figures_final/FIGURE_MANIFEST.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/FIGURE_MANIFEST.md)  
**Talk PNG home:** [`presentation_figures/`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures)  
**Limitations:** [`outputs/paper/LIMITATIONS.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/paper/LIMITATIONS.md)

> **Scope.** Everything *beyond* open/closed behavioral contrast forests: linear probes, gap-over-surface, RSA/CKA, representation↔behavior link, mixed-effects cells, checkpoint dissection, W3 steering + prose, W4 curriculum, W7 Bruneau, ToM vs contrast. Behavioral ladders/forests are only cited when they define an anchor for a non-behavioral claim.

> **Link convention.** Paths below are repo-relative. Click the GitHub blob/tree links. Prefer `presentation_figures/` PNGs for talks; prefer `outputs/figures_final/` + primary CSVs for analysis.

---

## One-paragraph claim (load-bearing)

Intent is **linearly decodable** from residual streams (peak intent ~0.67–0.88 last-token; belief_last often 0.85–0.98), yet **not used** for graded moral judgment: item-level probe↔contrast link is an informative null (slope +0.062, CI excludes +0.30), RSA geometry↔behavior is null (r=+0.098, n=28), residual-stream intent steering moves contrast ≤0.015 while outcome control moves 0.23–0.26, and models that *name* belief/intent in prose still rate by outcome. Post-training (SFT sufficient; locus recipe-dependent) induces the outcome bias. Prompt curriculum moves it only partially/inconsistently (3/6 clear L5). Passing BigToM false-belief coexists with inverted moral contrast.

---

## Navigation cheat sheet

| What you need | Start here |
|---|---|
| Full claim stack + tables | [`MENTOR_PACKET.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/MENTOR_PACKET.md) |
| This index (non-behavioral) | *you are here* |
| Talk slides (PNGs) | [`presentation_figures/README.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/README.md) |
| Regenerating figures | `code/experiments/53_figures_final_manifest.py`, `code/pack_presentation_figures.sh`, `code/60_talk_figures_aug6.py` |
| Claim constraints | [`LIMITATIONS.md`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/paper/LIMITATIONS.md) |

**Naming trap:** `outputs/figures_final/rsa_convergence_scatter.png` is the *old* model-level rep↔behavior scatter (n≈5). The RSA↔behavior pair test (n=28, r≈0.098) is **`rsa_vs_behavior_scatter.png`** under [`presentation_figures/mechanistic/`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures/mechanistic) and local `outputs/rsa/`.

**Do not use** anything under [`outputs/figures_final/_pending_rescore/`](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/figures_final/_pending_rescore) (STALE closed ladders).

---

## 1. Linear probing / layerwise curves

**Finding.** Intent and outcome are linearly decodable from last-token residuals. Peak **intent** CV accuracy ≈ **0.67–0.88** (small→7B); peak **outcome** ≈ **0.98–1.00**. Mentor-packet “0.85–0.98” refers mainly to **belief_last** position probes.

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/01_extract_activations.py`, `code/02_probe.py` | [01](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/01_extract_activations.py) · [02](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/02_probe.py) |
| Per-model CSVs | `outputs/probe/{Model}_probe.csv` (+ `_belief_last`, `_action_last`, `_mean`, permnull, withincell) | [probe/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/probe) |
| Figure (canonical) | `outputs/figures_final/layerwise_curves.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/layerwise_curves.png) |
| Talk copy | `presentation_figures/master_ladders/layerwise_curves.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/master_ladders/layerwise_curves.png) |
| Raw acts | `outputs/acts/` | **not on GitHub** (gitignored; regenerate locally) |

---

## 2. Gap-over-surface (probe − TF-IDF) + C2 source split

**Finding (supporting; downgraded).** Span-matched intent−outcome gap at `belief_last`: mean **+0.087** (8/8 models, sign test **p=0.008**). Does **not** support “intent early / outcome late.” C2: YS2008 vs YS2009 outcome at belief_last differs only **+0.046**; YS2009 still **~0.80** (not chance) → gap = semantic generalization over bag-of-words, not anticipation.

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/33_gap_dissociation_figure.py`, `44_span_matched_gaps.py`, `43_c2_source_split.py`, `21_surface_baseline.py` | [33](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/33_gap_dissociation_figure.py) · [44](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/44_span_matched_gaps.py) · [43](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/43_c2_source_split.py) · [21](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/21_surface_baseline.py) |
| Primary CSV | `outputs/probe/gap_over_surface_span_matched.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/probe/gap_over_surface_span_matched.csv) |
| Related CSVs | `gap_over_surface_within_model_paired.csv`, `gap_over_surface_by_pooling.csv`, `surface_baseline.csv`, `c2_source_split.csv` | [probe/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/probe) |
| Figure | `outputs/figures_final/gap_over_surface_span_matched.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/gap_over_surface_span_matched.png) |
| Talk copies | `presentation_figures/master_ladders/gap_over_surface_span_matched.png`, `presentation_figures/mechanistic/gap_over_surface_dissociation*.png` | [master](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/master_ladders/gap_over_surface_span_matched.png) · [mech](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures/mechanistic) |
| Writeups | `outputs/probe/C2_SOURCE_SPLIT.md`, `outputs/analysis/C2_SOURCE_SPLIT_BELIEF_LAST.md` | [C2](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/probe/C2_SOURCE_SPLIT.md) · [analysis](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/analysis/C2_SOURCE_SPLIT_BELIEF_LAST.md) |

---

## 3. RSA / CKA geometry + RSA ↔ behavior

**Finding.** Model-pair RSA Spearman vs |behavioral contrast difference|: **r = +0.098** [−0.31, +0.46], **n=28** pairs — **null** (“same answer, different route”). Within-family base↔instruct RDMs stay highly similar (e.g. OLMo RSA ~0.91) while behavior can diverge sharply.

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/24_rsa_cka.py`; talk redraw `code/60_talk_figures_aug6.py` | [24](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/24_rsa_cka.py) · [60](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/60_talk_figures_aug6.py) |
| Convergence test | `outputs/rsa/convergence_test.json` | [JSON](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/convergence_test.json) |
| Pair table | `outputs/rsa/convergence_pairs.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/convergence_pairs.csv) |
| Similarity matrix | `outputs/rsa/model_similarity.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/model_similarity.csv) |
| Other CSVs | `hypothesis_rdm.csv`, `base_vs_instruct_geometry.csv`, `rsa_permutation_null.csv` | [rsa/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/rsa) |
| Heatmap | `outputs/figures_final/rsa_similarity_heatmap.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/rsa_similarity_heatmap.png) |
| RSA↔behavior scatter (correct) | `presentation_figures/mechanistic/rsa_vs_behavior_scatter.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/mechanistic/rsa_vs_behavior_scatter.png) |
| Talk heatmaps | `presentation_figures/mechanistic/model_similarity_heatmap_{rsa_spearman,cka_linear}.png` | [mech](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures/mechanistic) |
| RDM `.npy` | `outputs/rsa/rdm_*.npy` | **not on GitHub** (regenerate via `24_rsa_cka.py`) |

---

## 4. Representation ↔ behavior (J2 item-level + model-level footnote)

**Finding.**
- **J2 (primary):** pooled slope **+0.062** [−0.146, +0.270]; excludes pre-stated meaningful **+0.30 SD**. Matched: **+0.029** [−0.165, +0.223].
- **Model-level (footnote only):** r ≈ **−0.209** [−0.80, +0.58] — uninformative (n≈8).

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/41_item_level_dissociation.py`, `code/04_link_analysis.py`, `code/experiments/23_build_intent_reliance_summary.py` | [41](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/41_item_level_dissociation.py) · [04](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/04_link_analysis.py) · [23](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/23_build_intent_reliance_summary.py) |
| J2 CSV | `outputs/link/item_level_dissociation.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/link/item_level_dissociation.csv) |
| Groups / model-level | `item_level_groups.csv`, `representation_vs_behavior.csv` | [link/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/link) |
| Figure J2 | `outputs/figures_final/item_level_dissociation.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/item_level_dissociation.png) |
| Model-level scatter | `outputs/link/rep_vs_behavior.png` · talk: `presentation_figures/mechanistic/rep_vs_behavior.png` | [link](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/link/rep_vs_behavior.png) · [talk](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/mechanistic/rep_vs_behavior.png) |
| Writeup | `outputs/link/ITEM_LEVEL_DISSOCIATION.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/link/ITEM_LEVEL_DISSOCIATION.md) |

---

## 5. Mixed-effects 2×2 interaction (J3)

**Finding.** **0/20** match human cell order (attempted > accidental); **14 inverted**, **6 tied**. Humans: b_intent=+0.90, b_outcome=+0.23, interaction=−0.20. Models: **18/20 outcome-saturating**. Negative interaction can mimic the human coefficient while cells are backwards.

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/39_mixed_effects_2x2.py` | [39](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/39_mixed_effects_2x2.py) |
| CSV | `outputs/stats/mixed_effects_2x2.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/stats/mixed_effects_2x2.csv) |
| Figure | `outputs/figures_final/interaction_forest.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/interaction_forest.png) |
| Talk | `presentation_figures/master_ladders/interaction_forest.png`, `presentation_figures/mechanistic/mixed_effects_interaction.png` | [forest](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/master_ladders/interaction_forest.png) · [ME](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/mechanistic/mixed_effects_interaction.png) |
| Writeup | `outputs/stats/MIXED_EFFECTS_2x2.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/stats/MIXED_EFFECTS_2x2.md) |

---

## 6. Checkpoint dissection (post-training locus)

**Finding.** SFT **sufficient** in OLMo / Tulu / Zephyr (base→SFT drops 0.15–0.59). SFT share: OLMo **85%**, Tulu **65%**, Zephyr **27%** (DPO **73%**). `b_outcome/b_intent` **3.0–7.7×** at non-base stages. All three bases now engaged (`rating_std` 0.03–0.10).

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/16_checkpoint_dissection.py`, `47_checkpoint_stage_shares.py` | [16](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/16_checkpoint_dissection.py) · [47](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/47_checkpoint_stage_shares.py) |
| CSVs | `outputs/experiments/checkpoint_dissection.csv`, `checkpoint_stage_shares.csv` | [dissection](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/checkpoint_dissection.csv) · [shares](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/checkpoint_stage_shares.csv) |
| Figure | `outputs/figures_final/checkpoint_dissection.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/checkpoint_dissection.png) |
| Talk | `presentation_figures/_headline/checkpoint_dissection.png`, `presentation_figures/master_ladders/checkpoint_dissection.png` | [headline](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/_headline/checkpoint_dissection.png) |
| Writeups | `CHECKPOINT_DISSECTION.md`, `CHECKPOINT_STAGE_SHARES.md` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |

---

## 7. W3 causal steering

**Finding.** Probe-weight intent steering: max |Δcontrast| **≤0.015** (OLMo 0.013, Qwen 0.016) while outcome control moves **0.232–0.259** (~15×). M1: margin displace **−3.18 / +7.21 SD**, decoding → chance, contrast flat. Diff-of-means “intent” effects = ceiling compression (all four cells up).

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/48_w3_causal_steering.py`, `50_w3_summary.py` | [48](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/48_w3_causal_steering.py) · [50](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/50_w3_summary.py) |
| Verdict MD | `outputs/experiments/W3_STEERING_SUMMARY.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W3_STEERING_SUMMARY.md) |
| Prespec | `outputs/experiments/W3_PRESPEC.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W3_PRESPEC.md) |
| CSVs | `w3_steering_*.csv`, `w3_layersweep_*.csv`, `w3_manipulation_*.csv`, `w3_calibration_*.csv` under `outputs/experiments/` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| Figures | `outputs/figures_final/w3_steering_dose_{OLMo,Qwen}.png`, `w3_layersweep_*.png`, `w3_manipulation_*.png` | [figures_final](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/figures_final) |
| Talk | `presentation_figures/w3_w4_w7/`, `presentation_figures/master_ladders/` | [w3_w4_w7](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures/w3_w4_w7) |

---

## 8. W3 prose / rating dissociation

**Finding.** Models name belief/intent in **~69–82%** of explanations (LLM coder primary) yet still rate by outcome in that subset (`b_outcome/b_intent` **5.7×–12.2×**; cell order inverted). Lexicon↔hand κ≈0.73–0.76; lexicon↔LLM κ≈0.42–0.49.

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/51_w3_prose_rating.py` | [51](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/51_w3_prose_rating.py) |
| Writeup | `outputs/experiments/W3_PROSE_RATING.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W3_PROSE_RATING.md) |
| CSVs | `w3_prose_items_*.csv`, `w3_prose_llm_labels_*.csv`, `w3_prose_manual_sample_*.csv` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| Figure | `outputs/figures_final/w3_prose_rating.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/w3_prose_rating.png) |
| Talk | `presentation_figures/master_ladders/w3_prose_rating.png`, `presentation_figures/w3_w4_w7/w3_prose_rating.png` | [master](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/master_ladders/w3_prose_rating.png) |

---

## 9. W4 prompt curriculum

**Finding.** **3/6** clear L5 bar (+0.15); need 4 → P3 fail. Stack gains mostly **accidental-blame suppression** (5/6). Intent-led movement **2/6**, only at principle-alone (L8). Principle-alone beats full stack in **3/6**; all sub-additive. Frame: bias **entrenched**, not “prompt-reachable.”

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/54_w4_prompt_curriculum.py`, `55_w4_summary.py` | [54](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/54_w4_prompt_curriculum.py) · [55](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/55_w4_summary.py) |
| Verdict | `outputs/experiments/W4_CURRICULUM.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W4_CURRICULUM.md) |
| Short summary | `outputs/experiments/W4_SUMMARY.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W4_SUMMARY.md) |
| Prespec / prompts | `W4_PRESPEC.md`, `W4_PROMPT_LEVELS.md` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| CSVs | `w4_prompt_curriculum.csv`, `w4_curriculum_cells.csv` | [curriculum](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/w4_prompt_curriculum.csv) |
| Figure | `outputs/figures_final/w4_curriculum.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/w4_curriculum.png) |
| Talk | `presentation_figures/{_headline,master_ladders,w3_w4_w7}/w4_curriculum.png` | [w3_w4_w7](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures/w3_w4_w7) |
| Bug archive | `outputs/experiments/_w4_prefix_fewshot_bug/` | [dir](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments/_w4_prefix_fewshot_bug) |

---

## 10. W7 Bruneau (appendix — harm selectivity)

**Finding.** Harm/outcome decoding high within physical / emotional / false-belief domains (often **0.9–1.0**). Cross-domain transfer substantial (Qwen within~0.95 / cross~0.77; OLMo 0.98 / 0.81) → largely **shared harm code**. Does **not** speak to intent-weighting (no attempt conditions).

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/56_w7_bruneau.py` | [56](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/56_w7_bruneau.py) |
| Writeup | `outputs/experiments/W7_BRUNEAU.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W7_BRUNEAU.md) |
| CSVs | `w7_bruneau_probes.csv`, `w7_bruneau_transfer.csv`, `w7_bruneau_interaction.csv` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| Figure | `outputs/figures_final/w7_bruneau_selectivity.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/w7_bruneau_selectivity.png) |
| Talk | `presentation_figures/{master_ladders,w3_w4_w7}/w7_bruneau_selectivity.png` | [w3_w4_w7](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/presentation_figures/w3_w4_w7) |
| Acts | `outputs/acts_bruneau/` | **not on GitHub** |

---

## 11. ToM vs moral contrast (J1) + closed BigToM

**Finding (open).** Engaged instruct models: BigToM FB (`init_belief=0`) **0.815–0.985** with moral contrast **−0.37 to −0.65**. Empty high-FB / positive-contrast cell. All-model *r* is a **confound** (cite table/scatter, not *r*). ToMi dropped (82% non-ToM in scored slice).

**Closed BigToM:** mostly **0.92–0.99** FB standalone. **Do not correlate** with v1-contaminated closed moral contrasts.

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/36_tom_benchmarks.py`, `42_tom_vs_contrast.py`, `45_tom_generative.py`, `46_rebuild_closed_tom.py` | [36](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/36_tom_benchmarks.py) · [42](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/42_tom_vs_contrast.py) · [45](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/45_tom_generative.py) · [46](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/46_rebuild_closed_tom.py) |
| Open joint CSV | `outputs/tom_benchmarks/tom_vs_contrast.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/tom_benchmarks/tom_vs_contrast.csv) |
| Accuracy tables | `tom_accuracy_by_model.csv`, `tom_accuracy_by_model_generative.csv` | [tom_benchmarks/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/tom_benchmarks) |
| Figure | `outputs/figures_final/tom_vs_contrast.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/figures_final/tom_vs_contrast.png) |
| Talk | `presentation_figures/{_headline,master_ladders}/tom_vs_contrast.png` | [headline](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/_headline/tom_vs_contrast.png) |
| Writeups | `TOM_VS_CONTRAST.md`, `CLOSED_TOM.md`, `TOMI_SCORING_AUDIT.md` | [tom_benchmarks/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/tom_benchmarks) |

---

## 12. Floors, prompt invariance, scoring parity (supporting)

| Analysis | Key number | Primary artifacts | GitHub |
|---|---|---|---|
| Engagement / effect floors (J4) | `rating_std` floor **0.2191**; effect floor **0.0988** | `outputs/stats/floor_derivation.csv`, `FLOOR_DERIVATION.md`, `floor_derivation.png` | [stats/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/stats) |
| Prompt invariance / factorial | **5/20** flip sign across 1–7 prompts; construct ≈0% variance | `outputs/analysis/prompt_invariance_decomposition.csv`, `prompt_factorial_*.csv`, `REVIEW_CHECKS.md` | [analysis/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/analysis) |
| Scoring parity EV vs sampling | 7B-I pass r>0.95 | `outputs/analysis/scoring_parity.csv`, `scoring_parity_writeup.md` | [analysis/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/analysis) |
| Base vs instruct pairs | Qwen ladder Δ ≈ −0.216, p≈0.020 | `outputs/analysis/base_vs_instruct_pairs.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/analysis/base_vs_instruct_pairs.csv) |
| Scale vs contrast | ρ ≈ −0.23, n.s. | `outputs/analysis/scale_vs_performance.csv` | [CSV](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/analysis/scale_vs_performance.csv) |

---

## 13. Human developmental anchors (context for non-behavioral claims)

**Finding.** Pre-spec primary = Naughty presented-first (youngest **+0.24**): **18/18** open non-degenerate models at/below. Punish secondary (+0.09): also **18/18**. Adult Young 2007 line kept with audit caveats (Exp.1 cannot anchor attempted−accidental from YS2011 alone).

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/29_dual_human_ladders.py`, `code/60_talk_figures_aug6.py` | [29](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/29_dual_human_ladders.py) · [60](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/60_talk_figures_aug6.py) |
| Open-only CSVs | `outputs/master_all_models_{digitized,punish,text_reported}_openonly.csv` | [outputs/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs) |
| Talk-safe ladder | `presentation_figures/master_ladders/master_developmental_ladder_talksafe.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/presentation_figures/master_ladders/master_developmental_ladder_talksafe.png) |
| Adult-anchor audit | `dataset/human_reference/ADULT_ANCHOR_MATCH_AUDIT.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/dataset/human_reference/ADULT_ANCHOR_MATCH_AUDIT.md) |
| Notes | `outputs/human_anchor_comparison.NOTES.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/human_anchor_comparison.NOTES.md) |

---

## 14. E1 layerwise hypothesis-RDM variance partitioning (methods note — undecidable as specified)

**Finding.** **No verdict is available, and that is the result.** The pre-registered gate P4 required a layer's lexical unique variance to be below the hypothesis RDMs' before its intent/outcome partition could be read; lexical distance dominates instead, so P4 excluded **36/36** gated (model, layer) cells under full-document TF-IDF *and* still **36/36** under the E5d span-matched control built on `44_span_matched_gaps.py` machinery. The correct write-up is a methods note: **surface-control gating is undefined when the experimental factors are lexically realised**, because intent and outcome are *stated in words* in these vignettes, so a lexical RDM is not a nuisance regressor but a partial re-encoding of the design. The 3–18× lexical dominance is labelled **post-hoc / M1** throughout and carries no confirmatory weight.

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/61_rdm_variance_partition.py` | [61](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/61_rdm_variance_partition.py) |
| Verdict MD | `outputs/rsa/E1_VARIANCE_PARTITION.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/E1_VARIANCE_PARTITION.md) |
| Prespec | `outputs/rsa/E1_PRESPEC.md` (amendment 1: `lexical_resid`; E5d: span-matched retry) | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/E1_PRESPEC.md) |
| CSVs | `outputs/rsa/e1_variance_partition*.csv` | [rsa/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/rsa) |
| Figures | `outputs/rsa/e1_variance_partition*.png` | [rsa/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/rsa) |

---

## 15. E2 checkpoint-stage RSA (underpowered; P1 withdrawn)

**Finding.** RDMs at base/SFT/DPO for OLMo and Tulu. The pre-registered coupling between cross-stage RDM shift and |Δcontrast| is **ρ = +0.870, n = 6, 95% bootstrap CI [+0.200, +1.000]** — a point estimate that looks strong on six observations, which is why the CI is quoted with it everywhere the number appears. Per prespec, **P1 is withdrawn** rather than claimed: the analysis is **underpowered in both directions**, and withdrawing P1 is not evidence for its converse. The split-half ceiling required the **Spearman-Brown** correction before cross-stage RSA could be compared against it.

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/62_checkpoint_stage_rsa.py` | [62](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/62_checkpoint_stage_rsa.py) |
| Verdict MD | `outputs/rsa/E2_CHECKPOINT_STAGE_RSA.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/E2_CHECKPOINT_STAGE_RSA.md) |
| Prespec | `outputs/rsa/E2_PRESPEC.md` (amendment: Spearman-Brown ceiling) | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/E2_PRESPEC.md) |
| CSVs | `outputs/rsa/e2_checkpoint_stage_*.csv` | [rsa/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/rsa) |
| Figure | `outputs/rsa/checkpoint_stage_rsa.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/rsa/checkpoint_stage_rsa.png) |

---

## 16. W8A (E3) amnesic outcome ablation — the manipulation result

**Finding.** Erasing the outcome direction with LEACE/INLP at the W3 peak layer **lets intent-sensitive judgment surface**: Δcontrast **+0.3182** (OLMo, L16), **+0.2312** (Qwen, L19), **+0.3480** (Tulu, L17), against a J4 floor of 0.0988 — the complement of W3, where *adding* intent signal moved nothing (≤0.015). The headline is **RECOVERY in 2 of 3 models**; Tulu meets the effect bar but fails the cell-spread rule (§18). Read **P2 before any Δcontrast**: the contrast is a diagonal, so erasing outcome moves it toward zero mechanically and only the pre-registered RECOVERY signature licenses the intent reading. **Renamed `W8` → `W8A`** so it cannot collide with "W8 — Paper assembly" (§`outputs/paper/`).

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/63_w8a_amnesic_outcome.py` | [63](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/63_w8a_amnesic_outcome.py) |
| Verdict MD | `outputs/experiments/W8A_AMNESIC.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W8A_AMNESIC.md) |
| Prespec | `outputs/experiments/W8A_PRESPEC.md` (rank calibration; V1a/V2 per-arm gating) | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/W8A_PRESPEC.md) |
| CSVs | `w8a_amnesic_*.csv`, `w8a_amnesic_verification_*.csv`, `w8a_amnesic_calibration_*.csv`, `w8a_amnesic_layer_selection_*.csv` under `outputs/experiments/` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| Figures | `outputs/experiments/w8a_amnesic_{OLMo-2-1124-7B-Instruct,Qwen2.5-7B-Instruct,Llama-3.1-Tulu-3-8B}.png` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |

---

## 17. E4 cross-cultural placement (Curtin et al. 2020, 10 societies)

**Finding.** Both pre-registered predictions **failed**, and the restatement is the usable result. P1: **4/6** engaged models (and 12/18 non-degenerate) fall below the lowest society, against a bar of all. P2: **9/10** societies show a positive intent effect with CI excluding zero, not 10/10 — **Yasawa is +0.0000, CI [−0.0265, +0.0227]**. So the claim may **not** be phrased "models sit outside the human range"; the defensible form is a distribution-overlap claim: *model intent effects fall below all 9 societies with a detectable intent effect (lowest +0.1016)*. Axis caveat: this is intent-at-constant-harm (`intentional − accidental`), **not** the project's `attempted − accidental` headline, which has no counterpart in that dataset.

| Role | Path | GitHub |
|---|---|---|
| Script | `code/experiments/64_crosscultural_placement.py` | [64](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/64_crosscultural_placement.py) |
| Verdict MD | `outputs/crosscultural/E4_CROSSCULTURAL.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/crosscultural/E4_CROSSCULTURAL.md) |
| Prespec | `dataset/human_reference/crosscultural/E4_PRESPEC.md` (amendment 1: population correction) | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/dataset/human_reference/crosscultural/E4_PRESPEC.md) |
| CSVs | `outputs/crosscultural/society_intent_effects.csv`, `model_placement.csv` | [crosscultural/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/crosscultural) |
| Method / caveats | `dataset/human_reference/crosscultural/README.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/dataset/human_reference/crosscultural/README.md) |

---

## 18. E5 controls on W8A — where the headline is bounded

**Finding, in the order that matters.** (a) **E5a's specificity control FAILED** in Qwen: a matched-norm perturbation along scenario identity moved the contrast **+0.1345**, past the 0.0988 floor, and the separation ratio fell to **1.72×** against a 2× bar (OLMo 10.26×, Tulu 4.31×). The honest consequence is that the outcome direction is the **most effective** direction found, not the **only** one. (b) **E5b's third model bounds the headline to 2 of 3**: Tulu moves **+0.3480**, the largest of the three, but retains only **38%** of its cell spread and is **COLLAPSE** under the pre-registered 50% rule — a substantive finding, not merely a failed gate (§19). (c) **E5c: OLMo's effect survives** cell-spread floors of 0.55 and 0.60 (largest surviving Δ **+0.2355**, n=3 doses each), so it is not an artefact of compression. (d) **E5d** fixed E2's reporting (n + bootstrap CI inline) and ran E1's span-matched retry.

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/65_e5_controls.py` (reporting), `63_w8a_amnesic_outcome.py --nuisance` (compute) | [65](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/65_e5_controls.py) · [63](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/63_w8a_amnesic_outcome.py) |
| Verdict MD | `outputs/experiments/E5_CONTROLS.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/E5_CONTROLS.md) |
| Prespec | `outputs/experiments/E5_PRESPEC.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/E5_PRESPEC.md) |
| CSVs | `w8a_nuisance_*.csv`, `w8a_nuisance_decodability_*.csv`, `w8a_spread_sensitivity.csv` under `outputs/experiments/` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| Figure | `outputs/experiments/w8a_nuisance_specificity.png` | [PNG](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/w8a_nuisance_specificity.png) |
| Content audit | `outputs/experiments/E5_CONTENT_AUDIT.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/E5_CONTENT_AUDIT.md) |

---

## 19. E6 orthogonalized specificity — a new control, not a re-run of E5a

**Finding.** **E5a's P1 verdict stands as a failure**; E6 is a differently-specified test reported beside it, and nothing here overturns it. E6 fixed two defects: E5a's norm-matched arms used `s > 1` (reflecting past the class mean, so the "erased" factor stayed decodable at 1.000 — perturbations, not erasures), and the nuisance directions were never orthogonalized against outcome. **E6b:** with Gram-Schmidt orthogonalization and `s ≤ 1` asserted, every nuisance arm is **inert in 3/3 models** — largest |Δcontrast| **0.0160** at any coherent dose — and separation rises to **9.9× / 22.3× / 24.6×** (OLMo/Qwen/Tulu) against E5a's 10.26× / 1.72× / 4.31×; **Qwen moves from 1.72× to 22.3×**. **E6a:** the geometric explanation was tested and **not supported** — Qwen's scenario–outcome overlap exceeds the others by only **+0.0389** against a pre-registered **0.10** bar (and the ordering reverses under the other estimator), exactly the "highest but by 0.04" outcome the margin was written to block. What that rules out is a *Qwen-specific* elevation in overlap, which is what a Qwen-specific failure would require; overlap is substantial in **all three** models in the chat-templated geometry the hook acts in (ρ = 0.44–0.66), and is small in all three in the raw-text rendering — a gap that is itself reported, since it means raw-text overlap statistics should not be used to predict what a chat-templated intervention will do. E5a's failure is therefore attributed to **over-projection (`s = 1.28`)** rather than to that model being geometrically unusual. E6b's own drive-to-chance **assertion aborted the first job** and exposed an error in E6's prespec (LEACE's projection is oblique, so the top singular vector of `I − P` does not erase the factor); amendment 1 records it, and E6a is reported under both estimators so the fix could not buy a verdict.

| Role | Path | GitHub |
|---|---|---|
| Scripts | `code/experiments/66_e6_controls.py` (reporting, E6a, and `--verify`). **The E6b GPU script (`67_e6b_orthogonalized.py`) was lost before it was ever committed** — see E6_CONTROLS.md §Provenance. E6b cannot currently be re-run from source; the CSVs plus `e6_directions_*.npz` are the record, and `--verify` re-derives the orthogonalization from the dumped vectors and reproduces every surviving fraction (all consistent). | [66](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/code/experiments/66_e6_controls.py) |
| Verdict MD | `outputs/experiments/E6_CONTROLS.md` | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/E6_CONTROLS.md) |
| Prespec | `outputs/experiments/E6_PRESPEC.md` (amendment 1: direction estimator; amendment 2: E6a-live) | [MD](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/blob/main/outputs/experiments/E6_PRESPEC.md) |
| CSVs | `e6_direction_overlap.csv`, `e6_orthogonalized_*.csv` under `outputs/experiments/` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |
| Figures | `outputs/experiments/e6_direction_overlap.png`, `e6_orthogonalized_specificity.png` | [experiments/](https://github.com/MisssAimeee/moral-judge-agents-tom-llm/tree/main/outputs/experiments) |

---

## 20. Incomplete / local-only (do not treat as finished)

| Item | Status | Notes |
|---|---|---|
| Closed reasoning dose | Incomplete; contrasts mostly NaN | `outputs/closed_reasoning/` often **untracked**; figure `reasoning_dose_response.png` missing |
| Gemini-2.5-pro closed behavioral | Stalled (~25%, quota/504) | Watchdog logs under `outputs/logs/` (local) |
| Raw activations / RDM `.npy` | Regenerable, gitignored | Run extract + `24_rsa_cka.py` / probe pipeline |

---

## Suggested next steps (for Claude / follow-on work)

1. **Paper assembly.** Pull claim stack from this file + `MENTOR_PACKET.md` into Results sections; enforce `LIMITATIONS.md` wording (especially adult anchor, ToM *r* confound, W4 frame).
2. **Closed behavioral v3.** Finish incomplete closed models; re-score STALE ladders; never join closed ToM to v1 moral contrasts.
3. **RSA scatter hygiene.** Prefer `rsa_vs_behavior_scatter.png` (n=28); retire or caption-rename `rsa_convergence_scatter.png` so it cannot be misread as RSA↔behavior.
4. **Optional extensions (only if pre-registered).** Threshold / non-linear J2; deeper W3 layers beyond peak; closed-model prose dissociation; finish reasoning-dose once API budget allows.
5. **Do not re-open** ToMi as a primary ToM measure without a new scored slice; do not upgrade gap-over-surface to “anticipation.”

---

## Quick table (analysis → number → CSV → PNG → MD)

| Analysis | Key number | Primary CSV/JSON | Preferred PNG | Summary MD |
|---|---|---|---|---|
| Layerwise probes | intent 0.67–0.88; outcome ~1.0 | `outputs/probe/*_probe.csv` | `figures_final/layerwise_curves.png` | mentor §4 |
| Gap / C2 | gap +0.087 | `gap_over_surface_span_matched.csv` | `figures_final/gap_over_surface_span_matched.png` | `C2_SOURCE_SPLIT.md` |
| RSA↔behavior | r=+0.098, n=28 | `rsa/convergence_test.json` | `presentation_figures/mechanistic/rsa_vs_behavior_scatter.png` | mentor §4 |
| RSA heatmap | within-family high | `rsa/model_similarity.csv` | `figures_final/rsa_similarity_heatmap.png` | — |
| J2 item link | slope +0.062 | `link/item_level_dissociation.csv` | `figures_final/item_level_dissociation.png` | `ITEM_LEVEL_DISSOCIATION.md` |
| Model-level link | r=−0.209 (footnote) | `link/representation_vs_behavior.csv` | `mechanistic/rep_vs_behavior.png` | same |
| Mixed-effects J3 | 0/20 human cell order | `stats/mixed_effects_2x2.csv` | `figures_final/interaction_forest.png` | `MIXED_EFFECTS_2x2.md` |
| Checkpoints | SFT 85/65/27% | `checkpoint_dissection.csv` | `figures_final/checkpoint_dissection.png` | `CHECKPOINT_*.md` |
| W3 steering | \|Δc\|≤0.015 vs 0.23–0.26 | `w3_steering_*.csv` | `figures_final/w3_steering_dose_*.png` | `W3_STEERING_SUMMARY.md` |
| W3 prose | name 69–82%; still outcome-led | `w3_prose_items_*.csv` | `figures_final/w3_prose_rating.png` | `W3_PROSE_RATING.md` |
| W4 curriculum | 3/6 clear L5 | `w4_prompt_curriculum.csv` | `figures_final/w4_curriculum.png` | `W4_CURRICULUM.md` |
| W7 Bruneau | shared harm code | `w7_bruneau_*.csv` | `figures_final/w7_bruneau_selectivity.png` | `W7_BRUNEAU.md` |
| ToM vs contrast | FB 0.82–0.99 & contrast −0.37…−0.65 | `tom_vs_contrast.csv` | `figures_final/tom_vs_contrast.png` | `TOM_VS_CONTRAST.md` |

---

*Generated as a navigation index for agents and collaborators. Numbers above match `MENTOR_PACKET.md` / experiment writeups as of the Aug 2026 talk pack; regenerate mentor packet via `code/experiments/49_mentor_packet.py` if artifacts change.*
