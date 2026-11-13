# Robustness plan: decoding over time, prompt fidelity, reliability

Written 2026-10-05. The goal is to separate what comes from the exact prompt wording from
what doesn't: model priors, base vs instruct, chat formatting, and per-story reliability.

## 0. Problems found while setting this up (fix before new claims)

| # | Problem | Consequence | Fix |
|---|---|---|---|
| P1 | `human_verbatim` is not what participants saw. YS2008 participants answered an item-specific stem ("Putting the substance in was:") on a 1 = Forbidden to 3 = Permissible scale. The model was asked "How permissible was X's action? 1 (completely permissible) to 3 (completely impermissible)": different wording and reversed anchors. YS2009 participants got item-specific questions ("How much blame does Matt deserve **for just watching as his cousin went into the kitchen**?", 1 = None to 4 = A Lot); the model got a generic one. | Model and human contrasts are not on the same instrument. | New `participant_exact` template (`84_participant_exact_prompts.py`). Audit: `outputs/analysis/participant_exact_prompts.csv`. |
| P2 | YS2011 stories are second person ("Imagine you..."), so `extract_agent()` returns "Imagine" and all 10 prompts ask about **"Imagine's action"**. | YS2011 is the least consistent source across models (R3), which may be this bug. | `participant_exact` uses the published "How morally wrong was the action?". |
| P3 | Qwen2.5 **base** tokenizers ship a chat template, so `03_behavioral.py` ran Qwen base models in chat format. OLMo, Llama, Gemma and Mistral base models got raw text. | The Qwen ladder, the main base-vs-instruct test, compares chat-formatted base models against instruct models. | `81 --chat off` for decoding. `03_behavioral.py --chat off --out_dir outputs/behavior_chatoff` (new flag; the default `auto` keeps old behaviour) re-rates the Qwen base ladder. |
| P4 | Every YS2008 **and** YS2009 story states the outcome-determining world state (foreshadow) **before** the belief. The comment in `submit_source_probes.sh` says YS2009 does not. | Outcome is "decodable at the belief clause" by construction, in both sources. Recheck the C2 source-split conclusion. | `no_foreshadow` stimulus variant (`stage_landmarks.py`). |
| P5 | `clause_offsets.csv` belief spans sometimes start at the second belief sentence (YS2009_01: intent text diverges ~65 characters before `belief_start`). | Pooling at the belief clause can miss part of the belief. | Stages now come from text diffs against flipped partners. All 288 YS2008/9 stories have a complete, consistently ordered set. |

## 1. Decoding study (new: D1–D4)

**Representations.** `81_extract_token_trajectories.py` saves hidden states at every token at 5
relative depths (0.25, 0.5, 0.65, 0.8, 1.0) for each prompt condition: bare story, story + one
of {`human_verbatim`, `participant_exact`, `blame_w1`, `wrong_w2`, `punish_w1`}, chat on or off,
original or `no_foreshadow`.

**Time axis.** Every story is cut into the same stages, `background | foreshadow | belief |
action | outcome | question`, and each stage is resampled to 6 bins. The boundaries come from
where each story differs from its intent-flipped and outcome-flipped partners.

**Decoders.** At each layer and bin, a logistic classifier is trained for intent and for
outcome. GroupKFold over `scenario_group` keeps all 4 cells of a scenario on the same side,
so train and test always use different stories. Each decoder is then applied to every other
bin, giving the temporal-generalization (RSM-like) matrix. Per bin there's also an RSA
regression: the distance between cells ~ intent difference + outcome difference.

| ID | Question | Readout (in `*__hypotheses.csv`) | Pass/fail logic |
|---|---|---|---|
| D1 | Does intent accuracy rise after the intention stage, then plateau? | `intent_rise`, `intent_late_mean`, `intent_late_slope` with a scenario-bootstrap CI | Plateau = slope CI includes 0. Sanity check: `intent_pre_onset` must be 0.50 (identical text). |
| D1b | Is intent flat through the outcome stage? | the same slope over the action and outcome bins, plus the temporal-generalization row trained at the end of the belief stage | A square block means a stable code; a fading or below-chance block means the code is rewritten when the outcome arrives. |
| D1c | Is outcome flat through the intention stage? | `outcome_intention_stage_slope` over the belief and action bins (original stimuli; the foreshadow has already given the outcome) | CI includes 0 = the foreshadowed outcome is held while intent is read; negative = intent overwrites it. |
| D2 | Can outcome be decoded from the intention stage? Is there a dissociation per story? | `outcome_in_belief`; `per_story.csv` gives P(correct) per story and stage | Original stimuli: high **by construction** (foreshadow). No-foreshadow: must be 0.50 by construction. |
| D3 | Does the model have priors that aren't in the prompt? | `harm_expectation`: the outcome decoder trained at the end of the story, applied during the belief stage, P(harm \| guilty) − P(harm \| innocent) within the true outcome, against `surface_harm_expectation` (TF-IDF) | In no-foreshadow the text says nothing about outcome, so model > surface means the model expects harm from bad intent. |
| D4 | How much of the intent code is prompt-specific? | `--cross`: train on prompt A's final token, test on prompt B's | Diagonal − off-diagonal = intent information tied to the wording. Compare base vs instruct. |

**Smoke test (Qwen2.5-0.5B, all 288 stories; `outputs/decoding_time/smoke_Qwen2.5-0.5B/`):**
- Intent before the belief stage: 0.50 at every layer. The null works.
- Intent peaks around 0.8 during the belief stage and decays to about 0.6. The slope over the action and outcome bins is −0.009 to −0.012 per bin (CI excludes 0), so it decays rather than plateauing.
- No-foreshadow outcome during the belief stage: 0.500.
- Harm expectation in no-foreshadow: +0.20 (CI 0.15–0.29) against +0.002 for the TF-IDF surface control.
- Cross-prompt intent at the final token: 0.51–0.59. Intent is barely present at the answer position and doesn't transfer across wordings.

These are 0.5B numbers and only validate the pipeline; the scientific runs are 7–8B base/instruct pairs.

## 2. Behavioural robustness (R1–R5)

`83_reliability_normalization.py` (CPU, existing ratings) writes `outputs/analysis/reliability/SUMMARY.md`.

| ID | Check | First result (29 models; Kimi-K3, Claude-3.5-Haiku and the partial Gemini-2.5-Pro run dropped as constant or partial) |
|---|---|---|
| R1 | Global normalization: endpoint vs observed min-max vs z (global, within source) vs rank | Model ordering of the contrast is preserved (Spearman 0.93–0.98). The only sign flips are models with a contrast of about 0 (OLMo-2-7B base +0.0008, Gemini-2.5-Flash +0.003). |
| R2 | Split-half reliability per source; raters = templates, models, or humans; raw and within-condition | Templates within a model: 0.89–0.95 within-condition. Instruct models agree with each other (0.83–0.90 within-condition); base models don't (0.34–0.56). |
| R3 | Per-story consistency, sorted | Inconsistency is concentrated in YS2011 (see P2) and in accidental items. |
| R4 | Clustering and variance shares | Stories cluster by **outcome** (ARI 0.62), not intent (0.02). Base models: condition 0.3% of variance, template 49%, scenario 16%. Instruct models: condition 56%. |
| R5 | `participant_exact` re-rating | Qwen2.5-0.5B-Instruct: changes the YS2008 level and contrast (−0.04 → −0.11). Run on every model. |

**Human reliability.** No per-participant ratings for YS2008/YS2009 are public: the papers
report cell means only (n = 10–17). The nearest open data is OpenNeuro
[ds000212](https://openneuro.org/datasets/ds000212) (Young, Chakroff, Saxe, Koster-Hale et al.;
CC0): 39 adults (NT and ASD), 60 Young & Saxe (2011)-style intentional/accidental scenarios
rated 1–4, with a per-trial key press. The item text isn't in the dataset, so it can't be joined
to master story IDs yet; it would need the experiment script or the Chakroff (2016) supplement.
`85_ds000212_human_reliability.py` gives the human benchmark: NT (n = 22) within-condition
split-half is physical harm 0.60, psychological harm 0.60, incest 0.42, pathogen 0.21 (raw:
0.72–0.95). On the same within-condition measure, instruct models agree with each other on
our items at 0.83–0.90. These are different stimuli, so the comparison is only indicative.

The variance shares answer part of the base-model question: base models do respond to the
story (scenario), but not to the 2×2 manipulation. That's the "no moral judgment" pattern, not
outcome-based judgment.

## 3. Run order

```bash
# CPU, now
python code/experiments/84_participant_exact_prompts.py --audit
python code/experiments/83_reliability_normalization.py

# GPU + CPU chain (6 models; ~15 GB scratch per model)
SCORE_EXACT=1 bash code/submit_decoding_time.sh
# afterwards: re-run 83 so participant_exact joins R1-R4
```

## 4. Submitted 2026-10-05
- Extraction + participant-exact rescoring (GPU, one job per model): 24894342 (Qwen2.5-7B), 24894353 (Qwen2.5-7B-Instruct), 24894363 (OLMo-2-7B), 24894373 (OLMo-2-7B-Instruct), 24894383 (Llama-3.1-8B), 24894394 (Llama-3.1-8B-Instruct). 9 CPU decoding jobs are chained behind each.
- Qwen base ladder re-rated with chat off: 24894404, writing to `outputs/behavior_chatoff/`.
- After they finish: rerun `83_reliability_normalization.py` (adds participant_exact), then compare `behavior_chatoff` with `behavior` for the Qwen base models.

## 5. Open items
1. Item text for ds000212 items 1–60 (from Saxe/Young lab scripts), so human item means can be joined to the YS2011 master items and used as a noise ceiling.
2. Pythia isn't in the HF cache; add it to `MODELS=` in `code/submit_decoding_time.sh` if wanted.
