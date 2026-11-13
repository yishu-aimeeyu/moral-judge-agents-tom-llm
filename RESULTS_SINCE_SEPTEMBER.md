# New results, September – November 2026

This is a handoff for a co-author joining the paper. It covers every analysis run since
1 September 2026, says where its numbers live, and says how each one changes what the paper can
claim. The older project README (`README.md`) still describes the pipeline and setup.

Snapshot date: 2026-11-12. No new analyses have run since 2026-10-19.

---

## 0. Where things are

| What | Path |
|---|---|
| Current manuscript (master version) | `paper_submission/main.tex`, `main.pdf`, `supplement.tex` |
| Venue cuts (CogSci 2027, CCN 2027 extended abstract, ICLR 2027 workshop) | `paper_submission/venues/*/` |
| Every manuscript claim mapped to the file that backs it | `paper_submission/CLAIM_EVIDENCE.md` |
| Argument map, research questions, required wording | `paper_submission/PAPER_FRAMEWORK.md` |
| Venue plan and checked requirements | `paper_submission/VENUE_ADAPTATION.md`, `paper_submission/venues/VENUE_REQUIREMENTS.md` |
| E8 pretraining-trajectory results | `outputs/rsa/E8_CHECKPOINT_EMERGENCE.md`, `outputs/rsa/E8G_OLMO_REPLICATION.md`, `outputs/rsa/E8_STEP0_GATE.md`, prespec `outputs/experiments/E8_PRESPEC.md` |
| October robustness plan and the 5 problems it found | `outputs/DECODING_ROBUSTNESS_PLAN.md` |
| Follow-up analyses T1–T7 (full write-up) | `outputs/lax_0929/REPORT.md` and the CSVs/figures beside it |
| Reliability and normalization (R1–R4) | `outputs/analysis/reliability/SUMMARY.md` |
| Status of older analyses (W0–W8A, E1–E7) | `outputs/RUN_STATUS.md`, `outputs/MASTER_SUMMARY.md` |

Scripts are in `code/experiments/`, numbered in run order. Everything from September onward is
`72`–`94`.

---

## 1. The project in one paragraph

We test whether language models use an agent's **intent** (belief) or the **outcome** when they
judge moral vignettes. The stimuli are the Young & Saxe belief × outcome 2×2 vignettes: 298 rows
from YS2008 (192), YS2009 (96) and YS2011 (10). These collapse to **53 scenario groups**, and all
grouped cross-validation and bootstraps use those groups, so the effective n is 53, not 298. The
four cells are *neutral* (innocent belief, no harm), *accidental* (innocent, harm), *attempted*
(guilty, no harm) and *intentional* (guilty, harm). Adults condemn attempted harm more than
accidental harm. The headline behavioural measure is the **attempted − accidental contrast** on
endpoint-normalized ratings in [0, 1]. It is positive for adults (+0.666 on the reference shape)
and positive but smaller for children. The open-weight roster has 20 models (9 base, 11
instruct, up to 14B; half are Qwen2.5). Open models are scored deterministically by logprob
expected value. Closed API models are sampled.

**State of the main claims at the start of September** (all still standing; details are in
`CLAIM_EVIDENCE.md`):
- **Open models.** 0/20 reproduce the adult ordering: 14 are inverted and 6 are tied. Instruct models are *more* outcome-driven than base models.
- **Explicit ToM vs. use.** Six engaged instruct models score 0.815–0.985 on hard BigToM false belief, yet their moral contrasts run from −0.370 to −0.646.
- **Intent is decodable but unused.** Intent decodes at 0.90–0.98 at the end of the belief clause. Neither item-level probe margins nor RSA predict behaviour.
- **Causal tests.** Intent steering moves the contrast by ≤ 0.016. An outcome-direction control moves it by 0.23–0.26. Outcome erasure recovers the contrast in 2/3 models. The E6/E7 controls rule out a pooled outcome-specific mechanism.
- **Post-training.** SFT accounts for most of the shift toward outcome in 3 public pipelines.

---

## 2. E8: when the structure emerges during pretraining (21–23 Sept)

**Question.** When does intent structure and outcome structure first appear in a model's
representational geometry during pretraining? Does it arrive gradually or in a jump?

**Design.** Pre-registered in `E8_PRESPEC.md` with three amendments, each committed before the
data it bears on. At each checkpoint, the 298×298 RDM is regressed on intent, outcome and nuisance
predictor RDMs (lexical, scenario, …). This gives a standardized β per factor, at 5 relative
depths × 2 read positions:
- `belief_last`: the end of the belief clause, before the outcome is stated.
- `last`: the end of the story.

Each cell gets 1000 permutations for its null and a cluster bootstrap over the 53 scenario
groups. Scripts: `72`–`80`.

### E8a–e: Pythia-1.4B, 26 checkpoints from step 0 to 143k

- **Step-0 gate passes.** In the randomly initialized network, 0/20 intent or outcome cells fall outside even the uncorrected 95% band. The RDM pipeline does not manufacture structure.
- **Each factor appears only at its own read position:**

  | | `belief_last` | `last` |
  |---|---|---|
  | intent emerges | step 3,000 | never |
  | intent β at step 143k | +0.104 | −0.007 |
  | outcome emerges | never | step 1,000 |
  | outcome β at step 143k | +0.000 | **+0.490** |

  The same pattern holds at every depth from 0.2 to 1.0, and with or without the lexical covariate.
- **Emergence is smooth, not a jump.** A sigmoid in log(step) beats a free-changepoint model in every informative cell (ΔAICc +16 to +42; R² 0.95–0.996). This is a continuous, unthresholded measure, so it avoids the threshold artifact behind many "emergence" claims (Schaeffer et al.).
- **The pre-registered predictions fail, and the write-up reports that openly:**
  - **P1 fails** ("outcome emerges before intent at the primary cell"). The primary cell was `belief_last` @ 0.8, where a causal LM cannot yet have read the outcome. This was a design error in the prespec, and the cell was deliberately not moved. The "run all cells" clause rescues the analysis.
  - **P2 fails.**
  - **P4 fails, in the smooth direction.**
- **Non-triviality check.** At `last`, outcome β rises 0 → 0.49 while intent stays flat for all 143k steps. This is not a case of "everything gets more structured".
- **E8f behavioural overlay is empty.** 0/26 checkpoints clear the engagement floor (`rating_std` ≥ 0.219; the maximum is 0.086). Pretrained-only Pythia never produces ratings with an estimable 2×2 contrast. Its geometry holds outcome structure from step 1k, but its behaviour has nothing to grade.

### E8g: OLMo-2-7B, 20 stage-1 revisions plus post-training stages

- **The Pythia result replicates** under the rule fixed in amendment 3.1. At step 928,646, the dissociation holds at 5/5 depths at `belief_last` and 4/5 at `last`; the rule required ≥ 3/5.
- **Onset is bracketed between 5B and 9B tokens** for both factors. Pythia's onset is about 2B tokens for outcome and about 6B for intent: the same order of magnitude.
- **Post-training stages, depth 0.8:**

  | stage | intent β at `belief_last` | outcome β at `last` |
  |---|---|---|
  | last stage-1 | +0.108 | +0.494 |
  | base | +0.080 | +0.402 |
  | SFT | +0.129 | +0.513 |
  | DPO | +0.131 | +0.499 |
  | Instruct | +0.131 | +0.497 |

  Both factors dip at `base` and recover at SFT; DPO and RLVR leave them unchanged. The off-target factor stays near 0 at every stage. Two things could explain the dip, and these data cannot separate them: the unsampled stage-2 annealing, or the float16/float32 seam between stage and revision activations.
- **Behaviour.** 0/20 stage-1 revisions are gradeable (maximum `rating_std` 0.062). **The first gradeable OLMo point is SFT.**
- **Missing control.** OLMo publishes no step-0 checkpoint, so this family has no step-0 gate.
- **Figures.** `outputs/rsa/checkpoint_trajectory_OLMo-2-1124-7B_joined.png`, `checkpoint_emergence_by_position.png`, and RDM snapshots (`rsm_snapshots_*`).

**What E8 adds to the paper.** Representational structure for both factors is in place early
in pretraining and grows smoothly. Gradeable moral behaviour does not appear until SFT.
Representation comes first and its use comes later. This is the "training-time bridge" in
`PAPER_FRAMEWORK.md`. It is descriptive geometry, not a causal claim.

---

## 3. October robustness pass: problems found in the existing pipeline (5 Oct)

`DECODING_ROBUSTNESS_PLAN.md` was written to check how much of the result is prompt wording. It
found five problems that matter for the paper:

| # | Problem | Status |
|---|---|---|
| P1 | `human_verbatim` is **not** what participants saw. YS2008 used item-specific stems on a 1 = Forbidden … 3 = Permissible scale, so the anchors are reversed relative to our prompt. YS2009 used item-specific blame questions. | New `participant_exact` template (script `84`). Results in §5. |
| P2 | YS2011 stories are in the second person, so the agent extractor returned "Imagine" and all 10 prompts asked about "Imagine's action". | `participant_exact` uses the published question. YS2011 is the least consistent source, which may be this bug. **The `human_verbatim` YS2011 rows are still affected.** |
| P3 | Qwen2.5 **base** tokenizers ship a chat template, so the Qwen base ladder was rated in chat format. Other base models got raw text. | New `--chat off` flag in `03_behavioral.py`. Qwen base ladder re-rated (§5). |
| P4 | Every YS2008/YS2009 story states the outcome-determining fact (the "foreshadow") **before** the belief. A code comment claimed YS2009 did not. | `no_foreshadow` stimulus variant. This is what the T4 result in §6 rests on. |
| P5 | Some belief spans in `clause_offsets.csv` start at the second belief sentence. | Stages are now taken from text diffs against the flipped partner (`stage_landmarks.py`). All 288 YS2008/9 stories are consistent. |

---

## 4. Decoding over time (scripts 81, 82, 92; T6)

Hidden states are saved at every token at 5 depths, for 3 base/instruct pairs (Qwen2.5-7B,
OLMo-2-7B, Llama-3.1-8B). Each story is cut into background / belief / action / outcome stages of
10 bins each, using GroupKFold by scenario and within-scenario shuffle nulls.

- **Intent** sits at exactly 0.50 before the belief clause; the text is identical there, so this is a clean null. It rises to **0.95–0.98** by the end of the belief clause, then **decays** to about 0.6–0.75 through the action and outcome clauses (late slope CI excludes 0). It recovers to 0.81–0.86 at the final token. TF-IDF on the same prefix stays ≤ 0.65.
- **Outcome** is already 0.85–0.90 at the foreshadow, drifts down, then jumps to about 1.0 in the outcome clause.
- **Base and instruct curves are nearly identical** in all three families. Whatever post-training changes about behaviour, it does not visibly change *when* these factors become decodable.
- **Temporal generalization.** Intent forms one block over the belief and early action bins. The outcome code from the foreshadow transfers only weakly to the outcome clause.
- **RDM over time.** Intent β ≈ 1.6 in the belief stage and 0.25–0.37 in the outcome stage. Outcome β is 1.46–1.57 in the background (foreshadow) and 1.86–1.92 in the outcome stage. The intent ∧ outcome conjunction term stays ≈ 0, so the condition geometry is **additive**. (The first run coded the interaction RDM so the betas were not identifiable; it was fixed in `e499adf` and fully rerun. Only corrected numbers exist.)

### Not yet in REPORT.md: harm expectation in the no-foreshadow stories (D3)

On `no_foreshadow` stimuli, the text before the outcome clause says nothing about the outcome. We
train the outcome decoder at the end of the story and apply it during the belief stage. The
statistic is P(harm | guilty belief) − P(harm | innocent belief), computed within the true
outcome. Values at the T6 best layers, from `outputs/decoding_time/*__no_foreshadow__story__chatoff__hypotheses.csv`:

| model | layer | harm expectation [95% CI] | TF-IDF surface |
|---|---|---|---|
| Qwen2.5-7B | 18 | +0.106 [0.077, 0.142] | +0.002 |
| Qwen2.5-7B-Instruct | 18 | +0.128 [0.096, 0.161] | +0.002 |
| OLMo-2-7B | 16 | +0.132 [0.105, 0.164] | +0.002 |
| OLMo-2-7B-Instruct | 16 | +0.114 [0.088, 0.145] | +0.002 |
| Llama-3.1-8B | 16 | +0.121 [0.092, 0.154] | +0.002 |
| Llama-3.1-8B-Instruct | 16 | +0.145 [0.111, 0.181] | +0.002 |

A guilty belief pushes the representation toward the "harm happened" side of the outcome code
before any outcome is stated, and surface text does not explain it. Base and instruct models are
similar. **Interpretation is open.** The obvious reading is that the model expects harm from bad
intent. An alternative is that the outcome direction is partly a general "badness" axis that
intent also loads on. The decoding pass itself is at chance (`outcome_in_belief` 0.498–0.502), as
it must be. This has not been written up or put in the manuscript.

### Not yet in REPORT.md: cross-prompt transfer at the answer token (D4)

Here the intent probe is trained at the final token of one prompt wording and tested on another.
The five wordings are `human_verbatim`, `participant_exact`, `blame_w1`, `wrong_w2` and
`punish_w1`. Values are means over the 5×5 matrix at the T6 best layer, from
`outputs/decoding_time/*__cross_prompt.csv`:

| model | same prompt | different prompt |
|---|---|---|
| Qwen2.5-7B / Instruct | 0.640 / 0.762 | 0.559 / 0.577 |
| OLMo-2-7B / Instruct | 0.696 / 0.842 | 0.559 / 0.702 |
| Llama-3.1-8B / Instruct | 0.677 / 0.866 | 0.542 / 0.699 |

At the answer position, intent is weakly present in base models and tied to the wording. Instruct
models carry more of it, and more of it transfers across wordings. This agrees with T5 below.

---

## 5. Prompt fidelity: participant-exact prompts and chat-off base models

### Participant-exact prompts (R5; script 84)

These are attempted − accidental contrasts on YS2008 + YS2009, endpoint-normalized. They come
from `outputs/behavior_participant_exact/raw_*.csv` and `outputs/behavior/raw_*.csv` and are not
yet in any write-up; this README computed them as mean(attempted) − mean(accidental):

| model | `human_verbatim` | `participant_exact` |
|---|---|---|
| Qwen2.5-0.5B-Instruct | −0.040 | −0.072 |
| Qwen2.5-7B | −0.077 | −0.093 |
| Qwen2.5-7B-Instruct | −0.281 | −0.308 |
| OLMo-2-7B | +0.002 | +0.000 |
| OLMo-2-7B-Instruct | −0.365 | −0.361 |
| Llama-3.1-8B | +0.004 | −0.002 |
| Llama-3.1-8B-Instruct | −0.116 | −0.078 |

**Giving models the exact question and scale that participants saw does not rescue the
contrast.** Every engaged model stays negative, and base models stay at about 0. P1 above is real,
but it does not drive the headline. These are point estimates with no CIs yet; the next step is
to rerun `83_reliability_normalization.py` so `participant_exact` joins R1–R4.

### Qwen base ladder with chat formatting off (P3)

Attempted − accidental averaged over 7 templates, chat auto → chat off, from
`outputs/lax_0929/normalization_robustness.csv`:

| model | chat auto | chat off |
|---|---|---|
| Qwen2.5-0.5B | +0.004 | +0.001 |
| Qwen2.5-1.5B | −0.013 | −0.001 |
| Qwen2.5-3B | −0.053 | −0.018 |
| Qwen2.5-7B | −0.060 | −0.045 |
| Qwen2.5-14B | −0.142 | −0.123 |

Chat formatting made the Qwen base models slightly more negative. Removing it shrinks their
contrasts toward 0 but keeps the ordering. The base-vs-instruct gap therefore gets slightly
*larger*, so the main claim is unaffected. The chat-off rows are kept out of the class means.

---

## 6. Follow-up work order T1–T7 (29 Sept – 19 Oct; scripts 86–94)

Full details, caveats and every CSV are in `outputs/lax_0929/REPORT.md`. The headline numbers:

**T1. Normalization.** The ratings come on 1–3, 1–4 and 1–7 scales. Four ways of putting them on
a common scale were compared: endpoint, global min-max, z-score within model × template × source,
and rank.
- The model ordering is stable: ρ ≥ 0.94 against endpoint.
- Instruct − base, by scheme:
  - endpoint: **−0.319** [−0.345, −0.291]
  - global min-max: −0.309
  - z-score: −0.898 SD
  - rank: −0.259
- All 9 base/instruct pairs are negative under endpoint and min-max. Qwen2.5-14B flips under z and rank.
- **Items cluster by outcome** (ARI 0.88), not by intent (≈ 0) or source (0.07). Source clusters only under `human_verbatim`, where source is confounded with scale.
- The YS2008 human scale and the 2007 adult anchor run in the **opposite direction** to our prompt. Results already account for this, but the methods section must say so.

**T1.5. MOR4 human scanner data.** This is a YS2008-design fMRI behavioural replication.
- 18 subjects, of whom 13 responded; 39.6% of trials are missing.
- Only 3 response keys were used, which matches the 1–3 scale.
- Attempted − accidental is +0.60 to +0.70 key units, about +0.30 to +0.35 in endpoint units.
- **Blocked:** there is no mapping from story numbers 1–48 to scenarios, so no per-item human comparison is possible.
- The trial-level and per-subject files are deliberately left out of this public repo (see §10).

**T2. Story variance.**
- Harm-outcome cells (accidental, intentional) have about **2×** the across-model variance of the no-harm cells.
- Story length predicts nothing (all p > 0.2).
- Variance across templates is about 5× smaller than variance across models.

**T3. Split-half reliability** (Spearman–Brown, after removing condition means):

| raters | YS2008 | YS2009 | YS2011 |
|---|---|---|---|
| instruct models | 0.83 | 0.84 | 0.93 |
| base models | 0.36 | 0.59 | 0.64 |
| templates within a model | 0.89 | 0.92 | 0.93 |
| MOR4 humans | 0.40 [0.10, 0.61] | — | — |

Instruct models agree with *each other* item by item more than humans do. ds000212 (OpenNeuro,
script 85) gives a human comparison on *different* items: neurotypical adults, within-condition
split-half 0.21–0.60. The closed-model sampling runs are deterministic, so their reliability of
1.0 is uninformative.

**T4. Outcome "decodable before the outcome."**
- At `belief_last`, outcome decodes at 0.70–0.87 across 14 models (TF-IDF 0.63; all p = 0.001).
- **Delete the foreshadow sentence and accuracy is exactly 0.50.** The prefixes of outcome partners are then byte-identical, so the 0.50 holds by construction.
- So "outcome at the intention stage" is a readout of the foreshadow, not an inference. It must never be written as "the model predicts an unseen outcome".
- This result is already in the abstract.

**T5. Intent probes across held-out templates** (Qwen2.5-7B ± I, Llama-3.1-8B ± I, last prompt token).
- Holding out a whole construct (blame / wrong / punish), peak accuracy is 0.755 / 0.825 / 0.639 / 0.814. That beats TF-IDF (≈ 0.60) for all four models.
- Transfer between single templates is weak for base models (0.56–0.57, below TF-IDF) and better for instruct models (0.58–0.75).
- Two pairs only. The Llama best layer was assumed.
- This result is already in the abstract.

**T6.** Decoding over time; see §4.

**T7. Priors and ablations** (2 pairs, 7 templates). Ablations: no story, background only, belief
removed, outcome removed.
- **Instruct models have stronger story-free priors** (prior strength +0.056 over base) and **far larger context sensitivity** (|full − no-story| 0.29 vs 0.06; difference +0.238 [0.219, 0.255]). Base models barely respond to the story.
- **Removing the belief doubles the instruct contrast's magnitude** (Qwen-I −0.24 → −0.45; Llama-I −0.20 → −0.47). Removing the outcome while keeping the foreshadow halves it.
- **Variance decomposition** (20 open models; R² 0.49):
  - The largest unique term is **condition × base/instruct (8.6%)**.
  - Type I shares: condition 17.3%, condition × class 11.0%, class 10.1%; template 2.2%, scale 0.1%.
  - Prompt wording is a small source of variance. What post-training does to the 2×2 is the large one.

**R1–R4** (`outputs/analysis/reliability/SUMMARY.md`; script 83) are the first pass of the same
checks over 29 models, including closed ones. They agree with T1 and T3. Variance shares per
model:
- Condition explains ≤ 0.5% of rating variance in 6 of 9 base models (Llama, Gemma, Mistral, OLMo, Qwen-0.5B, Qwen-1.5B). In those models, template and scenario explain the variance instead.
- The exceptions are the larger Qwen base models: 10% at 3B, 23% at 7B, 34% at 14B.
- In instruct models, condition explains 37–82%, except Qwen-0.5B-Instruct at 3%.

---

## 7. Closed-model behaviour (clean v3 rerun; data now in the repo)

The clean v3 rerun used 3 templates × 5 samples at T = 1, on the repaired stimuli with the fixed
parser. It finished at the end of July, but its raw data was never committed. It is now included
(`outputs/agents/behavior_v3/`). Values from `outputs/agents/stats/contrast_by_model.csv`:

| model | attempted − accidental [95% CI] |
|---|---|
| gpt-5.5 | **+0.528** [0.444, 0.610] |
| claude-opus-4-6 | +0.336 [0.236, 0.432] |
| claude-sonnet-4-6 | +0.211 [0.104, 0.324] |
| gemini-3.5-flash | +0.168 [0.081, 0.258] |
| gemini-2.5-flash | −0.029 [−0.118, 0.065] |
| claude-haiku-4-5 | −0.163 [−0.215, −0.106] |
| gpt-4o-mini | −0.237 [−0.279, −0.197] |
| gpt-4o | −0.269 [−0.354, −0.181] |
| gpt-5.4-mini | −0.694 [−0.759, −0.624] |

Four models are positive, four negative, and one includes 0. Gemini-2.5-Pro is partial and
excluded. **This is the main reason the paper cannot say "all LLMs fail".** The open-model
dissociation is strong, but closed frontier models are heterogeneous. No mechanistic access is
possible for them.

---

## 8. Paper assembly (9–19 Oct)

- **Master manuscript** in `paper_submission/`: "Beyond False-Belief Accuracy: …".
  - `scripts/build_assets.py` regenerates `generated/result_tables.tex` from the authoritative CSVs.
  - The `fig_T*` figures are copied from `outputs/lax_0929/`; the build script does not regenerate them.
- **Venue cuts:**
  - CogSci 2027: behaviour-first, "Passing False-Belief Tests Is Not Using Intent".
  - CCN 2027 extended abstract: mechanism-first.
  - ICLR 2027 workshop: "Readable but Unused".
- The planned order is NeurIPS 2027 full paper → journal → CogSci → CCN → ICLR workshop. Official deadlines and dual-submission rules are **not yet verified**, and `VENUE_GUIDELINES_VERIFIED.md` is an empty placeholder.
- **Uncommitted edits made during assembly, now committed:**
  - **W3/W4 framing** (`50_w3_summary.py`, `55_w4_summary.py`). The old text said intent was "unreachable by steering, reachable by prompting". That overclaimed. W4 clears its bar in only 3/6 models, its gains mostly come from accidental blame going down, and intent-led movement shows up in only 2/6. The new wording is "**entrenched**": resistant to representation-level intervention, and only partly and inconsistently responsive to instruction.
  - **Closed BigToM table** (`46_rebuild_closed_tom.py`). Runs that returned nothing (gemini-3.1-pro-preview: 400 attempted, 0 answered) now show as FAILED rows instead of disappearing.
  - **Reasoning-dose roster** (`52`, `59`). claude-opus-5 and o3 were dropped on cost (69% of the bill). As a result, no flagship reasoner is in the dose–response, which must be stated as a limitation. The run itself is **still not submitted**; it is waiting on API budget.

---

## 9. How the new results change what we can claim

**Strengthened**
- *Representation is present, use is not.* Intent is decodable early in pretraining (E8), stays decodable through every post-training stage (E8g), generalizes across held-out rating constructs (T5), and is cleanly time-locked to the belief clause (T6). Behaviour still goes the wrong way.
- *The headline is not an artifact of wording, scales or formatting.* It survives participant-exact prompts, 4 normalizations, chat-off base models, and a variance decomposition in which template and scale explain ≤ 2.2%.
- *Post-training is where the behaviour is decided.* Condition × base/instruct is the largest unique variance term. Instruct models are the context-sensitive ones (T7). Pretrained-only checkpoints have no gradeable behaviour (E8f/E8g), and the first gradeable point is SFT.

**Constrained (wording the draft must respect; also see `PAPER_FRAMEWORK.md`, "Required wording discipline")**
- Outcome information before the outcome clause comes from the foreshadow sentence (T4). Never write "anticipates" or "predicts the outcome".
- Emergence is smooth (E8 P4), so not "phase transition". The pre-registered P1 failed because of a prespec design error, and the paper should say so.
- Closed models are heterogeneous (§7), so not "LLMs fail".
- Steering vs. prompting: "entrenched", not "reachable by prompting" (§8).
- Human anchors are reference shapes, not item-matched. MOR4 cannot be item-matched until the story mapping is found.

**New and unexploited**
- D3 harm expectation (§4): a guilty belief shifts the outcome code toward harm with no outcome text. This could be a short subsection or a limitation on the outcome-erasure interpretation. It needs a control: train the outcome decoder only on intent-matched pairs, or compare the outcome direction with a generic-valence direction.
- D4 / T5: instruct models carry more transferable intent at the answer token, yet behave *worse*. This sharpens the "readable but unused" framing.

---

## 10. Open items and blockers

1. **MOR4 story mapping.** No decoder from story numbers 1–48 to YS2008 scenarios is available. It blocks per-item human comparisons, the human-vs-model variance correlation (T2), and an item-level noise ceiling. It needs the MOR4 experiment script from the Saxe lab.
2. **ds000212 item text** (Chakroff 2016 supplement or lab scripts). Without it, ds000212 can't be joined to the YS2011 items.
3. **Per-item matched adult ratings.** None are confirmed (`ADULT_ANCHOR_MATCH_AUDIT.md`). This is the most valuable missing human-data item.
4. **Reasoning dose–response.** The batch runner is built and validated (about $181 with the reduced roster) but has not been submitted.
5. **Gemini-2.5-Pro closed v3** is partial. `code/watch_google_n1_flash_pro.sh` exists to finish it at n = 1.
6. **Model coverage.** T5 and T7 have only 2 base/instruct pairs. OLMo has no T5 extraction and no step-0 gate. The Llama best layer was assumed.
7. **YS2011 "Imagine's action" bug (P2)** still affects `human_verbatim` YS2011 rows. This is only 10 rows, but footnote it.
8. **Manuscript metadata.** Authors, funding, ethics, data URL and venue deadlines are all unfilled.

---

## 11. What is in the repo and what isn't

`outputs/` is gitignored. The results listed in this README were added explicitly (`git add -f`):
the T1–T7 CSVs and figures, the reliability summary, the per-model decoding-time curves,
hypothesis, RSA and cross-prompt CSVs, the chat-off and participant-exact ratings, the closed v3
raw ratings, and the full `paper_submission/` folder.

**Deliberately left out, because this repository is public:**
- `outputs/lax_0929/T7/stimuli_*.csv` and `outputs/analysis/participant_exact_prompts.csv`. They contain Saxelab vignette text and item-specific question wording, which must not be redistributed (see `.gitignore`). Regenerate them locally with `93_lax_priors_ablation.py --build` and `84_participant_exact_prompts.py --audit`.
- `outputs/lax_0929/mor4_trials.csv`, `mor4_condition_means_by_subject.csv` and `mor4_missingness.csv`. These are trial- and subject-level unpublished lab scanner data. Only the aggregate condition means and key counts are included.
- `dataset/human_reference/children/raw/`. These are third-party OSF files; re-fetch them with `69_children_ingest.py --fetch`.
- Activations (`acts_ckpt/*.npz`, about 3.5 GB), temporal-generalization matrices (`*_tg.npz`), per-story decoding dumps (`*__per_story.csv`), per-model time-course PNGs and logs. These are large and can be regenerated.
- The `*_overleaf.zip` files in the root. They duplicate `paper_submission/` and `paper_submission/venues/`.

## 12. Re-running the new analyses

```bash
source .venv/bin/activate
# E8 (GPU for 73; the rest CPU)
python code/experiments/72_e8_predictors.py
python code/experiments/73_e8_checkpoint_extract.py   # see E8_PRESPEC for the checkpoint list
python code/experiments/74_e8_checkpoint_betas.py && python code/experiments/75_e8_shape_test.py
python code/experiments/76_e8_verdict.py && python code/experiments/77_e8_behavioral_overlay.py
python code/experiments/78_e8g_join_poststages.py && python code/experiments/79_e8g_verdict.py

# October robustness: trajectories + participant-exact + chat-off (GPU, Slurm)
SCORE_EXACT=1 bash code/submit_decoding_time.sh
python code/experiments/83_reliability_normalization.py
python code/experiments/85_ds000212_human_reliability.py

# T1–T7 (scripts 86–94; T5 and T7 need GPU)
python code/experiments/87_lax_normalization_audit.py
python code/experiments/88_lax_variance_reliability.py
python code/experiments/89_lax_outcome_from_intent_stage.py
python code/experiments/90_extract_template_last_token.py && python code/experiments/91_lax_template_heldout_decoding.py
python code/experiments/92_lax_time_decoding_clauses.py
python code/experiments/93_lax_priors_ablation.py --build && bash code/submit_lax_t7.sh && python code/experiments/93_lax_priors_ablation.py --analyze
python code/experiments/94_lax_t5_t6_figures.py

# Manuscript
cd paper_submission && python scripts/build_assets.py && latexmk -pdf main.tex
```

Each script's arguments are in its docstring. Commit messages from `d718ab7` to `e499adf` record
the decision behind each step.
