# Venue adaptation plan

Source: `Paper_submission_timeline.xlsx`. Official venue rules were checked on 9 Oct 2026; see `venues/VENUE_REQUIREMENTS.md`, which also corrects several spreadsheet entries (CogSci location and appendix rule, CCN track archival status, and the LEACE note). Formatted drafts are in `venues/`.

## Recommended sequence

1. **NeurIPS 2027 full paper:** strongest fit for the integrated behavioral, representational, intervention, and training-trajectory contribution.
2. **Rolling journal version:** Nature Communications, PNAS, or Nature Human Behaviour are plausible integrated targets, subject to scope and editorial fit.
3. **CogSci 2027:** behavior-first six-page adaptation.
4. **CCN 2027:** mechanism-first short or non-archival adaptation.
5. **ICLR workshop:** mechanism-first short adaptation.

The ICLR 2027 main-track opportunity shown in the spreadsheet has been skipped. The exact submission policies and dual-submission constraints require confirmation from the applicable calls.

## NeurIPS adaptation

Retain:
- full multi-level evaluation framework
- clean open/closed behavioral split
- BigToM-to-moral dissociation
- grouped decoding, RSA, and item-level linkage
- steering, erasure, and specificity controls
- Pythia/OLMo training trajectories
- audit and reproducibility details

Lead with:
- false-belief success does not guarantee applied intent use
- representation and behavioral application can dissociate
- mechanistic controls constrain stronger causal interpretations

Move to supplement if space requires:
- complete family-level tables
- all prompt analyses
- full prose/free-generation taxonomy
- extended cross-cultural context
- checkpoint-by-checkpoint numerical inventory

## CogSci adaptation

Center:
- the belief × outcome factorial design
- full four-cell profiles rather than a single interaction
- explicit versus applied ToM
- human developmental and cross-cultural context
- psychometric and stimulus-repair lessons

Compress:
- representation analyses into one summary figure
- steering and erasure into a brief converging-evidence subsection
- checkpoint trajectories into one paragraph or omit if required

Do not imply that the human references are item-matched norms.

## CCN adaptation

Center:
- grouped probe decoding
- item-level representation–behavior linkage
- RSA linkage
- intent/outcome steering controls
- LEACE erasure and layer specificity
- read-position-dependent emergence

Behavioral findings should establish the phenomenon, not consume the paper. State prominently that E1 is undecidable and E2 is underpowered.

## ICLR-workshop adaptation

Use a narrow mechanism question:

> When intent is linearly decodable from hidden states, does that information control the model's downstream moral judgment?

Retain the causal-control logic:
- intent steering
- outcome positive control
- outcome-prose placebo
- specificity audits
- erasure and label controls

Avoid presenting a workshop version as a comprehensive evaluation of all LLMs.

## Journal adaptation

The integrated journal version should retain the full chain:

1. capability
2. behavioral profile
3. robustness
4. representation
5. representation–behavior linkage
6. intervention
7. training-time emergence

Before submission, strengthen:
- archival provenance for clean closed-model data
- author/funding/competing-interest metadata
- item-matched human validation, if new data collection is authorized
- balance beyond the current open-model roster
- mechanistic access to newer model families, if available

The final target among Nature Communications, PNAS, and Nature Human Behaviour cannot be selected from the experiment evidence alone; editorial scope, article format, and current calls must be checked.

## Claims that must remain unchanged across venues

- Open and closed results are not inferentially pooled.
- No universal “all LLMs fail” claim is supported.
- Decodability is not equated with behavioral use.
- Internal representation claims apply only to open models.
- The cross-cultural result is “below all nine societies with detectable effects,” not “outside the human range.”
- The checkpoint analysis measures representational geometry; gradeable OLMo ratings begin only after post-training.
- The study does not adjudicate consciousness, genuine understanding, or a binary pattern-matching-versus-reasoning ontology.
