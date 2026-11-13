# Paper framework and argument map

## Core thesis

Passing explicit false-belief tests is insufficient evidence that a model reliably applies mental-state information in downstream moral judgment. The open-model results support a dissociation between representation and behavioral use; the clean closed-model reruns show that the behavioral pattern is heterogeneous rather than universal.

This is a claim about measured behavior and accessible representations—not consciousness, genuine understanding, or the absence of every possible ToM-like mechanism.

## Core research questions

1. Does explicit false-belief success generalize to intent-sensitive graded judgment?
2. Is intent/outcome weighting robust across families, sizes, prompts, scoring methods, and training stages?
3. Are intent and outcome represented internally, and do those representations predict behavior?
4. Does manipulating those representations alter judgment?
5. When do representational structure and gradeable behavior appear during training?

## Section-by-section logic

### Abstract
- Lead with the construct gap: explicit inference versus applied use.
- Name the multi-level design.
- Report the open and closed behavioral results separately.
- Give the two strongest representation/use statistics and the bounded intervention result.
- End with the narrow conclusion: representation and application can dissociate.

### Introduction
- Establish why benchmark accuracy cannot by itself distinguish reusable social inference from task-specific answer production.
- Separate capability, behavioral resemblance, and mechanism.
- Introduce moral belief × outcome judgment as a graded cognitive-science test.
- State the five research questions.
- Preview the non-universal result: strong open-model dissociation, heterogeneous closed-model behavior.

### Related work
- ToM benchmark successes: BigToM, ToMBench, PNAS and Nature Human Behaviour evaluations.
- Robustness concerns: trivial alterations and adversarial stress tests.
- Explicit/application gap: SimpleToM.
- Moral cognition: adult factorial, developmental shift, and cross-cultural variation.
- Internal representations and interventions: belief probes and LEACE.
- Emergence: continuous metrics versus threshold artifacts.

Logical transition: prior work motivates moving from “can answer” to “uses the information under controlled downstream demands.”

### Methods
- Explain the three-level analysis: capability, behavioral profile, representation/use.
- Document stimulus repair and effective sample size before model results.
- Keep open deterministic and closed sampled protocols separate.
- Define all four cells and clarify that attempted − accidental is a diagonal contrast.
- Describe human references as non-item-matched shapes.
- Explain grouped cross-validation and bootstrap units.
- Present probes, links, interventions, and checkpoint trajectories in the order needed to interpret causality.

Logical transition: each method rules out a narrower interpretation; no single method is treated as decisive.

### Results
1. Behavioral scope:
   - Open: 0/20 reference ordering, 14 inverted, six tied.
   - Closed: four positive, four negative, one interval crossing zero.
2. Explicit versus applied ToM:
   - Six engaged instruct models pass hard BigToM while retaining negative moral contrasts.
3. Post-training:
   - SFT sufficient in three pipelines; stage shares recipe-dependent.
4. Representation:
   - Intent and outcome decodable.
   - Item-level and RSA links do not support behavioral coupling at the specified scale.
   - E1 undecidable; E2 underpowered.
5. Intervention:
   - Intent steering small relative to outcome control.
   - Prose and prompt analyses provide behavioral support and bounds.
   - Outcome erasure recovers in 2/3; controls prevent an outcome-unique pooled claim.
6. Emergence:
   - Smooth, read-position-dependent geometry in Pythia and OLMo.
   - No gradeable pretraining ratings; first gradeable OLMo point is SFT.
7. Human-range qualification:
   - Models fall below nine societies with detectable effects, but not outside the full observed human range.

### Discussion
- Interpret false-belief success as capability evidence, not sufficient evidence of stable application.
- Treat decodability as availability, not use.
- Explain how pretraining geometry and post-training readout can coexist.
- State that closed-model positive behavior narrows the universal claim and motivates updated mechanistic access.
- Explain why the experiment cannot adjudicate a binary “pattern matching versus reasoning” ontology.
- Propose a four-layer evaluation standard: explicit inference, applied behavior, robustness, mechanism.

### Limitations
- 14B open-model ceiling; half the roster is Qwen.
- Human references are not item matched; child references cover two cells.
- Effective n = 53.
- Factors are lexically realized; no clean outcome-free belief position.
- Open/closed protocols differ.
- Internal evidence applies only to open models.
- Intervention and trajectory model counts are small.
- Clean closed data need archival provenance.
- Reasoning-dose and Gemini-2.5-Pro runs are incomplete.

### Conclusion
- Re-state the operational result, not a philosophical claim.
- Emphasize a test standard that distinguishes saying, behaving, representing, and using.

## Innovation points

1. **Triangulated construct validation.** Explicit benchmark performance, graded downstream behavior, internal representation, and intervention are evaluated as separate evidential levels.
2. **Representation-versus-use tests.** Decodability is paired with item-level linkage, RSA, positive-control steering, and erasure.
3. **Training-time bridge.** Continuous pretraining geometry is joined to public post-training stages and behavioral engagement.
4. **Cognitive-science factor decomposition.** Full four-cell profiles prevent a matching interaction coefficient from hiding reversed cell ordering.
5. **Audit as method.** Parser contamination, duplicate scenario groups, scale differences, and failed-measurement imputation are treated as scientific validity issues.
6. **Falsifier retention.** Failed controls and underpowered analyses remain visible and constrain the headline.

These are integration and procedure innovations. The draft does not claim priority over each component method.

## Research difficulties

- ToM is a construct, not a single observable.
- Human and model tasks are not psychometrically equivalent.
- Lexical controls can re-encode the manipulated factors.
- Model outputs and internal geometry require different uncertainty models.
- API-only models prevent mechanistic comparison.
- Checkpoint availability confounds family, recipe, and stage.
- Linear interventions have limited causal scope.
- Rapid model turnover can reverse a universal behavioral claim before mechanisms are replicated.

## Required wording discipline

Use:
- “behavior consistent with”
- “does not establish human-like ToM”
- “supports a dissociation between representation and behavioral use”
- “outcome was the most effective direction found”
- “recovery in two of three models”

Avoid:
- “all LLMs fail”
- “no model reaches an eight-year-old level”
- “outside the human range”
- “intent is unused everywhere”
- “only outcome matters”
- “SFT, not DPO, causes the bias”
- “the model predicts an unseen outcome”
