# E7 pre-registration

Written before any E7b compute. Committed on its own commit and tagged `prespec-e7`. Amendments
are appended to this file as further commits and never rewrite text above; `make check-prespec`
enforces that against git history.

Every threshold below is inherited from earlier prespecs rather than chosen now. Nothing in this
document is new except the direction being erased and the decision table.

---

## E7b — the symmetric residual control

### The question, and why it is the last one the mechanism thread needs

W8A found that erasing the outcome direction at the W3 site recovers the attempted−accidental
contrast: 2 of 3 models cross the J4 floor. E6b then asked the mirror-image question and found
that erasing nuisance directions *orthogonalized against outcome and intent* is inert in 3 of 3,
which rules out "any perturbation of this size moves the readout".

E6a-live leaves one live alternative that neither test addresses. Under chat templating the
outcome direction places **0.44–0.66 of its variance inside the k=10 scenario subspace**. So the
W8A result is consistent with two different mechanisms, and they have different implications:

* **(A) Outcome-carried.** The recovery comes from the outcome-specific component. The scenario
  overlap is incidental — outcome and scenario are correlated because harm scenarios differ from
  no-harm scenarios in content.
* **(B) Scenario-carried.** The recovery comes from the scenario-aligned component. In that case
  "erasing outcome" is partly erasing scenario identity, and the intervention is less specific
  than the W8A framing claims.

E7b separates them by erasing **the component of the outcome direction orthogonal to scenario
identity** and asking whether the recovery survives. It is the symmetric partner of E6b: E6b
removed outcome and intent from the nuisance directions, E7b removes scenario from the outcome
direction.

**Either result is publishable and the mechanism thread closes after this regardless of which one
we get.** (A) sharpens W8A's claim; (B) bounds it, and the bound is the more interesting finding.

### Construction (fixed here, not at run time)

1. **Site.** The W3 peak layers, inherited unchanged: OLMo-2-7B-Instruct L16, Qwen2.5-7B-Instruct
   L19, Llama-3.1-Tulu-3-8B L17. No re-selection of depth.
2. **Directions.** Class-mean difference (`meandiff`), per E6_PRESPEC amendment 1, on the
   chat-templated last-token residuals. `w_outcome` = mean(harm) − mean(no_harm), unit norm.
3. **Scenario subspace.** `S_k` = top **k = 10** principal components of the globally-centred
   per-scenario-group mean matrix (53 groups). k = 10 is E6a's primary k, inherited.
4. **The erased direction.** `w_out_perp = w_outcome − Proj_{S_k}(w_outcome)`, renormalised to
   unit length. Its surviving fraction `||w_out_perp|| / ||w_outcome||` is reported; from
   E6a-live's variance fractions this is expected in the range 0.58–0.75, and the observed value
   is recorded whatever it is.
5. **Erasure.** The same rank-1 directional projection E6b used, `h -> h − s((h−mu).n)n`, with
   **s ≤ 1 enforced by assertion**. s > 1 is the E5a over-projection bug and is refused.
6. **Matched norm.** Every arm is compared at equal displacement `mean(||Δh||/||h||)`, with the
   comparison arm scaled DOWN to the smaller displacement, never up.
7. **Matched coherence.** An arm counts only if `ppl / ppl_baseline ≤ 1.5`. Where the pre-specified
   dose breaks that, the largest `s ∈ {0.25, 0.50, 0.75, 1.00}` that satisfies it is used, and the
   substitution is reported per model.

### Arms

| arm | direction | purpose |
|---|---|---|
| `outcome_perp_scenario` | `w_out_perp`, s ∈ {0.25, 0.50, 0.75, 1.00} | the test |
| `outcome_full_dirmatched` | `w_outcome`, scaled to `outcome_perp_scenario`'s displacement | **positive control** |
| `outcome_in_scenario` | `Proj_{S_k}(w_outcome)`, renormalised, matched displacement | the complement: what (B) predicts carries the effect |
| `random_matched` | random unit direction, matched displacement, seeds 0–4 | null |

### Thresholds — all inherited, none new

| quantity | threshold | source |
|---|---|---|
| effect floor on \|Δcontrast\| | **0.0988** | q95 of the within-scenario permutation null, `floor_derivation.csv` |
| perplexity ratio | ≤ 1.5 | W8A_PRESPEC V3 |
| cell-SD retention | ≥ 0.50 | W8A_PRESPEC P2 |
| erased-factor in-sample accuracy | within 0.05 of majority | W8A_PRESPEC V1a |
| scenario subspace k | 10 | E6a primary |

### The decision table, fixed in advance

Read only after the positive control is checked. `Δc` is Δcontrast at the matched, coherent dose.

| positive control (`outcome_full_dirmatched`) | `outcome_perp_scenario` | conclusion |
|---|---|---|
| clears 0.0988 | clears 0.0988 in **3 of 3** | **(A)** The recovery is carried by the outcome-specific component. The scenario overlap is incidental and W8A's framing stands as written. |
| clears 0.0988 | clears in 1–2 of 3 | Model-dependent. Reported as split, with no pooled claim; the mechanism is not established in the models that fail. |
| clears 0.0988 | clears in **0 of 3** | **(B)** The recovery requires the scenario-aligned component. W8A's intervention is not outcome-specific, and this bound goes in LIMITATIONS.md and the abstract. |
| **fails** 0.0988 | anything | **The test is void, not negative.** The matched displacement is too small to move behaviour at all, so the comparison carries no information about (A) vs (B). This row is written down in advance precisely so a null cannot be read as support for (A). |

`random_matched` exceeding 0.0988 in any model also voids that model's test.

### Time-box and scope

One day of compute. Runs only after E7a establishes that the E6b readout is reproducible from a
committed script. No new claims are added to any existing document until the decision table above
selects a row.

### What would falsify the reading we expect

We expect (A). It is falsified by `outcome_perp_scenario` failing the floor while
`outcome_full_dirmatched` and `outcome_in_scenario` clear it in the same model at the same
displacement. That combination is recorded as (B) and reported as such, and this sentence exists
so that outcome cannot later be described as a surprise.

---

## Amendment 1 — V1a is reported, not gated (appended before any E7b compute)

The threshold table above inherits W8A_PRESPEC V1a: the erased factor's in-sample accuracy must
fall within 0.05 of majority. That gate was written for a **full** erasure, where the projection is
fitted to annihilate the factor and failing to reach chance means the erasure did not work.

Every E7b arm is instead a **rank-1 directional projection at a matched norm**, and at s < 1 it is
deliberately a partial displacement. Such an arm cannot drive outcome decoding to chance, and is
not intended to: the matched-norm design is what makes the test arm, the positive control, the
complement and the random nulls comparable to each other. Gating on V1a would therefore fail all
four arms mechanically, including the positive control, and void the experiment by construction
rather than by evidence.

So for E7b, in-sample outcome accuracy is **recorded per arm** (`insample_outcome_acc` in the
CSVs) and is **not** used as a pass/fail gate. The gates that do decide E7b are unchanged and
unrelaxed: the 0.0988 effect floor, the ≤ 1.5 perplexity ratio, the ≥ 0.50 cell-SD retention, and
the equal-displacement matching in item 6. The decision table is untouched.

This is written down before compute so that it cannot be mistaken for a gate dropped after seeing
that an arm failed it. The `s ≤ 1` assertion in item 5 remains live and hard.

---

## Amendment 2 — the matching in item 6 was degenerate, and the first run was aborted because of it

**This amendment is not blind.** A first E7b run (job 20259085, L40S, commit 9aea88f) was started,
produced output for OLMo-2-7B-Instruct, and was cancelled. Its log is kept at
`outputs/logs/e7b_aborted_run_20259085.log`. What was seen before cancelling is disclosed in full
below. Writing this amendment after seeing those numbers is worse than writing it before, and the
disclosure is here so the reader can discount it accordingly rather than take it on trust.

### The defect

Item 6 says arms are compared at equal displacement with the comparison arm "scaled DOWN to the
smaller displacement, never up". The implementation read that as a single global target: the minimum
`s = 1` displacement over **all** arms, random nulls included. Measured on OLMo, those displacements
are

| arm | displacement at s = 1 |
|---|---:|
| `outcome_full_dirmatched` | 0.0930 |
| `outcome_perp_scenario` (the test) | 0.0704 |
| `outcome_in_scenario` | 0.0666 |
| `random_matched`, seeds 0–4 | 0.0027 – 0.0041 |

A random unit direction in a 4096-dimensional residual stream is very nearly orthogonal to the
directions the data actually varies along, so projecting it out barely moves anything. Taking the
global minimum therefore pins **every** arm at 0.0027 — a 26-fold reduction for the test arm — and
at that dose nothing moves, the positive control included. The void row would then fire in all three
models, not because the matched displacement is uninformative about (A) versus (B), but because the
implementation chose a displacement no arm can be informative at. That is a defect in the code, not
a result, and reporting it as a null would be reporting an artefact.

### The fix

1. **Matching becomes pairwise**, which is what E6b's inherited machinery already did (E6b scaled the
   outcome arm down to *each* nuisance arm's own `s = 1` displacement, one comparison at a time).
   For every comparison arm `c`, both the test arm and `c` are emitted at
   `m = min(d_test, d_c)`, so each pair is at equal displacement and neither member is ever scaled
   above its own `s = 1`. Item 6's rule is unchanged; only the scope of "the comparison" is fixed to
   the pair it was always about.
2. **A covariance-matched random null is added**, because the isotropic one cannot be matched at
   all. `r = X_centred^T g` with `g ~ N(0, I)`, renormalised: a random direction drawn along the
   data's own covariance, so its displacement is comparable to the outcome direction's and it can be
   scaled down to meet it. Seeds 0–4, as before.
3. **The isotropic random arm is kept** and reported, flagged `capped`, at its own `s = 1`. It is no
   longer treated as a matched null, because it is a strictly *weaker* perturbation than the test
   arm and so cannot bound it. The voiding rule ("`random` exceeding 0.0988 voids that model") now
   attaches to the covariance-matched null, which is the one that can actually reach the floor.

Unchanged: the site, the estimator, k = 10, the 0.0988 floor, the ≤ 1.5 perplexity ratio, the
≥ 0.50 cell-SD retention, the `s ≤ 1` assertion, the dose grid `{0.25, 0.50, 0.75, 1.00}`, and the
decision table. Nothing that decides the outcome is touched.

### Full disclosure of what the aborted run showed

OLMo-2-7B-Instruct only; the run never reached Qwen or Tulu. Construction diagnostics: the outcome
direction places **0.4208** of its variance inside the k = 10 scenario subspace (E6a-live's range was
0.44–0.66, so this run sits just below it), surviving fraction **0.7610**, baseline contrast
**−0.4448**, baseline perplexity 10.857.

The dose grid on the test arm, which this amendment does not affect:

| s | Δcontrast | ≥ 0.0988 | ppl ratio |
|---:|---:|:--:|---:|
| 0.25 | +0.0395 | no | 1.002 |
| 0.50 | +0.0787 | no | 1.005 |
| 0.75 | +0.1172 | **yes** | 1.009 |
| 1.00 | +0.1549 | **yes** | 1.017 |

And the degenerate matched block, all seven arms pinned at 0.0027, which is the evidence for the
defect: test +0.0058, positive control +0.0067, complement +0.0046, isotropic randoms −0.0035,
−0.0028, −0.0007. The positive control is inert, so by the table this was a void.

**Why this is disclosable rather than fatal.** The pipeline is deterministic on fixed hardware —
E7a established that to the bit — and the dose grid is untouched by this amendment. So the corrected
run must reproduce the four OLMo dose numbers above exactly. If it does not, this amendment changed
something it claims not to have, and that is visible rather than arguable. The numbers this
amendment could bias are the pairwise matched arms, and for OLMo the direction of that bias is
stated in advance here: having seen the test arm clear the floor at s ≥ 0.75, the expectation going
in is now (A), and the falsification condition in "What would falsify the reading we expect" above
still applies unaltered.
