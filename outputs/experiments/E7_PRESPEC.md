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
