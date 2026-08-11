# E6 — resolving E5a's P1 failure with a correctly-specified test: pre-specified predictions

Written before any E6 compute, mirroring [`W3_PRESPEC.md`](W3_PRESPEC.md). Committed on its own
commit, with no E6 result in existence at the time of writing.

## What E6 is, and what it is not

E5a's pre-registered P1 **failed** ([`E5_CONTROLS.md`](E5_CONTROLS.md)): in
`Qwen2.5-7B-Instruct`, a matched-norm perturbation along scenario identity moved the
attempted−accidental contrast by **+0.1345**, past the J4 floor of **0.0988**. That verdict is
final and E6 does not revisit it. **E6 is a new test with a different specification, not a
reinterpretation of E5a**, and nothing in E6 can overturn E5a's failure — at most E6 can show
*why* E5a failed and whether a correctly-specified control passes.

Two defects in E5a's specification motivate E6, both identified after the fact and both recorded
in `E5_CONTROLS.md` before this file was written:

1. **The over-projection bug.** E5a's norm-matched arms scaled the projection by `s > 1`
   (2.47× gender, 1.53× length, 1.28× scenario in OLMo). `h ↦ h − s·((h−μ)·ŵ)ŵ` with `s > 1`
   reflects the residual *past* the class mean instead of onto it, so the factor becomes linearly
   decodable again with inverted sign. In-sample accuracy of the "erased" factor was **1.000**,
   not chance. Those arms were matched-magnitude perturbations, not erasures.
2. **No orthogonalization.** The nuisance directions were never orthogonalized against outcome
   and intent. If scenario identity overlaps the outcome direction, perturbing along scenario
   *partly perturbs outcome*, so a contrast movement is not evidence that a morally irrelevant
   direction moves judgment. E5a could not distinguish these.

E6 fixes both. Constants carried over unchanged and not re-derived: `J4_EFFECT_FLOOR = 0.0988`,
`V1_TOL = 0.05`, `PPL_RATIO_MAX = 1.5`, `CELL_SD_RETENTION = 0.50`, `OUTCOME_CHANCE = 0.5168`,
`GAMMAS = {0.25, 0.50, 0.75, 1.00}`, W3 sites L16 (OLMo) / L19 (Qwen), and the in-run-selected
L17 (Tulu, from `w8_amnesic_layer_selection_Llama-3.1-Tulu-3-8B.csv`).

---

## A representation mismatch, disclosed before it can be discovered later

The task specifies E6a on "stored direction vectors". **The direction vectors are not stored.**
The W8/E5 erasers were fitted in-run on GPU from *chat-templated* last-token residuals
(`63_w8_amnesic_outcome.py`, `fmt()` → `hs[k][:, -1, :]`), and only summary CSVs survive
(`w3_steering_directions_*.csv` holds norms and cosines, not vectors). What *is* on disk is
`outputs/acts/<tag>.npz` — **raw-text** last-token activations from `01_extract_activations.py`,
298 items × layers × d, for all three models.

This is handled by running the diagnostic twice rather than by picking one and hoping:

- **E6a-cached (primary, CPU, run first as instructed).** Directions re-fitted from
  `outputs/acts/`. Same items, same layers, same estimators; a *different* text rendering from
  the one the erasure hook acted on.
- **E6a-live (confirmatory, written by the E6b GPU job).** The E6b run dumps the actual
  chat-templated direction vectors it uses to
  `outputs/experiments/e6_directions_<tag>.npz`, and the identical diagnostic is recomputed on
  them.

**If the two disagree on E6a's prediction, E6a-live is authoritative for interpreting E6b**,
because it is the geometry the intervention actually acts in, and the disagreement itself is
reported as a finding about how much chat templating moves this geometry. E6a-cached is not
discarded; it is the pre-registered primary and is reported either way.

---

## E6a — direction-overlap diagnostic (CPU-only, run first)

### Definitions, fixed here

- **Binary factor direction.** For `outcome`, `intent`, `gender`, `length`: fit rank-1 LEACE for
  that factor at the site, then recover the erased direction as the top right-singular vector of
  `I − P`, unit-normalised. This is the direction the intervention removes, not a separately
  chosen estimator, so overlaps describe the actual manipulation.
- **Nuisance labels**, identical to E5a so the arms are comparable: `gender` = first gendered
  pronoun in `text`; `length` = median split on `word_count`.
- **Scenario subspace.** Let `M` be the (n_scenarios × d) matrix of per-`scenario_group` mean
  residuals, globally centred. `S_k` = span of its top `k` principal components. Reported for
  `k ∈ {1, 5, 10, 25}`; **`k = 10` is the pre-designated primary** (~1/7 of the ~75 groups,
  enough to carry scenario structure without spanning the item space).
- **Measure (i), cosine.** `|cos(w_scenario_1, w_outcome)|` and `|cos(w_scenario_1, w_intent)|`,
  where `w_scenario_1` is the top scenario PC.
- **Measure (ii), variance fraction.** For unit `w`, `ρ_k(w) = ‖P_{S_k} w‖²` ∈ [0, 1] — the
  fraction of the direction's variance lying in the scenario subspace, equal to `cos²` of the
  principal angle between `span(w)` and `S_k`. Also computed with `gender` and `length` as 1-d
  subspaces in place of `S_k`.

### Prediction, with the margin stated before looking

No overlap number has been computed. E5a's separation ratios are 10.26× (OLMo), 4.31× (Tulu) and
1.72× (Qwen) — Qwen leaks 2.4–6× more nuisance movement than the others. If geometric
entanglement explains that, the overlap gap should be substantial rather than marginal.

```
  P1 (PRIMARY, geometric explanation of the Qwen-specific failure).
      rho_10(w_outcome) computed against the scenario subspace is LARGER for Qwen than for
      BOTH OLMo and Tulu, by a margin of at least 0.10 in absolute terms on the 0-1 scale:
          rho_10_Qwen(outcome) - max(rho_10_OLMo(outcome), rho_10_Tulu(outcome)) >= 0.10
  P2 (secondary, cosine form). The same ordering holds for |cos(w_scenario_1, w_outcome)|
      with a margin of at least 0.05.
  P3 (disclosure, no verdict rides on it). rho_10 and the cosines are reported for intent,
      gender and length in the same table, so a reader can see whether any elevated overlap
      is specific to outcome or is a property of that model's geometry generally.

  FALSIFICATION. If the margin is not met -- including the case where Qwen is highest but by
  less than 0.10, and the case where Qwen is not highest at all -- then the geometric
  explanation for E5a's Qwen-specific failure is NOT supported. E6b is still run and still
  reported, but it must then be interpreted WITHOUT the overlap story, and the E5a failure is
  left unexplained rather than explained by a mechanism the data did not show. Reporting
  "Qwen is highest but by 0.04" as a confirmation is the specific failure mode P1's margin
  exists to prevent.
```

Output: `outputs/experiments/e6_direction_overlap.csv`, `outputs/experiments/e6_direction_overlap.png`.

---

## E6b — orthogonalized nuisance erasure

### The construction

For each nuisance direction `n` ∈ {gender, length, scenario_1}:

1. Orthonormalise the removal basis: `u₁ = ŵ_outcome`; `u₂ = ŵ_intent − (ŵ_intent·u₁)u₁`,
   renormalised. Intent is orthogonalised against outcome first so `{u₁, u₂}` is orthonormal and
   the Gram-Schmidt step below is a true projection.
2. `n⊥ = n − (n·u₁)u₁ − (n·u₂)u₂`, then renormalise to `n̂⊥`.
3. Report **`‖n⊥‖ / ‖n‖`**, the fraction of the nuisance direction surviving orthogonalization.
   A small value is itself a result: it means the nuisance direction was largely inside the
   outcome–intent plane, which is the mechanism E5a could not see.
4. Erase along the residual only: `h ↦ h − s·((h−μ)·n̂⊥)n̂⊥`, i.e. the same rank-1 directional
   projection E5a used (`rank1_dir`), applied to `n̂⊥` instead of `n`.

### `s ≤ 1`, enforced and asserted — and one place where the assertion cannot apply

`s ≤ 1` is enforced in code and asserted at runtime; `s > 1` raises rather than warns. The
over-projection check is: after erasure at `s = 1`, in-sample accuracy of the erased factor must
fall within `V1_TOL` of that factor's majority rate, and the run **fails loudly** otherwise.

**That assertion is applied to the plain, non-orthogonalized arm, and cannot be applied to the
orthogonalized arm. The reason is structural, not a relaxation of the standard.** A factor's
linear decodability is carried by its full direction `n`. Orthogonalization deliberately declines
to remove the part of `n` lying along outcome and intent, so whenever the overlap is non-zero the
factor *must* remain partly decodable after erasing `n⊥` — driving it to chance would require
removing exactly the components E6b exists to preserve. Demanding chance on the orthogonalized
arm would make every arm fail by construction and E6b would return nothing. So:

- **plain arm, `s = 1`**: accuracy must reach the majority rate within `V1_TOL`, asserted, run
  fails loudly if not. This is the machinery check E5a lacked.
- **orthogonalized arm**: accuracy before and after is *reported*, full chance is **not**
  required, and the reported drop is interpreted against `‖n⊥‖/‖n‖`.

### Matching, done by scaling the outcome arm DOWN

`s ≤ 1` and "matched perturbation norm" are in direct conflict if matching is done the way E5a
did it: gender's natural `s = 1` displacement is 0.0372 against outcome's 0.0918 in OLMo, so
matching upward requires `s = 2.47`, which is exactly the bug. **Matching is therefore done in
the other direction**: the *outcome* arm is scaled down to each nuisance arm's `s = 1`
displacement, via its own `γ`, and no nuisance arm is ever scaled above `s = 1`. Displacement is
linear in the scale, so the required `γ` is computed as `D_nuisance(s=1) / D_outcome(γ=1)` and run
explicitly rather than interpolated from the existing grid.

One consequence, stated so it cannot be mistaken later: a down-scaled outcome arm does **not**
satisfy V1a, because outcome is not fully erased at `γ < 1`. The matched comparison is therefore
a **magnitude-matched comparison of directions**, which is what the objection requires, and it is
*not* a second test of W8's P1. W8's P1 remains the full-erasure `γ = 1.0` result.

**Matched retained coherence.** Each arm's perplexity ratio is reported, and the primary
comparison is restricted to doses where **both** the nuisance arm and its matched outcome dose
satisfy `ppl ≤ 1.5`. This is the gate E5a's breaching dose failed.

```
  P1 (PRIMARY, orthogonalized nuisance inertness). For every model and every orthogonalized
      nuisance arm, at EVERY coherence-passing dose, |Delta-contrast| < 0.0988, while the
      outcome arm at the matched displacement exceeds 0.0988.
  P2 (separation). Separation ratio |Delta-contrast_outcome| / |Delta-contrast_nuisance| at
      matched displacement, reported per model ALONGSIDE E5a's (OLMo 10.26x, Tulu 4.31x,
      Qwen 1.72x) in the same table, so the change attributable to orthogonalization is
      visible. Bar: >= 2x, unchanged from E5a's P2.
  P3 (surviving fraction). ||n_perp|| / ||n|| reported per arm per model. No bar; it is the
      quantity that makes P1's result mechanistically interpretable.

  FALSIFICATION. If an orthogonalized nuisance arm still moves the contrast by >= 0.0988 at a
  coherence-passing dose, then the E5a failure is NOT explained by outcome overlap and is not
  repaired by orthogonalization. That is the strongest available form of the objection and it
  would mean the contrast responds to large perturbations largely regardless of direction.
  W8A's headline would then be reduced to a magnitude claim -- "the outcome direction is the
  most effective direction we found" -- with specificity abandoned rather than qualified, and
  that reduction goes in the abstract.
```

Outputs: `outputs/experiments/e6_orthogonalized_<tag>.csv`,
`outputs/experiments/e6_directions_<tag>.npz`, `outputs/experiments/E6_CONTROLS.md`.

---

## E6c — coherence-gated restatement (pre-registered secondary)

All E5a and E6b arms are restricted to coherence-passing doses (`ppl ≤ 1.5`, the pre-registered
V3 bound) and the primary comparison is reported on that subset.

```
  R1. The restatement is labelled a pre-registered SECONDARY throughout. E5a's literal P1
      verdict STANDS and is restated as failed in the same paragraph.
  R2. No sentence may imply that E5a's failure has been overturned, reinterpreted, explained
      away, or rescued. Permitted framing: E6 is a differently-specified test whose result is
      reported beside E5a's. Forbidden framing: "in fact", "the apparent failure", "once
      properly controlled", or any construction placing E5a's verdict in doubt.
  R3. Every E5a number in the restatement is read from the committed
      w8a_nuisance_<tag>.csv, not recomputed.
```

---

## E6d — repo hygiene, performed regardless of every E6 outcome

1. **`W8` → `W8A` for the ablation.** `W8_AMNESIC.md`, `W8_PRESPEC.md`, `w8_amnesic_*`,
   `w8_nuisance_*`, `w8_spread_sensitivity.csv`, `63_w8_amnesic_outcome.py` and figure names all
   move to `W8A`/`w8a`. `57_w8_paper_assembly.py` and `outputs/paper/` keep `W8`, which is the
   pre-existing meaning ("W8 — Paper assembly"). Renames use `git mv` so history follows.
2. **E1–E5 sections in `BEYOND_BEHAVIOR_MASTER.md`**, following the existing template exactly:
   a finding paragraph, then the role/path/GitHub table.
3. **`LIMITATIONS.md`** gains E5a's P1 failure with its three caveats framed as **open
   questions**, the 2/3-RECOVERY + Tulu-COLLAPSE bound, and E1's undecidability.
4. **`MENTOR_PACKET.md:5`** replaces "entrenched" with a formulation distinguishing
   prompt-entrenchment from representational recoverability.
5. **The Tulu result is written up as a substantive finding** — largest contrast movement of the
   three, lowest retained cell spread, i.e. an outcome direction more entangled with general
   rating behaviour — not merely as a failed gate.

---

## Standing rules honoured here

Every number in every E6 document is read from a committed CSV and never recomputed inline; the
one exception is E6a's overlap statistics, which are *produced* by E6a and committed as
`e6_direction_overlap.csv` before any prose cites them. Nothing under
`outputs/figures_final/_pending_rescore/` is read or written. Raw activations stay gitignored.

**Disclosure.** At the time of writing: no direction overlap, cosine, principal angle,
surviving-fraction, Δcontrast, cell mean, or perplexity ratio existed for any E6 arm. The E6a
margin (0.10 absolute, 0.05 cosine), the `k = 10` primary subspace, and the P2 bar of 2× were
chosen without knowledge of any E6 result. The only inputs consulted were the shapes and keys of
`outputs/acts/*.npz`, the column headers of `dataset/master/moral_2x2_master.csv`, and the
already-committed E5 numbers quoted above.

---

## Amendment 1, 2026-08-10 — the direction this prespec specified does not erase the factor

**This amendment was written between two GPU jobs, after the first aborted and before any E6b
behavioural number existed.** It is reconstructed here from the run logs, which fix the timeline
beyond dispute:

| time | event | evidence |
|---|---|---|
| 18:30 | this prespec committed | commit `3dacb29` |
| 18:42–18:43 | first E6b job **aborts on its own assertion** | `outputs/logs/gpu_20123946.log` |
| 18:53 | second E6b job runs to completion | `outputs/logs/gpu_20124997.log` |

The abort is the reason for the amendment, and it is the assertion in §E6b doing exactly the job
it was written to do:

```
ASSERT plain_gender s=1: in-sample acc 1.0000 vs majority 0.5034 (tol 0.05)
AssertionError: OLMo-2-1124-7B-Instruct plain_gender at s=1 did NOT drive gender to chance:
acc=1.0000, majority=0.5034. Refusing to report a behavioural number from an erasure that did
not erase (E6_PRESPEC E6b).
```

**What was wrong.** §E6a defined the factor direction as *"the top right-singular vector of
`I − P`"* where `P` is the rank-1 LEACE projection. That was an error on my part. LEACE's
projection is **oblique**, not orthogonal: it whitens, removes the direction in whitened space,
and unwhitens. The subspace `I − P` maps onto is therefore not the subspace whose removal along a
Euclidean projection erases the factor, and removing the top singular vector of `I − P` left
gender **perfectly decodable** — accuracy 1.0000, not 0.5034.

**The correction.** The factor direction becomes the **class-mean-difference direction**
(`meandiff`), unit-normalised: `ŵ = (μ₊ − μ₋)/‖μ₊ − μ₋‖`, which is the direction whose rank-1
Euclidean removal equalises the class means and hence drives linear accuracy to the majority rate.
After the change the assertion passes exactly, in every model and both binary factors — in-sample
accuracy lands on the majority rate to four decimals (`0.5034` gender, `0.5470` length).

**Why this does not buy E6a a verdict, and how that is guaranteed rather than asserted.** Changing
a direction estimator after seeing that the first one failed is precisely the move that can launder
a prediction, so E6a is reported under **both** estimators, in the same table, with `meandiff` named
as primary and `leace_sv` as originally specified. The margins are:

| estimator | P1 margin (bar +0.10) | P2 margin (bar +0.05) | P1 |
|---|---:|---:|---|
| `meandiff` (primary after this amendment) | +0.0389 | −0.0719 | **fails** |
| `leace_sv` (as originally specified) | −0.0355 | −0.0166 | **fails** |

P1 and P2 fail under both, so the correction cannot be the reason for E6a's verdict. Under the
original estimator Qwen is the **lowest** of the three models; under the corrected one it is the
highest but by 0.0389 against a bar of 0.10. Both are recorded as NOT SUPPORTED.

Unchanged by this amendment: every threshold (0.0988, ±0.05 V1 tolerance, 1.5× perplexity, 2×
separation, the 0.10/0.05 E6a margins), `k = 10` as primary, the `s ≤ 1` rule and its assertion,
the match-downward construction, and E6c's and E6d's reporting rules. The estimator is the only
change.

## Amendment 2, 2026-08-10 — E6a-live is completed from the dumped directions

§"A representation mismatch" promised a confirmatory **E6a-live**: the same diagnostic recomputed
on the chat-templated directions the erasure hook actually acts in, dumped by the E6b job to
`e6_directions_<tag>.npz`. The E6b job wrote those dumps, but the live rows were never added to
`e6_direction_overlap.csv`, so the confirmatory arm was left incomplete. It is completed now from
the committed `.npz` files, which contain both the chat-templated residuals (`X`, 298 × d) and the
`meandiff` direction vectors used by E6b.

This adds rows with `source = live_chat_template`; it changes no cached row and no threshold. Per
the original prespec, if cached and live disagree on E6a's prediction, **live is authoritative for
interpreting E6b** and the disagreement is itself reported.
