# E5 — controls hardening W8/E3: pre-specified predictions

Written before any E5 run, mirroring [`W3_PRESPEC.md`](W3_PRESPEC.md). W8's verdict
([`W8_AMNESIC.md`](W8_AMNESIC.md)) is the paper's headline: erasing the outcome direction at the
W3 peak layers moves the attempted−accidental contrast by **+0.3182** (OLMo, L16) and **+0.2312**
(Qwen, L19), both above the J4 effect floor of **0.0988**, with a monotone dose-response and the
intent coefficient preserved. E5 attacks the four weakest joints in that claim.

Constants carried over unchanged from W8 and not re-derived here: `J4_EFFECT_FLOOR = 0.0988`
(q95 of the within-scenario permutation null, `floor_derivation.csv`), `CELL_SD_RETENTION = 0.50`,
`V1_TOL = 0.05`, `V2_RETENTION = 0.90`, `OUTCOME_CHANCE = 0.5168`, `INTENT_CHANCE = 0.50`, the
four V3 coherence bounds, and `GAMMAS = {0.25, 0.50, 0.75, 1.00}`.

---

## E5a — decodable-nuisance erasure (highest priority)

**The objection.** W8's random arm is rank-matched but not decodability-matched, and O1 already
recorded that it is also not magnitude-matched (a 31–34× smaller perturbation). So W8 cannot yet
rule out: *erasing any highly decodable direction perturbs judgment.*

### A structural obstacle, measured before this prespec was written

The natural design is to erase a morally irrelevant direction whose probe accuracy is within
±0.05 of the outcome probe's. **In this stimulus set that is impossible, and the reason is a
property of the design rather than a shortfall of the candidates.** Measured on cached
`OLMo-2-1124-7B-Instruct` residuals at L16 under the same leave-scenario-group-out CV that W8
uses (no behavioural quantity was computed, and none existed for these arms):

| property | varies *within* a scenario group | leave-group-out CV accuracy |
|---|---:|---:|
| outcome (experimental factor) | 90.6% of groups | **0.9866** |
| intent (experimental factor) | 100% of groups | 0.8456 |
| protagonist gender | **0%** of groups | 0.5671 |
| narrative length bin | 13% of groups | 0.6174 |
| domain cluster (TF-IDF k=2) | 4% of groups | 0.5235 |
| source corpus | 45% of groups | 0.6376 |

The 2×2 is manipulated **within** scenario. Every morally irrelevant property is therefore a
property **of** the scenario, constant across its four cells, and can only be learned by
generalising across scenarios — which leave-group-out CV deliberately makes hard. Anything that
does vary within a group *is* the intent or outcome manipulation. So "a morally irrelevant
direction as decodable as outcome" does not exist here, and no candidate list would have found
one. This is the same root cause as E1's finding, and it is reported, not worked around.

### What is run instead, and why it still answers the objection

Matching moves to the quantity O1 identified as actually missing, **perturbation norm**, which is
also what the user's design specifies. Decodability is reported alongside rather than matched.

**Nuisance arms**, each fitted at the same W3 peak layer as the outcome arm:

1. `gender` — protagonist gender (first gendered pronoun), rank-1 LEACE.
2. `length` — narrative length bin (median split on word count), rank-1 LEACE.
3. `scenario` — scenario identity, the **most decodable structure in the representation** and the
   strongest available form of the objection. It dominates the RDM (E1) and is morally
   irrelevant to a within-scenario contrast. Erased as the top-1 direction of the between-scenario
   class-mean structure so it is rank-comparable to the other arms.

**Two dose conventions per arm, both reported:**

- **exact** (`s = 1`): erase the factor completely. Answers "does removing this nuisance factor
  move judgment?"
- **norm-matched** (`s = D_outcome / D_arm`): scale the erasure along the same direction until its
  mean displacement `|Δh|/|h|` equals the outcome arm's at γ=1.0. Answers "does a perturbation
  *this large* move judgment, wherever it points?" Scaling is exact because displacement is linear
  in `s` for `h ↦ h − s·((h−μ)·w)w`.

**Gating.** Each arm is gated on the factor **it** erases (V1a on that factor, V2 on the factor it
must preserve), which is the fix already applied in W8; V3 and V4 are unchanged. The `scenario`
arm has no leave-group-out decodability (held-out groups are unseen classes), so its V1a is
measured in-sample and this is stated wherever it is reported.

```
  P1 (nuisance inertness, PRIMARY). For every nuisance arm, at BOTH dose conventions,
      |Δcontrast| < 0.0988 (the J4 floor), while the outcome arm at the same site exceeds
      it. This is the prediction that hardens W8.
  P2 (norm-matched separation). At matched displacement, |Δcontrast_outcome| is at least
      2x the largest |Δcontrast| across nuisance arms. Bar set at 2x because W8's existing
      magnitude-matched intent-vs-outcome comparison already reaches 2.2x (OLMo) and 3.6x
      (Qwen); E5a must not fall below the weaker of those.
  P3 (decodability disclosure). The leave-group-out accuracy of every erased factor is
      reported in the same table as its Delta-contrast, so a reader can see exactly how far
      from decodability-matched each arm is. No verdict rides on this; it is an
      anti-overclaiming requirement on the writeup.

  FALSIFICATION. P1 fails if ANY nuisance arm moves the contrast by >= 0.0988 at either dose.
  That is publishable and damaging to W8: it would mean the contrast responds to large
  perturbations regardless of direction, and W8's effect could not be attributed to the
  outcome direction specifically. It would NOT be reported as a W8 failure but as a W8
  scope reduction, and the headline would be downgraded accordingly.
```

---

## E5b — third model

Replicate the full W8 erasure and dose sweep on `allenai/Llama-3.1-Tulu-3-8B`, chosen over Zephyr
because it is already in the E2 checkpoint pipeline **and** has cached activations, so no new
extraction is needed. Target n=3 for the headline.

**Site selection.** W3 fixed L16/L19 for OLMo/Qwen; Tulu has no W3 site and no probe sweep on
disk. The site is therefore chosen by the same rule W3 used, applied in-run and **before any
behavioural number is read**: the layer maximising leave-scenario-group-out intent probe accuracy
on the chat-templated last-token residual. That is the representation the erasure actually acts
on, so it is a stricter reading of the rule than reusing cached raw-text activations. The selected
layer and the full layer sweep are written to `w8_amnesic_layer_selection_Llama-3.1-Tulu-3-8B.csv`.

There is no W3 baseline contrast for Tulu, so the W8 drift cross-check is undefined and is
reported as `n/a` rather than as a passing check.

```
  P1 (replication). Delta-contrast >= +0.0988 at a gate-passing dose, with the intent
      coefficient preserved, i.e. the same P1 W8 applies to OLMo and Qwen.
  P2 (label). The RECOVERY / COLLAPSE / NO CHANGE label under W8's existing 50% cell-spread
      rule.

  FALSIFICATION. If Tulu shows NO CHANGE or COLLAPSE, the headline becomes 2 of 3 rather
  than 3 of 3 and must be stated that way in the abstract. A third-model null does not
  invalidate OLMo and Qwen, but it does bound the generality claim, and the bound goes in
  the headline sentence rather than in a limitations paragraph.
```

---

## E5c — spread-matched sensitivity for OLMo

OLMo's only gate-passing dose retains **0.5124** of the pre-erasure cell spread, just 0.0124 above
the 0.50 boundary at which W8's P2 label flips to COLLAPSE. The effect is therefore measured where
the cells are nearly maximally compressed, and it needs to be shown that this is not what produces
it.

Δcontrast is recomputed under progressively stricter spread floors. No new run: the dose sweep on
disk already contains the required rows.

```
  P1 (survival at 0.55). Restricted to doses retaining > 0.55 of cell spread, the largest
      Delta-contrast for the outcome arm is still >= 0.0988.
  P2 (survival at 0.60). The same, restricted to doses retaining > 0.60.

  CONSEQUENCE, fixed now so it cannot be negotiated later. If P2 fails, OLMo is DOWNGRADED
  to supporting evidence and the primary evidence becomes Qwen plus E5b's model. That
  downgrade is written into the W8 headline and the abstract, not only into a caveat.
  If P1 also fails, OLMo is withdrawn from the headline entirely.

  NOTE ON THE GATE. Doses below gamma=1.0 do not satisfy V1a (outcome is not fully erased),
  so they cannot count toward W8's P1. E5c is therefore explicitly a SENSITIVITY analysis,
  not a second P1 test, and its numbers are labelled as such wherever they appear.
```

---

## E5d — E2 reporting fix and E1 span-matched retry

### E2

The coupling statistic that withdrew P1 (`rho = +0.870`) is computed over **6** transitions. It
currently appears without n or an interval, which overstates it.

```
  R1. Every appearance of the coupling statistic -- verdict MD, figure subtitle, and any
      summary -- carries n and a bootstrap 95% CI.
  R2. One sentence states that 6 points is underpowered in BOTH directions: it can neither
      establish coupling nor establish its absence.
  R3. P1 remains withdrawn per the prespec's own rule. The writeup must NOT assert the
      converse (that geometry reshaping DRIVES the behavioural change); withdrawal of P1 is
      not evidence for P2's alternative. This is the specific overclaim R3 exists to block.
```

### E1

Retry P4's surface gate with a **span-matched** lexical control instead of full-document TF-IDF,
built on the non-factor span using `clause_offsets.csv` (the setup text preceding `belief_start`,
which is the portion that does not realise either factor).

```
  R4. If the span-matched control still excludes every layer, that is the FINAL answer and
      E1 is written up as a methods note: "surface-control gating is undefined when the
      experimental factors are lexically realised." The 3-18x outcome-over-intent dominance
      stays labelled post-hoc/M1 throughout, in every document.
  R5. If it does NOT exclude every layer, the pre-registered P1/P2 become evaluable and are
      reported against their ORIGINAL 7/8 bar, with the span-matched variant named as the
      one that made them evaluable.

  EXPECTATION, recorded so it cannot be claimed as a prediction afterwards: the setup span
  is identical across the four cells of a scenario, so its TF-IDF distance is close to the
  scenario-identity matrix, which E1 already showed dominates the RDM. R4 is the likely
  branch. Recording this in advance is the point.
```

---

## Outputs

| path | contents |
|---|---|
| `outputs/experiments/w8_nuisance_<tag>.csv` | E5a per-arm, per-dose contrast, cells, gates, displacement |
| `outputs/experiments/w8_nuisance_decodability_<tag>.csv` | E5a leave-group-out accuracy per erased factor |
| `outputs/experiments/w8_amnesic_*Llama-3.1-Tulu-3-8B*` | E5b, same schema as the other two W8 models |
| `outputs/experiments/w8_spread_sensitivity.csv` | E5c restricted-dose table |
| `outputs/experiments/W8_AMNESIC.md` | gains an E5a section, an E5b model section, and an E5c section |
| `outputs/rsa/E1_VARIANCE_PARTITION.md`, `E2_CHECKPOINT_STAGE_RSA.md` | E5d edits |

Nothing under `outputs/figures_final/_pending_rescore/` is read or written. Raw activations stay
gitignored. No E5 figure is promoted to `presentation_figures/` until this file's thresholds have
been reported against.

**Disclosure.** The decodability table in E5a was measured before this prespec was written, which
is why the structural obstacle could be described rather than predicted. It involves no
behavioural quantity: no contrast, cell mean, or Δcontrast existed for any E5 arm at the time of
writing, and none of E5's thresholds (0.0988, 2×, 0.55, 0.60) were chosen with knowledge of an
E5 result.
