# Content audit — what needs correcting after the W8 arm-gating fix and E5

Requested scope: `MENTOR_PACKET.md`, `BEYOND_BEHAVIOR_MASTER.md`, `outputs/paper/`, and the talk
decks, searched for any statement traceable to the **pre-fix W8 run** — the arm-gating bug that
printed *"Intent ≥ outcome, which contradicts P3"* because V1a/V2 were applied to the outcome
factor on every arm instead of to the factor each arm erases.

Nothing under `outputs/figures_final/_pending_rescore/` was read or written.

---

## 1. The thing that was searched for does not exist anywhere

**No document repeats the pre-fix W8 claim, and none ever did.** Verified three ways:

| check | result |
|---|---|
| `contradicts P3` in the working tree, ignores disabled | 1 hit, in `code/experiments/63_w8_amnesic_outcome.py` only |
| `git log --all -S'contradicts P3' -- outputs/` | **no commit ever added or removed it** |
| `git show fea5f51:outputs/experiments/W8_AMNESIC.md` | already contains the **post-fix** prose |

The single code hit is a conditional branch that fires only if intent ≥ outcome among
gate-passing doses. It does not fire in the current run: P3 reads *"not evaluable for this
model"* for OLMo and Tulu (their intent arms miss their own V1a tolerance by 0.0138 and 0.0305)
and *"Outcome > intent, as predicted"* for Qwen. The buggy sentence existed only in a
regenerated-over `W8_AMNESIC.md` and in `outputs/logs/gpu_20044740.log`, and was never committed
or cited downstream.

**`MENTOR_PACKET.md`, `BEYOND_BEHAVIOR_MASTER.md`, and all four `outputs/paper/` documents
contain zero occurrences of `amnesic`, `LEACE`, `erasure`, or `erase`.** They predate W8 and make
no W8 claim, correct or otherwise. There are no talk decks in markdown; the only talk artefact is
`outputs/master_developmental_ladder_talksafe.png`, a behavioural ladder figure that carries no
representational claim.

**So: no correction is required on the ground the audit was requested for.** The four items below
are real and were found while looking, but they are *staleness and naming* problems, not
propagated-bug problems, and they should not be described as bug fallout.

---

## 2. `MENTOR_PACKET.md` line 5 — the one-paragraph summary is now too strong

Current text asserts:

> Intent is linearly decodable from the residual stream at 0.85-0.98, yet residual-stream
> steering does not move the judgment [...] The bias is entrenched — not "unreachable by
> steering, reachable by prompting."

The first clause is still exactly right and is W3's result: *adding* intent signal moves the
contrast ≤ 0.015. But "the bias is entrenched" is now contradicted in one direction by W8: while
*adding* intent does nothing, **removing outcome** moves the contrast +0.2312 to +0.3480, and in 2
of 3 models the four cell means stay apart while it moves (the RECOVERY signature). Intent is
latent-but-suppressed in those two, not absent.

**Needs:** a sentence distinguishing additive steering (null, W3) from subtractive erasure
(effective, W8), and replacing "entrenched" with the bounded claim. Must carry E5's two bounds —
2 of 3 models, and E5a's specificity failure in Qwen — or it will overstate in the other
direction.

## 3. `BEYOND_BEHAVIOR_MASTER.md` — abstract contradicted, and the E-series is missing entirely

- **Line 18** states intent is *"**not used** for graded moral judgment"* on the strength of the
  probe↔contrast null, the RSA null, and W3. W8 does not overturn those, but it does show the
  judgment is *causally sensitive to the outcome direction's presence*, which the unqualified
  "not used" reads against.
- **Sections run 1–14 and stop at W7/ToM.** There is no entry for E1 (variance partitioning),
  E2 (checkpoint-stage RSA), E3/W8 (amnesic ablation), E4 (cross-cultural placement), or E5 (the
  controls). The index is the document's whole purpose, so these are load-bearing omissions.
- **Suggested next steps (line 244)** may still propose work that E1–E5 has since done.

**Needs:** five new sections in the existing table format with CSV/PNG/MD/prespec links, and the
line-18 qualifier.

## 4. `outputs/paper/LIMITATIONS.md` — no limitation covers what E5 found

The nine limitations predate W8 and E5. Two E5 results are exactly limitations-shaped and are
currently unrepresented:

1. **E5a's specificity failure.** A matched-magnitude perturbation along scenario identity moves
   Qwen's contrast +0.1345, past the J4 floor, so the outcome direction is the most effective
   direction found rather than the only one.
2. **E5b's 2-of-3 replication.** Tulu-3-8B is COLLAPSE, not RECOVERY, so the W8 headline is
   bounded at 2 of 3 and — per `E5_PRESPEC.md` — that bound belongs in the abstract's headline
   sentence, not in a caveat.

Also worth adding, since E1 is now a methods note: surface-control gating is undefined in this
stimulus design, so E1's registered P1/P2 are **undecidable, not refuted**, and the 3–18×
dominance is post-hoc (M1) everywhere.

## 5. `W8` means two different things in this repository

This is the one item that can actually cause a reader to misread a number.

| meaning | where |
|---|---|
| **paper assembly** (workstream 8) | `outputs/paper/README.md:1` "# W8 paper assembly", `57_w8_paper_assembly.py`, `RUN_STATUS.md:77`, `STATUS_REPORT.md:176,287` |
| **amnesic outcome ablation** (E3) | `W8_PRESPEC.md`, `W8_AMNESIC.md`, `63_w8_amnesic_outcome.py`, `w8_amnesic_*.csv`, `w8_nuisance_*.csv` |

`RUN_STATUS.md:77` currently reads "W8 — Paper assembly | complete", which a reader arriving from
`W8_AMNESIC.md` will read as "the ablation is complete and assembled". The two are unrelated.

**Needs:** rename one of them in prose (the experiment is already "E3" in the prespec's own
title, so "E3/W8" or just "E3" for the ablation is the cheaper fix), and disambiguate the
`RUN_STATUS.md` and `STATUS_REPORT.md` rows.

---

## Summary

| # | file | severity | is it pre-fix W8 fallout? |
|---|---|---|---|
| 1 | — | none | **no such statement exists anywhere** |
| 2 | `MENTOR_PACKET.md:5` | substantive — claim now too strong | no, predates W8 |
| 3 | `BEYOND_BEHAVIOR_MASTER.md:18,244` + missing sections | substantive — index incomplete | no, predates W8 |
| 4 | `outputs/paper/LIMITATIONS.md` | substantive — two E5 findings unrepresented | no, predates W8 |
| 5 | `RUN_STATUS.md:77`, `STATUS_REPORT.md:176,287`, `paper/README.md:1` | naming collision, misreadable | no, predates W8 |

Nothing in this list has been edited. The audit was asked for as a list, and changing the
mentor-facing summary and the paper's limitations table is a content decision rather than a
mechanical correction.
