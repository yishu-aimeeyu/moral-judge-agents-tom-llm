# E4 — cross-cultural placement: pre-specified analysis plan

Written before any per-society effect size was computed. Recorded verbatim by
`code/experiments/64_crosscultural_placement.py` to this path on the first `--run`, and never
overwritten.

**What was inspected before this file was written:** the dataset *schema only* — column
names, the distinct values of `society`, `bank`, `condition`, `question`, and the range of
`judgrev` (see `SCHEMA_NOTE.md`). No group means, no differences, no effect sizes. The
schema had to be read to name columns in a plan; the quantities under test had not been
computed when the thresholds below were fixed.

**Why this exists.** The project's human anchors are all Western: Young et al. (2007) adults
and the Killen/Cushman-style child bands. The adult anchor is already carrying a strong
caveat (`ADULT_ANCHOR_MATCH_AUDIT.md`: no confirmed item-and-condition match). A reviewer's
natural objection is that "models fall below the youngest measured child" might be an
artifact of one cultural sample. Curtin et al. (2020) re-analyse Barrett et al.'s 10-society
vignette data, which is the only source available with an intent manipulation measured the
same way across societies. E4 places model contrasts inside that human distribution.

## The dataset, and the one place it does not match us

Source: OSF node [`65krf`](https://osf.io/65krf/), *Data sets → Cross-cultural experimental
data set*, files `Intentions_KII_BarrettData.csv` (6,525 rows) and
`codebook_Intentions_KII_BarrettData.xlsx`. 10 societies, 322 subjects, judgments on a
5-point scale `judgrev ∈ {−2 … +2}`.

Relevant structure: `bank = INTENT` (3,853 rows) contains conditions `Intentional`,
`Accidental`, `Motivated`, `Anti-motivated`; `question ∈ {Badness, Punishment, Reputation}`.

**The mismatch, stated before any analysis.** There is **no failed-attempt condition** in
this dataset. The project's headline contrast, `blame(attempted) − blame(accidental)`,
therefore has **no counterpart here and will not be plotted against it.** E4 places models on
the one axis that does exist in both: the **intent effect at constant harm**,
`Intentional − Accidental`, where both conditions involve a bad outcome and only intent
differs. On our side that is the `cell_intentional − cell_accidental` difference from
`outputs/stats/mixed_effects_2x2.csv`, which is `b_intent` evaluated at outcome = harm.

Substituting the headline contrast here would repeat exactly the error
`ADULT_ANCHOR_MATCH_AUDIT.md` was written to stop: quoting a human number for a cell pair the
humans were never asked about. This is a placement on a *related* axis, and the figure and MD
must say so in those words.

## The pre-specified analysis

```
=========================
Written before any per-society effect size was computed. Recorded verbatim to
dataset/human_reference/crosscultural/E4_PRESPEC.md on first --run, never overwritten.

  MEASURE (human). Per society: restrict to bank = INTENT, question = Badness, conditions
      Intentional and Accidental. Compute each SUBJECT's mean judgrev per condition, take
      that subject's Intentional - Accidental difference, then average over subjects within
      society. Subject-level first, so societies with more items per subject do not get
      extra weight. 95% CI by bootstrap over subjects within society, 2000 draws, seeded.
  MEASURE (model). cell_intentional - cell_accidental from outputs/stats/mixed_effects_2x2.csv,
      spec = primary, on the existing 0-1 normalised blame scale.
  NORMALISATION (primary). Human differences are divided by the response-scale range
      (judgrev spans -2..+2, range 4) to put them in the same 0-1-per-unit-scale units as the
      model differences. This is a SCALE normalisation and nothing more: it does not make the
      stimuli, the task, or the populations comparable, and the figure must label it as such.
  NORMALISATION (secondary). Cohen's d within society (difference / pooled within-society SD
      of judgrev) is reported in the CSV and in a second panel. Models are NOT placed on the
      d axis, because a model's rating SD is not commensurable with a human sample's.

  P1 (placement). Every open-weight non-degenerate model's intent effect falls BELOW the
      LOWEST society's scale-normalised intent effect. Bar: at least 18 of the 18
      non-degenerate open models, matching the 18/18 convention already used for the child
      ladders. Models failing the engagement floor (rating_std < 0.2191) are excluded and
      the exclusion is listed by name.
  P2 (human universality of the direction). Every one of the 10 societies has a POSITIVE
      intent effect, i.e. intentional harm is judged worse than accidental harm, with a
      bootstrap CI excluding zero. This is the claim that makes P1 interesting: it is not
      "models differ from Westerners" but "models sit outside the whole measured human
      range".
  P3 (spread reporting). The human distribution is reported as min, max, median and the
      full per-society list, never as a single mean. Collapsing 10 societies into one number
      would reintroduce the single-anchor problem this experiment exists to address.
  P4 (question robustness). The same computation is repeated for question = Punishment and
      question = Reputation. Badness is primary because it is the wrongness judgment closest
      to our blame prompt. If the sign or the P1 verdict changes across questions, that is
      reported in the headline rather than as an appendix note.

  FALSIFICATION. P2 fails if any society's intent effect is <= 0 or has a CI spanning zero.
  That is a reportable and important outcome: it would mean some human societies also fail to
  weight intent on this measure, and the paper's claim must be restated as a distribution-
  overlap statement ("models sit at or below the least intent-weighting society") rather than
  as "outside the human range". P1 fails if any engaged model's intent effect exceeds the
  lowest society value, in which case that model is named in the headline.

  WHAT THIS CANNOT SHOW. Fixed in advance so it cannot be softened later:
    - Different stimuli. These are not the YS2008/YS2009 vignettes; no item is shared.
    - Different scale and task (5-point badness of a described act vs our normalised blame
      rating over 1-7 templates).
    - Different populations and administration (in-person translated interview vs prompted
      model).
    - The reference distribution is therefore a REFERENCE DISTRIBUTION, not item-matched
      ground truth, and no causal or developmental claim is licensed by proximity to it.
========================================================================================
```

## Outputs

| path | contents |
|---|---|
| `dataset/human_reference/crosscultural/README.md` | provenance, download command, checksums, the normalisation method, and the caveat list |
| `dataset/human_reference/crosscultural/SCHEMA_NOTE.md` | structure-only schema dump (already written, pre-analysis) |
| `outputs/crosscultural/society_intent_effects.csv` | society × question: raw difference, scale-normalised difference, Cohen's d, bootstrap CI, n subjects |
| `outputs/crosscultural/model_placement.csv` | model × intent effect, its percentile position in the 10-society distribution, engagement-floor flag |
| `outputs/figures_final/crosscultural_placement.png` | **new figure only** — 10-society human distribution with model contrasts placed inside it |
| `outputs/crosscultural/E4_CROSSCULTURAL.md` | verdict against P1–P4 plus the caveat block verbatim |

**Raw data is gitignored.** `raw/Intentions_KII_BarrettData.csv` and the codebook are not
committed; the README carries the OSF file IDs, the `curl` command, and MD5 checksums so the
download is reproducible. Derived per-society summaries are committed.

**No existing figure is modified.** `crosscultural_placement.png` is a new file. Nothing under
`outputs/figures_final/_pending_rescore/` is read or written, and no figure is copied into
`presentation_figures/` until `E4_CROSSCULTURAL.md` reports the verdict against P1–P4.
