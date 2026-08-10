# Cross-cultural human reference (Curtin et al. 2020 / Barrett et al. 2016)

Reference distribution of human intent-weighting across 10 societies, used to place model
contrasts inside a human range that is not exclusively Western. Pre-registered plan and
thresholds: [`E4_PRESPEC.md`](E4_PRESPEC.md). Structure-only schema dump:
[`SCHEMA_NOTE.md`](SCHEMA_NOTE.md).

> **This is a reference distribution, not item-matched ground truth.** Different stimuli,
> different response scale, different task, different populations and administration. Nothing
> in this folder licenses a claim that a model "matches" or "falls below" a specific human
> group on our own vignettes. Same caveat discipline as
> [`../ADULT_ANCHOR_MATCH_AUDIT.md`](../ADULT_ANCHOR_MATCH_AUDIT.md).

---

## Provenance

| field | value |
|---|---|
| OSF project | https://osf.io/65krf/ |
| Path in project | `Data sets / Cross-cultural experimental data set` |
| Data file | `Intentions_KII_BarrettData.csv` — OSF file id `5e1369f157341901d7803b2b` |
| Codebook | `codebook_Intentions_KII_BarrettData.xlsx` — OSF file id `5e1369f175458401c5f99a82` |
| Retrieved | 2026-08-09 |
| Rows | 6,525 |
| Societies | 10 (Hadza, Himba, Karo Batak, Los Angeles, Martu, Shuar, Storozhnitsa, Sursurunga, Tsimane, Yasawa) |
| Subjects | 322 |

Underlying experiment is the Barrett et al. vignette battery; the OSF release accompanies
Curtin et al. (2020) on kinship intensity and the use of mental states in moral judgment.
Cite the original papers for the data, not this repo.

### Reproducing the download

```bash
mkdir -p dataset/human_reference/crosscultural/raw
cd dataset/human_reference/crosscultural/raw
V="view_only=0ee63a2f9c6541c3b3ab4c4fe2663a13"
curl -sSL "https://osf.io/download/5e1369f157341901d7803b2b/?$V" -o Intentions_KII_BarrettData.csv
curl -sSL "https://osf.io/download/5e1369f175458401c5f99a82/?$V" -o codebook_Intentions_KII_BarrettData.xlsx
md5sum *
```

Expected checksums:

```
498ac62e4aa30e7f01f6adf6d9da865f  Intentions_KII_BarrettData.csv
c36d9e34e04064f737cf0db170280ad4  codebook_Intentions_KII_BarrettData.xlsx
```

`raw/` is gitignored. Only the derived per-society summary
(`outputs/crosscultural/society_intent_effects.csv`) and these docs are tracked.

---

## Relevant structure

| column | values used here |
|---|---|
| `society` | 10 societies |
| `bank` | `INTENT` (3,853 rows) — the intent manipulation. `MITIGAT` is a separate mitigating-circumstances battery and is **not used**. |
| `condition` | `Intentional`, `Accidental` used here. `Motivated` / `Anti-motivated` are a separate motive manipulation and are not used. |
| `question` | `Badness` (primary), `Punishment`, `Reputation` (robustness) |
| `judgrev` | judgment, 5-point, integer −2 … +2 |

---

## The measure, and why it is not our headline contrast

**There is no failed-attempt condition in this dataset.** Our headline contrast,
`blame(attempted) − blame(accidental)`, cannot be computed from it and is deliberately not
plotted against it. Doing so would repeat the exact error the adult-anchor audit was written
to prevent: quoting a human number for a cell pair the humans were never asked about.

The shared axis is the **intent effect at constant harm**:

```
human_intent_effect  = mean(judgrev | Intentional) − mean(judgrev | Accidental)
model_intent_effect  = cell_intentional − cell_accidental
```

Both hold the outcome constant (harm occurred in both conditions) and vary only whether the
agent intended it. On the model side this is `b_intent` evaluated at outcome = harm, taken
from `outputs/stats/mixed_effects_2x2.csv` (`spec = primary`).

### Aggregation

Subject-level first, then society mean:

1. For each subject, mean `judgrev` within `Intentional` and within `Accidental`.
2. That subject's difference.
3. Society mean over subject differences; 95% CI by bootstrap over subjects (2,000 draws,
   seeded).

Subject-first prevents societies with more items per subject from being weighted more heavily
than their subject count warrants.

### Normalisation to our scale

**Primary — scale-range normalisation.** `judgrev` spans −2 … +2, so its range is 4. Human
differences are divided by 4, putting them in "fraction of the response scale" units, which is
what the model contrasts already are (normalised 0–1 blame). A human effect of +1.0 raw scale
points becomes +0.25.

```
human_scaled = (mean_Intentional − mean_Accidental) / 4
```

This is a **units conversion and nothing more.** It does not make the stimuli, tasks, or
populations comparable. It makes two numbers plottable on one axis.

**Secondary — Cohen's d.** Difference divided by the pooled within-society SD of `judgrev`,
reported in the CSV and a second panel. Models are **not** placed on the d axis: a model's
rating SD across templates is not commensurable with a human sample's between-subject SD, so
standardising both by their own SDs would compare two different quantities.

---

## What the figure may and may not say

Allowed: "every engaged model's intent effect at constant harm falls below the lowest of 10
measured societies, on a scale-normalised axis, with different stimuli."

Not allowed: any statement that a model is "at the level of" a named society; any
developmental or causal inference; any use of this distribution as ground truth for our own
items; any merging of this axis with the `attempted − accidental` ladder figures.

New figure only: `outputs/figures_final/crosscultural_placement.png`. No existing figure is
modified, and nothing is copied to `presentation_figures/` until
`outputs/crosscultural/E4_CROSSCULTURAL.md` reports the verdict against the E4 thresholds.
