# E8c — step-0 sanity gate

**Verdict: PASS**

The rule, its Bonferroni correction and the consequence of failure were fixed in `outputs/experiments/E8_PRESPEC.md` section 4 before any checkpoint was downloaded. At random initialisation a network cannot have learned the moral belief × outcome factors, so `intent` and `outcome` must both sit inside their own within-scenario permutation null in every cell.

- cells: **10** (layer × position), so the per-cell two-sided threshold is the 99.75th percentile of the null (family-wise α = 0.05)
- cells exceeding the *uncorrected* 95% band: **0** of 20 predictor-cells, against **1.0** expected by chance on a clean pipeline
- cells failing the corrected rule: **0**

Relative depth [0.0] is excluded per amendment 2.1: at a single-token read position the layer-0 activation is a token-embedding lookup, and every vignette ends in `.` or `).`, so it takes 2 distinct values across 298 items at every checkpoint. The first run of this gate included it and also passed (0 of 24 predictor-cells outside the uncorrected band); dropping cells tightens a Bonferroni family rather than loosening it.

## All step-0 betas, including the covariates

`lexical` and `scenario` are **expected to be large here and that is not a failure**: a random network is close to a random projection, and random projections approximately preserve input similarity, so token-overlap structure survives into the hidden states. This was stated in the prespec before the numbers existed, so that a large step-0 lexical β could not later be misread as contamination. The gate is on `intent` and `outcome` only.

| position | layer | predictor | β | null q97.5(\|β\|) | inside null | p_perm |
|---|---:|---|---:|---:|---|---:|
| `belief_last` | 5 | `intent` | -0.0032 | 0.0101 | yes | 0.5267 |
| `belief_last` | 5 | `outcome` | -0.0130 | 0.0182 | yes | 0.3573 |
| `belief_last` | 5 | `interaction` | +0.0118 | 0.0314 | yes | 0.3953 |
| `belief_last` | 5 | `scenario` | +0.1286 | 0.1377 | yes | 1.0000 |
| `belief_last` | 5 | `lexical` | +0.4002 | 0.3746 | **no** | 0.0005 |
| `belief_last` | 10 | `intent` | -0.0015 | 0.0092 | yes | 0.7656 |
| `belief_last` | 10 | `outcome` | -0.0113 | 0.0169 | yes | 0.4833 |
| `belief_last` | 10 | `interaction` | +0.0068 | 0.0288 | yes | 0.6192 |
| `belief_last` | 10 | `scenario` | +0.1315 | 0.1391 | yes | 1.0000 |
| `belief_last` | 10 | `lexical` | +0.3880 | 0.3672 | **no** | 0.0005 |
| `belief_last` | 14 | `intent` | -0.0008 | 0.0094 | yes | 0.8826 |
| `belief_last` | 14 | `outcome` | -0.0100 | 0.0163 | yes | 0.5822 |
| `belief_last` | 14 | `interaction` | +0.0040 | 0.0293 | yes | 0.7816 |
| `belief_last` | 14 | `scenario` | +0.1376 | 0.1448 | yes | 1.0000 |
| `belief_last` | 14 | `lexical` | +0.3617 | 0.3429 | **no** | 0.0005 |
| `belief_last` | 19 | `intent` | -0.0007 | 0.0095 | yes | 0.8991 |
| `belief_last` | 19 | `outcome` | -0.0100 | 0.0164 | yes | 0.5697 |
| `belief_last` | 19 | `interaction` | +0.0032 | 0.0296 | yes | 0.8251 |
| `belief_last` | 19 | `scenario` | +0.1371 | 0.1441 | yes | 1.0000 |
| `belief_last` | 19 | `lexical` | +0.3640 | 0.3459 | **no** | 0.0005 |
| `belief_last` | 24 | `intent` | +0.0004 | 0.0093 | yes | 0.9480 |
| `belief_last` | 24 | `outcome` | -0.0087 | 0.0160 | yes | 0.7071 |
| `belief_last` | 24 | `interaction` | -0.0017 | 0.0289 | yes | 0.8981 |
| `belief_last` | 24 | `scenario` | +0.1375 | 0.1439 | yes | 1.0000 |
| `belief_last` | 24 | `lexical` | +0.3619 | 0.3461 | **no** | 0.0005 |
| `last` | 5 | `intent` | -0.0077 | 0.0078 | yes | 0.0300 |
| `last` | 5 | `outcome` | -0.0144 | 0.0180 | yes | 0.3593 |
| `last` | 5 | `interaction` | +0.0232 | 0.0230 | **no** | 0.0240 |
| `last` | 5 | `scenario` | +0.1235 | 0.1323 | yes | 1.0000 |
| `last` | 5 | `lexical` | +0.4688 | 0.4414 | **no** | 0.0005 |
| `last` | 10 | `intent` | -0.0067 | 0.0073 | yes | 0.0490 |
| `last` | 10 | `outcome` | -0.0138 | 0.0170 | yes | 0.3433 |
| `last` | 10 | `interaction` | +0.0194 | 0.0202 | yes | 0.0295 |
| `last` | 10 | `scenario` | +0.1284 | 0.1357 | yes | 1.0000 |
| `last` | 10 | `lexical` | +0.4474 | 0.4252 | **no** | 0.0005 |
| `last` | 14 | `intent` | -0.0067 | 0.0073 | yes | 0.0405 |
| `last` | 14 | `outcome` | -0.0136 | 0.0160 | yes | 0.2314 |
| `last` | 14 | `interaction` | +0.0192 | 0.0197 | yes | 0.0300 |
| `last` | 14 | `scenario` | +0.1352 | 0.1418 | yes | 1.0000 |
| `last` | 14 | `lexical` | +0.4185 | 0.3984 | **no** | 0.0005 |
| `last` | 19 | `intent` | -0.0069 | 0.0070 | yes | 0.0300 |
| `last` | 19 | `outcome` | -0.0138 | 0.0157 | yes | 0.1824 |
| `last` | 19 | `interaction` | +0.0201 | 0.0187 | **no** | 0.0190 |
| `last` | 19 | `scenario` | +0.1361 | 0.1421 | yes | 1.0000 |
| `last` | 19 | `lexical` | +0.4148 | 0.3965 | **no** | 0.0005 |
| `last` | 24 | `intent` | -0.0057 | 0.0068 | yes | 0.0815 |
| `last` | 24 | `outcome` | -0.0129 | 0.0153 | yes | 0.2899 |
| `last` | 24 | `interaction` | +0.0156 | 0.0179 | yes | 0.0500 |
| `last` | 24 | `scenario` | +0.1365 | 0.1418 | yes | 1.0000 |
| `last` | 24 | `lexical` | +0.4130 | 0.3968 | **no** | 0.0005 |
