# E8 — status (2026-09-22, 17:30): complete, including the full OLMo sweep

Nothing is in flight.

| stage | result | where |
|---|---|---|
| E8a–E8e (Pythia-1.4B, 26 revisions) | double dissociation by read position; P1, P4 falsified | `E8_CHECKPOINT_EMERGENCE.md` |
| E8c gate | passed | `E8_STEP0_GATE.md` |
| E8f Pythia behaviour | 0 of 26 gradeable (max rating_std 0.086) | `checkpoint_behavioral_overlay.png` |
| E8g OLMo-2-7B (20 revisions) | **replicates**: 5/5 at `belief_last`, 4/5 at `last`, same as the pre-registered 8-revision run; onset 5B–9B tokens at both positions | `E8G_OLMO_REPLICATION.md`, `checkpoint_trajectory_OLMo-2-1124-7B_joined.png` |
| E8f OLMo behaviour | 0 of 20 stage-1 revisions gradeable (max 0.062); first gradeable point is SFT | same doc and figure |
| RDM snapshots | both models, intent-first and outcome-first sorts | `rdm_snapshots_<model>_{intent,outcome}first.png` |

Deviations: `E8_PRESPEC.md` amendments 3, 3.1, 3.2, each committed before the compute it covers.

Optional follow-ups: a few stage-2 (annealing) revisions to separate the base-model dip
from the float16/float32 seam; denser revisions between step 1,000 and 2,000 to pin the onset.
