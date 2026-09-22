# E8 — status (2026-09-22): complete

All of E8a–E8g is done and committed. Nothing is in flight.

| stage | result | where |
|---|---|---|
| E8a–E8e (Pythia-1.4B) | double dissociation by read position; P1, P4 falsified | `E8_CHECKPOINT_EMERGENCE.md` |
| E8c gate | passed | `E8_STEP0_GATE.md` |
| E8f behavioural overlay | 0 of 26 checkpoints gradeable (max rating_std 0.086 vs 0.2191); panel empty | `checkpoint_behavioral_overlay.png`, section in `E8_CHECKPOINT_EMERGENCE.md` |
| E8g OLMo-2-7B | **replicates** under amendment 3.1: 5/5 depths at `belief_last`, 4/5 at `last` (same as Pythia) | `E8G_OLMO_REPLICATION.md`, `checkpoint_trajectory_OLMo-2-1124-7B_joined.png` |

Deviations are all in `E8_PRESPEC.md` amendments 3 and 3.1, committed before any OLMo β:
eight revisions not ~20, no step-0 gate (none published), n=8 shape fits not scored.

Optional follow-ups, none required by the prespec: the full ~20-revision OLMo sweep (would
narrow the 3B–9B-token onset bracket); a few stage-2 revisions to separate the annealing dip
from the float16/float32 seam.
