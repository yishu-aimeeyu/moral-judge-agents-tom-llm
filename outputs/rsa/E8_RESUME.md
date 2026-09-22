# E8 — status (2026-09-22, 16:15): full OLMo sweep in flight

E8a–E8g are complete on the first 8 OLMo revisions (`E8G_OLMO_REPLICATION.md`: replicates,
5/5 and 4/5). The user asked for the finer OLMo trajectory; amendment 3.2 (committed before
compute) extends to 20 stage-1 revisions and adds behavioural scoring on them.

## In flight
Slurm job **23498049** (`e8g_full`, pi_jhm, A100), log `outputs/logs/gpu_23498049.log`,
command file `outputs/logs/e8g_full_cmd.sh`:
1. extract the 12 new revisions into `outputs/acts_ckpt/` (resumable; skips those on disk)
2. behavioural logprob-EV on all 20 -> `outputs/rsa/checkpoint_behavioral_OLMo-2-1124-7B.csv`
3. delete the stage-1 revisions from the scratch HF cache
If preempted or killed, resubmit the same command: both steps skip finished revisions.

## When it finishes, in order
1. `outputs/acts_ckpt/OLMo*.npz` = 20 files; behavioural CSV has 20 rows.
2. CPU: `74_e8_checkpoint_betas.py --model OLMo-2-1124-7B --no-step0-gate --jobs 8`,
   then `75_e8_shape_test.py` with `--betas/--out` on the OLMo CSV.
3. `78_e8g_join_poststages.py --plot-only` (post-stage CSV already computed), with its
   bottom panel extended to draw stage-1 behavioural coefficients where gradeable.
4. `79_e8g_verdict.py`: the verdict is read at step 928,646 so it should not change; report
   the refined onset bracket. The 8-revision verdict stays the pre-registered one.
5. `80_e8_rdm_snapshots.py --model OLMo-2-1124-7B --sort intent|outcome`, with snapshots
   moved onto the new revisions around the onset; delete
   `rdm_snapshots_cache_OLMo-2-1124-7B.npz` first.
