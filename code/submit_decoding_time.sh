#!/usr/bin/env bash
# submit_decoding_time.sh -- D1-D3 + R5: token trajectories -> decoding over time.
#
# Per model, one GPU job extracts every prompt condition (81), then CPU jobs chained behind
# it run the temporal decoding (82) per condition and one cross-prompt job. A failed
# extraction cancels its own decoding jobs only.
#
#   bash code/submit_decoding_time.sh                     # default 3 base/instruct pairs
#   MODELS="Qwen/Qwen2.5-7B" bash code/submit_decoding_time.sh
#   SCORE_EXACT=1 bash code/submit_decoding_time.sh       # also rate with participant_exact
#
# Disk: ~1.5-2 GB per 7B model x prompt in $TOM_TRAJ_DIR (scratch), ~15 GB per model.

set -euo pipefail
PROJ=/home/aimeeyu/tom_project
cd "$PROJ"
export TOM_TRAJ_DIR="${TOM_TRAJ_DIR:-/orcd/scratch/orcd/007/$USER/acts_traj}"
MODELS="${MODELS:-Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct unsloth/Meta-Llama-3.1-8B unsloth/Meta-Llama-3.1-8B-Instruct}"
PROMPTS="story human_verbatim participant_exact blame_w1 wrong_w2 punish_w1"
EXP=code/experiments

for M in $MODELS; do
  TAG="${M##*/}"
  # chat=auto reproduces 03_behavioral.py (Qwen BASE tokenizers ship a chat template);
  # chat=off on the bare story gives the raw-text condition for every model.
  GPU_CMD="python $EXP/81_extract_token_trajectories.py --models $M --prompts $PROMPTS --chat auto \
&& python $EXP/81_extract_token_trajectories.py --models $M --prompts story --chat off \
&& python $EXP/81_extract_token_trajectories.py --models $M --prompts story --chat off --variant no_foreshadow"
  [ -n "${SCORE_EXACT:-}" ] && GPU_CMD="$GPU_CMD && python $EXP/84_participant_exact_prompts.py --run --models $M"
  JID=$(PARSABLE=1 JOBNAME="traj_$TAG" TIME=04:00:00 bash engaging/submit_gpu.sh "$GPU_CMD")
  echo "$TAG: extraction job $JID"

  for P in $PROMPTS; do
    DEP="afterok:$JID" JOBNAME="dec_${TAG}_$P" TIME=03:00:00 CPUS=16 \
      bash engaging/submit_cpu.sh "python $EXP/82_temporal_decoding.py --model $TAG --prompt $P --chat auto" >/dev/null
  done
  for V in original no_foreshadow; do
    DEP="afterok:$JID" JOBNAME="dec_${TAG}_${V}_raw" TIME=03:00:00 CPUS=16 \
      bash engaging/submit_cpu.sh "python $EXP/82_temporal_decoding.py --model $TAG --variant $V --prompt story --chat off" >/dev/null
  done
  # cross-prompt is only defined within one chat setting; story under chat=auto is included
  DEP="afterok:$JID" JOBNAME="cross_$TAG" TIME=02:00:00 CPUS=16 \
    bash engaging/submit_cpu.sh "python $EXP/82_temporal_decoding.py --model $TAG --cross --chat auto --prompts $PROMPTS" >/dev/null
  echo "$TAG: decoding jobs queued behind $JID"
done
