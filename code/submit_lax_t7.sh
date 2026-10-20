#!/usr/bin/env bash
# submit_lax_t7.sh -- Lax 9/29 T7: logprob ratings of ablated stories (93 --build) under the 7 factorial
# templates, one GPU job per base/instruct pair (~25-30 min each on an L40S), then 93 --analyze on CPU.
#
#   bash code/submit_lax_t7.sh
#   PAIRS="Qwen/Qwen2.5-7B,Qwen/Qwen2.5-7B-Instruct" bash code/submit_lax_t7.sh

set -euo pipefail
PROJ=/home/aimeeyu/tom_project
cd "$PROJ"
PAIRS="${PAIRS:-Qwen/Qwen2.5-7B,Qwen/Qwen2.5-7B-Instruct unsloth/Meta-Llama-3.1-8B,unsloth/Meta-Llama-3.1-8B-Instruct}"
T7=outputs/lax_0929/T7
TPL="human_verbatim blame_w1 blame_w2 wrong_w1 wrong_w2 punish_w1 punish_w2"
JIDS=()
for P in $PAIRS; do
  MODELS="${P//,/ }"
  CMD="python code/experiments/93_lax_priors_ablation.py --build"
  for A in no_story background_only belief_removed outcome_removed; do
    CMD="$CMD && python code/03_behavioral.py --backend hf --scoring logprob --models $MODELS --templates $TPL --csv $T7/stimuli_$A.csv --out_dir $T7/behavior_$A --skip_existing"
  done
  J=$(PARSABLE=1 JOBNAME="t7_${P%%,*}" TIME=01:00:00 bash engaging/submit_gpu.sh "$CMD")
  J=${J%%[^0-9]*}
  echo "$P: GPU job $J"
  JIDS+=("$J")
done
DEP="afterok:$(IFS=:; echo "${JIDS[*]}")" JOBNAME=t7_analyze TIME=01:00:00 CPUS=8 \
  bash engaging/submit_cpu.sh ".venv/bin/python code/experiments/93_lax_priors_ablation.py --analyze"
