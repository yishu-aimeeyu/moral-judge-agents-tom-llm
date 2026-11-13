#!/usr/bin/env bash
# Closed moral BEHAVIORAL rescore on master v3 — the job that updates the ladder figures.
#
# Priority: money goes here first. Does NOT run ToM, does NOT run thinking-dose.
#
# Why not "resume" the overnight closed_reasoning run: those rows went through the
# midpoint-imputing / range-clamping parser and are quarantined under
# outputs/closed_reasoning/_prebilling_imputed/. Resuming them would poison the figures.
# This write is a clean v3 pass into outputs/agents/behavior_v3/, with resume on its own
# raw_*.csv files if a provider dies mid-model.
#
# Roster (budget-friendly size ladder; opus-4-6 not opus-5; gemini-2.5-pro not 3.1-preview).
# Within each provider, figure-critical models (those in the STALE *_all.png ladders) run
# first, then the rest of the size ladder.
#
# Config: 3 templates matching the prior closed ladder × n_samples=5, sampling, T=1.
# Est. ~$4–8 compute; keys should have ~$10/$10/$15.
#
# Usage:
#   bash code/submit_closed_behavioral_v3.sh
#   PROVIDER=openai bash code/submit_closed_behavioral_v3.sh   # one provider only

set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env_agents ] && set -a && source .env_agents && set +a

OUT="outputs/agents/behavior_v3"
mkdir -p "$OUT" outputs/logs outputs/agents/stats
STAMP=$(date +%Y%m%d_%H%M%S)
PROVIDER="${PROVIDER:-all}"

TEMPLATES=(human_verbatim para_wrong7 punish7)
N_SAMPLES=5
TEMPERATURE=1.0

# Order = spend priority. Figure-critical first, then size-ladder completion.
OPENAI_MODELS=(gpt-4o gpt-4o-mini gpt-5.4-mini gpt-5.5)
ANTHROPIC_MODELS=(claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6)
GOOGLE_MODELS=(gemini-2.5-flash gemini-2.5-pro gemini-3.5-flash)

# Plan D: finish gpt-5.5 + opus + gemini-3.5-flash only. Skips gemini-2.5-pro
# (thinking-token burn). Usage: PLAN_D=1 bash code/submit_closed_behavioral_v3.sh
if [[ "${PLAN_D:-0}" == "1" ]]; then
  GOOGLE_MODELS=(gemini-3.5-flash)
  echo "PLAN_D=1 — Google roster: ${GOOGLE_MODELS[*]} (NOT running gemini-2.5-pro)"
fi

run_provider() {
  local backend="$1"; shift
  local log="outputs/logs/closed_behav_v3_${backend}_${STAMP}.log"
  echo "launching $backend (${#} models) -> $log"
  python -u code/03_behavioral.py \
    --backend "$backend" \
    --scoring sampling \
    --n_samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --templates "${TEMPLATES[@]}" \
    --out_dir "$OUT" \
    --skip_existing \
    --models "$@" \
    >"$log" 2>&1
  echo "  $backend exit=$?"
}

echo "=== closed behavioral v3 $STAMP ==="
echo "out_dir=$OUT  templates=${TEMPLATES[*]}  n_samples=$N_SAMPLES"
echo "NOT running: ToM, reasoning dose, opus-5, o3, gemini-3.1-pro-preview"
echo ""

pids=()
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "openai" ]]; then
  run_provider openai "${OPENAI_MODELS[@]}" &
  pids+=($!)
fi
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "anthropic" ]]; then
  run_provider anthropic "${ANTHROPIC_MODELS[@]}" &
  pids+=($!)
fi
if [[ "$PROVIDER" == "all" || "$PROVIDER" == "google" ]]; then
  run_provider google "${GOOGLE_MODELS[@]}" &
  pids+=($!)
fi

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "!! pid $pid failed"
    fail=1
  fi
done

echo ""
echo "=== rebuild agents stats + ladders from v3 ==="
# Point stats at the new directory. Old v1 agents/behavior stays untouched as provenance.
python -u code/06_stats.py \
  --behavior "$OUT" \
  --out outputs/agents/stats \
  2>&1 | tee "outputs/logs/closed_behav_v3_stats_${STAMP}.log"

python -u code/29_dual_human_ladders.py \
  2>&1 | tee -a "outputs/logs/closed_behav_v3_stats_${STAMP}.log" || true

# Promote restored *_all.png out of quarantine when closed rows are present in agents/stats
python -u code/collect_figures_final.sh 2>/dev/null || bash code/collect_figures_final.sh || true
python -u code/experiments/53_figures_final_manifest.py || true

echo "=== done $(date); fail=$fail ==="
echo "Check coverage: python -c \"import csv,glob; ...\" or ls $OUT/raw_*.csv"
exit $fail
