#!/usr/bin/env bash
# Keep gemini-3.5-flash → gemini-2.5-pro at n_samples=1 running until both hit 894
# story×template pairs, then rebuild stats/figures. Auto-restarts on death or stall.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[ -f .env_agents ] && set -a && source .env_agents && set +a

OUT=outputs/agents/behavior_v3
LOGDIR=outputs/logs
mkdir -p "$OUT" "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER="$LOGDIR/closed_behav_v3_watch_n1_${STAMP}.log"
EXPECTED=894
STALL_MIN=15          # no raw mtime change → restart
MAX_RESTARTS=80
# Default flash→pro. Pro-only: MODELS="gemini-2.5-pro" bash code/watch_google_n1_flash_pro.sh
MODELS=(${MODELS:-gemini-3.5-flash gemini-2.5-pro})

pairs() {
  local f="$1"
  python - "$f" "$EXPECTED" <<'PY'
import csv, os, sys
path, exp = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path) or os.path.getsize(path) == 0:
    print(0); raise SystemExit
with open(path, encoding="utf-8") as fh:
    print(len({(r["template"], r["story_id"]) for r in csv.DictReader(fh)}))
PY
}

raw_for() {
  # gemini-2.5-pro -> raw_gemini-2_5-pro.csv
  local m="$1"
  echo "$OUT/raw_$(echo "$m" | tr '.' '_').csv"
}

both_done() {
  local ok=1 line="coverage"
  for m in "${MODELS[@]}"; do
    local n
    n=$(pairs "$(raw_for "$m")")
    line+=" $m=$n/$EXPECTED"
    [[ "$n" -ge "$EXPECTED" ]] || ok=0
  done
  echo "$line" | tee -a "$MASTER"
  [[ "$ok" -eq 1 ]]
}

echo "=== watch n1 ${MODELS[*]} $STAMP ===" | tee "$MASTER"
echo "auto-restart on death/stall; n_samples=1; rate-limit 429s retry (not treated as billing)" | tee -a "$MASTER"

restarts=0
while ! both_done; do
  if [[ "$restarts" -ge "$MAX_RESTARTS" ]]; then
    echo "!! hit MAX_RESTARTS=$MAX_RESTARTS — stopping" | tee -a "$MASTER"
    exit 2
  fi
  RUNLOG="$LOGDIR/closed_behav_v3_google_n1_${STAMP}_r${restarts}.log"
  echo "$(date): launch restart=$restarts -> $RUNLOG" | tee -a "$MASTER"
  python -u code/03_behavioral.py \
    --backend google --scoring sampling --n_samples 1 --temperature 1.0 \
    --templates human_verbatim para_wrong7 punish7 \
    --out_dir "$OUT" --skip_existing \
    --models "${MODELS[@]}" \
    >"$RUNLOG" 2>&1 &
  pid=$!
  echo "  pid=$pid models=${MODELS[*]}" | tee -a "$MASTER"
  # Stall clock starts at launch (not at stale raw mtimes from hours ago).
  last_progress=$(date +%s)
  last_newest=0
  for m in "${MODELS[@]}"; do
    t=$(stat -c %Y "$(raw_for "$m")" 2>/dev/null || echo 0)
    [[ "$t" -gt "$last_newest" ]] && last_newest=$t
  done

  while kill -0 "$pid" 2>/dev/null; do
    if both_done; then
      echo "$(date): both done — killing worker $pid" | tee -a "$MASTER"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      break 2
    fi
    newest=0
    for m in "${MODELS[@]}"; do
      t=$(stat -c %Y "$(raw_for "$m")" 2>/dev/null || echo 0)
      [[ "$t" -gt "$newest" ]] && newest=$t
    done
    now=$(date +%s)
    if [[ "$newest" -gt "$last_newest" ]]; then
      last_newest=$newest
      last_progress=$now
      prog="$(date): progress"
      for m in "${MODELS[@]}"; do prog+=" $m=$(pairs "$(raw_for "$m")")"; done
      echo "$prog" | tee -a "$MASTER"
    fi
    if [[ $((now - last_progress)) -gt $((STALL_MIN * 60)) ]]; then
      echo "$(date): STALL ${STALL_MIN}m with no new rows — restarting pid=$pid" | tee -a "$MASTER"
      kill "$pid" 2>/dev/null || true
      sleep 2
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      break
    fi
    # Only abort on true money exhaustion (not RPM/RPD "quota" 429s).
    if grep -qiE 'credits are depleted|prepayment credits|insufficient funds' "$RUNLOG" 2>/dev/null; then
      echo "$(date): TRUE BILLING STOP — not restarting" | tee -a "$MASTER"
      tail -5 "$RUNLOG" | tee -a "$MASTER"
      exit 3
    fi
    sleep 60
  done
  wait "$pid" 2>/dev/null || true
  if both_done; then break; fi
  restarts=$((restarts + 1))
  echo "$(date): worker exited; sleep 20 then resume ($restarts)" | tee -a "$MASTER"
  sleep 20
done

echo "$(date): rebuilding stats + figures" | tee -a "$MASTER"
python -u code/06_stats.py --behavior "$OUT" --out outputs/agents/stats 2>&1 | tee -a "$MASTER"
python -u code/09_agent_figures.py 2>&1 | tee -a "$MASTER" || true
python -u code/29_dual_human_ladders.py 2>&1 | tee -a "$MASTER" || true
bash code/collect_figures_final.sh 2>&1 | tee -a "$MASTER" || true
python -u code/experiments/53_figures_final_manifest.py 2>&1 | tee -a "$MASTER" || true
echo "=== DONE $(date); restarts=$restarts ===" | tee -a "$MASTER"
both_done
echo "WATCH_N1_COMPLETE" | tee -a "$MASTER"
