#!/usr/bin/env bash
# 完了した通常学習runのmonitor sessionを止め、port 6001をslot 8追加学習runの
# monitorへ切り替える。trainerは起動しない。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_command tmux
readonly RUN_NAME="$(run_name_for_phase bucket8)"
readonly RUN_ROOT="$RUNS_ROOT/$RUN_NAME"
readonly MONITOR_ENV="$RUN_ROOT/config/monitor.env"
[[ -f "$RUN_ROOT/config/manifest.txt" ]] || fail "slot 8 runが準備されていません: $RUN_ROOT"
[[ -f "$MONITOR_ENV" ]] || fail "monitor.envがありません: $MONITOR_ENV"
[[ ! -e "$GATES_ROOT/$RUN_NAME/monitor.done" ]] || fail "slot 8 monitor gateは既に完了しています"
require_no_trainer_process

# 前段runのtrainerが終了済みであることを確認してから、そのmonitorだけを止める。
for previous in "$(run_name_for_phase base)" "$BASE_FINAL_RUN_NAME"; do
  previous_root="$RUNS_ROOT/$previous"
  [[ -d "$previous_root" ]] || continue
  if tmux has-session -t "train-$previous" 2>/dev/null; then
    fail "前段runのtrainer sessionが残っています: train-$previous"
  fi
  [[ -f "$previous_root/state/training.ended" ]] \
    || fail "前段runの終了記録がありません: $previous_root/state/training.ended"
  for session in "monitor-render-$previous" "monitor-http-$previous"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "[switch-monitor] 停止: $session"
    fi
  done
done
while command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$MONITOR_PORT" -sTCP:LISTEN >/dev/null 2>&1; do
  sleep 1
done

set -a
# shellcheck disable=SC1090
source "$MONITOR_ENV"
set +a
export TRAINING_PHASE=bucket8
exec "$EXPERIMENT_SCRIPT_DIR/run-monitor.sh"
