#!/usr/bin/env bash
# baseまたはbucket8 runを、認証付きmonitor実装で配信してreadbackする。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

run_name=$(require_training_phase)
[[ -n "${MONITOR_USER:-}" ]] || fail "MONITOR_USERを明示してください"
[[ -n "${MONITOR_PASSWORD:-}" ]] || fail "MONITOR_PASSWORDを明示してください"
[[ -n "${MONITOR_PUBLIC_URL:-}" ]] || fail "Vast.aiのmapped URLをMONITOR_PUBLIC_URLで明示してください"
[[ "$MONITOR_PUBLIC_URL" != */ ]] || fail "MONITOR_PUBLIC_URLの末尾にslashを付けないでください"
require_command curl
require_command python3
require_command sleep
require_command tmux
[[ -f "$MONITOR_IMPLEMENTATION_DIR/monitor.py" ]] || fail "monitor rendererがありません"
[[ -f "$MONITOR_IMPLEMENTATION_DIR/monitor_server.py" ]] || fail "monitor serverがありません"

readonly RUN_NAME="$run_name"
readonly RUN_ROOT="$RUNS_ROOT/$RUN_NAME"
readonly OUTPUT_DIR="$MONITOR_ROOT/$RUN_NAME"
readonly GATE_DIR="$GATES_ROOT/$RUN_NAME"
readonly RENDER_SESSION="monitor-render-$RUN_NAME"
readonly HTTP_SESSION="monitor-http-$RUN_NAME"

[[ ! -e "$GATE_DIR/monitor.done" ]] || fail "monitor gateは既に完了しています"
tmux has-session -t "$RENDER_SESSION" 2>/dev/null && fail "tmux sessionが既にあります: $RENDER_SESSION"
tmux has-session -t "$HTTP_SESSION" 2>/dev/null && fail "tmux sessionが既にあります: $HTTP_SESSION"
if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:$MONITOR_PORT -sTCP:LISTEN >/dev/null 2>&1; then
  fail "port $MONITOR_PORTは既に使用中です"
fi

mkdir -p "$OUTPUT_DIR" "$GATE_DIR" "$EXPERIMENT_ROOT/logs/monitor"
python3 "$MONITOR_IMPLEMENTATION_DIR/monitor.py" \
  --run-name "$RUN_NAME" --run-root "$RUN_ROOT" --output-dir "$OUTPUT_DIR" \
  --milestone-interval "$MONITOR_MILESTONE_INTERVAL" --once

printf -v render_command 'exec python3 %q --run-name %q --run-root %q --output-dir %q --milestone-interval %q >>%q 2>&1' \
  "$MONITOR_IMPLEMENTATION_DIR/monitor.py" "$RUN_NAME" "$RUN_ROOT" "$OUTPUT_DIR" \
  "$MONITOR_MILESTONE_INTERVAL" \
  "$EXPERIMENT_ROOT/logs/monitor/render-$RUN_NAME.log"
printf -v http_command 'exec python3 %q --root %q --bind 0.0.0.0 --port %q >>%q 2>&1' \
  "$MONITOR_IMPLEMENTATION_DIR/monitor_server.py" "$OUTPUT_DIR" "$MONITOR_PORT" \
  "$EXPERIMENT_ROOT/logs/monitor/http-$RUN_NAME.log"

tmux new-session -d -s "$RENDER_SESSION" "bash -lc $(printf '%q' "$render_command")"
tmux new-session -d -s "$HTTP_SESSION" \
  -e "MONITOR_USER=$MONITOR_USER" -e "MONITOR_PASSWORD=$MONITOR_PASSWORD" \
  "bash -lc $(printf '%q' "$http_command")"

until curl --silent --fail --user "$MONITOR_USER:$MONITOR_PASSWORD" \
  "http://127.0.0.1:$MONITOR_PORT/status.json" >/dev/null 2>&1; do
  tmux has-session -t "$RENDER_SESSION" 2>/dev/null \
    || fail "monitor rendererがreadback前に終了しました"
  tmux has-session -t "$HTTP_SESSION" 2>/dev/null \
    || fail "monitor HTTP serverがreadback前に終了しました"
  sleep 1
done
curl --silent --show-error --fail --user "$MONITOR_USER:$MONITOR_PASSWORD" \
  "http://127.0.0.1:$MONITOR_PORT/status.json" >/dev/null \
  || fail "monitor local readbackに失敗しました"
unauth_status=$(curl --silent --output /dev/null --write-out '%{http_code}' \
  "http://127.0.0.1:$MONITOR_PORT/")
[[ "$unauth_status" == 401 ]] \
  || fail "monitorの未認証requestが401になりません: $unauth_status"
curl --silent --show-error --fail --user "$MONITOR_USER:$MONITOR_PASSWORD" \
  "$MONITOR_PUBLIC_URL/status.json" >/dev/null \
  || fail "monitor mapped readbackに失敗しました: $MONITOR_PUBLIC_URL"

{
  printf 'checked_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'training_phase=%s\n' "$TRAINING_PHASE"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'run_root=%s\n' "$RUN_ROOT"
  printf 'port=%s\n' "$MONITOR_PORT"
  printf 'public_url=%s\n' "$MONITOR_PUBLIC_URL"
  printf 'authentication=basic\n'
  printf 'renderer_tmux=%s\n' "$RENDER_SESSION"
  printf 'http_tmux=%s\n' "$HTTP_SESSION"
} | write_manifest_atomic "$GATE_DIR/monitor.txt"
date -u +%FT%TZ >"$GATE_DIR/monitor.done"
echo "[monitor] $RUN_NAME のlocal/mapped readbackと未認証401を確認しました"
