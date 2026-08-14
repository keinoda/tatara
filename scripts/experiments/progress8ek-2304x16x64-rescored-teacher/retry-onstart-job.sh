#!/usr/bin/env bash
# operatorが原因を確認したbuild失敗markerだけを明示確認付きで解除する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"
require_command cat
require_command rm
require_command tmux

readonly JOB_NAME="build_tatara"
[[ "${CONFIRM_RETRY_JOB:-}" == "$JOB_NAME" ]] \
  || fail "CONFIRM_RETRY_JOB=$JOB_NAMEを明示してください"
readonly FAILED_MARKER="$STATE_DIR/$JOB_NAME.failed"
[[ -f "$FAILED_MARKER" ]] || fail "失敗markerがありません: $FAILED_MARKER"
[[ ! -f "$STATE_DIR/$JOB_NAME.done" ]] || fail "完了済みjobはretryしません"
tmux has-session -t "$JOB_NAME" 2>/dev/null && fail "jobのtmuxがまだ実行中です"

echo "[retry] 直前の失敗内容: $(cat "$FAILED_MARKER")"
rm -- "$FAILED_MARKER"
TATARA_COMMIT=$(manifest_value "$MANIFEST_DIR/instance.txt" tatara)
export TATARA_COMMIT
exec bash "$EXPERIMENT_ROOT/onstart.sh"
