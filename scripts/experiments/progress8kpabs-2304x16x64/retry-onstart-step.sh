#!/usr/bin/env bash
# operatorが原因確認後に、指定した失敗markerだけを明示的に解除してonstartを再実行する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ -n "${RETRY_STEP:-}" ]] || fail "RETRY_STEPを明示してください"
case "$RETRY_STEP" in
  build_tatara|build_rshogi|download_training|download_validation|download_progress|prepare_data) ;;
  *) fail "RETRY_STEPが対象外です: $RETRY_STEP" ;;
esac
[[ "${CONFIRM_RETRY_STEP:-}" == "$RETRY_STEP" ]] \
  || fail "CONFIRM_RETRY_STEP=$RETRY_STEPを同時に明示してください"
failed_marker="$STATE_DIR/$RETRY_STEP.failed"
[[ -f "$failed_marker" ]] || fail "失敗markerがありません: $failed_marker"
[[ ! -f "$STATE_DIR/$RETRY_STEP.done" ]] || fail "完了済みstepはretryしません"
tmux has-session -t "$RETRY_STEP" 2>/dev/null && fail "stepのtmuxがまだ実行中です"

echo "[retry] 直前の失敗内容: $(cat "$failed_marker")"
rm -- "$failed_marker"
TATARA_COMMIT=$(manifest_value "$MANIFEST_DIR/source-revisions.txt" tatara)
export TATARA_COMMIT
exec bash "$EXPERIMENT_ROOT/onstart.sh"
