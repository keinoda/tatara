#!/usr/bin/env bash
# onstartの各非同期stepをread-onlyで一覧表示する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

steps=(build_tatara build_rshogi download_training download_validation download_progress prepare_data)
printf '%-22s %-10s %s\n' step status detail
for step in "${steps[@]}"; do
  if [[ -f "$STATE_DIR/$step.done" ]]; then
    status=done
    detail=$(cat "$STATE_DIR/$step.done")
  elif [[ -f "$STATE_DIR/$step.failed" ]]; then
    status=failed
    detail=$(cat "$STATE_DIR/$step.failed")
  elif tmux has-session -t "$step" 2>/dev/null; then
    status=running
    detail="tmux=$step"
  else
    status=pending
    detail="-"
  fi
  printf '%-22s %-10s %s\n' "$step" "$status" "$detail"
done

echo
partial="$TRAIN_PSV.partial"
if [[ -f "$TRAIN_PSV" ]]; then
  training_bytes=$(file_size "$TRAIN_PSV")
  training_state=complete
elif [[ -f "$partial" ]]; then
  training_bytes=$(file_size "$partial")
  training_state=appending
else
  training_bytes=0
  training_state=pending
fi
append_markers=$(find "$STATE_DIR/training-append" -maxdepth 1 -type f -name '*.done' 2>/dev/null | awk 'END {print NR+0}')
printf 'training: state=%s appended_shards=%s/34 bytes=%s/%s survey_shard=%s\n' \
  "$training_state" "$append_markers" "$training_bytes" "$TRAIN_EXPECTED_BYTES" \
  "$([[ -f "$TRAIN_SURVEY_SHARD" ]] && printf retained || printf absent)"
echo
echo "logs: $ONSTART_LOG_DIR"
echo "bootstrap log: $(dirname "$EXPERIMENT_ROOT")/onstart.log"
