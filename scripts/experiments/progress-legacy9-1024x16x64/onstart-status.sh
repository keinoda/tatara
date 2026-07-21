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
echo "logs: $ONSTART_LOG_DIR"
echo "bootstrap log: $(dirname "$EXPERIMENT_ROOT")/onstart.log"
