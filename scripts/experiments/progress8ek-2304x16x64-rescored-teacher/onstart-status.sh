#!/usr/bin/env bash
# datasetへ触れず、instance準備jobと固定環境のreadbackだけを表示する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"
require_command cat
require_command tmux

job=build_tatara
if [[ -f "$STATE_DIR/$job.done" ]]; then
  status=done
  detail=$(cat "$STATE_DIR/$job.done")
elif [[ -f "$STATE_DIR/$job.failed" ]]; then
  status=failed
  detail=$(cat "$STATE_DIR/$job.failed")
elif tmux has-session -t "$job" 2>/dev/null; then
  status=running
  detail="tmux=$job"
else
  status=pending
  detail="-"
fi

printf '%-18s %-10s %s\n' job status detail
printf '%-18s %-10s %s\n' "$job" "$status" "$detail"
echo
if [[ -f "$MANIFEST_DIR/instance.txt" ]]; then
  cat "$MANIFEST_DIR/instance.txt"
else
  echo "instance manifest: missing"
fi
if [[ -f "$MANIFEST_DIR/capacity-at-create.txt" ]]; then
  cat "$MANIFEST_DIR/capacity-at-create.txt"
fi
echo
echo "dataset download: deferred"
echo "job log: $ONSTART_LOG_DIR/build_tatara.log"
echo "bootstrap log: $(dirname "$EXPERIMENT_ROOT")/onstart.log"
