#!/usr/bin/env bash
# 準備済みのslot 8追加学習runを、manifestとの再照合後にtmuxで開始する。
# 明示的に実行したときだけtrainerを起動する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_command tmux
require_command nvidia-smi
require_clean_experiment_checkout
readonly RUN_NAME="$(run_name_for_phase bucket8)"
readonly RUN_ROOT="$RUNS_ROOT/$RUN_NAME"
readonly MANIFEST="$RUN_ROOT/config/manifest.txt"
readonly COMMAND_FILE="$RUN_ROOT/config/command.sh"
readonly LAUNCH_FILE="$RUN_ROOT/config/launch.sh"
readonly GATE_DIR="$GATES_ROOT/$RUN_NAME"

[[ -f "$MANIFEST" ]] || fail "slot 8 runが準備されていません: $MANIFEST"
[[ ! -e "$RUN_ROOT/state/training.started" ]] || fail "このrunは既に開始されています: $RUN_ROOT"
tmux has-session -t "$BUCKET8_TRAINER_SESSION" 2>/dev/null \
  && fail "trainer sessionが既にあります: $BUCKET8_TRAINER_SESSION"
require_no_trainer_process
[[ -f "$GATE_DIR/monitor.done" ]] || fail "slot 8 monitor gateが未完了です: $GATE_DIR/monitor.done"
for session in "monitor-render-$RUN_NAME" "monitor-http-$RUN_NAME"; do
  tmux has-session -t "$session" 2>/dev/null || fail "monitor sessionがありません: $session"
done
compute_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
[[ -z "$compute_processes" ]] || fail "GPUに別のcompute processがあります: $compute_processes"

# 準備時点のmanifestと現在の実体を照合する。
[[ "$(manifest_value "$MANIFEST" tatara_commit)" == "$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)" ]] \
  || fail "Tatara commitが準備時点と異なります"
require_file_sha256 "$COMMAND_FILE" "$(manifest_value "$MANIFEST" command_sha256)" "command.sh"
require_file_sha256 "$LAUNCH_FILE" "$(manifest_value "$MANIFEST" launch_sha256)" "launch.sh"
require_file_sha256 "$NNUE_TRAIN" "$(manifest_value "$MANIFEST" nnue_train_sha256)" "nnue-train"
require_file_sha256 "$(manifest_value "$MANIFEST" base_network)" \
  "$(manifest_value "$MANIFEST" base_network_sha256)" "base network"
require_file_sha256 "$(manifest_value "$MANIFEST" progress_bin)" \
  "$(manifest_value "$MANIFEST" progress_sha256)" "progress.bin"
[[ "$(file_size "$ENTERING_KING_PSV")" == "$(manifest_value "$MANIFEST" training_bytes)" ]] \
  || fail "相入玉教師のbyte数が準備時点と異なります"

tmux new-session -d -s "$BUCKET8_TRAINER_SESSION" "bash -lc $(printf '%q' "bash $(printf '%q' "$LAUNCH_FILE")")"
until [[ -f "$RUN_ROOT/state/trainer.pid" ]] && pgrep -f "$NNUE_TRAIN" >/dev/null 2>&1; do
  tmux has-session -t "$BUCKET8_TRAINER_SESSION" 2>/dev/null \
    || fail "trainer sessionが起動直後に終了しました。$RUN_ROOT/logs/train.log を確認してください"
  sleep 1
done
{
  printf 'started_at=%s\n' "$(cat "$RUN_ROOT/state/training.started")"
  printf 'launcher_pid=%s\n' "$(cat "$RUN_ROOT/state/trainer.pid")"
  printf 'trainer_process=%s\n' "$(pgrep -f "$NNUE_TRAIN" | head -1)"
  printf 'trainer_session=%s\n' "$BUCKET8_TRAINER_SESSION"
} | write_manifest_atomic "$RUN_ROOT/state/start-readback.txt"
echo "[start-bucket8] $BUCKET8_TRAINER_SESSION で学習を開始しました"
echo "[start-bucket8] log: $RUN_ROOT/logs/train.log"
