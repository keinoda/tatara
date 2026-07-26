#!/usr/bin/env bash
# 全gate完了後に初回421 SB学習を専用tmuxで開始する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
require_source_revision
gpu=$(require_single_rtx5090)
require_command curl
require_command tmux
[[ -f "$STATE_DIR/prepare_data.done" ]] || fail "prepare_data.doneがありません"
[[ -x "$NNUE_TRAIN" ]] || fail "nnue-trainがありません"
[[ -f "$MANIFEST_DIR/survey-shard-cleanup.txt" ]] \
  || fail "部分取得survey用shardのcleanupが完了していません"
shopt -s nullglob
remaining_shards=("$TRAIN_SHARD_DIR"/split_*.bin)
(( ${#remaining_shards[@]} == 0 )) \
  || fail "本学習前に元shardを削除してください: ${remaining_shards[*]}"
[[ -n "${MONITOR_USER:-}" && -n "${MONITOR_PASSWORD:-}" ]] \
  || fail "monitor生存確認のためMONITOR_USERとMONITOR_PASSWORDを明示してください"

progress_bin=$(require_progress_approval)
progress_sha=$(sha256_file "$progress_bin")
readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
for gate in smoke precision resume export monitor; do
  require_gate "$GATE_DIR" "$gate"
done
precision=$(precision_from_gate "$GATE_DIR")
threads=$(manifest_value "$GATE_DIR/precision.approved.txt" threads)

# monitorはgate時だけでなくtrainer起動直前にも同じrun向けで生存確認する。
[[ "$(manifest_value "$GATE_DIR/monitor.txt" run_name)" == "$RUN_NAME" ]] \
  || fail "monitor gateのrun名が一致しません"
curl --silent --show-error --fail --user "$MONITOR_USER:$MONITOR_PASSWORD" \
  http://127.0.0.1:6001/status.json >/dev/null \
  || fail "monitorがport 6001で応答しません"

readonly PREPARED_MANIFEST="$MANIFEST_DIR/prepared-data.txt"
require_exact_size "$TRAIN_PSV" "$TRAIN_EXPECTED_BYTES" "教師PSV"
require_exact_size "$VALIDATION_PSV" "$VALIDATION_EXPECTED_BYTES" "validation PSV"
expected_train_sha=$(manifest_value "$PREPARED_MANIFEST" training_sha256)
expected_validation_sha=$(manifest_value "$PREPARED_MANIFEST" validation_sha256)
echo "[preflight] 教師PSVのSHA-256を再検証します"
actual_train_sha=$(sha256_file "$TRAIN_PSV")
actual_validation_sha=$(sha256_file "$VALIDATION_PSV")
[[ "$actual_train_sha" == "$expected_train_sha" ]] \
  || fail "教師PSVのSHA-256がprepared manifestと異なります"
[[ "$actual_validation_sha" == "$expected_validation_sha" ]] \
  || fail "validation PSVのSHA-256がprepared manifestと異なります"

readonly RUN_ROOT="$RUNS_ROOT/$RUN_NAME"
readonly CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
readonly CONFIG_DIR="$RUN_ROOT/config"
readonly LOG_DIR="$RUN_ROOT/logs"
readonly RUN_STATE_DIR="$RUN_ROOT/state"
readonly TRAIN_SESSION="train-$RUN_NAME"
[[ ! -e "$RUN_ROOT" ]] || fail "既存runを上書きしません: $RUN_ROOT"
tmux has-session -t "$TRAIN_SESSION" 2>/dev/null && fail "tmux sessionが既にあります: $TRAIN_SESSION"

COMMAND_DATA="$TRAIN_PSV"
COMMAND_OUTPUT="$CHECKPOINT_DIR"
COMMAND_NET_ID="$RUN_NAME"
COMMAND_SUPERBATCHES=421
COMMAND_BATCHES_PER_SB=6104
COMMAND_BATCH_SIZE=65536
COMMAND_THREADS="$threads"
COMMAND_PROGRESS="$progress_bin"
COMMAND_VALIDATION="$VALIDATION_PSV"
COMMAND_PRECISION="$precision"
COMMAND_SAVE_RATE=20
COMMAND_KEEP_CHECKPOINTS=2
COMMAND_RESUME=""
build_training_command
command=("${TRAINING_COMMAND[@]}")

mkdir -p "$CHECKPOINT_DIR" "$CONFIG_DIR" "$LOG_DIR" "$RUN_STATE_DIR"
write_command_file "$CONFIG_DIR/command.sh" "${command[@]}"
chmod 700 "$CONFIG_DIR/command.sh"
{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'tatara_commit=%s\n' "$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)"
  printf 'tatara_upstream_commit=%s\n' "$TATARA_UPSTREAM_COMMIT"
  printf 'container_image=%s@%s\n' "$CONTAINER_IMAGE" "$CONTAINER_IMAGE_DIGEST"
  printf 'training_psv=%s\n' "$TRAIN_PSV"
  printf 'training_sha256=%s\n' "$actual_train_sha"
  printf 'validation_psv=%s\n' "$VALIDATION_PSV"
  printf 'validation_file_positions=%s\n' "$VALIDATION_FILE_POSITIONS"
  printf 'validation_effective_positions=%s\n' "$VALIDATION_EFFECTIVE_POSITIONS"
  printf 'validation_sha256=%s\n' "$actual_validation_sha"
  printf 'progress_bin=%s\n' "$progress_bin"
  printf 'progress_sha256=%s\n' "$progress_sha"
  printf 'progress_approval=%s\n' "$PROGRESS_APPROVAL"
  printf 'precision=%s\n' "$precision"
  printf 'train_threads=%s\n' "$threads"
  printf 'training_order=disk_order_preserved_worker_prefetch_nondeterministic\n'
  printf 'gpu=%s\n' "$gpu"
  printf 'rustc=%s\n' "$(rustc --version)"
} | write_manifest_atomic "$CONFIG_DIR/manifest.txt"

printf -v tmux_body \
  'set -uo pipefail; echo $$ >%q; date -u +%%FT%%TZ >%q; set +e; bash %q 2>&1 | tee %q; rc=${PIPESTATUS[0]}; set -e; printf "%%s\n" "$rc" >%q; date -u +%%FT%%TZ >%q; exit "$rc"' \
  "$RUN_STATE_DIR/trainer.pid" "$RUN_STATE_DIR/trainer.started-at" \
  "$CONFIG_DIR/command.sh" "$LOG_DIR/train.log" \
  "$RUN_STATE_DIR/trainer.exit-code" "$RUN_STATE_DIR/trainer.ended-at"
tmux new-session -d -s "$TRAIN_SESSION" "bash -lc $(printf '%q' "$tmux_body")"
date -u +%FT%TZ >"$RUN_STATE_DIR/training.started"

echo "[train] tmux=$TRAIN_SESSION run=$RUN_NAME precision=$precision threads=$threads"
echo "[train] 421 SB × 6104 batch/SB × 65536 position/batch = 168,413,364,224局面"
echo "[train] 約10.009678 epoch。自動resume・自動stop・外部backupは行いません"
echo "[train] 監視: tmux attach -t $TRAIN_SESSION / tail -f $LOG_DIR/train.log"
