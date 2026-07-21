#!/usr/bin/env bash
# 完了済みrunのraw checkpointから、合意済み延長点まで別runとして手動resumeする。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
[[ -n "${PARENT_RUN_NAME:-}" ]] || fail "PARENT_RUN_NAMEを明示してください"
validate_run_name "$PARENT_RUN_NAME"
[[ -n "${RESUME_CHECKPOINT:-}" ]] || fail "RESUME_CHECKPOINTを明示してください"
[[ -n "${TARGET_SB:-}" ]] || fail "TARGET_SBを明示してください"
case "$TARGET_SB" in
  440|513|587|660|733) ;;
  *) fail "TARGET_SBは440, 513, 587, 660, 733のいずれかです: $TARGET_SB" ;;
esac
require_source_revision
gpu=$(require_single_rtx5090)
require_command curl
require_command tmux
[[ -n "${MONITOR_USER:-}" && -n "${MONITOR_PASSWORD:-}" ]] \
  || fail "monitor生存確認のためMONITOR_USERとMONITOR_PASSWORDを明示してください"

readonly PARENT_ROOT="$RUNS_ROOT/$PARENT_RUN_NAME"
readonly PARENT_MANIFEST="$PARENT_ROOT/config/manifest.txt"
[[ -f "$PARENT_MANIFEST" ]] || fail "親run manifestがありません: $PARENT_MANIFEST"
parent_exit=$(cat "$PARENT_ROOT/state/trainer.exit-code" 2>/dev/null || true)
[[ "$parent_exit" == 0 ]] || fail "親runが正常終了していません: exit=$parent_exit"
resume_real=$(canonical_file "$RESUME_CHECKPOINT")
[[ "$resume_real" == "$PARENT_ROOT/checkpoints/"*.ckpt ]] \
  || fail "RESUME_CHECKPOINTは親runのraw checkpointを指定してください: $resume_real"
[[ -s "$resume_real" ]] || fail "raw checkpointが空です: $resume_real"
checkpoint_sb=$(basename "$resume_real" .ckpt | sed -n 's/.*-\([0-9][0-9]*\)$/\1/p')
[[ -n "$checkpoint_sb" ]] || fail "checkpoint名からSB番号を取得できません"
(( TARGET_SB > checkpoint_sb )) || fail "TARGET_SBはcheckpoint SBより大きい必要があります"

progress_bin=$(require_progress_approval)
progress_sha=$(sha256_file "$progress_bin")
[[ "$progress_sha" == "$(manifest_value "$PARENT_MANIFEST" progress_sha256)" ]] \
  || fail "progress.binが親runと異なります"
precision=$(manifest_value "$PARENT_MANIFEST" precision)
threads=$(manifest_value "$PARENT_MANIFEST" train_threads)

readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
require_gate "$GATE_DIR" monitor
curl --silent --show-error --fail --user "$MONITOR_USER:$MONITOR_PASSWORD" \
  http://127.0.0.1:6001/status.json >/dev/null || fail "monitorが応答しません"

expected_train_sha=$(manifest_value "$PARENT_MANIFEST" training_sha256)
expected_validation_sha=$(manifest_value "$PARENT_MANIFEST" validation_sha256)
[[ "$(sha256_file "$TRAIN_PSV")" == "$expected_train_sha" ]] || fail "教師PSVが親runと異なります"
[[ "$(sha256_file "$VALIDATION_PSV")" == "$expected_validation_sha" ]] || fail "validation PSVが親runと異なります"

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
COMMAND_SUPERBATCHES="$TARGET_SB"
COMMAND_BATCHES_PER_SB=6104
COMMAND_BATCH_SIZE=65536
COMMAND_THREADS="$threads"
COMMAND_PROGRESS="$progress_bin"
COMMAND_VALIDATION="$VALIDATION_PSV"
COMMAND_PRECISION="$precision"
COMMAND_SAVE_RATE=20
COMMAND_KEEP_CHECKPOINTS=2
COMMAND_RESUME="$resume_real"
build_training_command
command=("${TRAINING_COMMAND[@]}")

mkdir -p "$CHECKPOINT_DIR" "$CONFIG_DIR" "$LOG_DIR" "$RUN_STATE_DIR"
write_command_file "$CONFIG_DIR/command.sh" "${command[@]}"
chmod 700 "$CONFIG_DIR/command.sh"
{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'parent_run_name=%s\n' "$PARENT_RUN_NAME"
  printf 'resume_checkpoint=%s\n' "$resume_real"
  printf 'resume_checkpoint_sha256=%s\n' "$(sha256_file "$resume_real")"
  printf 'resume_checkpoint_superbatch=%s\n' "$checkpoint_sb"
  printf 'target_superbatch=%s\n' "$TARGET_SB"
  printf 'tatara_commit=%s\n' "$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)"
  printf 'training_sha256=%s\n' "$expected_train_sha"
  printf 'validation_sha256=%s\n' "$expected_validation_sha"
  printf 'progress_bin=%s\n' "$progress_bin"
  printf 'progress_sha256=%s\n' "$progress_sha"
  printf 'precision=%s\n' "$precision"
  printf 'train_threads=%s\n' "$threads"
  printf 'gpu=%s\n' "$gpu"
} | write_manifest_atomic "$CONFIG_DIR/manifest.txt"

printf -v tmux_body \
  'set -uo pipefail; echo $$ >%q; date -u +%%FT%%TZ >%q; set +e; bash %q 2>&1 | tee %q; rc=${PIPESTATUS[0]}; set -e; printf "%%s\n" "$rc" >%q; date -u +%%FT%%TZ >%q; exit "$rc"' \
  "$RUN_STATE_DIR/trainer.pid" "$RUN_STATE_DIR/trainer.started-at" \
  "$CONFIG_DIR/command.sh" "$LOG_DIR/train.log" \
  "$RUN_STATE_DIR/trainer.exit-code" "$RUN_STATE_DIR/trainer.ended-at"
tmux new-session -d -s "$TRAIN_SESSION" "bash -lc $(printf '%q' "$tmux_body")"
date -u +%FT%TZ >"$RUN_STATE_DIR/training.started"
echo "[resume] tmux=$TRAIN_SESSION parent=$PARENT_RUN_NAME checkpoint_sb=$checkpoint_sb target_sb=$TARGET_SB"
