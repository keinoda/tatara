#!/usr/bin/env bash
# progress fixed8学習runbookの共通定数と検証関数。

set -Eeuo pipefail

EXPERIMENT_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
EXPERIMENT_ROOT=$(cd "$EXPERIMENT_SCRIPT_DIR/../../.." && pwd -P)
readonly EXPERIMENT_SCRIPT_DIR EXPERIMENT_ROOT

readonly EXPERIMENT_ID="progress-legacy9-1024x16x64"
readonly STATE_DIR="$EXPERIMENT_ROOT/.onstart"
readonly ONSTART_LOG_DIR="$EXPERIMENT_ROOT/logs/onstart"
readonly MANIFEST_DIR="$EXPERIMENT_ROOT/manifests"
readonly GATES_ROOT="$EXPERIMENT_ROOT/gates"
readonly RUNS_ROOT="$EXPERIMENT_ROOT/runs"
readonly SURVEY_ROOT="$EXPERIMENT_ROOT/survey"
readonly APPROVAL_ROOT="$EXPERIMENT_ROOT/progress/approved"
readonly TRAIN_SHARD_DIR="$EXPERIMENT_ROOT/data/training/shards"
readonly TRAIN_PSV="$EXPERIMENT_ROOT/data/training/public-teacher.psv"
readonly VALIDATION_PSV="$EXPERIMENT_ROOT/data/validation/floodgate.psv"
readonly NNUE_TRAIN="$EXPERIMENT_ROOT/target/release/nnue-train"
readonly NET_TO_YO="$EXPERIMENT_ROOT/target/release/net_to_yo"
readonly PROGRESS_SURVEY="$EXPERIMENT_ROOT/target/release/progress-bucket-survey"

readonly PSV_RECORD_BYTES=40
readonly TRAIN_EXPECTED_BYTES=586757977480
readonly TRAIN_EXPECTED_POSITIONS=14668949437
readonly VALIDATION_EXPECTED_BYTES=34276920
readonly VALIDATION_FILE_POSITIONS=856923
readonly VALIDATION_EFFECTIVE_POSITIONS=851968
readonly PROGRESS_EXPECTED_BYTES=1003104

readonly TATARA_UPSTREAM_COMMIT="da3ea68d46a5c1ac0c18c10a57fef52d02788879"
readonly RSHOGI_COMMIT="29245a1d8e4f198aba3fc832a506649221cb2f2c"
readonly YANEURAOU_COMMIT="771fe811f877859d6851ceccfd3e04c16454e689"
readonly TRAIN_DATASET_REVISION="5da309f4de4091cfb004eff94da97d49e3268aa2"
readonly VALIDATION_DATASET_REVISION="fdd5f602db82d888a87116f087d10dd5ea8313ab"
readonly CONTAINER_IMAGE="ghcr.io/keinoda/shogi-lab:cuda129-trt1011"
readonly CONTAINER_IMAGE_DIGEST="sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "必要なコマンド '$1' が見つかりません"
}

file_size() {
  if stat -c '%s' "$1" >/dev/null 2>&1; then
    stat -c '%s' "$1"
  else
    stat -f '%z' "$1"
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

canonical_file() {
  if command -v realpath >/dev/null 2>&1; then
    realpath "$1"
  else
    python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
  fi
}

manifest_value() {
  local manifest="$1" key="$2"
  [[ -f "$manifest" ]] || fail "manifestがありません: $manifest"
  local value
  value=$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {if (!found) exit 1}' "$manifest") \
    || fail "manifestに$keyがありません: $manifest"
  printf '%s\n' "$value"
}

require_exact_size() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || fail "$labelがありません: $path"
  local actual
  actual=$(file_size "$path")
  [[ "$actual" == "$expected" ]] \
    || fail "$labelのsizeが不正です: actual=$actual expected=$expected path=$path"
}

validate_run_name() {
  local run_name="$1"
  [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "RUN_NAMEは英数字で始め、英数字・ピリオド・アンダースコア・ハイフンだけを使ってください"
}

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || fail "RUN_NAMEを明示してください"
  validate_run_name "$RUN_NAME"
}

require_source_revision() {
  local source_manifest="$MANIFEST_DIR/source-revisions.txt"
  local expected actual
  expected=$(manifest_value "$source_manifest" tatara)
  actual=$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)
  [[ "$actual" == "$expected" ]] \
    || fail "Tatara HEADがonstartで固定したrevisionと異なります: actual=$actual expected=$expected"
  [[ -z "$(git -C "$EXPERIMENT_ROOT" status --porcelain)" ]] \
    || fail "Tatara checkoutに未保存の変更があります"
  [[ "$(manifest_value "$source_manifest" tatara_upstream)" == "$TATARA_UPSTREAM_COMMIT" ]] \
    || fail "Tatara upstream revisionがrunbook固定値と異なります"
  [[ "$(manifest_value "$source_manifest" rshogi)" == "$RSHOGI_COMMIT" ]] \
    || fail "rshogi revisionがrunbook固定値と異なります"
}

require_single_rtx5090() {
  require_command nvidia-smi
  local gpu_lines gpu_count gpu_name
  gpu_lines=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
  gpu_count=$(printf '%s\n' "$gpu_lines" | awk 'NF {n++} END {print n+0}')
  [[ "$gpu_count" == 1 ]] || fail "GPUはRTX 5090 1枚である必要があります: count=$gpu_count"
  gpu_name=$(printf '%s\n' "$gpu_lines" | cut -d, -f1)
  [[ "$gpu_name" == *"RTX 5090"* ]] || fail "GPUがRTX 5090ではありません: $gpu_lines"
  printf '%s\n' "$gpu_lines"
}

gate_dir_for_run() {
  local run_name="$1"
  validate_run_name "$run_name"
  printf '%s/%s\n' "$GATES_ROOT" "$run_name"
}

require_gate() {
  local gate_dir="$1" gate_name="$2"
  [[ -f "$gate_dir/$gate_name.done" ]] \
    || fail "gate '$gate_name'が完了していません: $gate_dir/$gate_name.done"
}

write_manifest_atomic() {
  local destination="$1"
  local parent tmp
  parent=$(dirname "$destination")
  mkdir -p "$parent"
  [[ ! -e "$destination" ]] || fail "既存manifestを上書きしません: $destination"
  tmp="$destination.tmp.$BASHPID"
  cat >"$tmp"
  mv "$tmp" "$destination"
}

approval_value() {
  local approval="$1" key="$2"
  manifest_value "$approval" "$key"
}

require_progress_approval() {
  [[ -n "${PROGRESS_APPROVAL:-}" ]] || fail "PROGRESS_APPROVALを明示してください"
  [[ -f "$PROGRESS_APPROVAL" ]] || fail "progress承認manifestがありません: $PROGRESS_APPROVAL"
  local selected_path selected_sha actual_sha selected_real
  selected_path=$(approval_value "$PROGRESS_APPROVAL" progress_bin)
  selected_sha=$(approval_value "$PROGRESS_APPROVAL" progress_sha256)
  [[ -f "$selected_path" ]] || fail "承認済みprogress.binがありません: $selected_path"
  require_exact_size "$selected_path" "$PROGRESS_EXPECTED_BYTES" "承認済みprogress.bin"
  selected_real=$(canonical_file "$selected_path")
  [[ "$selected_real" == "$EXPERIMENT_ROOT"/* ]] \
    || fail "承認済みprogress.binは学習専用folder直下に置いてください: $selected_real"
  actual_sha=$(sha256_file "$selected_path")
  [[ "$actual_sha" == "$selected_sha" ]] \
    || fail "progress.binのSHA-256が承認manifestと異なります: actual=$actual_sha expected=$selected_sha"
  printf '%s\n' "$selected_real"
}

precision_from_gate() {
  local gate_dir="$1"
  require_gate "$gate_dir" precision
  local mode
  mode=$(manifest_value "$gate_dir/precision.approved.txt" precision)
  [[ "$mode" == "fp32" || "$mode" == "all-optim" ]] \
    || fail "precision承認値が不正です: $mode"
  printf '%s\n' "$mode"
}

# Tataraの学習CLIを一箇所で組み立てる。呼び出し側はCOMMAND_*を明示する。
build_training_command() {
  : "${COMMAND_DATA:?COMMAND_DATA is required}"
  : "${COMMAND_OUTPUT:?COMMAND_OUTPUT is required}"
  : "${COMMAND_NET_ID:?COMMAND_NET_ID is required}"
  : "${COMMAND_SUPERBATCHES:?COMMAND_SUPERBATCHES is required}"
  : "${COMMAND_BATCHES_PER_SB:?COMMAND_BATCHES_PER_SB is required}"
  : "${COMMAND_BATCH_SIZE:?COMMAND_BATCH_SIZE is required}"
  : "${COMMAND_THREADS:?COMMAND_THREADS is required}"
  : "${COMMAND_PROGRESS:?COMMAND_PROGRESS is required}"
  : "${COMMAND_VALIDATION:?COMMAND_VALIDATION is required}"
  : "${COMMAND_PRECISION:?COMMAND_PRECISION is required}"

  TRAINING_COMMAND=(
    "$NNUE_TRAIN"
    --data "$COMMAND_DATA"
    --feature-set halfka-hm-merged
    --batch-size "$COMMAND_BATCH_SIZE"
    --batches-per-superbatch "$COMMAND_BATCHES_PER_SB"
    --superbatches "$COMMAND_SUPERBATCHES"
    --lr 8.75e-4
    --lr-schedule step
    --lr-gamma 0.992
    --lr-step 1
    --wdl 0.3333333
    --win-rate-model
    --wrm-in-scaling 340
    --wrm-in-offset 270
    --wrm-nnue2score 600
    --wrm-target-offset 270
    --wrm-target-scaling 380
    --loss-pow-exp 2.0
    --loss-qp-asymmetry 0
    --loss-weight-boost-w1 0
    --loss-weight-boost-w2 0.5
    --optimizer ranger
    --weight-decay 0
    --ft-factorize
    --test-data "$COMMAND_VALIDATION"
    --test-positions "$VALIDATION_EFFECTIVE_POSITIONS"
    --threads "$COMMAND_THREADS"
    --save-rate "${COMMAND_SAVE_RATE:-20}"
    --keep-checkpoints "${COMMAND_KEEP_CHECKPOINTS:-2}"
    --monitor-fp16-clamps
    --monitor-active-features
    --output "$COMMAND_OUTPUT"
    --net-id "$COMMAND_NET_ID"
    --experiment-name "$COMMAND_NET_ID"
  )
  if [[ "$COMMAND_PRECISION" == "all-optim" ]]; then
    TRAINING_COMMAND+=(--all-optim)
  elif [[ "$COMMAND_PRECISION" != "fp32" ]]; then
    fail "COMMAND_PRECISIONはfp32またはall-optimにしてください: $COMMAND_PRECISION"
  fi
  if [[ -n "${COMMAND_RESUME:-}" ]]; then
    TRAINING_COMMAND+=(--resume "$COMMAND_RESUME")
  fi
  TRAINING_COMMAND+=(
    layerstack
    --ft-out 1024
    --l1 16
    --l2 64
    --fv-scale 28
    --bucket-mode progress8kpabs
    --num-buckets 8
    --progress-coeff "$COMMAND_PROGRESS"
  )
}

write_command_file() {
  local path="$1"
  shift
  {
    printf '%q ' "$@"
    printf '\n'
  } | write_manifest_atomic "$path"
}
