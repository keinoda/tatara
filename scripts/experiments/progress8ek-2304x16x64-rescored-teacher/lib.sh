#!/usr/bin/env bash
# 再評価教師progress8ek学習のVast.ai準備とmonitorで共有する定数・検証関数。

set -Eeuo pipefail

EXPERIMENT_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
EXPERIMENT_ROOT=$(cd "$EXPERIMENT_SCRIPT_DIR/../../.." && pwd -P)
readonly EXPERIMENT_SCRIPT_DIR EXPERIMENT_ROOT

readonly NET_ID="nagisa-v5"
readonly STATE_DIR="$EXPERIMENT_ROOT/.onstart"
readonly ONSTART_LOG_DIR="$EXPERIMENT_ROOT/logs/onstart"
readonly MANIFEST_DIR="$EXPERIMENT_ROOT/manifests"
readonly GATES_ROOT="$EXPERIMENT_ROOT/gates"
readonly RUNS_ROOT="$EXPERIMENT_ROOT/runs"
readonly MONITOR_ROOT="$EXPERIMENT_ROOT/monitor"
readonly MONITOR_IMPLEMENTATION_DIR="$EXPERIMENT_ROOT/scripts/experiments/progress8kpabs-2304x16x64"
readonly MONITOR_PORT=6001
readonly SOURCE_SHARDS_MANIFEST="$EXPERIMENT_SCRIPT_DIR/source-shards.tsv"
readonly SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-/workspace/datasets/Knowledge_distilled_dataset_by_ponkotsuWCSC36}"
readonly SOURCE_VALIDATION_MARKER="${SOURCE_VALIDATION_MARKER:-/workspace/audits/ponkotsu-wcsc36/validation.done}"
readonly TRAINING_DATA_DIR="$EXPERIMENT_ROOT/data/training"
readonly ORDINARY_PSV="$TRAINING_DATA_DIR/ordinary.psv"
readonly ENTERING_KING_PSV="$TRAINING_DATA_DIR/entering-king.psv"
readonly PARTITION_METRICS="$TRAINING_DATA_DIR/metrics.json"
readonly PARTITION_BIN="$EXPERIMENT_ROOT/target/release/progress8ek-partition"
readonly PROGRESS_SURVEY="$EXPERIMENT_ROOT/target/release/progress-bucket-survey"
readonly BASELINE_PROGRESS="$EXPERIMENT_ROOT/progress/baseline/progress.bin"
readonly BASELINE_PROGRESS_SHA256="d77f47e874558d42fa2d87d173de3aba054eef51bcca9c1fc9f3a8daf93630d8"
readonly PROGRESS_EXPECTED_BYTES=1003104
readonly PSV_RECORD_BYTES=40
readonly SOURCE_EXPECTED_BYTES=586757977480
readonly SOURCE_EXPECTED_RECORDS=14668949437
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

manifest_value() {
  local manifest="$1"
  local key="$2"
  [[ -f "$manifest" ]] || fail "manifestがありません: $manifest"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {if (!found) exit 1}' "$manifest" \
    || fail "manifestに$keyがありません: $manifest"
}

write_manifest_atomic() {
  local destination="$1"
  local parent temporary
  parent=$(dirname "$destination")
  mkdir -p "$parent"
  [[ ! -e "$destination" ]] || fail "既存manifestを上書きしません: $destination"
  temporary="$destination.candidate.$BASHPID"
  cat >"$temporary"
  mv "$temporary" "$destination"
}

run_name_for_phase() {
  case "$1" in
    base) printf '%s-base\n' "$NET_ID" ;;
    bucket8) printf '%s-bucket8\n' "$NET_ID" ;;
    *) fail "TRAINING_PHASEはbaseまたはbucket8で指定してください: $1" ;;
  esac
}

require_training_phase() {
  [[ -n "${TRAINING_PHASE:-}" ]] || fail "TRAINING_PHASEをbaseまたはbucket8で明示してください"
  run_name_for_phase "$TRAINING_PHASE"
}
