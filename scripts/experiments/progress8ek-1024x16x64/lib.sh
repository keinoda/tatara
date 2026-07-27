#!/usr/bin/env bash

set -Eeuo pipefail

readonly PROGRESS8EK_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROGRESS8EK_ROOT="$(cd "$PROGRESS8EK_SCRIPT_DIR/../../.." && pwd -P)"
readonly LEGACY_ROOT="${LEGACY_ROOT:-/workspace/progress-legacy9-1024x16x64-training}"
readonly LEGACY_RUN_NAME="${LEGACY_RUN_NAME:-progress-legacy9-1024x16x64-20260721}"
readonly LEGACY_RUN_ROOT="$LEGACY_ROOT/runs/$LEGACY_RUN_NAME"
readonly LEGACY_TRAIN_PSV="$LEGACY_ROOT/data/training/public-teacher.psv"
readonly LEGACY_TRAIN_POSITIONS="${LEGACY_TRAIN_POSITIONS:-14668949437}"
readonly PSV_RECORD_BYTES=40
readonly BATCH_SIZE=65536
readonly TRAIN_THREADS=16
readonly FT_OUT="${FT_OUT:-1024}"
readonly SOURCE_SLOT="${SOURCE_SLOT:-7}"
readonly FINE_TUNE_WDL="${FINE_TUNE_WDL:-0.3333333}"
readonly FINE_TUNE_TARGET_EPOCHS="${FINE_TUNE_TARGET_EPOCHS:-10}"
readonly FINE_TUNE_SUPERBATCHES="${FINE_TUNE_SUPERBATCHES:-100}"
readonly FINE_TUNE_LR_SCHEDULE="${FINE_TUNE_LR_SCHEDULE:-one-cycle}"
readonly FINE_TUNE_LR_GAMMA="${FINE_TUNE_LR_GAMMA:-0.992}"
readonly FINE_TUNE_SAVE_RATE="${FINE_TUNE_SAVE_RATE:-}"
readonly FINE_TUNE_BATCH_ROUNDING="${FINE_TUNE_BATCH_ROUNDING:-ceil}"
readonly BASE_NETWORK="${BASE_NETWORK:-}"
readonly EXTRACTION_ROOT="$PROGRESS8EK_ROOT/extractions"
readonly RUNS_ROOT="$PROGRESS8EK_ROOT/runs/progress8ek"
readonly GATES_ROOT="$PROGRESS8EK_ROOT/gates/progress8ek"
readonly FILTER_BIN="$PROGRESS8EK_ROOT/target/release/progress8ek-filter"
readonly VERIFY_NETWORK_BIN="$PROGRESS8EK_ROOT/target/release/progress8ek-verify-network"
readonly NNUE_TRAIN="$PROGRESS8EK_ROOT/target/release/nnue-train"
readonly NET_TO_YO="$PROGRESS8EK_ROOT/target/release/net_to_yo"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "必要なcommandがありません: $1"
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
  local manifest="$1" key="$2" value
  [[ -f "$manifest" ]] || fail "manifestがありません: $manifest"
  value=$(awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {if (!found) exit 1}' "$manifest") \
    || fail "manifestに$keyがありません: $manifest"
  printf '%s\n' "$value"
}

validate_id() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || fail "IDは英数字で始め、英数字・ピリオド・アンダースコア・ハイフンだけを使ってください"
}

write_manifest_atomic() {
  local destination="$1" parent tmp
  parent=$(dirname "$destination")
  mkdir -p "$parent"
  [[ ! -e "$destination" ]] || fail "既存manifestを上書きしません: $destination"
  tmp="$destination.tmp.$BASHPID"
  tee "$tmp" >/dev/null
  mv "$tmp" "$destination"
}

require_clean_source() {
  [[ -z "$(git -C "$PROGRESS8EK_ROOT" status --porcelain)" ]] \
    || fail "progress8ek checkoutに未保存の変更があります"
}

validate_training_overrides() {
  [[ "$LEGACY_TRAIN_POSITIONS" =~ ^[1-9][0-9]*$ ]] \
    || fail "LEGACY_TRAIN_POSITIONSは1以上の整数にしてください"
  [[ "$FT_OUT" =~ ^[1-9][0-9]*$ ]] \
    || fail "FT_OUTは1以上の整数にしてください"
  [[ "$SOURCE_SLOT" =~ ^[0-7]$ ]] \
    || fail "SOURCE_SLOTは0から7の整数にしてください"
  [[ "$FINE_TUNE_SUPERBATCHES" =~ ^[1-9][0-9]*$ ]] \
    || fail "FINE_TUNE_SUPERBATCHESは1以上の整数にしてください"
  [[ -z "$FINE_TUNE_SAVE_RATE" || "$FINE_TUNE_SAVE_RATE" =~ ^[1-9][0-9]*$ ]] \
    || fail "FINE_TUNE_SAVE_RATEは空または1以上の整数にしてください"
  [[ "$FINE_TUNE_LR_SCHEDULE" == one-cycle || "$FINE_TUNE_LR_SCHEDULE" == step ]] \
    || fail "FINE_TUNE_LR_SCHEDULEはone-cycleまたはstepにしてください"
  [[ "$FINE_TUNE_BATCH_ROUNDING" == ceil || "$FINE_TUNE_BATCH_ROUNDING" == nearest ]] \
    || fail "FINE_TUNE_BATCH_ROUNDINGはceilまたはnearestにしてください"
  python3 - "$FINE_TUNE_WDL" "$FINE_TUNE_TARGET_EPOCHS" "$FINE_TUNE_LR_GAMMA" <<'PY' \
    || fail "WDL、epoch、gammaのいずれかが不正です"
import math
import sys

wdl = float(sys.argv[1])
epochs = float(sys.argv[2])
gamma = float(sys.argv[3])
valid = (
    math.isfinite(wdl)
    and 0.0 <= wdl <= 1.0
    and math.isfinite(epochs)
    and epochs > 0.0
    and math.isfinite(gamma)
    and gamma > 0.0
)
raise SystemExit(0 if valid else 1)
PY
}

legacy_progress_bin() {
  local manifest="$LEGACY_RUN_ROOT/config/manifest.txt" path expected actual
  path=$(manifest_value "$manifest" progress_bin)
  expected=$(manifest_value "$manifest" progress_sha256)
  [[ -f "$path" ]] || fail "現学習のprogress.binがありません: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || fail "現学習のprogress.bin SHA-256がmanifestと一致しません"
  printf '%s\n' "$path"
}

require_legacy_training_complete() {
  local state_dir="$LEGACY_RUN_ROOT/state" exit_code experiment status
  [[ -f "$state_dir/trainer.exit-code" ]] || fail "現学習はまだ終了していません"
  exit_code=$(tr -d '[:space:]' <"$state_dir/trainer.exit-code")
  [[ "$exit_code" == 0 ]] || fail "現学習が異常終了しています: rc=$exit_code"
  if pgrep -f "$LEGACY_ROOT/target/release/nnue-train.*$LEGACY_RUN_NAME" >/dev/null 2>&1; then
    fail "現学習processがまだ稼働しています"
  fi
  experiment=$(find "$LEGACY_RUN_ROOT/checkpoints/experiments" -maxdepth 1 -type f -name '*.json' | sort | tail -1)
  [[ -n "$experiment" ]] || fail "現学習のexperiment JSONがありません"
  status=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$experiment")
  [[ "$status" == completed ]] || fail "現学習のexperiment statusがcompletedではありません: $status"
}

require_legacy_training_data() {
  validate_training_overrides
  local expected_bytes=$((LEGACY_TRAIN_POSITIONS * PSV_RECORD_BYTES)) actual_bytes
  [[ -f "$LEGACY_TRAIN_PSV" ]] || fail "公開教師PSVがありません: $LEGACY_TRAIN_PSV"
  actual_bytes=$(file_size "$LEGACY_TRAIN_PSV")
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || fail "公開教師PSV sizeが不正です: actual=$actual_bytes expected=$expected_bytes"
}

require_single_rtx5090() {
  require_command nvidia-smi
  local rows count
  rows=$(nvidia-smi --query-gpu=name --format=csv,noheader)
  count=$(printf '%s\n' "$rows" | awk 'NF {count++} END {print count+0}')
  [[ "$count" == 1 && "$rows" == *"RTX 5090"* ]] \
    || fail "RTX 5090 1枚のinstanceが必要です: $rows"
}

latest_legacy_experiment_json() {
  find "$LEGACY_RUN_ROOT/checkpoints/experiments" -maxdepth 1 -type f -name '*.json' | sort | tail -1
}

cuda_gate_dir() {
  printf '%s/cuda-%s\n' "$GATES_ROOT" "$(git -C "$PROGRESS8EK_ROOT" rev-parse HEAD)"
}

require_cuda_gate() {
  local gate manifest commit
  gate=$(cuda_gate_dir)
  manifest="$gate/manifest.txt"
  [[ -f "$gate/done" && -f "$manifest" ]] || fail "CUDA gateが完了していません: $gate"
  commit=$(git -C "$PROGRESS8EK_ROOT" rev-parse HEAD)
  [[ "$(manifest_value "$manifest" tatara_commit)" == "$commit" ]] \
    || fail "CUDA gateのTatara revisionが一致しません"
}

require_extraction() {
  [[ -n "${EXTRACTION_ID:-}" ]] || fail "EXTRACTION_IDを明示してください"
  validate_id "$EXTRACTION_ID"
  readonly SELECTED_EXTRACTION_ROOT="$EXTRACTION_ROOT/$EXTRACTION_ID"
  readonly SELECTED_EXTRACTION_MANIFEST="$SELECTED_EXTRACTION_ROOT/manifest.txt"
  [[ -f "$SELECTED_EXTRACTION_ROOT/state/finalized-at" && -f "$SELECTED_EXTRACTION_MANIFEST" ]] \
    || fail "抽出結果が完了していません: $SELECTED_EXTRACTION_ROOT"
  readonly SELECTED_TRAIN_PSV="$(manifest_value "$SELECTED_EXTRACTION_MANIFEST" train_psv)"
  readonly SELECTED_HOLDOUT_PSV="$(manifest_value "$SELECTED_EXTRACTION_MANIFEST" holdout_psv)"
  readonly SELECTED_METRICS="$(manifest_value "$SELECTED_EXTRACTION_MANIFEST" metrics_json)"
  [[ "$(sha256_file "$SELECTED_TRAIN_PSV")" == "$(manifest_value "$SELECTED_EXTRACTION_MANIFEST" train_sha256)" ]] \
    || fail "抽出train PSVのSHA-256が一致しません"
  [[ "$(sha256_file "$SELECTED_HOLDOUT_PSV")" == "$(manifest_value "$SELECTED_EXTRACTION_MANIFEST" holdout_sha256)" ]] \
    || fail "抽出holdout PSVのSHA-256が一致しません"
  [[ "$(sha256_file "$SELECTED_METRICS")" == "$(manifest_value "$SELECTED_EXTRACTION_MANIFEST" metrics_sha256)" ]] \
    || fail "抽出metricsのSHA-256が一致しません"
}

select_base_network() {
  local output="$1"
  [[ ! -e "$output" ]] || fail "既存base選択reportを上書きしません: $output"
  if [[ -n "$BASE_NETWORK" ]]; then
    [[ -f "$BASE_NETWORK" ]] || fail "指定した基準networkがありません: $BASE_NETWORK"
    python3 - "$BASE_NETWORK" "$output" <<'PY'
import json
import pathlib
import sys

network = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
output.write_text(
    json.dumps({"selection": "explicit", "best": {"bin": str(network)}}, indent=2) + "\n",
    encoding="utf-8",
)
PY
    printf '%s\n' "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["best"]["bin"])' "$output")"
    return
  fi
  python3 "$LEGACY_ROOT/scripts/experiments/progress-legacy9-1024x16x64/select-saved-checkpoint.py" \
    --run-root "$LEGACY_RUN_ROOT" >"$output"
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["best"]["bin"])' "$output"
}

build_progress8ek_training_command() {
  validate_training_overrides
  : "${COMMAND_INIT_FROM:?COMMAND_INIT_FROM is required}"
  : "${COMMAND_DATA:?COMMAND_DATA is required}"
  : "${COMMAND_VALIDATION:?COMMAND_VALIDATION is required}"
  : "${COMMAND_TEST_POSITIONS:?COMMAND_TEST_POSITIONS is required}"
  : "${COMMAND_OUTPUT:?COMMAND_OUTPUT is required}"
  : "${COMMAND_NET_ID:?COMMAND_NET_ID is required}"
  : "${COMMAND_SUPERBATCHES:?COMMAND_SUPERBATCHES is required}"
  : "${COMMAND_BATCHES_PER_SB:?COMMAND_BATCHES_PER_SB is required}"
  : "${COMMAND_SAVE_RATE:?COMMAND_SAVE_RATE is required}"
  : "${COMMAND_PROGRESS:?COMMAND_PROGRESS is required}"
  [[ ${#COMMAND_LR_ARGS[@]} -gt 0 ]] || fail "COMMAND_LR_ARGSがありません"

  TRAINING_COMMAND=(
    "$NNUE_TRAIN"
    --init-from "$COMMAND_INIT_FROM"
    --data "$COMMAND_DATA"
    --feature-set halfka-hm-merged
    --batch-size "$BATCH_SIZE"
    --batches-per-superbatch "$COMMAND_BATCHES_PER_SB"
    --superbatches "$COMMAND_SUPERBATCHES"
    "${COMMAND_LR_ARGS[@]}"
    --wdl "$FINE_TUNE_WDL"
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
    --test-data "$COMMAND_VALIDATION"
    --test-positions "$COMMAND_TEST_POSITIONS"
    --threads "$TRAIN_THREADS"
    --save-rate "$COMMAND_SAVE_RATE"
    --keep-checkpoints 2
    --monitor-fp16-clamps
    --monitor-active-features
    --output "$COMMAND_OUTPUT"
    --net-id "$COMMAND_NET_ID"
    --experiment-name "$COMMAND_NET_ID"
    --all-optim
    layerstack
    --ft-out "$FT_OUT"
    --l1 16
    --l2 64
    --fv-scale 28
    --bucket-mode progress8ek
    --num-buckets 9
    --progress-coeff "$COMMAND_PROGRESS"
    --progress8ek-finetune
    --progress8ek-source-slot "$SOURCE_SLOT"
  )
}

write_command_file() {
  local destination="$1"
  shift
  [[ ! -e "$destination" ]] || fail "既存command fileを上書きしません: $destination"
  {
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\nexec'
    printf ' %q' "$@"
    printf '\n'
  } >"$destination"
  chmod 700 "$destination"
}
