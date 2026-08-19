#!/usr/bin/env bash
# 再評価教師progress8ek学習のVast.ai準備、monitor、slot 8追加学習で共有する定数・検証関数。

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
readonly PARTITION_ORDINARY_PSV="$TRAINING_DATA_DIR/ordinary.psv"
readonly ORDINARY_PSV="$TRAINING_DATA_DIR/ordinary-valid76.psv"
readonly ENTERING_KING_PSV="$TRAINING_DATA_DIR/entering-king.psv"
readonly VALIDATION_DATASET="takaoyamaoka/floodgate.hcpe"
readonly VALIDATION_DATASET_REVISION="fdd5f602db82d888a87116f087d10dd5ea8313ab"
readonly VALIDATION_DIR="$EXPERIMENT_ROOT/data/validation"
readonly VALIDATION_HCPE="$VALIDATION_DIR/floodgate.hcpe"
readonly VALIDATION_PSV="$VALIDATION_DIR/floodgate.psv"
readonly VALIDATION_HCPE_BYTES=32563074
readonly VALIDATION_HCPE_SHA256="fb9d60b283ade32cb5c5715fe27042476bc7169cc46b6d62a2d85975c1a945ac"
readonly VALIDATION_PSV_BYTES=34276920
readonly VALIDATION_PSV_SHA256="22e11b82fa4ac7d75e82806480b5bfdd7ba29d773bcfb91ed5b3dfe7a43d66b8"
readonly VALIDATION_FILE_POSITIONS=856923
readonly VALIDATION_EFFECTIVE_POSITIONS=851968
readonly RSHOGI_REPO="https://github.com/SH11235/rshogi.git"
readonly RSHOGI_COMMIT="29245a1d8e4f198aba3fc832a506649221cb2f2c"
readonly RSHOGI_DIR="$EXPERIMENT_ROOT/.runtime/rshogi"
readonly PARTITION_METRICS="$TRAINING_DATA_DIR/metrics.json"
readonly PARTITION_BIN="$EXPERIMENT_ROOT/target/release/progress8ek-partition"
readonly ACTIVE_AUDIT_BIN="$EXPERIMENT_ROOT/target/release/progress8ek-audit-psv"
readonly DATA_VALIDITY_AUDIT_DIR="${DATA_VALIDITY_AUDIT_DIR:-/workspace/audits/nagisa-v5-data-validity}"
readonly ORDINARY_ACTIVE_AUDIT="$DATA_VALIDITY_AUDIT_DIR/ordinary-active-audit.json"
readonly ENTERING_ACTIVE_AUDIT="$DATA_VALIDITY_AUDIT_DIR/entering-active-audit.json"
readonly PARTITION_DATA_MANIFEST="$MANIFEST_DIR/prepared-data.txt"
readonly PREPARED_DATA_MANIFEST="$MANIFEST_DIR/prepared-data-valid76.txt"
readonly PROGRESS_SURVEY="$EXPERIMENT_ROOT/target/release/progress-bucket-survey"
readonly NNUE_TRAIN="$EXPERIMENT_ROOT/target/release/nnue-train"
readonly BASELINE_PROGRESS="$EXPERIMENT_ROOT/progress/baseline/progress.bin"
readonly BASELINE_PROGRESS_SHA256="d77f47e874558d42fa2d87d173de3aba054eef51bcca9c1fc9f3a8daf93630d8"
readonly PROGRESS_EXPECTED_BYTES=1003104
readonly PSV_RECORD_BYTES=40
readonly SOURCE_EXPECTED_BYTES=586757977480
readonly SOURCE_EXPECTED_RECORDS=14668949437
readonly BASE_TRAIN_POSITIONS=14342411752
readonly BASE_BATCH_SIZE=65536
readonly BASE_BATCHES_PER_SUPERBATCH=10943
readonly BASE_SUPERBATCHES=800
readonly BASE_SAVE_RATE=100
readonly BASE_PRESENTED_POSITIONS=573728358400
readonly BASE_TARGET_EPOCHS=40
readonly MONITOR_MILESTONE_INTERVAL=100
readonly CONTAINER_IMAGE="ghcr.io/keinoda/shogi-lab:cuda129-trt1011"
readonly CONTAINER_IMAGE_DIGEST="sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876"
# slot 8（相入玉専用の第9 LayerStack）追加学習の固定値。
# base networkは通常学習の最終量子化networkで、既定は線形減衰tail runの最終SB。
readonly BASE_FINAL_RUN_NAME="$NET_ID-converge-sb500"
readonly BASE_FINAL_NETWORK_DEFAULT="$RUNS_ROOT/$BASE_FINAL_RUN_NAME/checkpoints/$NET_ID-600.bin"
readonly APPROVED_PROGRESS_SHA256="be0238267e9373bd318cb4fef119aabaf27b1468a6cda75c25f43fe67711bc81"
readonly ENTERING_KING_EXPECTED_BYTES=13061246280
readonly ENTERING_KING_SHA256="f7594e104c83e0e6d59a29ef00823f959e0f58412b5d4a8d5315fd4c3e3fd28f"
readonly VERIFY_NETWORK_BIN="$EXPERIMENT_ROOT/target/release/progress8ek-verify-network"
readonly NET_TO_YO="$EXPERIMENT_ROOT/target/release/net_to_yo"
readonly BUCKET8_SOURCE_SLOT=7
readonly BUCKET8_FILE_POSITIONS=326531157
# entering-king.psv末尾を同一file内のheld-out検証に予約する局面数（65,536の倍数）。
# 0を指定すると検証flagを付けず全件を学習に使う。
readonly BUCKET8_VALIDATION_TAIL_POSITIONS="${BUCKET8_VALIDATION_TAIL_POSITIONS:-851968}"
readonly BUCKET8_BATCH_SIZE=65536
readonly BUCKET8_BATCHES_PER_SUPERBATCH=250
readonly BUCKET8_SUPERBATCHES=800
readonly BUCKET8_SAVE_RATE=100
readonly BUCKET8_WDL=0.33
readonly BUCKET8_TARGET_EPOCHS=40
readonly BUCKET8_PRESENTED_POSITIONS=13107200000
readonly BUCKET8_TRAINER_SESSION="train-$NET_ID-bucket8"
readonly BUCKET8_SMOKE_NET_ID="$NET_ID-bucket8-smoke"

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

# 1 SBあたりの整数batch数が40 epochを下回らない最小値であることを確認する。
require_base_training_volume() {
  local target_positions previous_presented_positions
  target_positions=$((BASE_TRAIN_POSITIONS * BASE_TARGET_EPOCHS))
  previous_presented_positions=$((
    BASE_BATCH_SIZE * (BASE_BATCHES_PER_SUPERBATCH - 1) * BASE_SUPERBATCHES
  ))
  [[ "$BASE_PRESENTED_POSITIONS" == "$((
    BASE_BATCH_SIZE * BASE_BATCHES_PER_SUPERBATCH * BASE_SUPERBATCHES
  ))" ]] || fail "通常学習の提示局面数が固定値と一致しません"
  (( BASE_PRESENTED_POSITIONS >= target_positions )) \
    || fail "通常学習の提示局面数が40 epochを下回ります"
  (( previous_presented_positions < target_positions )) \
    || fail "通常学習のbatches/SBが40 epochを満たす最小値ではありません"
}

# 通常学習の固定CLIを組み立てる。progressだけは承認済みfileを呼び出し側が渡す。
build_base_training_command() {
  local progress_bin="$1"
  local output_dir="${2:-$RUNS_ROOT/$(run_name_for_phase base)/checkpoints}"
  [[ -n "$progress_bin" ]] || fail "承認済みprogress.binのpathを指定してください"
  require_base_training_volume

  BASE_TRAINING_COMMAND=(
    "$NNUE_TRAIN"
    --win-rate-model
    --batch-size "$BASE_BATCH_SIZE"
    --batches-per-superbatch "$BASE_BATCHES_PER_SUPERBATCH"
    --superbatches "$BASE_SUPERBATCHES"
    --lr 8.75e-4
    --lr-gamma 0.995
    --lr-step 1
    --weight-decay 0.0
    --wdl 0.0
    --scale 290
    --save-rate "$BASE_SAVE_RATE"
    --threads 16
    --all-optim
    --output "$output_dir"
    --net-id "$NET_ID"
    --data "$ORDINARY_PSV"
    --test-data "$VALIDATION_PSV"
    --test-positions "$VALIDATION_EFFECTIVE_POSITIONS"
    layerstack
    --ft-out 2304
    --l1 16
    --l2 64
    --bucket-mode progress8kpabs
    --num-buckets 8
    --progress-coeff "$progress_bin"
  )
}

# slot 8追加学習の提示局面数が、相入玉教師全件に対して40 epoch以上となる最小の
# 整数batches/SBであることを確認する。検証用に予約した末尾は学習から外れるため、
# 実epochは全件基準の値を下回らない。
require_bucket8_training_volume() {
  local target_positions previous_presented_positions
  target_positions=$((BUCKET8_FILE_POSITIONS * BUCKET8_TARGET_EPOCHS))
  previous_presented_positions=$((
    BUCKET8_BATCH_SIZE * (BUCKET8_BATCHES_PER_SUPERBATCH - 1) * BUCKET8_SUPERBATCHES
  ))
  [[ "$BUCKET8_PRESENTED_POSITIONS" == "$((
    BUCKET8_BATCH_SIZE * BUCKET8_BATCHES_PER_SUPERBATCH * BUCKET8_SUPERBATCHES
  ))" ]] || fail "slot 8追加学習の提示局面数が固定値と一致しません"
  (( BUCKET8_PRESENTED_POSITIONS >= target_positions )) \
    || fail "slot 8追加学習の提示局面数が40 epochを下回ります"
  (( previous_presented_positions < target_positions )) \
    || fail "slot 8追加学習のbatches/SBが40 epochを満たす最小値ではありません"
  [[ "$BUCKET8_VALIDATION_TAIL_POSITIONS" =~ ^[0-9]+$ ]] \
    || fail "BUCKET8_VALIDATION_TAIL_POSITIONSは0以上の整数にしてください"
  (( BUCKET8_VALIDATION_TAIL_POSITIONS % BUCKET8_BATCH_SIZE == 0 )) \
    || fail "BUCKET8_VALIDATION_TAIL_POSITIONSはbatch size $BUCKET8_BATCH_SIZE の倍数にしてください"
  (( BUCKET8_VALIDATION_TAIL_POSITIONS < BUCKET8_FILE_POSITIONS )) \
    || fail "BUCKET8_VALIDATION_TAIL_POSITIONSが相入玉教師件数以上です"
}

# 検証用末尾予約がある場合だけ、held-out flagをBUCKET8_TRAINING_COMMANDへ追加する。
append_bucket8_validation_args() {
  local test_positions="$1"
  (( BUCKET8_VALIDATION_TAIL_POSITIONS > 0 )) || return 0
  BUCKET8_TRAINING_COMMAND+=(
    --test-tail-positions "$BUCKET8_VALIDATION_TAIL_POSITIONS"
    --test-positions "$test_positions"
  )
}

# slot 8追加学習のCLIを組み立てる。通常学習commandから変わるのは、base networkの
# 初期化、相入玉教師、WDL、学習量、9-slot routing、slot限定更新flagだけである。
# 呼び出し側は base network・承認済みprogress.binを渡し、smokeだけが学習量・LR・
# 出力先を上書きする。
#   $1 base network (8 bucket量子化.bin)  $2 承認済みprogress.bin
#   $3 output directory  $4 net id  $5 superbatches  $6 batches/SB  $7 save rate
#   $8 検証pass局面数  $9.. LR引数
build_bucket8_command_core() {
  local base_network="$1" progress_bin="$2" output_dir="$3" net_id="$4"
  local superbatches="$5" batches_per_superbatch="$6" save_rate="$7"
  local test_positions="$8"
  shift 8
  local lr_args=("$@")
  [[ -n "$base_network" ]] || fail "base networkのpathを指定してください"
  [[ -n "$progress_bin" ]] || fail "承認済みprogress.binのpathを指定してください"

  BUCKET8_TRAINING_COMMAND=(
    "$NNUE_TRAIN"
    --init-from "$base_network"
    --win-rate-model
    --batch-size "$BUCKET8_BATCH_SIZE"
    --batches-per-superbatch "$batches_per_superbatch"
    --superbatches "$superbatches"
    "${lr_args[@]}"
    --weight-decay 0.0
    --wdl "$BUCKET8_WDL"
    --scale 290
    --save-rate "$save_rate"
    --threads 16
    --all-optim
    --output "$output_dir"
    --net-id "$net_id"
    --data "$ENTERING_KING_PSV"
  )
  append_bucket8_validation_args "$test_positions"
  BUCKET8_TRAINING_COMMAND+=(
    layerstack
    --ft-out 2304
    --l1 16
    --l2 64
    --bucket-mode progress8ek
    --num-buckets 9
    --progress-coeff "$progress_bin"
    --progress8ek-finetune
    --progress8ek-source-slot "$BUCKET8_SOURCE_SLOT"
  )
}

# production用slot 8追加学習command。
build_bucket8_training_command() {
  local base_network="$1" progress_bin="$2"
  local output_dir="${3:-$RUNS_ROOT/$(run_name_for_phase bucket8)/checkpoints}"
  require_bucket8_training_volume
  build_bucket8_command_core \
    "$base_network" "$progress_bin" "$output_dir" "$NET_ID" \
    "$BUCKET8_SUPERBATCHES" "$BUCKET8_BATCHES_PER_SUPERBATCH" "$BUCKET8_SAVE_RATE" \
    "$BUCKET8_VALIDATION_TAIL_POSITIONS" \
    --lr 8.75e-4 --lr-gamma 0.995 --lr-step 1
}

# preflight smoke用command。1 SB・2 batchだけ学習し、小さな定数LRで非対象parameterの
# 不変性とslot 8の更新を検査する。データ範囲・routing・flagはproductionと同じにする。
build_bucket8_smoke_command() {
  local base_network="$1" progress_bin="$2" output_dir="$3"
  require_bucket8_training_volume
  build_bucket8_command_core \
    "$base_network" "$progress_bin" "$output_dir" "$BUCKET8_SMOKE_NET_ID" \
    1 2 1 "$BUCKET8_BATCH_SIZE" \
    --lr 3.5e-5 --lr-schedule constant
}

require_file_sha256() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "$path" ]] || fail "$label がありません: $path"
  actual=$(sha256_file "$path")
  [[ "$actual" == "$expected" ]] \
    || fail "$label のSHA-256が期待値と一致しません: $path ($actual != $expected)"
}

require_no_trainer_process() {
  if pgrep -f "$NNUE_TRAIN" >/dev/null 2>&1; then
    fail "nnue-trainが稼働中です。学習プロセスを確認してください"
  fi
}

require_clean_experiment_checkout() {
  [[ -z "$(git -C "$EXPERIMENT_ROOT" status --porcelain)" ]] \
    || fail "Tatara checkoutに未保存の変更があります: $EXPERIMENT_ROOT"
}
