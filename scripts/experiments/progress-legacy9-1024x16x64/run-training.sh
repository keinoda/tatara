#!/usr/bin/env bash
# 公開教師を使う1024x16x64 fixed8 progress学習を手動開始する。
# progress係数はsurvey後にユーザーが選択したpathを必ず明示する。
set -Eeuo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
DERIVED_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd -P)
if [[ -n "${EXPERIMENT_ROOT:-}" ]]; then
  requested_root=$(cd "$EXPERIMENT_ROOT" && pwd -P)
  [[ "$requested_root" == "$DERIVED_ROOT" ]] \
    || fail "EXPERIMENT_ROOTがlauncherのGit checkoutと一致しません: $requested_root != $DERIVED_ROOT"
fi
readonly EXPERIMENT_ROOT="$DERIVED_ROOT"

[[ -n "${RUN_NAME:-}" ]] \
  || fail "RUN_NAMEを明示してください（例: progress-legacy9-1024x16x64-e10）"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fail "RUN_NAMEは英数字で始め、英数字・ピリオド・アンダースコア・ハイフンだけを使ってください"

readonly TRAIN_PSV="${TRAIN_PSV:-$EXPERIMENT_ROOT/data/training/public-teacher.psv}"
readonly VALIDATION_PSV="${VALIDATION_PSV:-$EXPERIMENT_ROOT/data/validation/floodgate.psv}"
[[ -n "${LEGACY_PROGRESS_BIN:-}" ]] \
  || fail "survey結果の採否決定後、LEGACY_PROGRESS_BINを明示してください。baselineも自動採用しません"
readonly LEGACY_PROGRESS_BIN
readonly TRAIN_THREADS="${TRAIN_THREADS:-30}"

[[ "$TRAIN_THREADS" =~ ^[1-9][0-9]*$ ]] \
  || fail "TRAIN_THREADSは1以上の整数にしてください: $TRAIN_THREADS"

readonly NNUE_TRAIN="$EXPERIMENT_ROOT/target/release/nnue-train"
readonly RUN_ROOT="$EXPERIMENT_ROOT/runs/$RUN_NAME"
readonly CHECKPOINT_DIR="$RUN_ROOT/checkpoints"
readonly CONFIG_DIR="$RUN_ROOT/config"
readonly LOG_DIR="$RUN_ROOT/logs"

[[ -x "$NNUE_TRAIN" ]] || fail "$NNUE_TRAINがありません。onstartのbuild_tataraを確認してください"
[[ -f "$TRAIN_PSV" ]] || fail "教師PSVがありません: $TRAIN_PSV"
[[ -f "$VALIDATION_PSV" ]] || fail "validation PSVがありません: $VALIDATION_PSV"
[[ -f "$LEGACY_PROGRESS_BIN" ]] || fail "progress.binがありません: $LEGACY_PROGRESS_BIN"
[[ $(stat -c '%s' "$TRAIN_PSV") == 586757977480 ]] \
  || fail "教師PSVのsizeが想定値と一致しません: $TRAIN_PSV"
[[ $(stat -c '%s' "$VALIDATION_PSV") == 34276920 ]] \
  || fail "validation PSVのsizeが想定値と一致しません: $VALIDATION_PSV"
[[ $(stat -c '%s' "$LEGACY_PROGRESS_BIN") == 1003104 ]] \
  || fail "progress.binは1,003,104 bytesである必要があります: $LEGACY_PROGRESS_BIN"

progress_real=$(realpath -e "$LEGACY_PROGRESS_BIN")
[[ "$progress_real" == "$EXPERIMENT_ROOT"/* ]] \
  || fail "LEGACY_PROGRESS_BINは学習専用folder直下のbaselineまたは承認済みcandidateを指定してください"

python3 - "$LEGACY_PROGRESS_BIN" <<'PY'
import math
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as stream:
    values = struct.iter_unpack("<d", stream.read())
    for index, (value,) in enumerate(values):
        if not math.isfinite(value):
            raise SystemExit(f"ERROR: progress.binに有限でない係数があります: index={index} value={value}")
PY

[[ ! -e "$RUN_ROOT" ]] || fail "$RUN_ROOTは既に存在します。既存runを上書きしません"
mkdir -p "$CONFIG_DIR" "$LOG_DIR"

progress_sha=$(sha256sum "$LEGACY_PROGRESS_BIN" | awk '{ print $1 }')
{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'tatara_commit=%s\n' "$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)"
  printf 'training_psv=%s\n' "$TRAIN_PSV"
  printf 'training_bytes=%s\n' "$(stat -c '%s' "$TRAIN_PSV")"
  printf 'validation_psv=%s\n' "$VALIDATION_PSV"
  printf 'validation_bytes=%s\n' "$(stat -c '%s' "$VALIDATION_PSV")"
  printf 'progress_bin=%s\n' "$LEGACY_PROGRESS_BIN"
  printf 'progress_sha256=%s\n' "$progress_sha"
  printf 'train_threads=%s\n' "$TRAIN_THREADS"
  printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',')"
  printf 'rustc=%s\n' "$(rustc --version)"
} >"$CONFIG_DIR/manifest.txt"

command=(
  "$NNUE_TRAIN"
  --data "$TRAIN_PSV"
  --feature-set halfka-hm-merged
  --batch-size 65536
  --batches-per-superbatch 6104
  --superbatches 367
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
  --all-optim
  --test-data "$VALIDATION_PSV"
  --test-positions 856923
  --threads "$TRAIN_THREADS"
  --save-rate 20
  --keep-checkpoints 2
  --monitor-fp16-clamps
  --monitor-active-features
  --output "$CHECKPOINT_DIR"
  --net-id "$RUN_NAME"
  --experiment-name "$RUN_NAME"
  layerstack
  --ft-out 1024
  --l1 16
  --l2 64
  --fv-scale 28
  --bucket-mode progress8kpabs
  --num-buckets 8
  --progress-coeff "$LEGACY_PROGRESS_BIN"
)

{
  printf '%q ' "${command[@]}"
  printf '\n'
} >"$CONFIG_DIR/command.txt"

echo "[train] run=$RUN_NAME threads=$TRAIN_THREADS progress_sha256=$progress_sha"
echo "[train] 367 SB × 6104 batch/SB × 65536 position/batch = 146,811,650,048局面"
echo "[train] 約10.0083 epoch。自動resumeと外部backupは行いません"

"${command[@]}" 2>&1 | tee "$LOG_DIR/train.log"
