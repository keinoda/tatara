#!/usr/bin/env bash
# Vast.aiで公開教師データを使う2304x16x64学習環境を準備する。
#
# 実施範囲:
#   - bootstrapが/workspace直下へcloneした指定Tatara commitを検証
#   - official upstreamの固定commitを設定し、専用commitがそれを含むことを検証
#   - 公開教師34 shardを1個ずつdownload・検証し、単一PSVの末尾へ順次追記
#   - legacy progress.binの固定commitからのdownloadとchecksum検証
#   - Tatara/rshogiのbuild、validation PSVの生成
#
# 本学習、progress係数の採用、外部backupは自動実行しない。
# shard全体と連結PSVを二重保持せず、1 TBの/workspace内で準備する。
set -Eeuo pipefail

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$WORKSPACE_ROOT/progress8kpabs-2304x16x64-training}"
export CARGO_HOME="${CARGO_HOME:-/opt/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}"
export HF_HOME="${HF_HOME:-$EXPERIMENT_ROOT/.runtime/huggingface}"
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$WORKSPACE_ROOT"
readonly ONSTART_LOG="$WORKSPACE_ROOT/onstart.log"
exec > >(tee -a "$ONSTART_LOG") 2>&1
echo "===== onstart bootstrap $(date -u +%FT%TZ) ====="

readonly TATARA_REPO="https://github.com/keinoda/tatara.git"
readonly TATARA_UPSTREAM_REPO="https://github.com/SH11235/tatara.git"
readonly TATARA_UPSTREAM_COMMIT="da3ea68d46a5c1ac0c18c10a57fef52d02788879"
readonly TATARA_COMMIT="${TATARA_COMMIT:?TATARA_COMMITをinstance作成時に明示してください}"
[[ "$TATARA_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "ERROR: TATARA_COMMITは40桁のGit SHAで指定してください" >&2; exit 1; }

readonly RSHOGI_REPO="https://github.com/SH11235/rshogi.git"
readonly RSHOGI_COMMIT="29245a1d8e4f198aba3fc832a506649221cb2f2c"
export RSHOGI_DIR="$EXPERIMENT_ROOT/.runtime/rshogi"

readonly YANEURAOU_COMMIT="771fe811f877859d6851ceccfd3e04c16454e689"
readonly CONTAINER_IMAGE="ghcr.io/keinoda/shogi-lab:cuda129-trt1011"
readonly CONTAINER_IMAGE_DIGEST="sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876"

readonly TRAIN_DATASET="sashimin/test20260726"
readonly TRAIN_DATASET_REVISION="8f461dd8dc4cb90c356392545a41e4e45c8f2418"
export TRAIN_DATA_DIR="$EXPERIMENT_ROOT/data/training"
export TRAIN_SHARD_DIR="$TRAIN_DATA_DIR/shards"
export TRAIN_PSV="$TRAIN_DATA_DIR/public-teacher.psv"
export TRAIN_PARTIAL_PSV="$TRAIN_PSV.partial"
export TRAIN_APPEND_STATE_DIR="$EXPERIMENT_ROOT/.onstart/training-append"
export TRAIN_SHARD_SPEC="$EXPERIMENT_ROOT/scripts/experiments/progress8kpabs-2304x16x64/training-shards.tsv"
export TRAIN_SURVEY_SHARD="$TRAIN_SHARD_DIR/split_000.bin"
readonly TRAIN_EXPECTED_SHARDS=34
readonly TRAIN_EXPECTED_BYTES=673002105840
readonly TRAIN_EXPECTED_POSITIONS=16825052646
readonly TRAIN_MAX_SHARD_BYTES=20000000000

readonly VALIDATION_DATASET="takaoyamaoka/floodgate.hcpe"
readonly VALIDATION_DATASET_REVISION="fdd5f602db82d888a87116f087d10dd5ea8313ab"
export VALIDATION_DIR="$EXPERIMENT_ROOT/data/validation"
export VALIDATION_HCPE="$VALIDATION_DIR/floodgate.hcpe"
export VALIDATION_PSV="$VALIDATION_DIR/floodgate.psv"
readonly VALIDATION_HCPE_BYTES=32563074
readonly VALIDATION_PSV_BYTES=34276920
readonly VALIDATION_POSITIONS=856923

readonly PROGRESS_SOURCE_COMMIT="771fe811f877859d6851ceccfd3e04c16454e689"
readonly PROGRESS_SOURCE_URL="https://raw.githubusercontent.com/keinoda/YaneuraOu/$PROGRESS_SOURCE_COMMIT/source/progress.bin"
export BASELINE_PROGRESS_DIR="$EXPERIMENT_ROOT/progress/baseline"
export BASELINE_PROGRESS_BIN="$BASELINE_PROGRESS_DIR/progress.bin"
readonly BASELINE_PROGRESS_BYTES=1003104
readonly BASELINE_PROGRESS_SHA256="d77f47e874558d42fa2d87d173de3aba054eef51bcca9c1fc9f3a8daf93630d8"

readonly PSV_RECORD_BYTES=40
readonly WORKSPACE_HEADROOM_BYTES=100000000000

# tmux内の子shellで参照する定数だけを明示的にexportする。
export RSHOGI_REPO RSHOGI_COMMIT
export TRAIN_DATASET TRAIN_DATASET_REVISION TRAIN_EXPECTED_SHARDS TRAIN_EXPECTED_BYTES TRAIN_EXPECTED_POSITIONS
export VALIDATION_DATASET VALIDATION_DATASET_REVISION VALIDATION_HCPE_BYTES VALIDATION_PSV_BYTES VALIDATION_POSITIONS
export PROGRESS_SOURCE_COMMIT PROGRESS_SOURCE_URL
export BASELINE_PROGRESS_BYTES BASELINE_PROGRESS_SHA256
export PSV_RECORD_BYTES

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "必要なコマンド '$1' が見つかりません"
}

file_size() {
  stat -c '%s' "$1"
}

for command_name in \
  awk bash cat chmod chown cmp curl cut date df find git grep head mv nproc paste python3 realpath rm \
  rustc service sha256sum sleep stat tail tee tmux touch truncate; do
  require_command "$command_name"
done

# 課金開始後にSSHできなくなる既知の権限事故を防ぐ。
[[ -f /root/.ssh/authorized_keys ]] \
  || fail "/root/.ssh/authorized_keysが存在しません"
chown root:root /root /root/.ssh /root/.ssh/authorized_keys
chmod go-w /root
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
service ssh start
touch /root/.no_auto_tmux

# `--onstart-cmd`のbootstrapがGitから取得したcheckoutだけを受理する。
# 本スクリプト本文をVast.aiへ貼り付けたり、ここで別revisionをcloneしたりしない。
[[ -d "$EXPERIMENT_ROOT/.git" ]] \
  || fail "Tatara checkoutがありません。RUNBOOKのgit clone bootstrapから起動してください: $EXPERIMENT_ROOT"
actual_origin=$(git -C "$EXPERIMENT_ROOT" remote get-url origin)
[[ "$actual_origin" == "$TATARA_REPO" ]] \
  || fail "Tatara originが想定外です: $actual_origin"
[[ -z "$(git -C "$EXPERIMENT_ROOT" status --porcelain)" ]] \
  || fail "Tatara checkoutに未保存の変更があります。上書きせず停止します"
[[ "$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)" == "$TATARA_COMMIT" ]] \
  || fail "Tatara checkoutが固定commitと異なります"

if git -C "$EXPERIMENT_ROOT" remote get-url upstream >/dev/null 2>&1; then
  actual_upstream=$(git -C "$EXPERIMENT_ROOT" remote get-url upstream)
  [[ "$actual_upstream" == "$TATARA_UPSTREAM_REPO" ]] \
    || fail "Tatara upstreamが想定外です: $actual_upstream"
else
  git -C "$EXPERIMENT_ROOT" remote add upstream "$TATARA_UPSTREAM_REPO"
fi
git -C "$EXPERIMENT_ROOT" fetch upstream "$TATARA_UPSTREAM_COMMIT"
git -C "$EXPERIMENT_ROOT" merge-base --is-ancestor \
  "$TATARA_UPSTREAM_COMMIT" HEAD \
  || fail "学習専用commitが固定upstream commitを含んでいません"

export STATE_DIR="$EXPERIMENT_ROOT/.onstart"
export LOG_DIR="$EXPERIMENT_ROOT/logs/onstart"
export MANIFEST_DIR="$EXPERIMENT_ROOT/manifests"
readonly PLAN_PATH="$EXPERIMENT_ROOT/docs/experiments/progress8kpabs-2304x16x64/PLAN.md"
readonly DECISIONS_PATH="$EXPERIMENT_ROOT/docs/experiments/progress8kpabs-2304x16x64/DECISIONS.md"
readonly RUNBOOK_PATH="$EXPERIMENT_ROOT/docs/experiments/progress8kpabs-2304x16x64/RUNBOOK.md"
readonly TRAIN_LAUNCHER="$EXPERIMENT_ROOT/scripts/experiments/progress8kpabs-2304x16x64/run-training.sh"

mkdir -p \
  "$STATE_DIR" \
  "$TRAIN_APPEND_STATE_DIR" \
  "$LOG_DIR" \
  "$MANIFEST_DIR" \
  "$HF_HOME" \
  "$TRAIN_SHARD_DIR" \
  "$VALIDATION_DIR" \
  "$BASELINE_PROGRESS_DIR" \
  "$EXPERIMENT_ROOT/progress/candidates" \
  "$EXPERIMENT_ROOT/survey" \
  "$EXPERIMENT_ROOT/runs"

echo "===== onstart $(date -u +%FT%TZ) ====="

[[ -f "$PLAN_PATH" ]] || fail "学習計画がありません: $PLAN_PATH"
[[ -f "$DECISIONS_PATH" ]] || fail "決定台帳がありません: $DECISIONS_PATH"
[[ -f "$RUNBOOK_PATH" ]] || fail "実行手順がありません: $RUNBOOK_PATH"
[[ -x "$TRAIN_LAUNCHER" ]] || fail "学習launcherがありません: $TRAIN_LAUNCHER"
[[ -f "$TRAIN_SHARD_SPEC" ]] || fail "教師shard仕様がありません: $TRAIN_SHARD_SPEC"
spec_shards=$(awk 'NF {count++} END {print count+0}' "$TRAIN_SHARD_SPEC")
spec_bytes=$(awk 'NF {total += $2} END {printf "%.0f\n", total}' "$TRAIN_SHARD_SPEC")
(( spec_shards == TRAIN_EXPECTED_SHARDS )) \
  || fail "教師shard仕様の件数が不正です: actual=$spec_shards expected=$TRAIN_EXPECTED_SHARDS"
(( spec_bytes == TRAIN_EXPECTED_BYTES )) \
  || fail "教師shard仕様の合計sizeが不正です: actual=$spec_bytes expected=$TRAIN_EXPECTED_BYTES"

require_command cargo
require_command hf
require_command nvidia-smi
require_command rustup

if command -v llc-22 >/dev/null 2>&1; then
  llc-22 --version | head -n 1
elif command -v llc-21 >/dev/null 2>&1; then
  llc-21 --version | head -n 1
else
  fail "Tataraのbuildに必要なllc-21以上が見つかりません"
fi
if ! command -v clang-22 >/dev/null 2>&1 && ! command -v clang-21 >/dev/null 2>&1; then
  fail "Tataraのbuildに必要なclang-21以上が見つかりません"
fi
[[ -e /usr/local/cuda/lib64/libcublas.so ]] \
  || fail "/usr/local/cuda/lib64/libcublas.soが見つかりません"
gpu_lines=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
gpu_count=$(printf '%s\n' "$gpu_lines" | awk 'NF {n++} END {print n+0}')
(( gpu_count == 1 )) || fail "GPUはRTX 5090 1枚である必要があります: count=$gpu_count"
[[ "$(printf '%s\n' "$gpu_lines" | cut -d, -f1)" == *"RTX 5090"* ]] \
  || fail "GPUがRTX 5090ではありません: $gpu_lines"
printf '%s\n' "$gpu_lines"

tatara_revision=$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)
[[ "$tatara_revision" == "$TATARA_COMMIT" ]] || fail "Tatara checkoutのcommitが固定値と異なります"
source_manifest="$MANIFEST_DIR/source-revisions.txt"
source_manifest_tmp="$source_manifest.tmp.$BASHPID"
{
  printf 'tatara=%s\n' "$tatara_revision"
  printf 'tatara_upstream=%s\n' "$TATARA_UPSTREAM_COMMIT"
  printf 'rshogi=%s\n' "$RSHOGI_COMMIT"
  printf 'yaneuraou=%s\n' "$YANEURAOU_COMMIT"
  printf 'training_dataset=%s\n' "$TRAIN_DATASET"
  printf 'training_dataset_revision=%s\n' "$TRAIN_DATASET_REVISION"
  printf 'training_shard_spec_sha256=%s\n' "$(sha256sum "$TRAIN_SHARD_SPEC" | awk '{print $1}')"
  printf 'validation_dataset=%s\n' "$VALIDATION_DATASET"
  printf 'validation_dataset_revision=%s\n' "$VALIDATION_DATASET_REVISION"
  printf 'progress_source_commit=%s\n' "$PROGRESS_SOURCE_COMMIT"
  printf 'container_image=%s@%s\n' "$CONTAINER_IMAGE" "$CONTAINER_IMAGE_DIGEST"
} >"$source_manifest_tmp"
if [[ ! -e "$source_manifest" ]]; then
  mv "$source_manifest_tmp" "$source_manifest"
else
  cmp -s "$source_manifest_tmp" "$source_manifest" \
    || fail "既存source manifestが今回の固定revision群と異なります"
  rm -- "$source_manifest_tmp"
fi

if [[ -e "$TRAIN_PSV" ]]; then
  existing_psv_bytes=$(file_size "$TRAIN_PSV")
  (( existing_psv_bytes == TRAIN_EXPECTED_BYTES )) \
    || fail "既存の連結PSVが不完全です: $TRAIN_PSV (${existing_psv_bytes}B)。上書きしません"
  required_remaining_bytes=$WORKSPACE_HEADROOM_BYTES
elif [[ -e "$TRAIN_PARTIAL_PSV" ]]; then
  existing_psv_bytes=$(file_size "$TRAIN_PARTIAL_PSV")
  (( existing_psv_bytes <= TRAIN_EXPECTED_BYTES )) \
    || fail "追記中PSVが想定sizeを超えています: $TRAIN_PARTIAL_PSV (${existing_psv_bytes}B)"
  required_remaining_bytes=$((
    TRAIN_EXPECTED_BYTES - existing_psv_bytes
    + 2 * TRAIN_MAX_SHARD_BYTES
    + WORKSPACE_HEADROOM_BYTES
  ))
else
  required_remaining_bytes=$((
    TRAIN_EXPECTED_BYTES
    + 2 * TRAIN_MAX_SHARD_BYTES
    + WORKSPACE_HEADROOM_BYTES
  ))
fi
workspace_available_bytes=$(df -PB1 "$WORKSPACE_ROOT" | awk 'NR == 2 { print $4 }')
[[ "$workspace_available_bytes" =~ ^[0-9]+$ ]] \
  || fail "/workspaceの空き容量を取得できませんでした"
if (( workspace_available_bytes < required_remaining_bytes )); then
  fail "/workspaceの空き容量不足: available=${workspace_available_bytes}B required=${required_remaining_bytes}B。1TB級volumeを指定してください"
fi
echo "[onstart] disk preflight: available=${workspace_available_bytes}B required=${required_remaining_bytes}B"

# tmux内で長時間stepを実行し、成功/失敗markerと個別logを残す。失敗は自動再試行しない。
start_step() {
  local step_name="$1"
  local step_body="$2"
  local done_file="$STATE_DIR/$step_name.done"
  local failed_file="$STATE_DIR/$step_name.failed"
  local log_file="$LOG_DIR/$step_name.log"

  if [[ -f "$done_file" ]]; then
    echo "[onstart] step '$step_name' は完了済みです"
    return
  fi
  if [[ -f "$failed_file" ]]; then
    fail "step '$step_name' は前回失敗済みです。retry-onstart-step.shで確認後に再試行してください: $failed_file"
  fi
  if tmux has-session -t "$step_name" 2>/dev/null; then
    echo "[onstart] tmux '$step_name' は実行中です"
    return
  fi

  local quoted_body outer_script tmux_command
  printf -v quoted_body '%q' "set -Eeuo pipefail"$'\n'"$step_body"
  printf -v outer_script \
    'set -uo pipefail; exec >>%q 2>&1; echo "===== %s start $(date -u +%%FT%%TZ) ====="; set +e; bash -lc %s; rc=$?; set -e; if (( rc != 0 )); then printf "%%s rc=%%s\n" "$(date -u +%%FT%%TZ)" "$rc" >%q; echo "[%s] failed rc=$rc"; exit "$rc"; fi; date -u +%%FT%%TZ >%q; echo "===== %s done $(date -u +%%FT%%TZ) ====="' \
    "$log_file" "$step_name" "$quoted_body" "$failed_file" "$step_name" "$done_file" "$step_name"
  printf -v tmux_command 'bash -lc %q' "$outer_script"
  tmux new-session -d -s "$step_name" "$tmux_command"
  echo "[onstart] tmux '$step_name' を開始しました: $log_file"
}

read -r -d '' build_tatara_body <<'STEP' || true
cd "$EXPERIMENT_ROOT"
bash scripts/setup-cuda-oxide.sh
bash scripts/build-kernels.sh
cargo build --release -p nnue-trainer -p net-to-yo -p progress-bucket-survey
cargo test --release -p net-to-yo -p nnue-format -p progress-bucket-survey
cargo test --release -p nnue-trainer --no-default-features
target/release/nnue-train layerstack --help | grep -F -- '--num-buckets'
target/release/nnue-train layerstack --help | grep -F 'progress8kpabs'
target/release/net_to_yo --help | grep -F 'assume-progress8kpabs'
STEP
start_step build_tatara "$build_tatara_body"

read -r -d '' build_rshogi_body <<'STEP' || true
if [[ -d "$RSHOGI_DIR/.git" ]]; then
  actual_origin=$(git -C "$RSHOGI_DIR" remote get-url origin)
  [[ "$actual_origin" == "$RSHOGI_REPO" ]] \
    || { echo "ERROR: rshogi originが想定外です: $actual_origin" >&2; exit 1; }
  [[ -z "$(git -C "$RSHOGI_DIR" status --porcelain)" ]] \
    || { echo "ERROR: rshogi checkoutに未保存の変更があります" >&2; exit 1; }
  [[ "$(git -C "$RSHOGI_DIR" rev-parse HEAD)" == "$RSHOGI_COMMIT" ]] \
    || { echo "ERROR: rshogi revisionが固定値と異なります" >&2; exit 1; }
else
  git clone --no-checkout "$RSHOGI_REPO" "$RSHOGI_DIR"
  git -C "$RSHOGI_DIR" checkout --detach "$RSHOGI_COMMIT"
fi
rshogi_revision=$(git -C "$RSHOGI_DIR" rev-parse HEAD)
[[ "$rshogi_revision" == "$RSHOGI_COMMIT" ]] \
  || { echo "ERROR: rshogi checkoutのcommitが固定値と異なります" >&2; exit 1; }
cd "$RSHOGI_DIR"
cargo build --release -p tools --no-default-features --features nnue-arch --bin hcpe_to_psv --bin psv_to_jsonl
STEP
start_step build_rshogi "$build_rshogi_body"

read -r -d '' download_training_body <<'STEP' || true
if [[ -e "$TRAIN_PSV" ]]; then
  [[ ! -e "$TRAIN_PARTIAL_PSV" ]] \
    || { echo "ERROR: 完成PSVと追記中PSVが同時に存在します" >&2; exit 1; }
  actual_bytes=$(stat -c '%s' "$TRAIN_PSV")
  (( actual_bytes == TRAIN_EXPECTED_BYTES )) \
    || { echo "ERROR: 完成PSV bytes=$actual_bytes expected=$TRAIN_EXPECTED_BYTES" >&2; exit 1; }
  echo "[download_training] 完成済みPSVを再利用します"
  exit 0
fi

mkdir -p "$TRAIN_APPEND_STATE_DIR" "$TRAIN_SHARD_DIR"
if [[ ! -e "$TRAIN_PARTIAL_PSV" ]]; then
  shopt -s nullglob
  existing_markers=("$TRAIN_APPEND_STATE_DIR"/*.done)
  (( ${#existing_markers[@]} == 0 )) \
    || { echo "ERROR: 追記中PSVが無いのに確定markerが残っています" >&2; exit 1; }
  : >"$TRAIN_PARTIAL_PSV"
fi

# markerで確定した連続prefixだけを正とする。追記中に中断した末尾はその境界へ戻す。
committed_bytes=0
missing_seen=0
while IFS=$'\t' read -r shard_name expected_bytes expected_sha; do
  [[ -n "$shard_name" ]] || continue
  marker="$TRAIN_APPEND_STATE_DIR/$shard_name.done"
  expected_marker="$shard_name"$'\t'"$expected_bytes"$'\t'"$expected_sha"$'\t'"$((committed_bytes + expected_bytes))"
  if [[ -f "$marker" ]]; then
    (( missing_seen == 0 )) \
      || { echo "ERROR: shard確定markerに欠番があります: $marker" >&2; exit 1; }
    [[ "$(cat "$marker")" == "$expected_marker" ]] \
      || { echo "ERROR: shard確定markerの内容が不正です: $marker" >&2; exit 1; }
    committed_bytes=$((committed_bytes + expected_bytes))
  else
    missing_seen=1
  fi
done <"$TRAIN_SHARD_SPEC"

partial_bytes=$(stat -c '%s' "$TRAIN_PARTIAL_PSV")
(( partial_bytes >= committed_bytes )) \
  || { echo "ERROR: 追記中PSVが確定済みprefixより短いです: actual=$partial_bytes committed=$committed_bytes" >&2; exit 1; }
if (( partial_bytes > committed_bytes )); then
  echo "[download_training] 中断した未確定末尾を切り戻します: $partial_bytes -> $committed_bytes"
  truncate --size "$committed_bytes" "$TRAIN_PARTIAL_PSV"
fi

while IFS=$'\t' read -r shard_name expected_bytes expected_sha; do
  [[ -n "$shard_name" ]] || continue
  shard="$TRAIN_SHARD_DIR/$shard_name"
  marker="$TRAIN_APPEND_STATE_DIR/$shard_name.done"
  if [[ -f "$marker" ]]; then
    if [[ -e "$shard" ]]; then
      actual_bytes=$(stat -c '%s' "$shard")
      actual_sha=$(sha256sum "$shard" | awk '{print $1}')
      [[ "$actual_bytes" == "$expected_bytes" && "$actual_sha" == "$expected_sha" ]] \
        || { echo "ERROR: 確定済みshardの残存fileが仕様と異なります: $shard" >&2; exit 1; }
      if [[ "$shard" != "$TRAIN_SURVEY_SHARD" ]]; then
        rm -- "$shard"
      fi
    fi
    continue
  fi

  hf download "$TRAIN_DATASET" "$shard_name" \
    --repo-type dataset \
    --revision "$TRAIN_DATASET_REVISION" \
    --local-dir "$TRAIN_SHARD_DIR"
  [[ -f "$shard" ]] || { echo "ERROR: shardを取得できませんでした: $shard" >&2; exit 1; }
  actual_bytes=$(stat -c '%s' "$shard")
  (( actual_bytes == expected_bytes )) \
    || { echo "ERROR: shard bytes=$actual_bytes expected=$expected_bytes path=$shard" >&2; exit 1; }
  (( actual_bytes % PSV_RECORD_BYTES == 0 )) \
    || { echo "ERROR: shardが40-byte PSV境界に揃っていません: $shard" >&2; exit 1; }
  actual_sha=$(sha256sum "$shard" | awk '{print $1}')
  [[ "$actual_sha" == "$expected_sha" ]] \
    || { echo "ERROR: shard SHA-256=$actual_sha expected=$expected_sha path=$shard" >&2; exit 1; }

  before_bytes=$(stat -c '%s' "$TRAIN_PARTIAL_PSV")
  expected_after=$((before_bytes + expected_bytes))
  python3 - "$TRAIN_PARTIAL_PSV" "$shard" <<'PY'
import os
import sys

output, source = sys.argv[1:]
with open(output, "ab", buffering=0) as destination:
    with open(source, "rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            destination.write(chunk)
    os.fsync(destination.fileno())
PY
  after_bytes=$(stat -c '%s' "$TRAIN_PARTIAL_PSV")
  (( after_bytes == expected_after )) \
    || { echo "ERROR: shard追記後のsizeが不正です: actual=$after_bytes expected=$expected_after" >&2; exit 1; }

  marker_tmp="$marker.tmp.$BASHPID"
  printf '%s\t%s\t%s\t%s\n' "$shard_name" "$expected_bytes" "$expected_sha" "$after_bytes" >"$marker_tmp"
  mv "$marker_tmp" "$marker"
  if [[ "$shard" == "$TRAIN_SURVEY_SHARD" ]]; then
    echo "[download_training] $shard_nameは部分取得survey用に一時保持します"
  else
    rm -- "$shard"
  fi
  echo "[download_training] $shard_nameを検証・追記しました: cumulative=$after_bytes"
done <"$TRAIN_SHARD_SPEC"

final_bytes=$(stat -c '%s' "$TRAIN_PARTIAL_PSV")
(( final_bytes == TRAIN_EXPECTED_BYTES )) \
  || { echo "ERROR: 連結PSV bytes=$final_bytes expected=$TRAIN_EXPECTED_BYTES" >&2; exit 1; }

append_manifest="$MANIFEST_DIR/training-inline-append.tsv"
append_tmp="$append_manifest.tmp.$BASHPID"
committed_bytes=0
: >"$append_tmp"
while IFS=$'\t' read -r shard_name expected_bytes expected_sha; do
  [[ -n "$shard_name" ]] || continue
  committed_bytes=$((committed_bytes + expected_bytes))
  marker="$TRAIN_APPEND_STATE_DIR/$shard_name.done"
  printf '%s\t%s\t%s\t%s\n' "$shard_name" "$expected_bytes" "$expected_sha" "$committed_bytes" >>"$append_tmp"
  cmp -s "$marker" <(tail -n 1 "$append_tmp") \
    || { echo "ERROR: 最終marker検証に失敗しました: $marker" >&2; exit 1; }
done <"$TRAIN_SHARD_SPEC"
if [[ -e "$append_manifest" ]]; then
  cmp -s "$append_tmp" "$append_manifest" \
    || { echo "ERROR: 既存training-inline-append.tsvが再検証結果と異なります" >&2; exit 1; }
  rm -- "$append_tmp"
else
  mv "$append_tmp" "$append_manifest"
fi

mv "$TRAIN_PARTIAL_PSV" "$TRAIN_PSV"
echo "[download_training] $TRAIN_EXPECTED_SHARDS shard / $final_bytes bytesを順次検証・追記しました"
STEP
start_step download_training "$download_training_body"

read -r -d '' download_validation_body <<'STEP' || true
hf download "$VALIDATION_DATASET" \
  --repo-type dataset \
  --revision "$VALIDATION_DATASET_REVISION" \
  --include floodgate.hcpe \
  --local-dir "$VALIDATION_DIR"
actual_bytes=$(stat -c '%s' "$VALIDATION_HCPE")
(( actual_bytes == VALIDATION_HCPE_BYTES )) \
  || { echo "ERROR: floodgate.hcpe bytes=$actual_bytes expected=$VALIDATION_HCPE_BYTES" >&2; exit 1; }
validation_checksum="$MANIFEST_DIR/validation-hcpe.sha256"
validation_tmp="$validation_checksum.tmp.$BASHPID"
sha256sum "$VALIDATION_HCPE" >"$validation_tmp"
if [[ -e "$validation_checksum" ]]; then
  cmp -s "$validation_tmp" "$validation_checksum" \
    || { echo "ERROR: 既存validation-hcpe.sha256が再検証結果と異なります" >&2; exit 1; }
else
  mv "$validation_tmp" "$validation_checksum"
fi
STEP
start_step download_validation "$download_validation_body"

read -r -d '' download_progress_body <<'STEP' || true
if [[ -e "$BASELINE_PROGRESS_BIN" ]]; then
  actual_bytes=$(stat -c '%s' "$BASELINE_PROGRESS_BIN")
  actual_sha=$(sha256sum "$BASELINE_PROGRESS_BIN" | awk '{ print $1 }')
  (( actual_bytes == BASELINE_PROGRESS_BYTES )) \
    || { echo "ERROR: 既存progress.binのsizeが不正です: ${actual_bytes}B" >&2; exit 1; }
  [[ "$actual_sha" == "$BASELINE_PROGRESS_SHA256" ]] \
    || { echo "ERROR: 既存progress.binのSHA-256が不正です: $actual_sha" >&2; exit 1; }
else
  curl --fail --location --retry 5 \
    --output "$BASELINE_PROGRESS_BIN" "$PROGRESS_SOURCE_URL"
fi
actual_bytes=$(stat -c '%s' "$BASELINE_PROGRESS_BIN")
actual_sha=$(sha256sum "$BASELINE_PROGRESS_BIN" | awk '{ print $1 }')
(( actual_bytes == BASELINE_PROGRESS_BYTES )) \
  || { echo "ERROR: progress.bin size=$actual_bytes expected=$BASELINE_PROGRESS_BYTES" >&2; exit 1; }
[[ "$actual_sha" == "$BASELINE_PROGRESS_SHA256" ]] \
  || { echo "ERROR: progress.bin SHA-256=$actual_sha expected=$BASELINE_PROGRESS_SHA256" >&2; exit 1; }
baseline_manifest="$MANIFEST_DIR/baseline-progress.txt"
baseline_tmp="$baseline_manifest.tmp.$BASHPID"
{
  printf 'source_commit=%s\n' "$PROGRESS_SOURCE_COMMIT"
  printf 'bytes=%s\n' "$actual_bytes"
  printf 'sha256=%s\n' "$actual_sha"
} >"$baseline_tmp"
if [[ -e "$baseline_manifest" ]]; then
  cmp -s "$baseline_tmp" "$baseline_manifest" \
    || { echo "ERROR: 既存baseline-progress.txtが再検証結果と異なります" >&2; exit 1; }
else
  mv "$baseline_tmp" "$baseline_manifest"
fi
STEP
start_step download_progress "$download_progress_body"

read -r -d '' prepare_data_body <<'STEP' || true
wait_for_step() {
  local dependency="$1"
  while [[ ! -f "$STATE_DIR/$dependency.done" ]]; do
    if [[ -f "$STATE_DIR/$dependency.failed" ]]; then
      echo "ERROR: dependency '$dependency' failed" >&2
      exit 1
    fi
    sleep 10
  done
}

wait_for_step download_training
wait_for_step download_validation
wait_for_step download_progress
wait_for_step build_rshogi
wait_for_step build_tatara

[[ -f "$TRAIN_PSV" ]] || { echo "ERROR: 順次追記済み教師PSVがありません: $TRAIN_PSV" >&2; exit 1; }
actual_bytes=$(stat -c '%s' "$TRAIN_PSV")
(( actual_bytes == TRAIN_EXPECTED_BYTES )) \
  || { echo "ERROR: 教師PSV bytes=$actual_bytes expected=$TRAIN_EXPECTED_BYTES" >&2; exit 1; }
actual_positions=$((actual_bytes / PSV_RECORD_BYTES))
(( actual_positions == TRAIN_EXPECTED_POSITIONS )) \
  || { echo "ERROR: 教師PSV positions=$actual_positions expected=$TRAIN_EXPECTED_POSITIONS" >&2; exit 1; }
training_sha=$(sha256sum "$TRAIN_PSV" | awk '{print $1}')

if [[ -e "$VALIDATION_PSV" ]]; then
  actual_validation_bytes=$(stat -c '%s' "$VALIDATION_PSV")
  (( actual_validation_bytes == VALIDATION_PSV_BYTES )) \
    || { echo "ERROR: 既存floodgate PSVのsizeが不正です: ${actual_validation_bytes}B。上書きしません" >&2; exit 1; }
else
  "$RSHOGI_DIR/target/release/hcpe_to_psv" \
    --input "$VALIDATION_HCPE" \
    --output "$VALIDATION_PSV"
fi

actual_validation_bytes=$(stat -c '%s' "$VALIDATION_PSV")
(( actual_validation_bytes == VALIDATION_PSV_BYTES )) \
  || { echo "ERROR: floodgate PSV bytes=$actual_validation_bytes expected=$VALIDATION_PSV_BYTES" >&2; exit 1; }
actual_validation_positions=$((actual_validation_bytes / PSV_RECORD_BYTES))
(( actual_validation_positions == VALIDATION_POSITIONS )) \
  || { echo "ERROR: floodgate PSV positions=$actual_validation_positions expected=$VALIDATION_POSITIONS" >&2; exit 1; }
validation_sha=$(sha256sum "$VALIDATION_PSV" | awk '{print $1}')

prepared_manifest="$MANIFEST_DIR/prepared-data.txt"
prepared_tmp="$prepared_manifest.tmp.$BASHPID"
{
  printf 'training_psv=%s\n' "$TRAIN_PSV"
  printf 'training_bytes=%s\n' "$actual_bytes"
  printf 'training_positions=%s\n' "$actual_positions"
  printf 'training_sha256=%s\n' "$training_sha"
  printf 'training_order=split_000_to_split_033_inline_append_no_reshuffle\n'
  printf 'training_append_manifest=%s\n' "$MANIFEST_DIR/training-inline-append.tsv"
  printf 'training_append_manifest_sha256=%s\n' "$(sha256sum "$MANIFEST_DIR/training-inline-append.tsv" | awk '{print $1}')"
  printf 'survey_shard=%s\n' "$TRAIN_SURVEY_SHARD"
  printf 'validation_psv=%s\n' "$VALIDATION_PSV"
  printf 'validation_bytes=%s\n' "$actual_validation_bytes"
  printf 'validation_effective_positions=851968\n'
  printf 'validation_sha256=%s\n' "$validation_sha"
} >"$prepared_tmp"
if [[ -e "$prepared_manifest" ]]; then
  cmp -s "$prepared_tmp" "$prepared_manifest" \
    || { echo "ERROR: 既存prepared-data.txtが再検証結果と異なります" >&2; exit 1; }
else
  mv "$prepared_tmp" "$prepared_manifest"
fi
STEP
start_step prepare_data "$prepare_data_body"

cat <<SUMMARY
[onstart] 準備stepを開始しました。本学習は自動開始していません。

状態確認:
  tail -f $ONSTART_LOG
  tmux ls
  ls -la $STATE_DIR
  tail -f $LOG_DIR/download_training.log
  tail -f $LOG_DIR/prepare_data.log
  $EXPERIMENT_ROOT/scripts/experiments/progress8kpabs-2304x16x64/onstart-status.sh

計画:
  less $PLAN_PATH
  less $DECISIONS_PATH
  less $RUNBOOK_PATH

次のgate:
  1. prepare_data.doneを確認
  2. 完成shardから校正/検証sampleを抽出してbaseline分布とaffine候補をsurvey
  3. survey結果を提示し、使用するprogress.binをユーザーが明示選択
  4. GPU smoke / resume / converter / monitor試験を各scriptで実行
  5. RUN_NAMEとPROGRESS_APPROVALを明示して$TRAIN_LAUNCHERを手動実行

固定学習値:
  batch-size=65536, batches-per-superbatch=6104, superbatches=421
  lr=0.000875, schedule=step, gamma=0.992, step=1
  architecture=2304x16x64, 8 training buckets, fixed 8-way progress routing
  export=net_to_yoがbucket 7を未使用の第9slotへ複製
  validation=ファイル856923局面、実効851968局面（65536×13 full batches）

注意:
  - 公開教師はsplit番号順に末尾追記し、再shuffleしません。
  - split_000.binだけは部分取得survey用に一時保持し、本学習前に明示削除します。
  - baseline progress係数も自動採用しません。
  - 既存checkout、dataset、runを上書きしません。
  - 外部backup、自動resume、自動再起動は行いません。
SUMMARY

echo "===== onstart end $(date -u +%FT%TZ) ====="
