#!/usr/bin/env bash
# Vast.ai上でprogress8ek 2304x16x64学習に必要なTatara実行環境だけを準備する。
# 教師データ、validation、progress.binの取得と学習開始は行わない。

set -Eeuo pipefail

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$WORKSPACE_ROOT/progress8ek-2304x16x64-rescored-teacher}"
export CARGO_HOME="${CARGO_HOME:-/opt/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}"
export PATH="$CARGO_HOME/bin:/usr/local/cuda/bin:$PATH"

readonly TATARA_REPO="https://github.com/keinoda/tatara.git"
readonly TATARA_UPSTREAM_REPO="https://github.com/SH11235/tatara.git"
readonly TATARA_UPSTREAM_COMMIT="da3ea68d46a5c1ac0c18c10a57fef52d02788879"
readonly TATARA_COMMIT="${TATARA_COMMIT:?TATARA_COMMITをinstance作成時に明示してください}"
readonly CONTAINER_IMAGE="ghcr.io/keinoda/shogi-lab:cuda129-trt1011"
readonly CONTAINER_IMAGE_DIGEST="sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876"
readonly STATE_DIR="$EXPERIMENT_ROOT/.onstart"
readonly LOG_DIR="$EXPERIMENT_ROOT/logs/onstart"
readonly MANIFEST_DIR="$EXPERIMENT_ROOT/manifests"
readonly ONSTART_LOG="$WORKSPACE_ROOT/onstart.log"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "必要なコマンド '$1' が見つかりません"
}

[[ "$TATARA_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || fail "TATARA_COMMITは40桁の小文字Git SHAで指定してください"

mkdir -p "$WORKSPACE_ROOT" "$STATE_DIR" "$LOG_DIR" "$MANIFEST_DIR" \
  "$EXPERIMENT_ROOT/runs" "$EXPERIMENT_ROOT/gates" "$EXPERIMENT_ROOT/monitor"
exec > >(tee -a "$ONSTART_LOG") 2>&1
echo "===== onstart $(date -u +%FT%TZ) ====="

for command_name in \
  awk bash chmod chown cmp cut date df git grep head lscpu mkdir mv nproc \
  nvidia-smi rm rustc service sha256sum tee tmux touch; do
  require_command "$command_name"
done
require_command cargo
require_command rustup

# SSH鍵のowner/modeを固定し、Vast.aiのlogin tmuxを無効化する。
[[ -f /root/.ssh/authorized_keys ]] \
  || fail "/root/.ssh/authorized_keysが存在しません"
chown root:root /root /root/.ssh /root/.ssh/authorized_keys
chmod go-w /root
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
service ssh start
touch /root/.no_auto_tmux

# Web UIへ貼るbootstrapが取得したcleanなdetached checkoutだけを受理する。
[[ -d "$EXPERIMENT_ROOT/.git" ]] \
  || fail "Tatara checkoutがありません: $EXPERIMENT_ROOT"
actual_origin=$(git -C "$EXPERIMENT_ROOT" remote get-url origin)
[[ "$actual_origin" == "$TATARA_REPO" ]] \
  || fail "Tatara originが想定外です: $actual_origin"
[[ -z "$(git -C "$EXPERIMENT_ROOT" status --porcelain)" ]] \
  || fail "Tatara checkoutに未保存の変更があります"
actual_commit=$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)
[[ "$actual_commit" == "$TATARA_COMMIT" ]] \
  || fail "Tatara checkoutが固定commitと異なります: actual=$actual_commit expected=$TATARA_COMMIT"

if git -C "$EXPERIMENT_ROOT" remote get-url upstream >/dev/null 2>&1; then
  actual_upstream=$(git -C "$EXPERIMENT_ROOT" remote get-url upstream)
  [[ "$actual_upstream" == "$TATARA_UPSTREAM_REPO" ]] \
    || fail "Tatara upstreamが想定外です: $actual_upstream"
else
  git -C "$EXPERIMENT_ROOT" remote add upstream "$TATARA_UPSTREAM_REPO"
fi
git -C "$EXPERIMENT_ROOT" fetch upstream "$TATARA_UPSTREAM_COMMIT"
git -C "$EXPERIMENT_ROOT" merge-base --is-ancestor "$TATARA_UPSTREAM_COMMIT" HEAD \
  || fail "学習専用commitが固定upstream commitを含んでいません"

if command -v llc-22 >/dev/null 2>&1; then
  llc_version_output=$(llc-22 --version)
elif command -v llc-21 >/dev/null 2>&1; then
  llc_version_output=$(llc-21 --version)
else
  fail "Tataraのbuildに必要なllc-21以上が見つかりません"
fi
llc_version=${llc_version_output%%$'\n'*}
if command -v clang-22 >/dev/null 2>&1; then
  clang_version_output=$(clang-22 --version)
elif command -v clang-21 >/dev/null 2>&1; then
  clang_version_output=$(clang-21 --version)
else
  fail "Tataraのbuildに必要なclang-21以上が見つかりません"
fi
clang_version=${clang_version_output%%$'\n'*}
[[ -e /usr/local/cuda/lib64/libcublas.so ]] \
  || fail "/usr/local/cuda/lib64/libcublas.soが見つかりません"

gpu_lines=$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader)
gpu_count=$(printf '%s\n' "$gpu_lines" | awk 'NF {n++} END {print n+0}')
(( gpu_count == 1 )) || fail "GPUはRTX 5090 1枚である必要があります: count=$gpu_count"
gpu_name=$(printf '%s\n' "$gpu_lines" | cut -d, -f1)
[[ "$gpu_name" == *"RTX 5090"* ]] || fail "GPUがRTX 5090ではありません: $gpu_lines"

cpu_model=$(lscpu | awk -F: '$1 ~ /^Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')
[[ "$cpu_model" == *"AMD Ryzen 9 9950X"* ]] \
  || fail "CPUがAMD Ryzen 9 9950Xではありません: ${cpu_model:-unknown}"
available_cpus=$(nproc)
(( available_cpus >= 16 )) \
  || fail "学習threads 16に必要なCPUが割り当てられていません: nproc=$available_cpus"
awk -v target="$WORKSPACE_ROOT" '$2 == target {found=1} END {exit !found}' /proc/mounts \
  || fail "$WORKSPACE_ROOTが独立volumeのmount pointではありません"
workspace_mount_target="$WORKSPACE_ROOT"
read -r workspace_total_bytes workspace_available_bytes < <(
  df -PB1 "$WORKSPACE_ROOT" | awk 'NR == 2 {print $2, $4}'
)
[[ "$workspace_total_bytes" =~ ^[0-9]+$ ]] \
  || fail "/workspaceの総容量を取得できませんでした"
[[ "$workspace_available_bytes" =~ ^[0-9]+$ ]] \
  || fail "/workspaceの空き容量を取得できませんでした"

printf '%s\n' "$gpu_lines"
printf 'cpu=%s nproc=%s\n' "$cpu_model" "$available_cpus"
printf 'workspace_total_bytes=%s workspace_available_bytes=%s\n' \
  "$workspace_total_bytes" "$workspace_available_bytes"
printf 'llvm=%s\nclang=%s\n' "$llc_version" "$clang_version"

instance_manifest="$MANIFEST_DIR/instance.txt"
instance_manifest_candidate="$MANIFEST_DIR/instance.candidate.$BASHPID"
{
  printf 'tatara=%s\n' "$actual_commit"
  printf 'tatara_upstream=%s\n' "$TATARA_UPSTREAM_COMMIT"
  printf 'container_image=%s@%s\n' "$CONTAINER_IMAGE" "$CONTAINER_IMAGE_DIGEST"
  printf 'gpu=%s\n' "$gpu_lines"
  printf 'cpu=%s\n' "$cpu_model"
  printf 'available_cpus=%s\n' "$available_cpus"
  printf 'dataset_download=deferred\n'
} >"$instance_manifest_candidate"
if [[ -e "$instance_manifest" ]]; then
  cmp -s "$instance_manifest_candidate" "$instance_manifest" \
    || fail "既存instance manifestが現在の実行環境と異なります"
  rm -- "$instance_manifest_candidate"
else
  mv "$instance_manifest_candidate" "$instance_manifest"
fi
capacity_manifest="$MANIFEST_DIR/capacity-at-create.txt"
if [[ ! -e "$capacity_manifest" ]]; then
  capacity_candidate="$MANIFEST_DIR/capacity-at-create.candidate.$BASHPID"
  {
    printf 'workspace_mount_target=%s\n' "$workspace_mount_target"
    printf 'workspace_total_bytes=%s\n' "$workspace_total_bytes"
    printf 'workspace_available_bytes=%s\n' "$workspace_available_bytes"
  } >"$capacity_candidate"
  mv "$capacity_candidate" "$capacity_manifest"
fi

# 長時間buildは独立tmuxで実行し、失敗時は自動再試行しない。
start_job() {
  local job_name="$1"
  local job_body="$2"
  local done_file="$STATE_DIR/$job_name.done"
  local failed_file="$STATE_DIR/$job_name.failed"
  local log_file="$LOG_DIR/$job_name.log"

  if [[ -f "$done_file" ]]; then
    echo "[onstart] job '$job_name' は完了済みです"
    return
  fi
  [[ ! -f "$failed_file" ]] \
    || fail "job '$job_name' は失敗済みです。原因確認後にretry-onstart-job.shを使ってください"
  if tmux has-session -t "$job_name" 2>/dev/null; then
    echo "[onstart] tmux '$job_name' は実行中です"
    return
  fi

  local quoted_body outer_script tmux_command
  printf -v quoted_body '%q' "set -Eeuo pipefail"$'\n'"$job_body"
  printf -v outer_script \
    'set -uo pipefail; exec >>%q 2>&1; echo "===== %s start $(date -u +%%FT%%TZ) ====="; set +e; bash -lc %s; rc=$?; set -e; if (( rc != 0 )); then printf "%%s rc=%%s\n" "$(date -u +%%FT%%TZ)" "$rc" >%q; echo "[%s] failed rc=$rc"; exit "$rc"; fi; date -u +%%FT%%TZ >%q; echo "===== %s done $(date -u +%%FT%%TZ) ====="' \
    "$log_file" "$job_name" "$quoted_body" "$failed_file" "$job_name" "$done_file" "$job_name"
  printf -v tmux_command 'bash -lc %q' "$outer_script"
  tmux new-session -d -s "$job_name" "$tmux_command"
  echo "[onstart] tmux '$job_name' を開始しました: $log_file"
}

read -r -d '' build_tatara_body <<'JOB' || true
cd "$EXPERIMENT_ROOT"
bash scripts/setup-cuda-oxide.sh
bash scripts/build-kernels.sh
cargo build --release \
  -p nnue-trainer -p net-to-yo -p progress-bucket-survey -p progress8ek-filter
cargo test --release \
  -p net-to-yo -p nnue-format -p progress-bucket-survey -p progress8ek-filter
cargo test --release -p nnue-trainer --no-default-features
cargo test --release -p nnue-trainer progress8ek_finetune_updates_only_slot8 -- --nocapture
if [[ -s nnue_train.ll ]]; then
  kernel_dir=.
elif [[ -s bins/nnue_train/nnue_train.ll ]]; then
  kernel_dir=bins/nnue_train
else
  echo "ERROR: nnue_train.llが生成されていません" >&2
  exit 1
fi
test -s "$kernel_dir/nnue_train.ptx"
target/release/nnue-train layerstack --help | grep -F -- '--progress8ek-finetune'
target/release/nnue-train layerstack --help | grep -F -- '--progress8ek-source-slot'
target/release/nnue-train layerstack --help | grep -F 'progress8kpabs'
target/release/nnue-train layerstack --help | grep -F 'progress8ek'
target/release/net_to_yo --help | grep -F 'assume-progress8ek'
target/release/progress8ek-filter --help >/dev/null
sha256sum "$kernel_dir/nnue_train.ll" "$kernel_dir/nnue_train.ptx" target/release/nnue-train \
  target/release/net_to_yo target/release/progress8ek-filter
JOB
start_job build_tatara "$build_tatara_body"

echo "[onstart] 教師データのdownloadは開始していません"
echo "[onstart] 状態確認: $EXPERIMENT_ROOT/scripts/experiments/progress8ek-2304x16x64-rescored-teacher/onstart-status.sh"
