#!/usr/bin/env bash
# slot 8追加学習のrun directoryを作成し、production command・launcher・manifestを
# 固定する。入力hash・gate完了・学習量を照合するだけで、trainerもmonitorも起動しない。
#
# 必須環境変数:
#   FULL_CI_LOG  この固定commitで全crate local CIを通したlogのpath（末尾PASS）

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

if (( $# != 2 )); then
  fail "usage: FULL_CI_LOG=<local-ci.log> $0 <base-network.bin> <approved-progress.bin>"
fi
base_network=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
progress_bin=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
[[ -n "${FULL_CI_LOG:-}" ]] || fail "FULL_CI_LOGに全crate local CIのlog pathを指定してください"
[[ -f "$FULL_CI_LOG" ]] || fail "FULL_CI_LOGがありません: $FULL_CI_LOG"
grep -qx 'PASS' "$FULL_CI_LOG" || fail "FULL_CI_LOGにPASS行がありません: $FULL_CI_LOG"

require_command nvidia-smi
require_command python3
require_clean_experiment_checkout
cd "$EXPERIMENT_ROOT"
readonly COMMIT="$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)"
readonly RUN_NAME="$(run_name_for_phase bucket8)"
readonly RUN_ROOT="$RUNS_ROOT/$RUN_NAME"
readonly GATE_ROOT="$GATES_ROOT/$RUN_NAME"
readonly CUDA_GATE_DIR="$GATE_ROOT/cuda-$COMMIT"
readonly SMOKE_ROOT="$GATE_ROOT/smoke-$COMMIT"
readonly BASE_RUN_ROOT="$RUNS_ROOT/$(run_name_for_phase base)"
readonly BASE_MONITOR_ENV="$BASE_RUN_ROOT/config/monitor.env"

[[ -f "$CUDA_GATE_DIR/done" ]] || fail "同じcommitのCUDA gateが完了していません: $CUDA_GATE_DIR"
[[ -f "$SMOKE_ROOT/done" ]] || fail "同じcommitのpreflight smokeが完了していません: $SMOKE_ROOT"
[[ "$(manifest_value "$SMOKE_ROOT/manifest.txt" base_network)" == "$base_network" ]] \
  || fail "smokeで使ったbase networkと指定が異なります"
[[ ! -e "$RUN_ROOT" ]] || fail "既存runを上書きしません: $RUN_ROOT"
[[ -x "$NNUE_TRAIN" ]] || fail "nnue-trainのrelease binaryがありません: $NNUE_TRAIN"
[[ -f "$base_network" ]] || fail "base networkがありません: $base_network"
[[ -f "$BASE_MONITOR_ENV" ]] || fail "base runのmonitor.envがありません: $BASE_MONITOR_ENV"
require_file_sha256 "$progress_bin" "$APPROVED_PROGRESS_SHA256" "承認済みprogress.bin"
[[ "$(file_size "$ENTERING_KING_PSV")" == "$ENTERING_KING_EXPECTED_BYTES" ]] \
  || fail "相入玉教師のbyte数が固定値と一致しません: $ENTERING_KING_PSV"
[[ "$(manifest_value "$PREPARED_DATA_MANIFEST" entering_king_sha256)" == "$ENTERING_KING_SHA256" ]] \
  || fail "prepared-data manifestの相入玉SHA-256が固定値と一致しません"
echo "[prepare-bucket8] 相入玉教師のSHA-256を再計算しています ($ENTERING_KING_PSV)"
require_file_sha256 "$ENTERING_KING_PSV" "$ENTERING_KING_SHA256" "相入玉教師"
entering_verified_at=$(date -u +%FT%TZ)
gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
[[ -n "$gpu_info" ]] || fail "GPUを検出できません"

build_bucket8_training_command "$base_network" "$progress_bin" "$RUN_ROOT/checkpoints"
presented="$BUCKET8_PRESENTED_POSITIONS"
train_positions=$((BUCKET8_FILE_POSITIONS - BUCKET8_VALIDATION_TAIL_POSITIONS))
read -r epochs_file epochs_train < <(python3 - "$presented" "$BUCKET8_FILE_POSITIONS" "$train_positions" <<'PY'
import sys
presented, file_positions, train_positions = (int(value) for value in sys.argv[1:4])
print(f"{presented / file_positions:.12f} {presented / train_positions:.12f}")
PY
)

umask 077
mkdir -p "$RUN_ROOT/config" "$RUN_ROOT/logs" "$RUN_ROOT/state" "$RUN_ROOT/checkpoints"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n\ncd %q\n\nexec' "$EXPERIMENT_ROOT"
  printf ' \\\n  %q' "${BUCKET8_TRAINING_COMMAND[@]}"
  printf '\n'
} >"$RUN_ROOT/config/command.sh"
chmod 700 "$RUN_ROOT/config/command.sh"
cat >"$RUN_ROOT/config/launch.sh" <<LAUNCH
#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly run=$(printf '%q' "$RUN_ROOT")

printf '%s\\n' "\$\$" >"\$run/state/trainer.pid"
date -u +%FT%TZ >"\$run/state/training.started"

set +e
bash "\$run/config/command.sh" 2>&1 | tee "\$run/logs/train.log"
rc=\${PIPESTATUS[0]}
set -e

printf '%s\\n' "\$rc" >"\$run/state/trainer.exit-code"
date -u +%FT%TZ >"\$run/state/training.ended"
exit "\$rc"
LAUNCH
chmod 700 "$RUN_ROOT/config/launch.sh"
bash -n "$RUN_ROOT/config/command.sh" "$RUN_ROOT/config/launch.sh"
install -m 600 "$BASE_MONITOR_ENV" "$RUN_ROOT/config/monitor.env"
monitor_public_url=$(awk -F= '$1 == "MONITOR_PUBLIC_URL" {sub(/^[^=]*=/, ""); print}' "$RUN_ROOT/config/monitor.env")

{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'training_phase=bucket8\n'
  printf 'tatara_commit=%s\n' "$COMMIT"
  printf 'tatara_origin_branch=codex/progress8ek-rescored-teacher-operations\n'
  printf 'full_ci_log=%s\n' "$FULL_CI_LOG"
  printf 'full_ci_log_sha256=%s\n' "$(sha256_file "$FULL_CI_LOG")"
  printf 'cuda_gate_manifest=%s\n' "$CUDA_GATE_DIR/manifest.txt"
  printf 'cuda_gate_manifest_sha256=%s\n' "$(sha256_file "$CUDA_GATE_DIR/manifest.txt")"
  printf 'smoke_manifest=%s\n' "$SMOKE_ROOT/manifest.txt"
  printf 'smoke_manifest_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/manifest.txt")"
  printf 'nnue_train=%s\n' "$NNUE_TRAIN"
  printf 'nnue_train_sha256=%s\n' "$(sha256_file "$NNUE_TRAIN")"
  printf 'base_network=%s\n' "$base_network"
  printf 'base_network_sha256=%s\n' "$(sha256_file "$base_network")"
  printf 'base_network_role=init_from_8bucket_quantised_network_optimizer_state_reset\n'
  printf 'training_psv=%s\n' "$ENTERING_KING_PSV"
  printf 'training_file_records=%s\n' "$BUCKET8_FILE_POSITIONS"
  printf 'training_bytes=%s\n' "$ENTERING_KING_EXPECTED_BYTES"
  printf 'training_sha256=%s\n' "$ENTERING_KING_SHA256"
  printf 'training_sha256_live_verified_at=%s\n' "$entering_verified_at"
  printf 'validation=same_file_tail\n'
  printf 'validation_tail_positions=%s\n' "$BUCKET8_VALIDATION_TAIL_POSITIONS"
  printf 'training_records_after_tail_reservation=%s\n' "$train_positions"
  printf 'progress_bin=%s\n' "$progress_bin"
  printf 'progress_sha256=%s\n' "$APPROVED_PROGRESS_SHA256"
  printf 'progress_transform=affine\n'
  printf 'progress_a=1.2959360695965043\n'
  printf 'progress_b=-0.564310958286911\n'
  printf 'architecture=2304x16x64\n'
  printf 'bucket_mode=progress8ek\n'
  printf 'num_buckets=9\n'
  printf 'source_slot=%s\n' "$BUCKET8_SOURCE_SLOT"
  printf 'updated_parameters=slot8_l1_l2_l3_only\n'
  printf 'batch_size=%s\n' "$BUCKET8_BATCH_SIZE"
  printf 'batches_per_superbatch=%s\n' "$BUCKET8_BATCHES_PER_SUPERBATCH"
  printf 'superbatches=%s\n' "$BUCKET8_SUPERBATCHES"
  printf 'save_rate=%s\n' "$BUCKET8_SAVE_RATE"
  printf 'lr_schedule=step\n'
  printf 'lr=8.75e-4\n'
  printf 'lr_gamma=0.995\n'
  printf 'lr_step=1\n'
  printf 'wdl=%s\n' "$BUCKET8_WDL"
  printf 'weight_decay=0.0\n'
  printf 'presented_positions=%s\n' "$presented"
  printf 'target_epochs=%s\n' "$BUCKET8_TARGET_EPOCHS"
  printf 'actual_epochs_over_file=%s\n' "$epochs_file"
  printf 'actual_epochs_over_training_records=%s\n' "$epochs_train"
  printf 'command_file=%s\n' "$RUN_ROOT/config/command.sh"
  printf 'command_sha256=%s\n' "$(sha256_file "$RUN_ROOT/config/command.sh")"
  printf 'launch_file=%s\n' "$RUN_ROOT/config/launch.sh"
  printf 'launch_sha256=%s\n' "$(sha256_file "$RUN_ROOT/config/launch.sh")"
  printf 'trainer_session=%s\n' "$BUCKET8_TRAINER_SESSION"
  printf 'monitor_env=%s\n' "$RUN_ROOT/config/monitor.env"
  printf 'monitor_public_url=%s\n' "$monitor_public_url"
  printf 'monitor_milestone_interval_superbatches=%s\n' "$MONITOR_MILESTONE_INTERVAL"
  printf 'gpu=%s\n' "$gpu_info"
  printf 'workspace_available_bytes_at_prepare=%s\n' "$(df --output=avail -B1 "$EXPERIMENT_ROOT" | tail -1 | tr -d ' ')"
  printf 'trainer_started=false\n'
  printf 'auto_resume=false\n'
  printf 'external_backup=false\n'
} | write_manifest_atomic "$RUN_ROOT/config/manifest.txt"

echo "[prepare-bucket8] $RUN_ROOT を準備しました（学習は未開始）"
echo "[prepare-bucket8] 次: switch-monitor-to-bucket8.sh → start-bucket8-training.sh"
