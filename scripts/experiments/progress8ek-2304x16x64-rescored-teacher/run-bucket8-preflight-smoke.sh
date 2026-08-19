#!/usr/bin/env bash
# 実際の相入玉教師とbase networkで1 SB・2 batchのslot 8追加学習を行い、
# 共有FT・L1f・slot 0–7の不変性、slot 8の更新、YaneuraOu形式への変換を検査する。
# production学習は開始しない。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

if (( $# != 2 )); then
  fail "usage: $0 <base-network.bin> <approved-progress.bin>"
fi
base_network=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
progress_bin=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")

require_command nvidia-smi
require_command python3
require_clean_experiment_checkout
cd "$EXPERIMENT_ROOT"
readonly COMMIT="$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)"
readonly GATE_ROOT="$GATES_ROOT/$(run_name_for_phase bucket8)"
readonly CUDA_GATE_DIR="$GATE_ROOT/cuda-$COMMIT"
readonly SMOKE_ROOT="$GATE_ROOT/smoke-$COMMIT"
readonly CHECKPOINTS="$SMOKE_ROOT/checkpoints"

[[ -f "$CUDA_GATE_DIR/done" ]] || fail "同じcommitのCUDA gateが完了していません: $CUDA_GATE_DIR"
[[ ! -e "$SMOKE_ROOT" ]] || fail "既存smoke gateを上書きしません: $SMOKE_ROOT"
[[ -x "$NNUE_TRAIN" && -x "$VERIFY_NETWORK_BIN" && -x "$NET_TO_YO" ]] \
  || fail "nnue-train / progress8ek-verify-network / net_to_yo のrelease binaryがありません"
[[ -f "$base_network" ]] || fail "base networkがありません: $base_network"
require_file_sha256 "$progress_bin" "$APPROVED_PROGRESS_SHA256" "承認済みprogress.bin"
[[ -f "$ENTERING_KING_PSV" ]] || fail "相入玉教師がありません: $ENTERING_KING_PSV"
[[ "$(file_size "$ENTERING_KING_PSV")" == "$ENTERING_KING_EXPECTED_BYTES" ]] \
  || fail "相入玉教師のbyte数が固定値と一致しません: $ENTERING_KING_PSV"
require_no_trainer_process
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ -n "$gpu_name" ]] || fail "GPUを検出できません"

mkdir -p "$CHECKPOINTS" "$SMOKE_ROOT/config" "$SMOKE_ROOT/logs"
build_bucket8_smoke_command "$base_network" "$progress_bin" "$CHECKPOINTS"
{
  printf '#!/usr/bin/env bash\nset -Eeuo pipefail\ncd %q\nexec' "$EXPERIMENT_ROOT"
  printf ' %q' "${BUCKET8_TRAINING_COMMAND[@]}"
  printf '\n'
} >"$SMOKE_ROOT/config/command.sh"
chmod 700 "$SMOKE_ROOT/config/command.sh"

set +e
bash "$SMOKE_ROOT/config/command.sh" 2>&1 | tee "$SMOKE_ROOT/logs/train.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" >"$SMOKE_ROOT/train-exit-code"
[[ "$rc" == 0 ]] || fail "slot 8 preflight smokeが失敗しました: rc=$rc ($SMOKE_ROOT/logs/train.log)"

readonly CANDIDATE="$CHECKPOINTS/$BUCKET8_SMOKE_NET_ID-1.bin"
[[ -f "$CANDIDATE" ]] || fail "smokeの量子化checkpointが生成されませんでした: $CANDIDATE"
"$VERIFY_NETWORK_BIN" \
  --base "$base_network" \
  --candidate "$CANDIDATE" \
  --ft-out 2304 \
  --l1 16 \
  --l2 64 \
  --source-slot "$BUCKET8_SOURCE_SLOT" \
  --require-slot8-difference \
  >"$SMOKE_ROOT/network-invariance.json"
python3 - "$SMOKE_ROOT/network-invariance.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
required = {
    "shared_parameters_bit_identical": True,
    "slots_0_through_7_bit_identical": True,
    "slot8_has_difference_from_source_slot": True,
}
for key, expected in required.items():
    if report.get(key) is not expected:
        raise SystemExit(f"ERROR: network invariance report: {key} != {expected}")
print("network invariance: shared/slot0-7 bit-identical, slot 8 updated")
PY

readonly YANEURAOU_NETWORK="$SMOKE_ROOT/progress8ek.nn.bin"
"$NET_TO_YO" \
  --input "$CANDIDATE" \
  --output "$YANEURAOU_NETWORK" \
  --assume-progress8ek 2>&1 | tee "$SMOKE_ROOT/logs/net_to_yo.log"
[[ -s "$YANEURAOU_NETWORK" ]] || fail "YaneuraOu形式の変換結果が空です"

{
  printf 'tatara_commit=%s\n' "$COMMIT"
  printf 'gpu=%s\n' "$gpu_name"
  printf 'cuda_gate=%s\n' "$CUDA_GATE_DIR/manifest.txt"
  printf 'cuda_gate_sha256=%s\n' "$(sha256_file "$CUDA_GATE_DIR/manifest.txt")"
  printf 'base_network=%s\n' "$base_network"
  printf 'base_network_sha256=%s\n' "$(sha256_file "$base_network")"
  printf 'progress_bin=%s\n' "$progress_bin"
  printf 'progress_sha256=%s\n' "$APPROVED_PROGRESS_SHA256"
  printf 'training_psv=%s\n' "$ENTERING_KING_PSV"
  printf 'validation_tail_positions=%s\n' "$BUCKET8_VALIDATION_TAIL_POSITIONS"
  printf 'source_slot=%s\n' "$BUCKET8_SOURCE_SLOT"
  printf 'smoke_superbatches=1\n'
  printf 'smoke_batches_per_superbatch=2\n'
  printf 'command_file=%s\n' "$SMOKE_ROOT/config/command.sh"
  printf 'command_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/config/command.sh")"
  printf 'candidate_network=%s\n' "$CANDIDATE"
  printf 'candidate_network_sha256=%s\n' "$(sha256_file "$CANDIDATE")"
  printf 'shared_and_slots_0_7_bit_identical=true\n'
  printf 'slot8_changed=true\n'
  printf 'network_invariance_report=%s\n' "$SMOKE_ROOT/network-invariance.json"
  printf 'network_invariance_report_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/network-invariance.json")"
  printf 'yaneuraou_network=%s\n' "$YANEURAOU_NETWORK"
  printf 'yaneuraou_network_sha256=%s\n' "$(sha256_file "$YANEURAOU_NETWORK")"
  printf 'yaneuraou_conversion=assume-progress8ek\n'
  printf 'train_log=%s\n' "$SMOKE_ROOT/logs/train.log"
  printf 'train_log_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/logs/train.log")"
} | write_manifest_atomic "$SMOKE_ROOT/manifest.txt"
date -u +%FT%TZ >"$SMOKE_ROOT/done"
echo "[bucket8-smoke] PASS $SMOKE_ROOT"
