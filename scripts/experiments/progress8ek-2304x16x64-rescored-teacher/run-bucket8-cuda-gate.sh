#!/usr/bin/env bash
# slot 8追加学習の前に、progress8ek専用kernelとslot限定更新の不変性testを
# 実行GPU上で通し、結果をgateとして固定する。学習は開始しない。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_command cargo
require_command nvidia-smi
require_clean_experiment_checkout
cd "$EXPERIMENT_ROOT"

readonly COMMIT="$(git -C "$EXPERIMENT_ROOT" rev-parse HEAD)"
readonly GATE_DIR="$GATES_ROOT/$(run_name_for_phase bucket8)/cuda-$COMMIT"
[[ ! -e "$GATE_DIR" ]] || fail "既存CUDA gateを上書きしません: $GATE_DIR"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
[[ -n "$gpu_name" ]] || fail "GPUを検出できません"
mkdir -p "$GATE_DIR"

set +e
{
  cargo test --manifest-path "$EXPERIMENT_ROOT/Cargo.toml" --release \
    -p nnue-trainer progress8ek_finetune_updates_only_slot8 -- --nocapture &&
  cargo test --manifest-path "$EXPERIMENT_ROOT/Cargo.toml" --release \
    -p progress8ek-filter &&
  cargo test --manifest-path "$EXPERIMENT_ROOT/Cargo.toml" --release \
    -p net-to-yo
} 2>&1 | tee "$GATE_DIR/gate.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" >"$GATE_DIR/exit-code"
[[ "$rc" == 0 ]] || fail "CUDA gateが失敗しました: rc=$rc ($GATE_DIR/gate.log)"
grep -q 'progress8ek_finetune_updates_only_slot8 \.\.\. ok' "$GATE_DIR/gate.log" \
  || fail "slot限定更新のGPU不変性testの成功行がlogにありません"

{
  printf 'tatara_commit=%s\n' "$COMMIT"
  printf 'gpu=%s\n' "$gpu_name"
  printf 'progress8ek_gpu_invariance_test=passed\n'
  printf 'progress8ek_filter_release_tests=passed\n'
  printf 'net_to_yo_release_tests=passed\n'
  printf 'log=%s\n' "$GATE_DIR/gate.log"
  printf 'log_sha256=%s\n' "$(sha256_file "$GATE_DIR/gate.log")"
} | write_manifest_atomic "$GATE_DIR/manifest.txt"
date -u +%FT%TZ >"$GATE_DIR/done"
echo "[bucket8-cuda-gate] PASS $GATE_DIR"
