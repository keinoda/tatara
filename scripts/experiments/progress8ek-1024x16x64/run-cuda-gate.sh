#!/usr/bin/env bash
# 現学習完了後、progress8ek専用kernelとslot限定更新をRTX 5090で検証する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_legacy_training_complete
require_clean_source
require_single_rtx5090
cd "$PROGRESS8EK_ROOT"
readonly COMMIT="$(git -C "$PROGRESS8EK_ROOT" rev-parse HEAD)"
readonly GATE_DIR="$GATES_ROOT/cuda-$COMMIT"
[[ ! -e "$GATE_DIR" ]] || fail "既存CUDA gateを上書きしません: $GATE_DIR"
mkdir -p "$GATE_DIR"

set +e
{
  bash "$PROGRESS8EK_ROOT/scripts/local-ci.sh" &&
  cargo build --manifest-path "$PROGRESS8EK_ROOT/Cargo.toml" --release \
    -p nnue-trainer -p net-to-yo -p progress8ek-filter &&
  cargo test --manifest-path "$PROGRESS8EK_ROOT/Cargo.toml" --release \
    -p nnue-trainer progress8ek_finetune_updates_only_slot8 -- --nocapture &&
  cargo test --manifest-path "$PROGRESS8EK_ROOT/Cargo.toml" --release \
    -p progress8ek-filter &&
  cargo test --manifest-path "$PROGRESS8EK_ROOT/Cargo.toml" --release \
    -p net-to-yo
} 2>&1 | tee "$GATE_DIR/gate.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" >"$GATE_DIR/exit-code"
[[ "$rc" == 0 ]] || fail "CUDA gateが失敗しました: rc=$rc"
{
  printf 'tatara_commit=%s\n' "$COMMIT"
  printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  printf 'local_ci=passed\n'
  printf 'progress8ek_gpu_invariance_test=passed\n'
  printf 'filter_release_tests=passed\n'
  printf 'net_to_yo_release_tests=passed\n'
  printf 'log_sha256=%s\n' "$(sha256_file "$GATE_DIR/gate.log")"
} | write_manifest_atomic "$GATE_DIR/manifest.txt"
date -u +%FT%TZ >"$GATE_DIR/done"
echo "[cuda-gate] PASS $GATE_DIR"
