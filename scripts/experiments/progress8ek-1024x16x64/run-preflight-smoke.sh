#!/usr/bin/env bash
# 実教師と基準networkで短い追加学習を行い、非対象parameterの不変性を検査する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_legacy_training_complete
require_clean_source
require_single_rtx5090
require_cuda_gate
require_extraction
[[ -x "$NNUE_TRAIN" && -x "$VERIFY_NETWORK_BIN" && -x "$NET_TO_YO" ]] \
  || fail "progress8ek release binaryがありません"

readonly COMMIT="$(git -C "$PROGRESS8EK_ROOT" rev-parse HEAD)"
readonly SMOKE_ROOT="$GATES_ROOT/smoke-$COMMIT-$EXTRACTION_ID"
readonly CHECKPOINTS="$SMOKE_ROOT/checkpoints"
[[ ! -e "$SMOKE_ROOT" ]] || fail "既存smoke gateを上書きしません: $SMOKE_ROOT"
mkdir -p "$CHECKPOINTS" "$SMOKE_ROOT/config" "$SMOKE_ROOT/logs"

base_network=$(select_base_network "$SMOKE_ROOT/config/base-selection.json")
[[ -f "$base_network" ]] || fail "選択した基準networkがありません: $base_network"
progress_bin=$(legacy_progress_bin)
COMMAND_INIT_FROM="$base_network"
COMMAND_DATA="$SELECTED_TRAIN_PSV"
COMMAND_VALIDATION="$SELECTED_HOLDOUT_PSV"
COMMAND_TEST_POSITIONS="$BATCH_SIZE"
COMMAND_OUTPUT="$CHECKPOINTS"
COMMAND_NET_ID="progress8ek-preflight-smoke"
COMMAND_SUPERBATCHES=1
COMMAND_BATCHES_PER_SB=2
COMMAND_SAVE_RATE=1
COMMAND_PROGRESS="$progress_bin"
COMMAND_LR_ARGS=(--lr 3.5e-5 --lr-schedule constant)
build_progress8ek_training_command
write_command_file "$SMOKE_ROOT/config/command.sh" "${TRAINING_COMMAND[@]}"

set +e
bash "$SMOKE_ROOT/config/command.sh" 2>&1 | tee "$SMOKE_ROOT/logs/train.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" >"$SMOKE_ROOT/train-exit-code"
[[ "$rc" == 0 ]] || fail "progress8ek preflight smokeが失敗しました: rc=$rc"
candidate=$(find "$CHECKPOINTS" -maxdepth 1 -type f -name '*-1.bin' | sort | tail -1)
[[ -n "$candidate" ]] || fail "smokeの量子化checkpointが生成されませんでした"
readonly YANEURAOU_NETWORK="$SMOKE_ROOT/progress8ek.nn.bin"
"$VERIFY_NETWORK_BIN" \
  --base "$base_network" \
  --candidate "$candidate" \
  --ft-out "$FT_OUT" \
  --source-slot "$SOURCE_SLOT" \
  --require-slot8-difference \
  >"$SMOKE_ROOT/network-invariance.json"
"$NET_TO_YO" \
  --input "$candidate" \
  --output "$YANEURAOU_NETWORK" \
  --assume-progress8ek
[[ -s "$YANEURAOU_NETWORK" ]] || fail "YaneuraOu形式の変換結果が空です"

{
  printf 'tatara_commit=%s\n' "$COMMIT"
  printf 'extraction_id=%s\n' "$EXTRACTION_ID"
  printf 'extraction_manifest_sha256=%s\n' "$(sha256_file "$SELECTED_EXTRACTION_MANIFEST")"
  printf 'base_network=%s\n' "$base_network"
  printf 'base_network_sha256=%s\n' "$(sha256_file "$base_network")"
  printf 'candidate_network=%s\n' "$candidate"
  printf 'candidate_network_sha256=%s\n' "$(sha256_file "$candidate")"
  printf 'shared_and_slots_0_7_bit_identical=true\n'
  printf 'source_slot=%s\n' "$SOURCE_SLOT"
  printf 'slot8_changed=true\n'
  printf 'network_invariance_report=%s\n' "$SMOKE_ROOT/network-invariance.json"
  printf 'network_invariance_report_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/network-invariance.json")"
  printf 'yaneuraou_network=%s\n' "$YANEURAOU_NETWORK"
  printf 'yaneuraou_network_sha256=%s\n' "$(sha256_file "$YANEURAOU_NETWORK")"
  printf 'yaneuraou_conversion=passed\n'
} | write_manifest_atomic "$SMOKE_ROOT/manifest.txt"
date -u +%FT%TZ >"$SMOKE_ROOT/done"
echo "[progress8ek-smoke] PASS $SMOKE_ROOT"
