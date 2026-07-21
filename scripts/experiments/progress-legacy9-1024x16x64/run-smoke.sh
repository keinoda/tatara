#!/usr/bin/env bash
# 1024x16x64実構成でFP32/all-optimとthread候補を比較し、採用は行わない。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
require_source_revision
require_single_rtx5090 >/dev/null
[[ -f "$STATE_DIR/prepare_data.done" ]] || fail "prepare_data.doneがありません"
[[ -x "$NNUE_TRAIN" ]] || fail "nnue-trainがありません"
progress_bin=$(require_progress_approval)

readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
readonly SMOKE_ROOT="$GATE_DIR/smoke"
[[ ! -e "$SMOKE_ROOT" ]] || fail "既存smokeを上書きしません: $SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"

# 16は提示command、30はNNUE Lab基準runに由来する比較対象。変更時は両方明示する。
readonly THREAD_CANDIDATES="${THREAD_CANDIDATES:-16 30}"
read -r -a thread_candidates <<<"$THREAD_CANDIDATES"
(( ${#thread_candidates[@]} >= 1 )) || fail "THREAD_CANDIDATESが空です"
for threads in "${thread_candidates[@]}"; do
  [[ "$threads" =~ ^[1-9][0-9]*$ ]] || fail "thread候補が不正です: $threads"
done

readonly SMOKE_BATCHES="${SMOKE_BATCHES:-8}"
[[ "$SMOKE_BATCHES" =~ ^[1-9][0-9]*$ ]] || fail "SMOKE_BATCHESは1以上の整数にしてください"
smoke_positions=$((65536 * SMOKE_BATCHES))
readonly SMOKE_PSV="$SMOKE_ROOT/smoke.psv"
dd if="$TRAIN_PSV" of="$SMOKE_PSV" bs=40 count="$smoke_positions" status=none
require_exact_size "$SMOKE_PSV" "$((smoke_positions * PSV_RECORD_BYTES))" "smoke PSV"

run_dirs=()
for precision in fp32 all-optim; do
  for threads in "${thread_candidates[@]}"; do
    run_id="${precision}-t${threads}"
    run_dir="$SMOKE_ROOT/$run_id"
    mkdir -p "$run_dir/checkpoints"
    COMMAND_DATA="$SMOKE_PSV"
    COMMAND_OUTPUT="$run_dir/checkpoints"
    COMMAND_NET_ID="smoke-$RUN_NAME-$run_id"
    COMMAND_SUPERBATCHES=1
    COMMAND_BATCHES_PER_SB="$SMOKE_BATCHES"
    COMMAND_BATCH_SIZE=65536
    COMMAND_THREADS="$threads"
    COMMAND_PROGRESS="$progress_bin"
    COMMAND_VALIDATION="$VALIDATION_PSV"
    COMMAND_PRECISION="$precision"
    COMMAND_SAVE_RATE=1
    COMMAND_KEEP_CHECKPOINTS=1
    COMMAND_RESUME=""
    build_training_command
    command=("${TRAINING_COMMAND[@]}")
    write_command_file "$run_dir/command.txt" "${command[@]}"
    echo "[smoke] precision=$precision threads=$threads"
    "${command[@]}" 2>&1 | tee "$run_dir/train.log"
    run_dirs+=("$run_dir")
  done
done

python3 "$EXPERIMENT_SCRIPT_DIR/summarize-smoke.py" \
  --output "$SMOKE_ROOT/report.json" "${run_dirs[@]}"
{
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'progress_sha256=%s\n' "$(sha256_file "$progress_bin")"
  printf 'smoke_positions=%s\n' "$smoke_positions"
  printf 'thread_candidates=%s\n' "$THREAD_CANDIDATES"
  printf 'report=%s\n' "$SMOKE_ROOT/report.json"
  printf 'report_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/report.json")"
  printf 'automatic_selection=false\n'
} | write_manifest_atomic "$SMOKE_ROOT/manifest.txt"
date -u +%FT%TZ >"$GATE_DIR/smoke.done"
echo "[smoke] 比較完了。report.jsonを確認してapprove-smoke.shを実行してください"
