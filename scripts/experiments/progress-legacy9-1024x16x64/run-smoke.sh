#!/usr/bin/env bash
# 確定済みall-optim / 16 threadsを1024x16x64実構成で健全性確認する。

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

readonly SMOKE_BATCHES="${SMOKE_BATCHES:-8}"
[[ "$SMOKE_BATCHES" =~ ^[1-9][0-9]*$ ]] || fail "SMOKE_BATCHESは1以上の整数にしてください"
smoke_positions=$((65536 * SMOKE_BATCHES))
readonly SMOKE_PSV="$SMOKE_ROOT/smoke.psv"
dd if="$TRAIN_PSV" of="$SMOKE_PSV" bs=40 count="$smoke_positions" status=none
require_exact_size "$SMOKE_PSV" "$((smoke_positions * PSV_RECORD_BYTES))" "smoke PSV"

readonly run_id="${FIXED_TRAIN_PRECISION}-t${FIXED_TRAIN_THREADS}"
readonly run_dir="$SMOKE_ROOT/$run_id"
mkdir -p "$run_dir/checkpoints"
COMMAND_DATA="$SMOKE_PSV"
COMMAND_OUTPUT="$run_dir/checkpoints"
COMMAND_NET_ID="smoke-$RUN_NAME-$run_id"
COMMAND_SUPERBATCHES=1
COMMAND_BATCHES_PER_SB="$SMOKE_BATCHES"
COMMAND_BATCH_SIZE=65536
COMMAND_THREADS="$FIXED_TRAIN_THREADS"
COMMAND_PROGRESS="$progress_bin"
COMMAND_VALIDATION="$VALIDATION_PSV"
COMMAND_PRECISION="$FIXED_TRAIN_PRECISION"
COMMAND_SAVE_RATE=1
COMMAND_KEEP_CHECKPOINTS=1
COMMAND_RESUME=""
build_training_command
command=("${TRAINING_COMMAND[@]}")
write_command_file "$run_dir/command.txt" "${command[@]}"
echo "[smoke] precision=$FIXED_TRAIN_PRECISION threads=$FIXED_TRAIN_THREADS"
"${command[@]}" 2>&1 | tee "$run_dir/train.log"

python3 "$EXPERIMENT_SCRIPT_DIR/summarize-smoke.py" \
  --output "$SMOKE_ROOT/report.json" "$run_dir"
python3 - "$SMOKE_ROOT/report.json" "$FIXED_TRAIN_PRECISION" "$FIXED_TRAIN_THREADS" <<'PY'
import json
import math
import sys

path, precision, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, encoding="utf-8") as stream:
    report = json.load(stream)
runs = report.get("runs", [])
if len(runs) != 1:
    raise SystemExit(f"ERROR: expected one fixed smoke run, got {len(runs)}")
run = runs[0]
if run.get("precision") != precision or run.get("threads") != threads:
    raise SystemExit(f"ERROR: fixed smoke settings mismatch: {run}")
if run.get("status") != "complete":
    raise SystemExit(f"ERROR: fixed smoke did not complete: {run}")
for key in ("mean_pos_per_sec", "loss", "test_loss", "max_fp16_clamp_ratio"):
    value = run.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"ERROR: non-finite smoke metric: {key}={value}")
PY
{
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'progress_sha256=%s\n' "$(sha256_file "$progress_bin")"
  printf 'smoke_positions=%s\n' "$smoke_positions"
  printf 'precision=%s\n' "$FIXED_TRAIN_PRECISION"
  printf 'threads=%s\n' "$FIXED_TRAIN_THREADS"
  printf 'cpu_model=AMD Ryzen 9 9950X\n'
  printf 'report=%s\n' "$SMOKE_ROOT/report.json"
  printf 'report_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/report.json")"
  printf 'setting_source=user_decision_before_smoke\n'
} | write_manifest_atomic "$SMOKE_ROOT/manifest.txt"

{
  printf 'fixed_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'precision=%s\n' "$FIXED_TRAIN_PRECISION"
  printf 'threads=%s\n' "$FIXED_TRAIN_THREADS"
  printf 'cpu_model=AMD Ryzen 9 9950X\n'
  printf 'smoke_report=%s\n' "$SMOKE_ROOT/report.json"
  printf 'smoke_report_sha256=%s\n' "$(sha256_file "$SMOKE_ROOT/report.json")"
  printf 'decision_source=user_fixed_all_optim_threads_delegated_for_16_physical_cores\n'
} | write_manifest_atomic "$GATE_DIR/precision.approved.txt"
date -u +%FT%TZ >"$GATE_DIR/smoke.done"
date -u +%FT%TZ >"$GATE_DIR/precision.done"
echo "[smoke] all-optim / 16 threadsの健全性確認を完了しました"
