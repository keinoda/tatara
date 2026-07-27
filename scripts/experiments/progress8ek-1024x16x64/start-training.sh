#!/usr/bin/env bash
# CUDA gateと実教師smokeに合格した構成で追加学習を開始する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ -n "${RUN_NAME:-}" ]] || fail "RUN_NAMEを明示してください"
validate_id "$RUN_NAME"
require_legacy_training_complete
require_clean_source
require_single_rtx5090
require_cuda_gate
require_extraction
require_command tmux
readonly COMMIT="$(git -C "$PROGRESS8EK_ROOT" rev-parse HEAD)"
readonly SMOKE_ROOT="$GATES_ROOT/smoke-$COMMIT-$EXTRACTION_ID"
[[ -f "$SMOKE_ROOT/done" && -f "$SMOKE_ROOT/manifest.txt" ]] \
  || fail "同じrevisionと抽出結果のpreflight smokeが完了していません: $SMOKE_ROOT"
[[ "$(manifest_value "$SMOKE_ROOT/manifest.txt" extraction_manifest_sha256)" == "$(sha256_file "$SELECTED_EXTRACTION_MANIFEST")" ]] \
  || fail "preflight smokeと抽出manifestが一致しません"

readonly RUN_ROOT="$RUNS_ROOT/$RUN_NAME"
readonly CHECKPOINTS="$RUN_ROOT/checkpoints"
readonly CONFIG="$RUN_ROOT/config"
readonly LOGS="$RUN_ROOT/logs"
readonly STATE="$RUN_ROOT/state"
readonly TRAIN_SESSION="train-$RUN_NAME"
[[ ! -e "$RUN_ROOT" ]] || fail "既存runを上書きしません: $RUN_ROOT"
tmux has-session -t "$TRAIN_SESSION" 2>/dev/null && fail "tmux sessionが既にあります: $TRAIN_SESSION"
mkdir -p "$CHECKPOINTS" "$CONFIG" "$LOGS" "$STATE"

base_network=$(manifest_value "$SMOKE_ROOT/manifest.txt" base_network)
[[ "$(sha256_file "$base_network")" == "$(manifest_value "$SMOKE_ROOT/manifest.txt" base_network_sha256)" ]] \
  || fail "基準network SHA-256がpreflight smokeと一致しません"
progress_bin=$(legacy_progress_bin)
plan_args=(
  --metrics "$SELECTED_METRICS"
  --output "$CONFIG/training-plan.json"
  --target-epochs "$FINE_TUNE_TARGET_EPOCHS"
  --superbatches "$FINE_TUNE_SUPERBATCHES"
  --lr-schedule "$FINE_TUNE_LR_SCHEDULE"
  --lr-gamma "$FINE_TUNE_LR_GAMMA"
  --batch-rounding "$FINE_TUNE_BATCH_ROUNDING"
)
if [[ -n "$FINE_TUNE_SAVE_RATE" ]]; then
  plan_args+=(--save-rate "$FINE_TUNE_SAVE_RATE")
fi
python3 "$PROGRESS8EK_SCRIPT_DIR/plan-training.py" \
  "${plan_args[@]}"
readarray -t plan < <(python3 - "$CONFIG/training-plan.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("superbatches", "batches_per_superbatch", "validation_positions", "save_rate", "total_positions", "actual_epochs"):
    print(d[key])
PY
)
[[ ${#plan[@]} == 6 ]] || fail "training planを読めません"

COMMAND_INIT_FROM="$base_network"
COMMAND_DATA="$SELECTED_TRAIN_PSV"
COMMAND_VALIDATION="$SELECTED_HOLDOUT_PSV"
COMMAND_TEST_POSITIONS="${plan[2]}"
COMMAND_OUTPUT="$CHECKPOINTS"
COMMAND_NET_ID="$RUN_NAME"
COMMAND_SUPERBATCHES="${plan[0]}"
COMMAND_BATCHES_PER_SB="${plan[1]}"
COMMAND_SAVE_RATE="${plan[3]}"
COMMAND_PROGRESS="$progress_bin"
if [[ "$FINE_TUNE_LR_SCHEDULE" == step ]]; then
  COMMAND_LR_ARGS=(
    --lr 8.75e-4
    --lr-schedule step
    --lr-gamma "$FINE_TUNE_LR_GAMMA"
    --lr-step 1
  )
else
  COMMAND_LR_ARGS=(
    --lr 8.75e-4
    --lr-schedule one-cycle
    --lr-warmup-pct 0.05
    --lr-div-factor 25
    --lr-final-div-factor 1000
  )
fi
build_progress8ek_training_command
write_command_file "$CONFIG/command.sh" "${TRAINING_COMMAND[@]}"
{
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'tatara_commit=%s\n' "$COMMIT"
  printf 'extraction_id=%s\n' "$EXTRACTION_ID"
  printf 'extraction_manifest=%s\n' "$SELECTED_EXTRACTION_MANIFEST"
  printf 'extraction_manifest_sha256=%s\n' "$(sha256_file "$SELECTED_EXTRACTION_MANIFEST")"
  printf 'base_network=%s\n' "$base_network"
  printf 'base_network_sha256=%s\n' "$(sha256_file "$base_network")"
  printf 'progress_bin=%s\n' "$progress_bin"
  printf 'progress_sha256=%s\n' "$(sha256_file "$progress_bin")"
  printf 'batch_size=%s\n' "$BATCH_SIZE"
  printf 'batches_per_superbatch=%s\n' "${plan[1]}"
  printf 'superbatches=%s\n' "${plan[0]}"
  printf 'total_positions=%s\n' "${plan[4]}"
  printf 'actual_epochs=%s\n' "${plan[5]}"
  printf 'validation_positions=%s\n' "${plan[2]}"
  printf 'lr_schedule=%s\n' "$FINE_TUNE_LR_SCHEDULE"
  printf 'lr=0.000875\n'
  if [[ "$FINE_TUNE_LR_SCHEDULE" == step ]]; then
    printf 'lr_gamma=%s\n' "$FINE_TUNE_LR_GAMMA"
    printf 'lr_step=1\n'
  else
    printf 'lr_warmup_pct=0.05\n'
    printf 'lr_div_factor=25\n'
    printf 'lr_final_div_factor=1000\n'
  fi
  printf 'wdl=%s\n' "$FINE_TUNE_WDL"
  printf 'source_slot=%s\n' "$SOURCE_SLOT"
  printf 'target_slot=8\n'
  printf 'weight_decay=0\n'
  printf 'precision=all-optim\n'
  printf 'threads=%s\n' "$TRAIN_THREADS"
  printf 'updated_parameters=slot8_l1_l2_l3_only\n'
} | write_manifest_atomic "$CONFIG/manifest.txt"

printf -v tmux_body \
  'set -uo pipefail; echo $$ >%q; date -u +%%FT%%TZ >%q; set +e; bash %q 2>&1 | tee %q; rc=${PIPESTATUS[0]}; set -e; printf "%%s\n" "$rc" >%q; date -u +%%FT%%TZ >%q; exit "$rc"' \
  "$STATE/trainer.pid" "$STATE/trainer.started-at" "$CONFIG/command.sh" \
  "$LOGS/train.log" "$STATE/trainer.exit-code" "$STATE/trainer.ended-at"
tmux new-session -d -s "$TRAIN_SESSION" "bash -lc $(printf '%q' "$tmux_body")"
date -u +%FT%TZ >"$STATE/training.started"
echo "[progress8ek-train] tmux=$TRAIN_SESSION run=$RUN_NAME"
echo "[progress8ek-train] ${plan[0]} SB x ${plan[1]} batches/SB x $BATCH_SIZE = ${plan[4]} positions (${plan[5]} epochs)"
