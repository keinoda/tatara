#!/usr/bin/env bash
# 確定済みall-optim / 16 threadsを2304x16x64実構成で健全性確認する。

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
readonly SMOKE_BATCHES="${SMOKE_BATCHES:-8}"
readonly FINALIZE_EXISTING_SMOKE="${FINALIZE_EXISTING_SMOKE:-0}"
[[ "$SMOKE_BATCHES" =~ ^[1-9][0-9]*$ ]] || fail "SMOKE_BATCHESは1以上の整数にしてください"
[[ "$FINALIZE_EXISTING_SMOKE" =~ ^[01]$ ]] \
  || fail "FINALIZE_EXISTING_SMOKEは0または1にしてください"
smoke_positions=$((65536 * SMOKE_BATCHES))
readonly SMOKE_PSV="$SMOKE_ROOT/smoke.psv"
readonly run_id="${FIXED_TRAIN_PRECISION}-t${FIXED_TRAIN_THREADS}"
readonly run_dir="$SMOKE_ROOT/$run_id"
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

for gate_artifact in \
  "$SMOKE_ROOT/manifest.txt" \
  "$GATE_DIR/precision.approved.txt" \
  "$GATE_DIR/smoke.done" \
  "$GATE_DIR/precision.done"; do
  [[ ! -e "$gate_artifact" ]] || fail "既存smoke gateを上書きしません: $gate_artifact"
done

if [[ "$FINALIZE_EXISTING_SMOKE" == 0 ]]; then
  [[ ! -e "$SMOKE_ROOT" ]] || fail \
    "既存smokeを上書きしません。完走済み成果物のgate確定だけを行う場合はFINALIZE_EXISTING_SMOKE=1を明示してください: $SMOKE_ROOT"
  mkdir -p "$run_dir/checkpoints"
  dd if="$TRAIN_PSV" of="$SMOKE_PSV" bs=40 count="$smoke_positions" status=none
  require_exact_size "$SMOKE_PSV" "$((smoke_positions * PSV_RECORD_BYTES))" "smoke PSV"
  write_command_file "$run_dir/command.txt" "${command[@]}"
  echo "[smoke] precision=$FIXED_TRAIN_PRECISION threads=$FIXED_TRAIN_THREADS"
  "${command[@]}" 2>&1 | tee "$run_dir/train.log"

  python3 "$EXPERIMENT_SCRIPT_DIR/summarize-smoke.py" \
    --output "$SMOKE_ROOT/report.json" "$run_dir"
  finalization_mode="fresh-smoke"
else
  [[ -d "$SMOKE_ROOT" ]] || fail "確定対象の既存smoke directoryがありません: $SMOKE_ROOT"
  [[ -d "$run_dir/checkpoints" ]] || fail "既存smokeのcheckpoint directoryがありません: $run_dir/checkpoints"
  require_exact_size "$SMOKE_PSV" "$((smoke_positions * PSV_RECORD_BYTES))" "既存smoke PSV"
  [[ -s "$run_dir/command.txt" ]] || fail "既存smokeのcommand.txtがありません: $run_dir/command.txt"
  cmp -s "$run_dir/command.txt" <(printf '%q ' "${command[@]}"; printf '\n') \
    || fail "既存smokeのcommand.txtが現在の固定contractと一致しません"
  [[ -s "$run_dir/train.log" ]] || fail "既存smokeのtrain.logがありません: $run_dir/train.log"
  [[ -s "$SMOKE_ROOT/report.json" ]] || fail "既存smokeのreport.jsonがありません: $SMOKE_ROOT/report.json"
  [[ -s "$run_dir/checkpoints/$COMMAND_NET_ID-1.bin" ]] \
    || fail "既存smokeの量子化checkpointがありません"
  [[ -s "$run_dir/checkpoints/$COMMAND_NET_ID-1.ckpt" ]] \
    || fail "既存smokeのraw checkpointがありません"
  finalization_mode="existing-completed-smoke"
  echo "[smoke] 既存の完走済み成果物を再実行せずに検証します"
fi

trainer_commit=$(python3 - \
  "$SMOKE_ROOT/report.json" \
  "$FIXED_TRAIN_PRECISION" \
  "$FIXED_TRAIN_THREADS" \
  "$run_dir" \
  "$COMMAND_NET_ID" \
  "$SMOKE_BATCHES" <<'PY'
import json
import math
from pathlib import Path
import sys

path = Path(sys.argv[1])
precision, threads = sys.argv[2], int(sys.argv[3])
run_dir = Path(sys.argv[4]).resolve()
net_id, smoke_batches = sys.argv[5], int(sys.argv[6])
with path.open(encoding="utf-8") as stream:
    report = json.load(stream)
runs = report.get("runs", [])
if len(runs) != 1:
    raise SystemExit(f"ERROR: expected one fixed smoke run, got {len(runs)}")
run = runs[0]
if run.get("precision") != precision or run.get("threads") != threads:
    raise SystemExit(f"ERROR: fixed smoke settings mismatch: {run}")
if run.get("status") != "completed":
    raise SystemExit(f"ERROR: fixed smoke did not complete: {run}")
for key in ("mean_pos_per_sec", "loss", "test_loss", "test_accuracy", "max_fp16_clamp_ratio"):
    value = run.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SystemExit(f"ERROR: non-finite smoke metric: {key}={value}")
if run["mean_pos_per_sec"] <= 0:
    raise SystemExit(f"ERROR: non-positive smoke throughput: {run['mean_pos_per_sec']}")
if not 0.0 <= run["test_accuracy"] <= 1.0:
    raise SystemExit(f"ERROR: smoke test_accuracy is outside [0, 1]: {run['test_accuracy']}")
if run["max_fp16_clamp_ratio"] < 0.0:
    raise SystemExit(f"ERROR: negative clamp ratio: {run['max_fp16_clamp_ratio']}")

experiment_path = Path(str(run.get("experiment", ""))).resolve()
experiment_root = (run_dir / "checkpoints" / "experiments").resolve()
try:
    experiment_path.relative_to(experiment_root)
except ValueError as error:
    raise SystemExit(f"ERROR: experiment JSON is outside fixed smoke run: {experiment_path}") from error
experiments = sorted(experiment_root.glob("*.json"))
if len(experiments) != 1 or experiments[0].resolve() != experiment_path:
    raise SystemExit(f"ERROR: expected the reported single experiment JSON: {experiments}")
with experiment_path.open(encoding="utf-8") as stream:
    experiment = json.load(stream)
if experiment.get("status") != "completed":
    raise SystemExit(f"ERROR: experiment status is not completed: {experiment.get('status')}")
if (experiment.get("results") or {}).get("interrupted") is not False:
    raise SystemExit("ERROR: completed smoke is marked interrupted")
params = experiment.get("params") or {}
expected_params = {
    "tf32": True,
    "threads": threads,
    "batch_size": 65536,
    "batches_per_superbatch": smoke_batches,
    "superbatches": 1,
    "start_superbatch": 1,
    "architecture": "LayerStack-2304-16-64-8bucket",
    "feature_set": "halfka-hm-merged",
    "bucket_mode": "progress8kpabs",
    "num_buckets": 8,
}
for key, expected in expected_params.items():
    if params.get(key) != expected:
        raise SystemExit(f"ERROR: experiment param mismatch: {key}={params.get(key)!r}, expected={expected!r}")
history = experiment.get("history") or []
if len(history) != 1 or history[0].get("superbatch") != 1:
    raise SystemExit(f"ERROR: smoke history is not exactly SB1: {history}")
expected_checkpoints = {f"{net_id}-1.bin", f"{net_id}-1.ckpt"}
if set(experiment.get("checkpoints") or []) != expected_checkpoints:
    raise SystemExit(f"ERROR: smoke checkpoint records mismatch: {experiment.get('checkpoints')}")
for name in expected_checkpoints:
    checkpoint = run_dir / "checkpoints" / name
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise SystemExit(f"ERROR: recorded smoke checkpoint is missing or empty: {checkpoint}")
commit = experiment.get("commit")
if (
    not isinstance(commit, str)
    or not 7 <= len(commit) <= 40
    or any(c not in "0123456789abcdef" for c in commit)
):
    raise SystemExit(f"ERROR: invalid trainer commit in experiment JSON: {commit!r}")
print(commit)
PY
)

git -C "$EXPERIMENT_ROOT" cat-file -e "$trainer_commit^{commit}" 2>/dev/null \
  || fail "smoke trainer commitをGit objectとして確認できません: $trainer_commit"
trainer_commit_full=$(git -C "$EXPERIMENT_ROOT" rev-parse --verify "$trainer_commit^{commit}")
git -C "$EXPERIMENT_ROOT" merge-base --is-ancestor "$trainer_commit_full" HEAD \
  || fail "smoke trainer commitは現在のsource revisionのancestorではありません: $trainer_commit"
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
  printf 'trainer_commit=%s\n' "$trainer_commit_full"
  printf 'experiment_commit=%s\n' "$trainer_commit"
  printf 'finalization_mode=%s\n' "$finalization_mode"
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
