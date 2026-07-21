#!/usr/bin/env bash
# raw checkpointを1 SBで作り、同じ設定でSB2へtrue resumeできることを確認する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
require_source_revision
require_single_rtx5090 >/dev/null
progress_bin=$(require_progress_approval)
readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
require_gate "$GATE_DIR" smoke
precision=$(precision_from_gate "$GATE_DIR")
threads=$(manifest_value "$GATE_DIR/precision.approved.txt" threads)
readonly DRILL_ROOT="$GATE_DIR/resume-drill"
readonly SMOKE_PSV="$GATE_DIR/smoke/smoke.psv"
[[ -f "$SMOKE_PSV" ]] || fail "smoke PSVがありません: $SMOKE_PSV"
[[ ! -e "$DRILL_ROOT" ]] || fail "既存resume drillを上書きしません: $DRILL_ROOT"
mkdir -p "$DRILL_ROOT/initial" "$DRILL_ROOT/resumed"

COMMAND_DATA="$SMOKE_PSV"
COMMAND_OUTPUT="$DRILL_ROOT/initial"
COMMAND_NET_ID="resume-$RUN_NAME-initial"
COMMAND_SUPERBATCHES=1
COMMAND_BATCHES_PER_SB=1
COMMAND_BATCH_SIZE=65536
COMMAND_THREADS="$threads"
COMMAND_PROGRESS="$progress_bin"
COMMAND_VALIDATION="$VALIDATION_PSV"
COMMAND_PRECISION="$precision"
COMMAND_SAVE_RATE=1
COMMAND_KEEP_CHECKPOINTS=1
COMMAND_RESUME=""
build_training_command
initial_command=("${TRAINING_COMMAND[@]}")
write_command_file "$DRILL_ROOT/initial-command.txt" "${initial_command[@]}"
"${initial_command[@]}" 2>&1 | tee "$DRILL_ROOT/initial.log"

initial_ckpt="$DRILL_ROOT/initial/resume-$RUN_NAME-initial-1.ckpt"
[[ -s "$initial_ckpt" ]] || fail "SB1 raw checkpointが生成されませんでした: $initial_ckpt"

COMMAND_OUTPUT="$DRILL_ROOT/resumed"
COMMAND_NET_ID="resume-$RUN_NAME-resumed"
COMMAND_SUPERBATCHES=2
COMMAND_RESUME="$initial_ckpt"
build_training_command
resumed_command=("${TRAINING_COMMAND[@]}")
write_command_file "$DRILL_ROOT/resumed-command.txt" "${resumed_command[@]}"
"${resumed_command[@]}" 2>&1 | tee "$DRILL_ROOT/resumed.log"

resumed_ckpt="$DRILL_ROOT/resumed/resume-$RUN_NAME-resumed-2.ckpt"
[[ -s "$resumed_ckpt" ]] || fail "SB2 raw checkpointが生成されませんでした: $resumed_ckpt"
grep -F "resuming from $initial_ckpt at superbatch 2" "$DRILL_ROOT/resumed.log" >/dev/null \
  || fail "resume開始SBをlogで確認できませんでした"

python3 - "$DRILL_ROOT/resumed/experiments" <<'PY'
import json, pathlib, sys
paths = list(pathlib.Path(sys.argv[1]).glob("*.json"))
if len(paths) != 1:
    raise SystemExit(f"ERROR: expected one resumed experiment, got {len(paths)}")
with paths[0].open(encoding="utf-8") as stream:
    doc = json.load(stream)
lineage = doc.get("lineage") or {}
if lineage.get("resumed_from_superbatch") != 1:
    raise SystemExit(f"ERROR: resumed_from_superbatch != 1: {lineage}")
history = doc.get("history") or []
if len(history) != 1 or history[0].get("superbatch") != 2:
    raise SystemExit(f"ERROR: resumed history is not exactly SB2: {history}")
if doc.get("status") != "complete":
    raise SystemExit(f"ERROR: resumed experiment is not complete: {doc.get('status')}")
PY

{
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'precision=%s\n' "$precision"
  printf 'threads=%s\n' "$threads"
  printf 'initial_checkpoint=%s\n' "$initial_ckpt"
  printf 'initial_checkpoint_sha256=%s\n' "$(sha256_file "$initial_ckpt")"
  printf 'resumed_checkpoint=%s\n' "$resumed_ckpt"
  printf 'resumed_checkpoint_sha256=%s\n' "$(sha256_file "$resumed_ckpt")"
  printf 'resumed_from_superbatch=1\n'
  printf 'resumed_history_first_superbatch=2\n'
} | write_manifest_atomic "$DRILL_ROOT/manifest.txt"
date -u +%FT%TZ >"$GATE_DIR/resume.done"
echo "[resume-drill] optimizer stateを含むraw checkpointのSB1→SB2 resumeを確認しました"
