#!/usr/bin/env bash
# smoke結果を見たユーザーがprecisionとthread数を明示承認する入口。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
[[ -n "${TRAIN_PRECISION:-}" ]] || fail "TRAIN_PRECISION=fp32またはall-optimを明示してください"
[[ "$TRAIN_PRECISION" == "fp32" || "$TRAIN_PRECISION" == "all-optim" ]] \
  || fail "TRAIN_PRECISIONが不正です: $TRAIN_PRECISION"
[[ -n "${TRAIN_THREADS:-}" && "$TRAIN_THREADS" =~ ^[1-9][0-9]*$ ]] \
  || fail "TRAIN_THREADSを1以上の整数で明示してください"
[[ -n "${APPROVAL_NOTE:-}" ]] || fail "APPROVAL_NOTEに採用理由を明示してください"

readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
require_gate "$GATE_DIR" smoke
readonly REPORT="$GATE_DIR/smoke/report.json"
python3 - "$REPORT" "$TRAIN_PRECISION" "$TRAIN_THREADS" <<'PY'
import json, sys
path, precision, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, encoding="utf-8") as stream:
    report = json.load(stream)
matches = [run for run in report["runs"] if run["precision"] == precision and run["threads"] == threads]
if len(matches) != 1:
    raise SystemExit(f"ERROR: smoke result is missing for precision={precision} threads={threads}")
run = matches[0]
if run["status"] != "complete":
    raise SystemExit(f"ERROR: selected smoke run did not complete: {run}")
for key in ("mean_pos_per_sec", "loss", "test_loss"):
    value = run[key]
    if value is None or not isinstance(value, (int, float)):
        raise SystemExit(f"ERROR: selected smoke metric is missing: {key}={value}")
PY

{
  printf 'approved_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'precision=%s\n' "$TRAIN_PRECISION"
  printf 'threads=%s\n' "$TRAIN_THREADS"
  printf 'smoke_report=%s\n' "$REPORT"
  printf 'smoke_report_sha256=%s\n' "$(sha256_file "$REPORT")"
  printf 'approval_note=%s\n' "$APPROVAL_NOTE"
} | write_manifest_atomic "$GATE_DIR/precision.approved.txt"
date -u +%FT%TZ >"$GATE_DIR/precision.done"
echo "[approve-smoke] precision=$TRAIN_PRECISION threads=$TRAIN_THREADSを承認しました"
