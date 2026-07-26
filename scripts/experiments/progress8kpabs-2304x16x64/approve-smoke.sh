#!/usr/bin/env bash
# 互換用: 確定済みall-optim / 16 threadsのsmoke gateをread-only確認する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
require_gate "$GATE_DIR" smoke
mode=$(precision_from_gate "$GATE_DIR")
threads=$(manifest_value "$GATE_DIR/precision.approved.txt" threads)
echo "[approve-smoke] 追加承認は不要です: precision=$mode threads=$threads"
