#!/usr/bin/env bash
# 現学習完了後に相入玉教師の抽出を新しいtmuxで開始する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ -n "${FILTER_ID:-}" ]] || fail "FILTER_IDを明示してください"
validate_id "$FILTER_ID"
require_legacy_training_complete
require_legacy_training_data
require_clean_source
require_command tmux
[[ -x "$FILTER_BIN" ]] || fail "release build済みprogress8ek-filterがありません: $FILTER_BIN"

readonly FILTER_ROOT="$EXTRACTION_ROOT/$FILTER_ID"
readonly FILTER_OUTPUT="$FILTER_ROOT/output"
readonly FILTER_CONFIG="$FILTER_ROOT/config"
readonly FILTER_LOG="$FILTER_ROOT/logs/filter.log"
readonly FILTER_STATE="$FILTER_ROOT/state"
readonly FILTER_SESSION="filter-$FILTER_ID"
[[ ! -e "$FILTER_ROOT" ]] || fail "既存抽出runを上書きしません: $FILTER_ROOT"
tmux has-session -t "$FILTER_SESSION" 2>/dev/null && fail "tmux sessionが既にあります: $FILTER_SESSION"

progress_bin=$(legacy_progress_bin)
command=(
  "$FILTER_BIN"
  --data "$LEGACY_TRAIN_PSV"
  --progress "$progress_bin"
  --output-dir "$FILTER_OUTPUT"
  --seed "${FILTER_SEED:-20260722}"
  --holdout-per-mille "${HOLDOUT_PER_MILLE:-100}"
  --threads "${FILTER_THREADS:-16}"
)
if [[ -n "${FILTER_MAX_RECORDS:-}" ]]; then
  [[ "$FILTER_MAX_RECORDS" =~ ^[1-9][0-9]*$ ]] || fail "FILTER_MAX_RECORDSは1以上の整数にしてください"
  command+=(--max-records "$FILTER_MAX_RECORDS")
fi

mkdir -p "$FILTER_CONFIG" "$(dirname "$FILTER_LOG")" "$FILTER_STATE"
printf '#!/usr/bin/env bash\nset -Eeuo pipefail\nexec' >"$FILTER_CONFIG/command.sh"
printf ' %q' "${command[@]}" >>"$FILTER_CONFIG/command.sh"
printf '\n' >>"$FILTER_CONFIG/command.sh"
chmod 700 "$FILTER_CONFIG/command.sh"
{
  printf 'filter_id=%s\n' "$FILTER_ID"
  printf 'tatara_commit=%s\n' "$(git -C "$PROGRESS8EK_ROOT" rev-parse HEAD)"
  printf 'legacy_run_name=%s\n' "$LEGACY_RUN_NAME"
  printf 'input_psv=%s\n' "$LEGACY_TRAIN_PSV"
  printf 'input_positions=%s\n' "$LEGACY_TRAIN_POSITIONS"
  printf 'progress_bin=%s\n' "$progress_bin"
  printf 'progress_sha256=%s\n' "$(sha256_file "$progress_bin")"
  printf 'seed=%s\n' "${FILTER_SEED:-20260722}"
  printf 'holdout_per_mille=%s\n' "${HOLDOUT_PER_MILLE:-100}"
  printf 'threads=%s\n' "${FILTER_THREADS:-16}"
  printf 'max_records=%s\n' "${FILTER_MAX_RECORDS:-all}"
} | write_manifest_atomic "$FILTER_CONFIG/manifest.txt"

printf -v tmux_body \
  'set -uo pipefail; date -u +%%FT%%TZ >%q; set +e; bash %q 2>&1 | tee %q; rc=${PIPESTATUS[0]}; if [[ "$rc" == 0 ]]; then bash %q %q; rc=$?; fi; printf "%%s\n" "$rc" >%q; date -u +%%FT%%TZ >%q; exit "$rc"' \
  "$FILTER_STATE/started-at" "$FILTER_CONFIG/command.sh" "$FILTER_LOG" \
  "$PROGRESS8EK_SCRIPT_DIR/finalize-filter.sh" "$FILTER_ROOT" \
  "$FILTER_STATE/exit-code" "$FILTER_STATE/ended-at"
tmux new-session -d -s "$FILTER_SESSION" "bash -lc $(printf '%q' "$tmux_body")"
echo "[filter] tmux=$FILTER_SESSION root=$FILTER_ROOT"
