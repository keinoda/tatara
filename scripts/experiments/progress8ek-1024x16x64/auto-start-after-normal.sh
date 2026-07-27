#!/usr/bin/env bash

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

: "${AUTO_STATE_ROOT:?AUTO_STATE_ROOT is required}"
: "${SAMPLE_FILTER_ID:?SAMPLE_FILTER_ID is required}"
: "${FULL_FILTER_ID:?FULL_FILTER_ID is required}"
: "${AUTO_RUN_NAME:?AUTO_RUN_NAME is required}"

validate_id "$SAMPLE_FILTER_ID"
validate_id "$FULL_FILTER_ID"
validate_id "$AUTO_RUN_NAME"
validate_training_overrides

mkdir -p "$AUTO_STATE_ROOT"
readonly AUTO_LOG="$AUTO_STATE_ROOT/auto-start.log"
[[ ! -e "$AUTO_STATE_ROOT/done" && ! -e "$AUTO_STATE_ROOT/failed" ]] \
  || fail "自動実行は既に終了しています: $AUTO_STATE_ROOT"
exec > >(tee -a "$AUTO_LOG") 2>&1

on_error() {
  local rc="$1" line="$2"
  {
    printf 'exit_code=%s\n' "$rc"
    printf 'line=%s\n' "$line"
    printf 'ended_at=%s\n' "$(date -u +%FT%TZ)"
  } >"$AUTO_STATE_ROOT/failed"
}
trap 'on_error "$?" "$LINENO"' ERR
trap 'rc=$?; if [[ "$rc" != 0 && ! -e "$AUTO_STATE_ROOT/failed" ]]; then on_error "$rc" "exit"; fi' EXIT

wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    sleep 60
  done
}

wait_for_filter() {
  local id="$1" root="$EXTRACTION_ROOT/$id" rc
  wait_for_file "$root/state/exit-code"
  rc=$(tr -d '[:space:]' <"$root/state/exit-code")
  [[ "$rc" == 0 ]] || fail "抽出が失敗しました: id=$id rc=$rc"
  [[ -f "$root/state/finalized-at" && -f "$root/manifest.txt" ]] \
    || fail "抽出の完了manifestがありません: $root"
}

date -u +%FT%TZ >"$AUTO_STATE_ROOT/started-at"
echo "[auto] 通常学習の正常完了を待機します: $LEGACY_RUN_ROOT"
wait_for_file "$LEGACY_RUN_ROOT/state/trainer.exit-code"
require_legacy_training_complete
require_legacy_training_data
[[ -n "$BASE_NETWORK" && -f "$BASE_NETWORK" ]] \
  || fail "指定した基準networkがありません: $BASE_NETWORK"

echo "[auto] CUDA gateを実行します"
bash "$PROGRESS8EK_SCRIPT_DIR/run-cuda-gate.sh"

echo "[auto] 400万局面の抽出smokeを実行します"
FILTER_ID="$SAMPLE_FILTER_ID" \
FILTER_MAX_RECORDS=4000000 \
  bash "$PROGRESS8EK_SCRIPT_DIR/start-filter.sh"
wait_for_filter "$SAMPLE_FILTER_ID"

echo "[auto] 全教師から相入玉局面を抽出します"
FILTER_ID="$FULL_FILTER_ID" \
  bash "$PROGRESS8EK_SCRIPT_DIR/start-filter.sh"
wait_for_filter "$FULL_FILTER_ID"

echo "[auto] 実教師preflightを実行します"
EXTRACTION_ID="$FULL_FILTER_ID" \
  bash "$PROGRESS8EK_SCRIPT_DIR/run-preflight-smoke.sh"

echo "[auto] 追加学習を開始します"
EXTRACTION_ID="$FULL_FILTER_ID" \
RUN_NAME="$AUTO_RUN_NAME" \
  bash "$PROGRESS8EK_SCRIPT_DIR/start-training.sh"

readonly TRAIN_STATE="$RUNS_ROOT/$AUTO_RUN_NAME/state"
[[ -f "$TRAIN_STATE/training.started" ]] \
  || fail "追加学習の開始markerがありません: $TRAIN_STATE"
tmux has-session -t "train-$AUTO_RUN_NAME" 2>/dev/null \
  || fail "追加学習のtmuxがありません: train-$AUTO_RUN_NAME"
date -u +%FT%TZ >"$AUTO_STATE_ROOT/done"
echo "[auto] 追加学習を開始しました: $AUTO_RUN_NAME"
