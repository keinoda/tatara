#!/usr/bin/env bash
# 完走した抽出結果を再検査し、SHA-256付きmanifestを作る。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ $# == 1 ]] || fail "usage: finalize-filter.sh FILTER_ROOT"
readonly FILTER_ROOT="$1"
readonly OUTPUT="$FILTER_ROOT/output"
readonly CONFIG="$FILTER_ROOT/config/manifest.txt"
readonly METRICS="$OUTPUT/metrics.json"
readonly TRAIN="$OUTPUT/entering-king-train.psv"
readonly HOLDOUT="$OUTPUT/entering-king-holdout.psv"
for path in "$CONFIG" "$METRICS" "$TRAIN" "$HOLDOUT"; do
  [[ -f "$path" ]] || fail "抽出成果物がありません: $path"
done

readarray -t values < <(python3 - "$METRICS" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
print(d["scanned_records"])
print(d["matched_records"])
print(d["train"]["records"])
print(d["holdout"]["records"])
print(d["train"]["verified_entering_king_records"])
print(d["holdout"]["verified_entering_king_records"])
PY
)
[[ ${#values[@]} == 6 ]] || fail "metrics.jsonの必須値を読めません"
[[ "${values[2]}" == "${values[4]}" && "${values[3]}" == "${values[5]}" ]] \
  || fail "出力PSVの相入玉再検証数が一致しません"
[[ "$(file_size "$TRAIN")" == "$((values[2] * PSV_RECORD_BYTES))" ]] \
  || fail "train PSVのsizeがrecord数と一致しません"
[[ "$(file_size "$HOLDOUT")" == "$((values[3] * PSV_RECORD_BYTES))" ]] \
  || fail "holdout PSVのsizeがrecord数と一致しません"

{
  printf 'filter_id=%s\n' "$(manifest_value "$CONFIG" filter_id)"
  printf 'tatara_commit=%s\n' "$(manifest_value "$CONFIG" tatara_commit)"
  printf 'scanned_records=%s\n' "${values[0]}"
  printf 'matched_records=%s\n' "${values[1]}"
  printf 'train_records=%s\n' "${values[2]}"
  printf 'holdout_records=%s\n' "${values[3]}"
  printf 'predicate=black_rank_le_5_and_white_rank_ge_5_inclusive\n'
  printf 'split_seed=%s\n' "$(manifest_value "$CONFIG" seed)"
  printf 'holdout_per_mille=%s\n' "$(manifest_value "$CONFIG" holdout_per_mille)"
  printf 'train_psv=%s\n' "$TRAIN"
  printf 'train_sha256=%s\n' "$(sha256_file "$TRAIN")"
  printf 'holdout_psv=%s\n' "$HOLDOUT"
  printf 'holdout_sha256=%s\n' "$(sha256_file "$HOLDOUT")"
  printf 'metrics_json=%s\n' "$METRICS"
  printf 'metrics_sha256=%s\n' "$(sha256_file "$METRICS")"
  printf 'progress_bin=%s\n' "$(manifest_value "$CONFIG" progress_bin)"
  printf 'progress_sha256=%s\n' "$(manifest_value "$CONFIG" progress_sha256)"
  printf 'input_psv=%s\n' "$(manifest_value "$CONFIG" input_psv)"
  printf 'input_positions=%s\n' "$(manifest_value "$CONFIG" input_positions)"
  printf 'input_order_preserved=true\n'
  printf 'automatic_adoption=false\n'
} | write_manifest_atomic "$FILTER_ROOT/manifest.txt"
date -u +%FT%TZ >"$FILTER_ROOT/state/finalized-at"
echo "[filter] finalized: $FILTER_ROOT/manifest.txt"
