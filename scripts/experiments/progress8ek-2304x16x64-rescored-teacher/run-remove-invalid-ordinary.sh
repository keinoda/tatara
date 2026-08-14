#!/usr/bin/env bash
# 全件監査で特定した全ゼロ区間だけを除き、affine調整と通常学習用PSVを作る。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"
cd "$EXPERIMENT_ROOT"

for command_name in awk cmp dd df mkdir python3 sha256sum stat; do
  require_command "$command_name"
done
[[ -f "$PARTITION_DATA_MANIFEST" ]] || fail "教師分割manifestがありません: $PARTITION_DATA_MANIFEST"
[[ -f "$PARTITION_ORDINARY_PSV" ]] || fail "分割済みordinary.psvがありません: $PARTITION_ORDINARY_PSV"
[[ -f "$ENTERING_KING_PSV" ]] || fail "相入玉PSVがありません: $ENTERING_KING_PSV"
[[ -f "$ORDINARY_ACTIVE_AUDIT" ]] || fail "ordinary全件監査がありません: $ORDINARY_ACTIVE_AUDIT"
[[ -f "$ENTERING_ACTIVE_AUDIT" ]] || fail "相入玉全件監査がありません: $ENTERING_ACTIVE_AUDIT"
[[ ! -e "$ORDINARY_PSV" ]] || fail "既存の適格ordinary PSVを上書きしません: $ORDINARY_PSV"
[[ ! -e "$PREPARED_DATA_MANIFEST" ]] \
  || fail "既存の適格教師manifestを上書きしません: $PREPARED_DATA_MANIFEST"

raw_records=$(manifest_value "$PARTITION_DATA_MANIFEST" ordinary_records)
raw_bytes=$(manifest_value "$PARTITION_DATA_MANIFEST" ordinary_bytes)
raw_sha=$(manifest_value "$PARTITION_DATA_MANIFEST" ordinary_sha256)
entering_records=$(manifest_value "$PARTITION_DATA_MANIFEST" entering_king_records)
entering_bytes=$(manifest_value "$PARTITION_DATA_MANIFEST" entering_king_bytes)
entering_sha=$(manifest_value "$PARTITION_DATA_MANIFEST" entering_king_sha256)
[[ "$(file_size "$PARTITION_ORDINARY_PSV")" == "$raw_bytes" ]] \
  || fail "ordinary.psvのsizeが分割manifestと異なります"
[[ "$(sha256_file "$PARTITION_ORDINARY_PSV")" == "$raw_sha" ]] \
  || fail "ordinary.psvのSHA-256が分割manifestと異なります"
[[ "$(file_size "$ENTERING_KING_PSV")" == "$entering_bytes" ]] \
  || fail "entering-king.psvのsizeが分割manifestと異なります"
[[ "$(sha256_file "$ENTERING_KING_PSV")" == "$entering_sha" ]] \
  || fail "entering-king.psvのSHA-256が分割manifestと異なります"

read -r first_invalid end_invalid removed_records eligible_records < <(
  python3 - \
    "$ORDINARY_ACTIVE_AUDIT" "$ENTERING_ACTIVE_AUDIT" \
    "$PARTITION_ORDINARY_PSV" "$ENTERING_KING_PSV" \
    "$raw_records" "$entering_records" <<'PY'
import json
import os
import sys

ordinary_path, entering_path = sys.argv[3], sys.argv[4]
raw_records, entering_records = int(sys.argv[5]), int(sys.argv[6])
with open(sys.argv[1], encoding="utf-8") as stream:
    ordinary = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    entering = json.load(stream)

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)

require(ordinary["exhaustive"] is True, "ordinary audit is not exhaustive")
require(ordinary["expected_active_indices"] == 76, "ordinary audit active count is not 76")
require(ordinary["records"] == raw_records, "ordinary audit record count mismatch")
require(ordinary["scanned_records"] == raw_records, "ordinary audit scan count mismatch")
require(os.path.realpath(ordinary["data"]) == os.path.realpath(ordinary_path), "ordinary audit path mismatch")
removed = ordinary["ineligible_records"]
require(removed > 0, "ordinary audit has no rejected records")
require(removed == ordinary["zero_records"], "ordinary rejected records are not all zero records")
require(
    ordinary["active_index_histogram"]
    == {"76": ordinary["eligible_records"], "108": removed},
    "ordinary active histogram contains an unexpected class",
)
first = ordinary["first_ineligible_index"]
last = ordinary["last_ineligible_index"]
require(first is not None and last is not None, "ordinary rejected range is missing")
require(last - first + 1 == removed, "ordinary rejected records are not one contiguous range")
require(ordinary["eligible_records"] + removed == raw_records, "ordinary audit is not exhaustive")

require(entering["exhaustive"] is True, "entering audit is not exhaustive")
require(entering["expected_active_indices"] == 76, "entering audit active count is not 76")
require(entering["records"] == entering_records, "entering audit record count mismatch")
require(entering["scanned_records"] == entering_records, "entering audit scan count mismatch")
require(os.path.realpath(entering["data"]) == os.path.realpath(entering_path), "entering audit path mismatch")
require(entering["ineligible_records"] == 0, "entering audit contains rejected records")
require(entering["zero_records"] == 0, "entering audit contains zero records")
require(
    entering["active_index_histogram"] == {"76": entering_records},
    "entering active histogram contains an unexpected class",
)

print(first, last + 1, removed, ordinary["eligible_records"])
PY
)

first_invalid_byte=$((first_invalid * PSV_RECORD_BYTES))
end_invalid_byte=$((end_invalid * PSV_RECORD_BYTES))
removed_bytes=$((removed_records * PSV_RECORD_BYTES))
eligible_bytes=$((eligible_records * PSV_RECORD_BYTES))
(( raw_bytes - removed_bytes == eligible_bytes )) \
  || fail "除外後ordinaryの計算sizeが一致しません"
available_bytes=$(df -PB1 "$EXPERIMENT_ROOT" | awk 'NR == 2 {print $4}')
[[ "$available_bytes" =~ ^[0-9]+$ ]] || fail "出力先filesystemの空き容量を取得できませんでした"
(( available_bytes >= eligible_bytes )) \
  || fail "適格ordinary PSVを作る空き容量がありません: available=$available_bytes required=$eligible_bytes"

mkdir -p "$(dirname "$ORDINARY_PSV")" "$MANIFEST_DIR"
printf '[remove-invalid-ordinary] source=%s records=%s remove=[%s,%s) removed=%s\n' \
  "$PARTITION_ORDINARY_PSV" "$raw_records" "$first_invalid" "$end_invalid" "$removed_records"
dd \
  if="$PARTITION_ORDINARY_PSV" \
  of="$ORDINARY_PSV" \
  bs=64M \
  count="$first_invalid_byte" \
  iflag=count_bytes \
  status=progress
dd \
  if="$PARTITION_ORDINARY_PSV" \
  of="$ORDINARY_PSV" \
  bs=64M \
  skip="$end_invalid_byte" \
  iflag=skip_bytes \
  oflag=append \
  conv=notrunc,fsync \
  status=progress

[[ "$(file_size "$ORDINARY_PSV")" == "$eligible_bytes" ]] \
  || fail "適格ordinary PSVのsizeが不一致です"
cmp --silent --bytes="$first_invalid_byte" "$PARTITION_ORDINARY_PSV" "$ORDINARY_PSV" \
  || fail "適格ordinary PSVの前半が入力と一致しません"
cmp --silent --ignore-initial="$end_invalid_byte:$first_invalid_byte" \
  "$PARTITION_ORDINARY_PSV" "$ORDINARY_PSV" \
  || fail "適格ordinary PSVの後半が入力と一致しません"

ordinary_sha=$(sha256_file "$ORDINARY_PSV")
ordinary_audit_sha=$(sha256_file "$ORDINARY_ACTIVE_AUDIT")
entering_audit_sha=$(sha256_file "$ENTERING_ACTIVE_AUDIT")
{
  printf 'partition_manifest=%s\n' "$PARTITION_DATA_MANIFEST"
  printf 'partition_manifest_sha256=%s\n' "$(sha256_file "$PARTITION_DATA_MANIFEST")"
  printf 'ordinary_source_psv=%s\n' "$PARTITION_ORDINARY_PSV"
  printf 'ordinary_source_sha256=%s\n' "$raw_sha"
  printf 'ordinary_removed_first_index=%s\n' "$first_invalid"
  printf 'ordinary_removed_end_exclusive=%s\n' "$end_invalid"
  printf 'ordinary_removed_records=%s\n' "$removed_records"
  printf 'ordinary_removed_reason=all_zero_and_progress_active_108\n'
  printf 'ordinary_psv=%s\n' "$ORDINARY_PSV"
  printf 'ordinary_records=%s\n' "$eligible_records"
  printf 'ordinary_bytes=%s\n' "$eligible_bytes"
  printf 'ordinary_sha256=%s\n' "$ordinary_sha"
  printf 'entering_king_psv=%s\n' "$ENTERING_KING_PSV"
  printf 'entering_king_records=%s\n' "$entering_records"
  printf 'entering_king_bytes=%s\n' "$entering_bytes"
  printf 'entering_king_sha256=%s\n' "$entering_sha"
  printf 'ordinary_active_audit=%s\n' "$ORDINARY_ACTIVE_AUDIT"
  printf 'ordinary_active_audit_sha256=%s\n' "$ordinary_audit_sha"
  printf 'entering_active_audit=%s\n' "$ENTERING_ACTIVE_AUDIT"
  printf 'entering_active_audit_sha256=%s\n' "$entering_audit_sha"
  printf 'progress_active_indices=76\n'
  printf 'copy_verification=full_prefix_and_suffix_cmp\n'
  printf 'input_order_preserved=true\n'
} | write_manifest_atomic "$PREPARED_DATA_MANIFEST"

echo "[remove-invalid-ordinary] affine調整と通常学習に使う適格ordinary PSVを固定しました"
