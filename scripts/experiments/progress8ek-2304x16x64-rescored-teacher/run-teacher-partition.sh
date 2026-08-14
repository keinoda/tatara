#!/usr/bin/env bash
# 固定済みWCSC36教師を通常局面と相入玉模様局面の2ファイルへ全件分割する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

for command_name in awk df mkdir python3 sha256sum stat; do
  require_command "$command_name"
done
[[ -x "$PARTITION_BIN" ]] || fail "progress8ek-partitionがありません: $PARTITION_BIN"
[[ -f "$SOURCE_VALIDATION_MARKER" ]] \
  || fail "教師shardの全SHA-256検証markerがありません: $SOURCE_VALIDATION_MARKER"
[[ ! -e "$TRAINING_DATA_DIR" ]] \
  || fail "既存の教師出力を上書きしません: $TRAINING_DATA_DIR"
[[ ! -e "$PARTITION_DATA_MANIFEST" ]] \
  || fail "既存の教師分割manifestを上書きしません: $PARTITION_DATA_MANIFEST"

data_args=()
source_bytes=0
source_records=0
source_files=0
while IFS=$'\t' read -r repository revision filename expected_bytes expected_sha score_state; do
  [[ "$repository" == "penguinkumimanu/Knowledge_distilled_dataset_by_ponkotsuWCSC36" ]] \
    || fail "source manifestのrepositoryが想定外です: $repository"
  [[ "$revision" == "526bd42a59cdd961ef0c42e6068499625811ffee" ]] \
    || fail "source manifestのrevisionが想定外です: $revision"
  [[ "$score_state" == "rescored" ]] || fail "source manifestのscore stateが想定外です: $score_state"
  [[ "$expected_sha" =~ ^[0-9a-f]{64}$ ]] || fail "source manifestのSHA-256が不正です: $filename"
  path="$SOURCE_DATASET_DIR/$filename"
  [[ -f "$path" ]] || fail "教師shardがありません: $path"
  actual_bytes=$(file_size "$path")
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || fail "教師shardのsizeがmanifestと異なります: path=$path actual=$actual_bytes expected=$expected_bytes"
  (( expected_bytes % PSV_RECORD_BYTES == 0 )) \
    || fail "教師shardが40-byte PSV境界に揃っていません: $path"
  data_args+=(--data "$path")
  source_bytes=$((source_bytes + expected_bytes))
  source_records=$((source_records + expected_bytes / PSV_RECORD_BYTES))
  source_files=$((source_files + 1))
done <"$SOURCE_SHARDS_MANIFEST"

(( source_files == 30 )) || fail "source shard数が不一致です: actual=$source_files expected=30"
(( source_bytes == SOURCE_EXPECTED_BYTES )) \
  || fail "source合計byte数が不一致です: actual=$source_bytes expected=$SOURCE_EXPECTED_BYTES"
(( source_records == SOURCE_EXPECTED_RECORDS )) \
  || fail "source合計局面数が不一致です: actual=$source_records expected=$SOURCE_EXPECTED_RECORDS"

available_bytes=$(df -PB1 "$EXPERIMENT_ROOT" | awk 'NR == 2 {print $4}')
required_bytes=$((SOURCE_EXPECTED_BYTES * 2))
[[ "$available_bytes" =~ ^[0-9]+$ ]] || fail "出力先filesystemの空き容量を取得できませんでした"
(( available_bytes >= required_bytes )) \
  || fail "並列partと最終2ファイルに必要な空き容量がありません: available=$available_bytes required=$required_bytes"

mkdir -p "$(dirname "$TRAINING_DATA_DIR")" "$MANIFEST_DIR"
printf '[teacher-partition] source_files=%s source_bytes=%s source_records=%s available_bytes=%s\n' \
  "$source_files" "$source_bytes" "$source_records" "$available_bytes"
printf '[teacher-partition] command:'
printf ' %q' "$PARTITION_BIN" "${data_args[@]}" --output-dir "$TRAINING_DATA_DIR" --threads 16
printf '\n'
"$PARTITION_BIN" "${data_args[@]}" --output-dir "$TRAINING_DATA_DIR" --threads 16

[[ -f "$PARTITION_ORDINARY_PSV" && -f "$ENTERING_KING_PSV" && -f "$PARTITION_METRICS" ]] \
  || fail "分割後の2ファイルまたはmetricsが不足しています"
read -r scanned ordinary_records entering_king_records verified_ordinary verified_entering_king < <(
  python3 - "$PARTITION_METRICS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    metrics = json.load(stream)
print(
    metrics["scanned_records"],
    metrics["ordinary"]["records"],
    metrics["entering_king"]["records"],
    metrics["ordinary"]["verified_records"],
    metrics["entering_king"]["verified_records"],
)
PY
)
(( scanned == SOURCE_EXPECTED_RECORDS )) || fail "分割走査局面数が不一致です: $scanned"
(( ordinary_records + entering_king_records == SOURCE_EXPECTED_RECORDS )) \
  || fail "通常局面と相入玉局面の合計がsourceと一致しません"
(( verified_ordinary == ordinary_records && verified_entering_king == entering_king_records )) \
  || fail "出力PSVの全件述語検査が完了していません"

ordinary_bytes=$(file_size "$PARTITION_ORDINARY_PSV")
entering_king_bytes=$(file_size "$ENTERING_KING_PSV")
(( ordinary_bytes == ordinary_records * PSV_RECORD_BYTES )) || fail "ordinary.psvのsizeが不一致です"
(( entering_king_bytes == entering_king_records * PSV_RECORD_BYTES )) \
  || fail "entering-king.psvのsizeが不一致です"
ordinary_sha=$(sha256_file "$PARTITION_ORDINARY_PSV")
entering_king_sha=$(sha256_file "$ENTERING_KING_PSV")
metrics_sha=$(sha256_file "$PARTITION_METRICS")

{
  printf 'source_manifest=%s\n' "$SOURCE_SHARDS_MANIFEST"
  printf 'source_manifest_sha256=%s\n' "$(sha256_file "$SOURCE_SHARDS_MANIFEST")"
  printf 'source_validation_marker=%s\n' "$SOURCE_VALIDATION_MARKER"
  printf 'source_files=%s\n' "$source_files"
  printf 'source_bytes=%s\n' "$source_bytes"
  printf 'source_records=%s\n' "$source_records"
  printf 'predicate=black_rank_one_based<=5_and_white_rank_one_based>=5\n'
  printf 'ordinary_psv=%s\n' "$PARTITION_ORDINARY_PSV"
  printf 'ordinary_records=%s\n' "$ordinary_records"
  printf 'ordinary_bytes=%s\n' "$ordinary_bytes"
  printf 'ordinary_sha256=%s\n' "$ordinary_sha"
  printf 'entering_king_psv=%s\n' "$ENTERING_KING_PSV"
  printf 'entering_king_records=%s\n' "$entering_king_records"
  printf 'entering_king_bytes=%s\n' "$entering_king_bytes"
  printf 'entering_king_sha256=%s\n' "$entering_king_sha"
  printf 'partition_metrics=%s\n' "$PARTITION_METRICS"
  printf 'partition_metrics_sha256=%s\n' "$metrics_sha"
  printf 'predicate_verification=all_records\n'
  printf 'input_order_preserved=true\n'
} | write_manifest_atomic "$PARTITION_DATA_MANIFEST"

echo "[teacher-partition] 2ファイルへの排他的全件分割とSHA-256記録が完了しました"
