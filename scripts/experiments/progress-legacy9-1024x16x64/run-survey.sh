#!/usr/bin/env bash
# 取得完了済みの公開教師shardを一度だけ読み、明示目標に対する単調affine係数を最適化する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ -n "${SURVEY_ID:-}" ]] || fail "SURVEY_IDを明示してください"
validate_run_name "$SURVEY_ID"
[[ -n "${SURVEY_SEED:-}" && "$SURVEY_SEED" =~ ^[0-9]+$ ]] \
  || fail "SURVEY_SEEDを0以上の整数で明示してください"
[[ -x "$PROGRESS_SURVEY" ]] || fail "progress-bucket-surveyがありません: $PROGRESS_SURVEY"

readonly SURVEY_DIR="$SURVEY_ROOT/$SURVEY_ID"
[[ ! -e "$SURVEY_DIR" ]] || fail "既存surveyを上書きしません: $SURVEY_DIR"

readonly BASELINE_PROGRESS="$EXPERIMENT_ROOT/progress/baseline/progress.bin"
require_exact_size "$BASELINE_PROGRESS" "$PROGRESS_EXPECTED_BYTES" "baseline progress.bin"

shards=()
expected_shard_sizes=()
input_source_manifest=""
if [[ -n "${SURVEY_INPUT_MANIFEST:-}" ]]; then
  [[ -f "$SURVEY_INPUT_MANIFEST" ]] || fail "SURVEY_INPUT_MANIFESTがありません: $SURVEY_INPUT_MANIFEST"
  input_source_manifest=$(canonical_file "$SURVEY_INPUT_MANIFEST")
  while IFS= read -r line; do
    [[ "$line" == shard=*" bytes="*" records="* ]] || continue
    shard_path=${line#shard=}
    shard_path=${shard_path%% bytes=*}
    expected_bytes=${line#* bytes=}
    expected_bytes=${expected_bytes%% records=*}
    [[ "$expected_bytes" =~ ^[1-9][0-9]*$ ]] \
      || fail "SURVEY_INPUT_MANIFESTのshard sizeが不正です: $line"
    shards+=("$shard_path")
    expected_shard_sizes+=("$expected_bytes")
  done <"$input_source_manifest"
else
  shopt -s nullglob
  shards=("$TRAIN_SHARD_DIR"/dlsuisho_unique_*.bin)
fi
(( ${#shards[@]} >= 1 )) || fail "取得完了済みの公開教師shardがありません"
shard_sizes=()
total_shard_bytes=0
for index in "${!shards[@]}"; do
  shard=${shards[$index]}
  [[ -f "$shard" ]] || fail "survey入力shardがありません: $shard"
  shard_bytes=$(file_size "$shard")
  (( shard_bytes > 0 && shard_bytes % PSV_RECORD_BYTES == 0 )) \
    || fail "公開教師shardが40-byte PSV境界に揃っていません: path=$shard bytes=$shard_bytes"
  if [[ -n "$input_source_manifest" ]]; then
    [[ "$shard_bytes" == "${expected_shard_sizes[$index]}" ]] \
      || fail "survey入力shardのsizeがsnapshotと異なります: path=$shard actual=$shard_bytes expected=${expected_shard_sizes[$index]}"
  fi
  shard_sizes+=("$shard_bytes")
  total_shard_bytes=$((total_shard_bytes + shard_bytes))
done
total_shard_positions=$((total_shard_bytes / PSV_RECORD_BYTES))
(( total_shard_positions >= 4000000 )) \
  || fail "取得完了済みshardが400万局面に達していません: actual=$total_shard_positions"
data_arg=$(IFS=,; printf '%s' "${shards[*]}")

optimized_candidate_name="${OPTIMIZED_CANDIDATE_NAME:-optimized-uniform}"
validate_run_name "$optimized_candidate_name"
optimizer_target="uniform"
optimizer_objective="uniform-bucket-mse"
if [[ -n "${OPTIMIZER_TARGET_PERCENTAGES:-}" ]]; then
  [[ -n "${OPTIMIZED_CANDIDATE_NAME:-}" ]] \
    || fail "OPTIMIZER_TARGET_PERCENTAGES指定時はOPTIMIZED_CANDIDATE_NAMEも明示してください"
  optimizer_target="$OPTIMIZER_TARGET_PERCENTAGES"
  optimizer_objective="explicit-target-bucket-mse"
fi

command=(
  "$PROGRESS_SURVEY"
  --data "$data_arg"
  --progress "$BASELINE_PROGRESS"
  --output-dir "$SURVEY_DIR"
  --seed "$SURVEY_SEED"
  --num-buckets 8
  --optimize-affine
  --optimize-split calibration
  --optimized-candidate-name "$optimized_candidate_name"
  --optimizer-grid-points 257
  --optimizer-refinements 6
)
if [[ "$optimizer_target" != uniform ]]; then
  command+=(--optimizer-target-percentages "$optimizer_target")
fi

# 3集合の配分は未決定なので推測しない。合計400万を保ち、3値すべて明示する。
[[ -n "${CALIBRATION_SAMPLES:-}" ]] || fail "CALIBRATION_SAMPLESを明示してください"
[[ -n "${SELECTION_SAMPLES:-}" ]] || fail "SELECTION_SAMPLESを明示してください"
[[ -n "${FINAL_TEST_SAMPLES:-}" ]] || fail "FINAL_TEST_SAMPLESを明示してください"
readonly CALIBRATION_SAMPLES SELECTION_SAMPLES FINAL_TEST_SAMPLES
for count in "$CALIBRATION_SAMPLES" "$SELECTION_SAMPLES" "$FINAL_TEST_SAMPLES"; do
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || fail "survey split数は1以上の整数にしてください: $count"
done
(( CALIBRATION_SAMPLES + SELECTION_SAMPLES + FINAL_TEST_SAMPLES == 4000000 )) \
  || fail "survey 3集合の合計は4,000,000局面にしてください"
command+=(
  --split "calibration:$CALIBRATION_SAMPLES"
  --split "selection:$SELECTION_SAMPLES"
  --split "final-test:$FINAL_TEST_SAMPLES"
)

# 明示候補はoptimizerとは別の補助比較に限る。通常の係数決定には指定しない。
# 例: AFFINE_CANDIDATES='reference:1.00:-0.25'
if [[ -n "${AFFINE_CANDIDATES:-}" ]]; then
  read -r -a affine_candidates <<<"$AFFINE_CANDIDATES"
  for candidate in "${affine_candidates[@]}"; do
    command+=(--candidate "$candidate")
  done
fi

mkdir -p "$SURVEY_ROOT"
printf '[survey] command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

[[ -f "$SURVEY_DIR/metrics.json" ]] || fail "survey metricsが生成されませんでした"
input_shards_manifest="$SURVEY_DIR/input-shards.txt"
{
  printf 'dataset_revision=%s\n' "$TRAIN_DATASET_REVISION"
  printf 'completed_shards=%s\n' "${#shards[@]}"
  printf 'total_bytes=%s\n' "$total_shard_bytes"
  printf 'total_positions=%s\n' "$total_shard_positions"
  if [[ -n "$input_source_manifest" ]]; then
    printf 'source_manifest=%s\n' "$input_source_manifest"
    printf 'source_manifest_sha256=%s\n' "$(sha256_file "$input_source_manifest")"
  else
    printf 'source_manifest=completed-shards-at-start\n'
  fi
  for index in "${!shards[@]}"; do
    printf 'shard=%s bytes=%s records=%s\n' \
      "${shards[$index]}" "${shard_sizes[$index]}" "$((shard_sizes[index] / PSV_RECORD_BYTES))"
  done
} | write_manifest_atomic "$input_shards_manifest"
{
  printf 'survey_id=%s\n' "$SURVEY_ID"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'seed=%s\n' "$SURVEY_SEED"
  printf 'metrics=%s\n' "$SURVEY_DIR/metrics.json"
  printf 'metrics_sha256=%s\n' "$(sha256_file "$SURVEY_DIR/metrics.json")"
  printf 'sample_plan_sha256=%s\n' "$(sha256_file "$SURVEY_DIR/sample-plan.bin")"
  printf 'input_shards=%s\n' "$input_shards_manifest"
  printf 'input_shards_sha256=%s\n' "$(sha256_file "$input_shards_manifest")"
  printf 'affine_optimization=%s\n' "$optimizer_objective"
  printf 'optimization_split=calibration\n'
  printf 'optimizer_target_percentages=%s\n' "$optimizer_target"
  printf 'optimized_candidate=%s\n' "$optimized_candidate_name"
  printf 'optimizer_grid_points=257\n'
  printf 'optimizer_refinements=6\n'
  printf 'teacher_data_passes=1\n'
  printf 'automatic_adoption=false\n'
} | write_manifest_atomic "$SURVEY_DIR/manifest.txt"

echo "[survey] 最適化まで完了しました。metrics.jsonを提示してからapprove-progress.shを実行してください"
