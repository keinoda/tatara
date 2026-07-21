#!/usr/bin/env bash
# 完成した公開教師shardから、明示候補だけを比較する再現可能surveyを実行する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ -n "${SURVEY_ID:-}" ]] || fail "SURVEY_IDを明示してください"
validate_run_name "$SURVEY_ID"
[[ -n "${SURVEY_SEED:-}" && "$SURVEY_SEED" =~ ^[0-9]+$ ]] \
  || fail "SURVEY_SEEDを0以上の整数で明示してください"
[[ -f "$STATE_DIR/prepare_data.done" ]] || fail "prepare_data.doneがありません"
[[ -x "$PROGRESS_SURVEY" ]] || fail "progress-bucket-surveyがありません: $PROGRESS_SURVEY"

readonly SURVEY_DIR="$SURVEY_ROOT/$SURVEY_ID"
[[ ! -e "$SURVEY_DIR" ]] || fail "既存surveyを上書きしません: $SURVEY_DIR"

readonly BASELINE_PROGRESS="$EXPERIMENT_ROOT/progress/baseline/progress.bin"
require_exact_size "$BASELINE_PROGRESS" "$PROGRESS_EXPECTED_BYTES" "baseline progress.bin"

shopt -s nullglob
shards=("$TRAIN_SHARD_DIR"/dlsuisho_unique_*.bin)
(( ${#shards[@]} == 30 )) || fail "公開教師shardは30個必要です: actual=${#shards[@]}"
data_arg=$(IFS=,; printf '%s' "${shards[*]}")

command=(
  "$PROGRESS_SURVEY"
  --data "$data_arg"
  --progress "$BASELINE_PROGRESS"
  --output-dir "$SURVEY_DIR"
  --seed "$SURVEY_SEED"
  --num-buckets 8
)

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

# 例: AFFINE_CANDIDATES='wide-a:0.85:-0.20 wide-b:1.00:-0.25'
# 未指定ならbaseline分布だけを取得する。候補範囲は自動決定しない。
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
{
  printf 'survey_id=%s\n' "$SURVEY_ID"
  printf 'created_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'seed=%s\n' "$SURVEY_SEED"
  printf 'metrics=%s\n' "$SURVEY_DIR/metrics.json"
  printf 'metrics_sha256=%s\n' "$(sha256_file "$SURVEY_DIR/metrics.json")"
  printf 'sample_plan_sha256=%s\n' "$(sha256_file "$SURVEY_DIR/sample-plan.bin")"
  printf 'automatic_adoption=false\n'
} | write_manifest_atomic "$SURVEY_DIR/manifest.txt"

echo "[survey] 完了しました。metrics.jsonを提示してからapprove-progress.shを実行してください"
