#!/usr/bin/env bash
# 非相入玉教師だけを使い、基準progress.binのa,bを前回と同じ方法で再調整する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

for command_name in mkdir python3 sha256sum stat; do
  require_command "$command_name"
done
[[ -x "$PROGRESS_SURVEY" ]] || fail "progress-bucket-surveyがありません: $PROGRESS_SURVEY"
[[ -f "$MANIFEST_DIR/prepared-data.txt" ]] || fail "教師分割manifestがありません"
[[ -f "$ORDINARY_PSV" ]] || fail "非相入玉教師がありません: $ORDINARY_PSV"
[[ -f "$BASELINE_PROGRESS" ]] || fail "基準progress.binがありません: $BASELINE_PROGRESS"
[[ "$(file_size "$BASELINE_PROGRESS")" == "$PROGRESS_EXPECTED_BYTES" ]] \
  || fail "基準progress.binのsizeが不正です"
actual_baseline_sha=$(sha256_file "$BASELINE_PROGRESS")
[[ "$actual_baseline_sha" == "$BASELINE_PROGRESS_SHA256" ]] \
  || fail "基準progress.binのSHA-256が不一致です: actual=$actual_baseline_sha expected=$BASELINE_PROGRESS_SHA256"

readonly SURVEY_ID="wcsc36-ordinary-affine"
readonly SURVEY_DIR="$EXPERIMENT_ROOT/survey/$SURVEY_ID"
readonly OPTIMIZED_CANDIDATE="optimized-wcsc36-ordinary-center"
readonly TARGET_PERCENTAGES="11,12,13,14,14,13,12,11"
readonly SURVEY_SEED=20260726
[[ ! -e "$SURVEY_DIR" ]] || fail "既存surveyを上書きしません: $SURVEY_DIR"

expected_ordinary_bytes=$(manifest_value "$MANIFEST_DIR/prepared-data.txt" ordinary_bytes)
expected_ordinary_sha=$(manifest_value "$MANIFEST_DIR/prepared-data.txt" ordinary_sha256)
[[ "$(file_size "$ORDINARY_PSV")" == "$expected_ordinary_bytes" ]] \
  || fail "ordinary.psvのsizeが分割manifestと異なります"
[[ "$(sha256_file "$ORDINARY_PSV")" == "$expected_ordinary_sha" ]] \
  || fail "ordinary.psvのSHA-256が分割manifestと異なります"
mkdir -p "$(dirname "$SURVEY_DIR")"

command=(
  "$PROGRESS_SURVEY"
  --data "$ORDINARY_PSV"
  --progress "$BASELINE_PROGRESS"
  --output-dir "$SURVEY_DIR"
  --seed "$SURVEY_SEED"
  --num-buckets 8
  --split calibration:2000000
  --split selection:1000000
  --split final-test:1000000
  --candidate previous-selected:1.2980837735881936:-0.5975424282106219
  --optimize-affine
  --optimize-split calibration
  --optimized-candidate-name "$OPTIMIZED_CANDIDATE"
  --optimizer-target-percentages "$TARGET_PERCENTAGES"
  --optimizer-grid-points 257
  --optimizer-refinements 6
)
printf '[progress-affine-survey] command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"

readonly METRICS="$SURVEY_DIR/metrics.json"
readonly CANDIDATE_PROGRESS="$SURVEY_DIR/progress-$OPTIMIZED_CANDIDATE.bin"
[[ -f "$METRICS" && -f "$CANDIDATE_PROGRESS" ]] \
  || fail "affine surveyのmetricsまたは候補progress.binが生成されませんでした"
[[ "$(file_size "$CANDIDATE_PROGRESS")" == "$PROGRESS_EXPECTED_BYTES" ]] \
  || fail "候補progress.binのsizeが不正です"
read -r optimized_a optimized_b < <(
  python3 - "$METRICS" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    metrics = json.load(stream)
optimization = metrics["affine_optimization"]
print(repr(optimization["a"]), repr(optimization["b"]))
PY
)

{
  printf 'survey_id=%s\n' "$SURVEY_ID"
  printf 'ordinary_psv=%s\n' "$ORDINARY_PSV"
  printf 'ordinary_sha256=%s\n' "$expected_ordinary_sha"
  printf 'baseline_progress=%s\n' "$BASELINE_PROGRESS"
  printf 'baseline_progress_sha256=%s\n' "$actual_baseline_sha"
  printf 'sample_seed=%s\n' "$SURVEY_SEED"
  printf 'calibration_samples=2000000\n'
  printf 'selection_samples=1000000\n'
  printf 'final_test_samples=1000000\n'
  printf 'target_percentages=%s\n' "$TARGET_PERCENTAGES"
  printf 'transform=affine\n'
  printf 'a=%s\n' "$optimized_a"
  printf 'b=%s\n' "$optimized_b"
  printf 'candidate_progress=%s\n' "$CANDIDATE_PROGRESS"
  printf 'candidate_progress_sha256=%s\n' "$(sha256_file "$CANDIDATE_PROGRESS")"
  printf 'metrics=%s\n' "$METRICS"
  printf 'metrics_sha256=%s\n' "$(sha256_file "$METRICS")"
  printf 'automatic_adoption=false\n'
} | write_manifest_atomic "$SURVEY_DIR/manifest.txt"

echo "[progress-affine-survey] a,b候補を生成しました。metrics.jsonの3 splitを確認してください"
