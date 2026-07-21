#!/usr/bin/env bash
# survey結果を確認したユーザーが、baselineまたは候補を明示承認する入口。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ -n "${SURVEY_ID:-}" ]] || fail "SURVEY_IDを明示してください"
validate_run_name "$SURVEY_ID"
[[ -n "${CANDIDATE_NAME:-}" ]] || fail "CANDIDATE_NAMEを明示してください（baselineも可）"
validate_run_name "$CANDIDATE_NAME"
[[ -n "${APPROVAL_NOTE:-}" ]] || fail "APPROVAL_NOTEに採否理由を明示してください"

readonly SURVEY_DIR="$SURVEY_ROOT/$SURVEY_ID"
readonly METRICS="$SURVEY_DIR/metrics.json"
[[ -f "$METRICS" ]] || fail "survey metricsがありません: $METRICS"

selection=$(python3 - "$METRICS" "$CANDIDATE_NAME" <<'PY'
import json, sys
metrics_path, name = sys.argv[1:]
with open(metrics_path, encoding="utf-8") as stream:
    report = json.load(stream)
for candidate in report["candidates"]:
    if candidate["name"] == name:
        path = candidate["generated_progress_bin"]
        if name == "baseline":
            path = "__BASELINE__"
        print(path)
        print(candidate["boundary_fixture_psv"])
        print("true" if candidate["boundary_fixture_complete"] else "false")
        break
else:
    raise SystemExit(f"ERROR: candidate '{name}' is not present in {metrics_path}")
PY
)
selected_path=$(printf '%s\n' "$selection" | sed -n '1p')
boundary_fixture=$(printf '%s\n' "$selection" | sed -n '2p')
boundary_complete=$(printf '%s\n' "$selection" | sed -n '3p')
if [[ "$selected_path" == "__BASELINE__" ]]; then
  selected_path="$EXPERIMENT_ROOT/progress/baseline/progress.bin"
fi
require_exact_size "$selected_path" "$PROGRESS_EXPECTED_BYTES" "選択progress.bin"
selected_path=$(canonical_file "$selected_path")
selected_sha=$(sha256_file "$selected_path")

mkdir -p "$APPROVAL_ROOT"
approval="$APPROVAL_ROOT/${SURVEY_ID}-${CANDIDATE_NAME}-${selected_sha:0:12}.txt"
{
  printf 'approved_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'survey_id=%s\n' "$SURVEY_ID"
  printf 'candidate_name=%s\n' "$CANDIDATE_NAME"
  printf 'survey_metrics=%s\n' "$METRICS"
  printf 'survey_metrics_sha256=%s\n' "$(sha256_file "$METRICS")"
  printf 'progress_bin=%s\n' "$selected_path"
  printf 'progress_sha256=%s\n' "$selected_sha"
  printf 'boundary_fixture_psv=%s\n' "$boundary_fixture"
  printf 'boundary_fixture_complete=%s\n' "$boundary_complete"
  printf 'approval_note=%s\n' "$APPROVAL_NOTE"
} | write_manifest_atomic "$approval"

echo "[approve-progress] 承認manifest: $approval"
echo "[approve-progress] 次工程では PROGRESS_APPROVAL=$approval を明示してください"
