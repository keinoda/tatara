#!/usr/bin/env bash
# 選択済み2304x16x64 smoke networkを9-slotへ変換し、固定YaneuraOuでload/evalする。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_run_name
require_source_revision
require_command git
require_command mktemp
progress_bin=$(require_progress_approval)
readonly GATE_DIR="$(gate_dir_for_run "$RUN_NAME")"
require_gate "$GATE_DIR" smoke
require_gate "$GATE_DIR" precision
require_gate "$GATE_DIR" resume
precision=$(precision_from_gate "$GATE_DIR")
threads=$(manifest_value "$GATE_DIR/precision.approved.txt" threads)

boundary_complete=$(approval_value "$PROGRESS_APPROVAL" boundary_fixture_complete)
[[ "$boundary_complete" == "true" ]] \
  || fail "選択候補の7境界上下fixtureが14件揃っていません。sample/candidateを見直してください"
boundary_psv=$(approval_value "$PROGRESS_APPROVAL" boundary_fixture_psv)
require_exact_size "$boundary_psv" 560 "7境界上下fixture PSV"

readonly EXPORT_ROOT="$GATE_DIR/export-test"
readonly CONTINUE_EXISTING_EXPORT="${CONTINUE_EXISTING_EXPORT:-0}"
readonly EXPORT_TRANSCRIPT_NAME="${EXPORT_TRANSCRIPT_NAME:-yaneuraou.log}"
[[ "$CONTINUE_EXISTING_EXPORT" =~ ^[01]$ ]] \
  || fail "CONTINUE_EXISTING_EXPORTは0または1にしてください"
[[ "$EXPORT_TRANSCRIPT_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.log$ ]] \
  || fail "EXPORT_TRANSCRIPT_NAMEは安全なbasename（*.log）にしてください"
transcript="$EXPORT_ROOT/$EXPORT_TRANSCRIPT_NAME"
for gate_artifact in "$EXPORT_ROOT/manifest.txt" "$GATE_DIR/export.done"; do
  [[ ! -e "$gate_artifact" ]] || fail "既存export gateを上書きしません: $gate_artifact"
done
[[ ! -e "$transcript" ]] || fail "既存のYaneuraOu探索logを上書きしません: $transcript"

input_bin="$GATE_DIR/smoke/${precision}-t${threads}/checkpoints/smoke-$RUN_NAME-${precision}-t${threads}-1.bin"
[[ -s "$input_bin" ]] || fail "選択smoke networkがありません: $input_bin"
output_bin="$EXPORT_ROOT/eval/nn.bin"
readonly RSHOGI_DIR="$EXPERIMENT_ROOT/.runtime/rshogi"
readonly PSV_TO_JSONL="$RSHOGI_DIR/target/release/psv_to_jsonl"
[[ -x "$PSV_TO_JSONL" ]] || fail "psv_to_jsonlがありません。onstartのbuild_rshogiを確認してください"
fixtures_jsonl="$EXPORT_ROOT/boundary-fixtures.jsonl"

if [[ "$CONTINUE_EXISTING_EXPORT" == 0 ]]; then
  [[ ! -e "$EXPORT_ROOT" ]] || fail \
    "既存export testを上書きしません。変換後の成果物から再開する場合はCONTINUE_EXISTING_EXPORT=1を明示してください: $EXPORT_ROOT"
  mkdir -p "$EXPORT_ROOT/eval"
  "$NET_TO_YO" --input "$input_bin" --output "$output_bin" --assume-progress8kpabs
  [[ -s "$output_bin" ]] || fail "YaneuraOu networkが生成されませんでした"
  "$PSV_TO_JSONL" --input "$boundary_psv" --output "$fixtures_jsonl" --limit 14
  finalization_mode="fresh-export-test"
else
  [[ -d "$EXPORT_ROOT/eval" ]] || fail "既存export testのeval directoryがありません"
  [[ -s "$output_bin" ]] || fail "既存の変換済みYaneuraOu networkがありません: $output_bin"
  [[ -s "$fixtures_jsonl" ]] || fail "既存のboundary fixture JSONLがありません: $fixtures_jsonl"
  finalization_mode="existing-conversion-artifacts"
  echo "[export-test] 既存の変換成果物を再生成せず、engine build以降を再開します"
fi
[[ "$(wc -l <"$fixtures_jsonl" | tr -d ' ')" == 14 ]] || fail "boundary fixture JSONLが14行ではありません"

readonly YANEURAOU_DIR="$EXPERIMENT_ROOT/.runtime/YaneuraOu-$YANEURAOU_COMMIT"
if [[ -d "$YANEURAOU_DIR/.git" ]]; then
  [[ "$(git -C "$YANEURAOU_DIR" remote get-url origin)" == "$YANEURAOU_REPO" ]] \
    || fail "YaneuraOu originが想定外です"
  [[ "$(git -C "$YANEURAOU_DIR" rev-parse HEAD)" == "$YANEURAOU_COMMIT" ]] \
    || fail "YaneuraOu revisionが固定値と異なります"
  [[ -z "$(git -C "$YANEURAOU_DIR" status --porcelain --untracked-files=no)" ]] \
    || fail "YaneuraOu checkoutにtracked変更があります"
else
  [[ ! -e "$YANEURAOU_DIR" ]] || fail "$YANEURAOU_DIRはGit repositoryではありません"
  [[ -n "${YANEURAOU_GITHUB_TOKEN:-}" ]] \
    || fail "YaneuraOu-privateをcloneするfine-grained tokenをYANEURAOU_GITHUB_TOKENへ指定してください"
  askpass=$(mktemp "$EXPERIMENT_ROOT/.runtime/yaneuraou-askpass.XXXXXX")
  trap 'rm -f -- "$askpass"' EXIT
  cat >"$askpass" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *Password*) printf '%s\n' "${YANEURAOU_GITHUB_TOKEN:?}" ;;
  *) exit 1 ;;
esac
ASKPASS
  chmod 700 "$askpass"
  GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
    git clone --no-checkout "$YANEURAOU_REPO" "$YANEURAOU_DIR"
  rm -f -- "$askpass"
  trap - EXIT
  git -C "$YANEURAOU_DIR" checkout --detach "$YANEURAOU_COMMIT"
fi

readonly ENGINE="$YANEURAOU_DIR/source/YaneuraOu-by-gcc"
if [[ ! -x "$ENGINE" ]]; then
  make -C "$YANEURAOU_DIR/source" -j"$(nproc)" PYTHON=python3 normal \
    YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_halfkahm2_2304_15_64_ls9 \
    TARGET_CPU=AVX2 COMPILER=g++
fi
[[ -x "$ENGINE" ]] || fail "YaneuraOu engine buildに失敗しました"

python3 "$EXPERIMENT_SCRIPT_DIR/yaneuraou-smoke.py" \
  --engine "$ENGINE" \
  --eval-dir "$EXPORT_ROOT/eval" \
  --progress "$progress_bin" \
  --fixtures-jsonl "$fixtures_jsonl" \
  --transcript "$transcript" \
  --nodes 100
[[ "$(grep -c '^bestmove ' "$transcript")" == 15 ]] \
  || fail "startpos + 14境界fixtureのbestmoveを確認できませんでした"

{
  printf 'completed_at=%s\n' "$(date -u +%FT%TZ)"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'tatara_input=%s\n' "$input_bin"
  printf 'tatara_input_sha256=%s\n' "$(sha256_file "$input_bin")"
  printf 'yaneuraou_network=%s\n' "$output_bin"
  printf 'yaneuraou_network_sha256=%s\n' "$(sha256_file "$output_bin")"
  printf 'yaneuraou_repo=%s\n' "$YANEURAOU_REPO"
  printf 'yaneuraou_commit=%s\n' "$YANEURAOU_COMMIT"
  printf 'yaneuraou_engine_sha256=%s\n' "$(sha256_file "$ENGINE")"
  printf 'progress_sha256=%s\n' "$(sha256_file "$progress_bin")"
  printf 'finalization_mode=%s\n' "$finalization_mode"
  printf 'transcript=%s\n' "$transcript"
  printf 'transcript_sha256=%s\n' "$(sha256_file "$transcript")"
  printf 'boundary_positions=14\n'
  printf 'bestmoves=15\n'
} | write_manifest_atomic "$EXPORT_ROOT/manifest.txt"
date -u +%FT%TZ >"$GATE_DIR/export.done"
echo "[export-test] 8→9変換、2304x16x64 engine load、startpos+14境界局面を確認しました"
