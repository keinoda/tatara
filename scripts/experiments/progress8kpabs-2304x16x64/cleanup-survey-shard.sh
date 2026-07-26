#!/usr/bin/env bash
# 係数決定後、本学習前に部分取得survey用shardだけを明示削除する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

[[ "${CONFIRM_REMOVE_SURVEY_SHARD:-}" == "$TRAIN_SURVEY_SHARD" ]] \
  || fail "CONFIRM_REMOVE_SURVEY_SHARD=$TRAIN_SURVEY_SHARDを明示してください"
[[ -n "${PROGRESS_APPROVAL:-}" ]] || fail "採用済み係数を示すPROGRESS_APPROVALを明示してください"
require_progress_approval >/dev/null
require_exact_size "$TRAIN_SURVEY_SHARD" "$TRAIN_SURVEY_SHARD_BYTES" "survey用教師shard"
actual_sha=$(sha256_file "$TRAIN_SURVEY_SHARD")
[[ "$actual_sha" == "$TRAIN_SURVEY_SHARD_SHA256" ]] \
  || fail "survey用教師shardのSHA-256が固定値と異なります: actual=$actual_sha"

survey_found=0
while IFS= read -r input_manifest; do
  if grep -F "shard=$TRAIN_SURVEY_SHARD " "$input_manifest" >/dev/null; then
    survey_found=1
    break
  fi
done < <(find "$SURVEY_ROOT" -mindepth 2 -maxdepth 2 -type f -name input-shards.txt -print)
(( survey_found == 1 )) \
  || fail "保持shardを入力にした完了済みsurveyが見つかりません"

cleanup_manifest="$MANIFEST_DIR/survey-shard-cleanup.txt"
[[ ! -e "$cleanup_manifest" ]] || fail "cleanup manifestが既にあります: $cleanup_manifest"
removed_at=$(date -u +%FT%TZ)
rm -- "$TRAIN_SURVEY_SHARD"
{
  printf 'removed_at=%s\n' "$removed_at"
  printf 'path=%s\n' "$TRAIN_SURVEY_SHARD"
  printf 'bytes=%s\n' "$TRAIN_SURVEY_SHARD_BYTES"
  printf 'sha256=%s\n' "$actual_sha"
  printf 'progress_approval=%s\n' "$(canonical_file "$PROGRESS_APPROVAL")"
} | write_manifest_atomic "$cleanup_manifest"

echo "[cleanup] survey用教師shardだけを削除しました: $TRAIN_SURVEY_SHARD"
