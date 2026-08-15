#!/usr/bin/env bash
# 固定revisionのfloodgate HCPEを取得し、同じrshogiでPSVへ変換する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

require_command cargo
require_command git
require_command hf
require_command sha256sum

verify_file() {
  local path="$1"
  local expected_bytes="$2"
  local expected_sha="$3"
  local label="$4"
  [[ -f "$path" ]] || fail "$label がありません: $path"
  [[ "$(file_size "$path")" == "$expected_bytes" ]] \
    || fail "$label のsizeが一致しません: $path"
  [[ "$(sha256_file "$path")" == "$expected_sha" ]] \
    || fail "$label のSHA-256が一致しません: $path"
}

mkdir -p "$VALIDATION_DIR" "$(dirname "$RSHOGI_DIR")" "$MANIFEST_DIR"
if [[ ! -e "$VALIDATION_HCPE" ]]; then
  HF_HOME="$EXPERIMENT_ROOT/.runtime/huggingface" \
    hf download "$VALIDATION_DATASET" \
      --repo-type dataset \
      --revision "$VALIDATION_DATASET_REVISION" \
      --include floodgate.hcpe \
      --local-dir "$VALIDATION_DIR"
fi
verify_file \
  "$VALIDATION_HCPE" "$VALIDATION_HCPE_BYTES" "$VALIDATION_HCPE_SHA256" \
  "floodgate HCPE"

if [[ -e "$RSHOGI_DIR" ]]; then
  [[ -d "$RSHOGI_DIR/.git" ]] || fail "rshogi pathがGit checkoutではありません: $RSHOGI_DIR"
  [[ "$(git -C "$RSHOGI_DIR" remote get-url origin)" == "$RSHOGI_REPO" ]] \
    || fail "rshogi originが想定外です"
  [[ -z "$(git -C "$RSHOGI_DIR" status --porcelain)" ]] \
    || fail "rshogi checkoutに未保存の変更があります"
else
  git clone --no-checkout "$RSHOGI_REPO" "$RSHOGI_DIR"
  git -C "$RSHOGI_DIR" checkout --detach "$RSHOGI_COMMIT"
fi
[[ "$(git -C "$RSHOGI_DIR" rev-parse HEAD)" == "$RSHOGI_COMMIT" ]] \
  || fail "rshogi commitが固定値と異なります"

cargo build \
  --manifest-path "$RSHOGI_DIR/Cargo.toml" \
  --release -p tools --no-default-features --features nnue-arch --bin hcpe_to_psv

if [[ ! -e "$VALIDATION_PSV" ]]; then
  "$RSHOGI_DIR/target/release/hcpe_to_psv" \
    --input "$VALIDATION_HCPE" \
    --output "$VALIDATION_PSV"
fi
verify_file \
  "$VALIDATION_PSV" "$VALIDATION_PSV_BYTES" "$VALIDATION_PSV_SHA256" \
  "floodgate PSV"
validation_positions=$((VALIDATION_PSV_BYTES / PSV_RECORD_BYTES))
[[ "$validation_positions" == "$VALIDATION_FILE_POSITIONS" ]] \
  || fail "floodgate PSVの局面数が一致しません"
validation_sha="$VALIDATION_PSV_SHA256"

manifest="$MANIFEST_DIR/floodgate-validation.txt"
candidate="$manifest.candidate.$BASHPID"
{
  printf 'validation_dataset=%s\n' "$VALIDATION_DATASET"
  printf 'validation_dataset_revision=%s\n' "$VALIDATION_DATASET_REVISION"
  printf 'validation_hcpe=%s\n' "$VALIDATION_HCPE"
  printf 'validation_hcpe_bytes=%s\n' "$VALIDATION_HCPE_BYTES"
  printf 'validation_hcpe_sha256=%s\n' "$VALIDATION_HCPE_SHA256"
  printf 'rshogi_repo=%s\n' "$RSHOGI_REPO"
  printf 'rshogi_commit=%s\n' "$RSHOGI_COMMIT"
  printf 'validation_psv=%s\n' "$VALIDATION_PSV"
  printf 'validation_psv_bytes=%s\n' "$VALIDATION_PSV_BYTES"
  printf 'validation_psv_sha256=%s\n' "$validation_sha"
  printf 'validation_file_positions=%s\n' "$VALIDATION_FILE_POSITIONS"
  printf 'validation_effective_positions=%s\n' "$VALIDATION_EFFECTIVE_POSITIONS"
} >"$candidate"
if [[ -e "$manifest" ]]; then
  cmp -s "$candidate" "$manifest" \
    || fail "既存floodgate manifestが現在の固定入力と異なります"
  rm -- "$candidate"
else
  mv "$candidate" "$manifest"
fi

printf 'validation_psv=%s\n' "$VALIDATION_PSV"
printf 'validation_file_positions=%s\n' "$VALIDATION_FILE_POSITIONS"
printf 'validation_effective_positions=%s\n' "$VALIDATION_EFFECTIVE_POSITIONS"
printf 'validation_psv_sha256=%s\n' "$validation_sha"
