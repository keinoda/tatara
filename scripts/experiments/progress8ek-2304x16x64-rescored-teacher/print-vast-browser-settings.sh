#!/usr/bin/env bash
# Vast.ai Web UIへ入力する固定値とclone-onlyのOn-start Scriptを表示する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

readonly TATARA_BRANCH="codex/progress8ek-rescored-teacher-operations"
readonly REMOTE_BRANCH="refs/heads/$TATARA_BRANCH"

[[ -n "${TATARA_COMMIT:-}" ]] || fail "TATARA_COMMITを40桁SHAで明示してください"
[[ "$TATARA_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || fail "TATARA_COMMITは40桁の小文字Git SHAで指定してください"

require_command git
remote_commit=$(git -C "$EXPERIMENT_ROOT" ls-remote origin "$REMOTE_BRANCH" | awk 'NR == 1 {print $1}')
[[ "$remote_commit" == "$TATARA_COMMIT" ]] \
  || fail "originの専用branch先端とTATARA_COMMITが一致しません: remote=${remote_commit:-missing} requested=$TATARA_COMMIT"

read -r -d '' bootstrap_template <<'BOOTSTRAP' || true
set -Eeuo pipefail
readonly repo="https://github.com/keinoda/tatara.git"
readonly branch="codex/progress8ek-rescored-teacher-operations"
readonly target="/workspace/progress8ek-2304x16x64-rescored-teacher"
readonly commit="__TATARA_COMMIT__"

touch /root/.no_auto_tmux

if [[ -e "$target" ]]; then
  [[ -d "$target/.git" ]] || { echo "ERROR: clone target is not a Git checkout: $target" >&2; exit 1; }
  actual_origin=$(git -C "$target" remote get-url origin)
  [[ "$actual_origin" == "$repo" ]] || { echo "ERROR: unexpected origin: $actual_origin" >&2; exit 1; }
  [[ -z "$(git -C "$target" status --porcelain)" ]] || { echo "ERROR: checkout has uncommitted changes" >&2; exit 1; }
  [[ "$(git -C "$target" rev-parse HEAD)" == "$commit" ]] || { echo "ERROR: existing checkout is not the pinned commit" >&2; exit 1; }
else
  git clone --branch "$branch" --single-branch --no-checkout "$repo" "$target"
  branch_head=$(git -C "$target" rev-parse "refs/remotes/origin/$branch")
  [[ "$branch_head" == "$commit" ]] || { echo "ERROR: cloned branch tip is not the pinned commit" >&2; exit 1; }
  git -C "$target" checkout --detach "$commit"
fi

exec env TATARA_COMMIT="$commit" bash "$target/onstart.sh"
BOOTSTRAP
bootstrap=${bootstrap_template/__TATARA_COMMIT__/$TATARA_COMMIT}

readonly image_reference="$CONTAINER_IMAGE@$CONTAINER_IMAGE_DIGEST"
readonly docker_options="-p $MONITOR_PORT:$MONITOR_PORT"

cat <<SETTINGS
Vast.ai Web UI settings

Offer:
1x RTX 5090 / AMD Ryzen 9 9950X / at least 16 allocated CPU threads

Launch mode:
SSH / Direct connections enabled

Image:
$image_reference

Container disk:
40 GB

Volume:
任意の容量を /workspace にmount

Docker Options:
$docker_options

On-start Script:
$bootstrap
SETTINGS
