#!/usr/bin/env bash
# Vast.aiのWeb画面へ入力する固定値とOn-start Scriptを表示する。
# 互換のためfile名は維持するが、CLI作成commandは表示・実行しない。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

readonly TATARA_REPO="https://github.com/keinoda/tatara.git"
readonly TATARA_BRANCH="codex/progress8kpabs-2304x16x64-training"
readonly REMOTE_BRANCH="refs/heads/$TATARA_BRANCH"

[[ -n "${TATARA_COMMIT:-}" ]] || fail "TATARA_COMMITを40桁SHAで明示してください"
[[ "$TATARA_COMMIT" =~ ^[0-9a-f]{40}$ ]] \
  || fail "TATARA_COMMITは40桁の小文字Git SHAで指定してください"

require_command git
remote_commit=$(git -C "$EXPERIMENT_ROOT" ls-remote origin "$REMOTE_BRANCH" | awk 'NR == 1 {print $1}')
[[ "$remote_commit" == "$TATARA_COMMIT" ]] \
  || fail "originの専用branch先端とTATARA_COMMITが一致しません。push完了後のbranch先端を指定してください: remote=${remote_commit:-missing} requested=$TATARA_COMMIT"

read -r -d '' bootstrap <<'BOOTSTRAP' || true
set -Eeuo pipefail
readonly repo="https://github.com/keinoda/tatara.git"
readonly branch="codex/progress8kpabs-2304x16x64-training"
readonly target="/workspace/progress8kpabs-2304x16x64-training"
: "${TATARA_COMMIT:?TATARA_COMMIT is missing from the Vast.ai environment}"
[[ "$TATARA_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid TATARA_COMMIT" >&2; exit 1; }

# Disable the Vast.ai automatic login tmux before cloning the runbook.
touch /root/.no_auto_tmux

if [[ -e "$target" ]]; then
  [[ -d "$target/.git" ]] || { echo "ERROR: clone target is not a Git checkout: $target" >&2; exit 1; }
  actual_origin=$(git -C "$target" remote get-url origin)
  [[ "$actual_origin" == "$repo" ]] || { echo "ERROR: unexpected origin: $actual_origin" >&2; exit 1; }
  [[ -z "$(git -C "$target" status --porcelain)" ]] || { echo "ERROR: checkout has uncommitted changes" >&2; exit 1; }
  [[ "$(git -C "$target" rev-parse HEAD)" == "$TATARA_COMMIT" ]] || { echo "ERROR: existing checkout is not the pinned commit" >&2; exit 1; }
else
  git clone --branch "$branch" --single-branch --no-checkout "$repo" "$target"
  branch_head=$(git -C "$target" rev-parse "refs/remotes/origin/$branch")
  [[ "$branch_head" == "$TATARA_COMMIT" ]] || { echo "ERROR: cloned branch tip is not the pinned commit" >&2; exit 1; }
  git -C "$target" checkout --detach "$TATARA_COMMIT"
fi

exec env TATARA_COMMIT="$TATARA_COMMIT" bash "$target/onstart.sh"
BOOTSTRAP

readonly image_reference="$CONTAINER_IMAGE@$CONTAINER_IMAGE_DIGEST"
readonly vast_env="-p 6001:6001 -e TATARA_COMMIT=$TATARA_COMMIT"

cat <<SETTINGS
Vast.ai Web UI settings

Launch mode:
SSH / Direct connections enabled

Image:
$image_reference

Container disk:
40 GB

Volume:
1000 GB mounted at /workspace

Docker Options:
$vast_env

On-start Script:
$bootstrap
SETTINGS
