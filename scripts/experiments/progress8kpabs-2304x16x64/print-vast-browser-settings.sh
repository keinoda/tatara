#!/usr/bin/env bash
# Vast.ai Web UIへ貼る設定を表示する正式入口。

set -Eeuo pipefail
readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$script_dir/print-vast-create-command.sh"
