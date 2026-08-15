#!/usr/bin/env bash
# 通常学習の固定CLIをshell実行可能な形式で表示する。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

if (( $# < 1 || $# > 2 )); then
  fail "usage: $0 <approved-progress.bin> [output-directory]"
fi

build_base_training_command "$@"
printf '%q ' "${BASE_TRAINING_COMMAND[@]}"
printf '\n'
