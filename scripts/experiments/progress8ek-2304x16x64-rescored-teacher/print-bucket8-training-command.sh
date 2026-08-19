#!/usr/bin/env bash
# slot 8追加学習の固定CLIをshell実行可能な形式で表示する。学習は開始しない。

set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/lib.sh"

if (( $# < 2 || $# > 3 )); then
  fail "usage: $0 <base-network.bin> <approved-progress.bin> [output-directory]"
fi

build_bucket8_training_command "$@"
printf '%q ' "${BUCKET8_TRAINING_COMMAND[@]}"
printf '\n'
