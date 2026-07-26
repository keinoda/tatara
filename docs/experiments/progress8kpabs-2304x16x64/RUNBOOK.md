# Vast.ai実行手順

## 固定対象

- branch: `codex/progress8kpabs-2304x16x64-training`
- instance checkout: `/workspace/progress8kpabs-2304x16x64-training`
- image: `ghcr.io/keinoda/shogi-lab:cuda129-trt1011`
- GPU: RTX 5090 1枚
- CPU: Ryzen 9 9950X、学習threads 16
- volume: 1000GB、`/workspace`
- monitor port: 6001

branch先端の40桁SHAを`TATARA_COMMIT`へ指定する。branch名だけで起動せず、
clone直後にremote branch先端とSHAが一致することを検証してからdetached checkoutする。

## Vast.ai Web UI

ローカルでbranchをpush済みの場合は次で入力値を生成できる。

```bash
cd /Users/keinoda/Documents/Tatara
TATARA_COMMIT="$(git rev-parse origin/codex/progress8kpabs-2304x16x64-training)" \
  scripts/experiments/progress8kpabs-2304x16x64/print-vast-browser-settings.sh
```

Web UIには次を設定する。

| 項目 | 値 |
|---|---|
| Image | scriptが表示するdigest付きimage |
| Container disk | 40GB |
| Volume | 1000GB、`/workspace` |
| Docker Options | `-p 6001:6001 -e TATARA_COMMIT=<40桁SHA>` |

On-start Scriptは次の形にする。

```bash
set -Eeuo pipefail
readonly repo="https://github.com/keinoda/tatara.git"
readonly branch="codex/progress8kpabs-2304x16x64-training"
readonly target="/workspace/progress8kpabs-2304x16x64-training"
: "${TATARA_COMMIT:?TATARA_COMMIT is missing from the Vast.ai environment}"
[[ "$TATARA_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "ERROR: invalid TATARA_COMMIT" >&2; exit 1; }

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
```

## onstartの監視

SSH接続後:

```bash
cd /workspace/progress8kpabs-2304x16x64-training
scripts/experiments/progress8kpabs-2304x16x64/onstart-status.sh
tmux ls
tail -f /workspace/onstart.log
```

個別log:

```bash
tail -f logs/onstart/download_training.log
tail -f logs/onstart/build_tatara.log
tail -f logs/onstart/build_rshogi.log
tail -f logs/onstart/prepare_data.log
```

`download_training`はshardを1個ずつ取得し、SHA-256検証後に
`data/training/public-teacher.psv.partial`へ追記する。確定済みshardは
`.onstart/training-append/*.done`で確認できる。`split_000.bin`だけはsurvey用に残し、
他の元shardは追記確定後に削除される。

失敗markerを確認して同じstepを再開する場合:

```bash
RETRY_STEP=download_training \
CONFIRM_RETRY_STEP=download_training \
  scripts/experiments/progress8kpabs-2304x16x64/retry-onstart-step.sh
```

`download_training`の再実行は確定markerを読み、未確定末尾だけを自動で確定境界へ
切り戻す。`.partial`やmarkerを手作業で削除しない。

## progress係数survey

次の2条件が揃えば、全34 shardの完了前でも開始できる。

```bash
test -f .onstart/training-append/split_000.bin.done
test -x target/release/progress-bucket-survey
```

前回採用候補も補助比較に含め、400万局面を固定して新候補を最適化する。

```bash
SURVEY_ID=teacher-20260726-center \
SURVEY_SEED=20260726 \
CALIBRATION_SAMPLES=2000000 \
SELECTION_SAMPLES=1000000 \
FINAL_TEST_SAMPLES=1000000 \
OPTIMIZER_TARGET_PERCENTAGES=11,12,13,14,14,13,12,11 \
OPTIMIZED_CANDIDATE_NAME=optimized-center-gentle \
AFFINE_CANDIDATES='previous-selected:1.2980837735881936:-0.5975424282106219' \
  scripts/experiments/progress8kpabs-2304x16x64/run-survey.sh
```

結果:

```bash
python3 -m json.tool survey/teacher-20260726-center/metrics.json | less
cat survey/teacher-20260726-center/manifest.txt
cat survey/teacher-20260726-center/input-shards.txt
```

baseline、`previous-selected`、`optimized-center-gentle`のcalibration・selection・
final-test分布と境界fixtureを提示する。採用候補はユーザー確認後に指定する。

```bash
SURVEY_ID=teacher-20260726-center \
CANDIDATE_NAME=<baseline|previous-selected|optimized-center-gentle> \
APPROVAL_NOTE='<採用理由>' \
  scripts/experiments/progress8kpabs-2304x16x64/approve-progress.sh
```

出力された承認manifestの絶対pathを以降の`PROGRESS_APPROVAL`に使う。

## 準備完了確認

```bash
test -f .onstart/prepare_data.done
cat manifests/source-revisions.txt
cat manifests/training-inline-append.tsv
cat manifests/prepared-data.txt
stat -c '%n %s' data/training/public-teacher.psv
```

期待値:

- 教師PSV: 673,002,105,840 bytes
- 教師局面: 16,825,052,646
- validation PSV: 34,276,920 bytes
- validation file局面: 856,923
- validation実効局面: 851,968

## GPU・resume・export gate

共通値:

```bash
export RUN_NAME=progress8kpabs-2304x16x64-10epoch
export PROGRESS_APPROVAL=/workspace/progress8kpabs-2304x16x64-training/progress/approved/<承認manifest>.txt
```

GPU smoke:

```bash
scripts/experiments/progress8kpabs-2304x16x64/run-smoke.sh
```

中断後、完走済みsmoke成果物からgate確定だけを再開する場合:

```bash
FINALIZE_EXISTING_SMOKE=1 \
  scripts/experiments/progress8kpabs-2304x16x64/run-smoke.sh
```

resume:

```bash
scripts/experiments/progress8kpabs-2304x16x64/run-resume-drill.sh
```

YaneuraOu形式変換と固定node探索:

```bash
scripts/experiments/progress8kpabs-2304x16x64/run-export-test.sh
```

変換済み成果物を残してengine build以降だけを再開する場合:

```bash
CONTINUE_EXISTING_EXPORT=1 \
EXPORT_TRANSCRIPT_NAME=yaneuraou-retry.log \
  scripts/experiments/progress8kpabs-2304x16x64/run-export-test.sh
```

export gateは次を確認する。

- Tatara 8 bucket networkをYaneuraOu 9 slotへ変換
- `halfkahm2_2304_15_64_ls9` engineが`nn.bin`と外部progress.binを読込
- startpos 1局面とprogress境界上下14局面で`bestmove`を返す
- 定跡は`BookFile=no_book`

## monitor gate

mapped port 6001のURLとBasic認証値を明示する。

```bash
export MONITOR_USER='<user>'
export MONITOR_PASSWORD='<password>'
export MONITOR_PUBLIC_URL='https://<mapped-host>'
scripts/experiments/progress8kpabs-2304x16x64/run-monitor.sh
```

local readback、mapped readback、未認証401を確認したときだけgateが完了する。

## survey用shardの削除

progress承認後、本学習前に保持していた`split_000.bin`だけを削除する。

```bash
CONFIRM_REMOVE_SURVEY_SHARD=/workspace/progress8kpabs-2304x16x64-training/data/training/shards/split_000.bin \
PROGRESS_APPROVAL="$PROGRESS_APPROVAL" \
  scripts/experiments/progress8kpabs-2304x16x64/cleanup-survey-shard.sh
```

`manifests/survey-shard-cleanup.txt`が生成され、`data/training/shards`に
`split_*.bin`が残っていないことを確認する。

## 本学習

全gateとcleanupの完了後に手動開始する。

```bash
RUN_NAME="$RUN_NAME" \
PROGRESS_APPROVAL="$PROGRESS_APPROVAL" \
MONITOR_USER="$MONITOR_USER" \
MONITOR_PASSWORD="$MONITOR_PASSWORD" \
  scripts/experiments/progress8kpabs-2304x16x64/run-training.sh
```

固定commandは次の内容を含む。

```text
--batch-size 65536
--batches-per-superbatch 6104
--superbatches 421
--lr 8.75e-4
--lr-schedule step
--lr-gamma 0.992
--lr-step 1
--all-optim
--threads 16
--save-rate 20
--keep-checkpoints 2
layerstack --ft-out 2304 --l1 16 --l2 64
--bucket-mode progress8kpabs --num-buckets 8
```

監視:

```bash
tmux attach -t "train-$RUN_NAME"
tail -f "runs/$RUN_NAME/logs/train.log"
```

自動resume、自動instance停止、外部backupは行わない。

## 完走後の選択と延長

保存済みcheckpointとvalidationを一覧する。

```bash
python3 scripts/experiments/progress8kpabs-2304x16x64/select-saved-checkpoint.py \
  --run-root "runs/$RUN_NAME"
```

延長は親runを変更せず別runにする。

```bash
PARENT_RUN_NAME="$RUN_NAME" \
RUN_NAME=progress8kpabs-2304x16x64-<target>sb \
RESUME_CHECKPOINT=/workspace/progress8kpabs-2304x16x64-training/runs/<parent>/checkpoints/<raw>.ckpt \
TARGET_SB=<505|589|673|757|841> \
PROGRESS_APPROVAL="$PROGRESS_APPROVAL" \
MONITOR_USER="$MONITOR_USER" \
MONITOR_PASSWORD="$MONITOR_PASSWORD" \
  scripts/experiments/progress8kpabs-2304x16x64/resume-training.sh
```

候補は約12、14、16、18、20 epochに対応する。延長判断とcheckpoint採用は
validation曲線を提示してから行う。
