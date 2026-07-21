# Vast.ai実行手順

以下は課金前にlocalで確認し、Vast.aiでは上から順に実行する。`<...>`はoperatorが実値へ置き換える。
既存file、checkout、run、tmuxをscriptが自動削除する手順はない。

## 0. 課金前のlocal確認

```bash
cd /Users/keinoda/Documents/Tatara
git switch codex/progress-legacy9-1024x16x64-training
git status --short
git fetch origin codex/progress-legacy9-1024x16x64-training
git rev-parse HEAD
git rev-parse origin/codex/progress-legacy9-1024x16x64-training
```

working treeがcleanで、最後の2 SHAが一致していることを確認する。この40桁SHAを
`TATARA_COMMIT`とする。

## 1. Vast.aiのWeb画面から起動

Vast.aiのWeb UIで対象offerを選び、template設定を次の値にする。

| Web UI項目 | 入力値 |
|---|---|
| Launch Mode | SSH |
| Direct connections | ON |
| Image | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011@sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876` |
| Container Disk | 40 GB |
| Volume | 1400 GB |
| Volume Mount Path | `/workspace` |
| Docker Options | `-p 6001:6001 -e TATARA_COMMIT=<専用branch先端の40桁SHA>` |

`On-start Script`欄には次だけを貼る。`<...>`は置換せず、`TATARA_COMMIT`はDocker Optionsから
渡す。

```bash
set -Eeuo pipefail
readonly repo="https://github.com/keinoda/tatara.git"
readonly branch="codex/progress-legacy9-1024x16x64-training"
readonly target="/workspace/progress-legacy9-1024x16x64-training"
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

このOn-start Scriptは次のbootstrapだけで、repository側の`onstart.sh`本文は含まない。

1. `/root/.no_auto_tmux`を作り、Vast.aiのlogin時自動tmuxを止める。
2. 専用branchを`/workspace/progress-legacy9-1024x16x64-training`へcloneする。
3. cloneしたbranch先端と`TATARA_COMMIT`が一致することを確認する。
4. detached checkoutしたrepository内の`onstart.sh`を実行する。

再起動時にclone先が既にあれば、origin・clean状態・HEADを検査し、すべて一致する場合だけ再利用する。
`pull`、別revisionへのcheckout、既存directory削除は行わない。Web UIの設定内容と選択offerを
確認してから、Web UIのRent/Launch操作でinstanceを作成する。ローカルのVast CLIは使わない。

同じ設定をterminal上で確認したい場合だけ、次を実行する。fileへ保存する必要はない。

```bash
TATARA_COMMIT=<40桁SHA> \
  scripts/experiments/progress-legacy9-1024x16x64/print-vast-browser-settings.sh
```

## 2. onstart状態確認

SSH接続後、自動tmuxに入らないことを確認する。

```bash
test -f /root/.no_auto_tmux
cd /workspace/progress-legacy9-1024x16x64-training
git status --short
git rev-parse HEAD
scripts/experiments/progress-legacy9-1024x16x64/onstart-status.sh
tail -f /workspace/onstart.log
```

`onstart.sh`はTatara/rshogi build、30 shard download、validation download、progress.bin取得を
別tmuxで開始する。rshogiのconverterは
`--no-default-features --features nnue-arch`でbuildする。依存step完了後、再shuffleせずPSVを
連結する。本学習は自動開始しない。

```bash
tmux ls
ls -la .onstart
tail -f logs/onstart/download_training.log
tail -f logs/onstart/prepare_data.log
```

T0/T1の完了条件は`.onstart/prepare_data.done`である。失敗markerがある場合はlogとartifactを
調査し、原因を確定するまで次へ進まない。

T2のsurveyだけはT1の完了を待たず、Tatara build・baseline progress.bin取得と、公開教師shard
1個以上の取得完了後に開始できる。30 shard全部と連結PSVはT1以降のgateで確認する。

## 3. onstart失敗時の明示retry

```bash
RETRY_STEP=<失敗step> \
CONFIRM_RETRY_STEP=<同じ失敗step> \
  scripts/experiments/progress-legacy9-1024x16x64/retry-onstart-step.sh
```

`prepare_data`の生成途中PSVだけを除く必要がある場合に限り、同じcommandへ次を追加する。

```bash
CONFIRM_REMOVE_PARTIAL_PSV=/workspace/progress-legacy9-1024x16x64-training/data/training/public-teacher.psv.partial
```

このscriptは指定したfailed markerと、明示承認された生成途中PSV以外を削除しない。完成済みPSV、
dataset、manifestは上書きしない。checksum不一致はtransient failureとしてretryしない。

## 4. T2: 400万局面survey

T1の完了を待たず、取得完了済みshardの合計が400万局面以上なら実行できる。scriptは開始時点の
確定済み`.bin`だけを入力として固定し、実行中に追加で完了したshardを混ぜない。使用したfile・
size・局面数は`survey/<SURVEY_ID>/input-shards.txt`へ記録する。各surveyで400万局面は1回だけ読み、
calibrationのbaseline logitを使って`a > 0`の候補を自動生成する。目標比率を省略した場合は
12.5%ずつの`optimized-uniform`、明示した場合はその比率との平均二乗誤差を最小化する。
selectionとfinal-testは最適化に使わない。

```bash
SURVEY_ID=<新しいsurvey名> \
SURVEY_SEED=20260721 \
CALIBRATION_SAMPLES=2000000 \
SELECTION_SAMPLES=1000000 \
FINAL_TEST_SAMPLES=1000000 \
  scripts/experiments/progress-legacy9-1024x16x64/run-survey.sh
```

中央を穏やかに厚くした比率を同じ再現可能なoptimizerで比較する場合は、候補名と8 bucketの合計100%に
なる目標を両方明示する。目標は探索入力であり、実測比率がその値と一致する保証ではない。

```bash
SURVEY_ID=<新しいsurvey名> \
SURVEY_SEED=20260721 \
CALIBRATION_SAMPLES=2000000 \
SELECTION_SAMPLES=1000000 \
FINAL_TEST_SAMPLES=1000000 \
OPTIMIZED_CANDIDATE_NAME=optimized-center-gentle \
OPTIMIZER_TARGET_PERCENTAGES='11,12,13,14,14,13,12,11' \
SURVEY_INPUT_MANIFEST=survey/<比較元のSURVEY_ID>/input-shards.txt \
  scripts/experiments/progress-legacy9-1024x16x64/run-survey.sh
```

`AFFINE_CANDIDATES='<候補名>:<a>:<b>'`は既知係数を同じsample上で補助比較する場合だけ追加する。
通常の係数決定には指定せず、baseline取得、最適化、3 split評価、候補`progress.bin`生成までを
上の1回で完了する。

失敗再現などで同じ入力snapshotを再利用する必要がある場合だけ、上のcommandへ次を追加する。

```bash
SURVEY_INPUT_MANIFEST=survey/<baselineのSURVEY_ID>/input-shards.txt
```

`survey/<SURVEY_ID>/metrics.json`をユーザーへ提示する。最適化された`a,b`、明示目標と12.5%からの
MSE・最大乖離、bucket 0–7、migration、境界crossing、total variation、飽和、3 splitの差を確認する。
どの生成候補も自動採用しない。

```bash
SURVEY_ID=<survey名> \
CANDIDATE_NAME=<baselineまたは候補名> \
APPROVAL_NOTE='<提示結果に基づく採否理由>' \
  scripts/experiments/progress-legacy9-1024x16x64/approve-progress.sh
```

表示された承認manifestの絶対pathを以後の`PROGRESS_APPROVAL`に使う。

## 5. T3: 固定precision/thread smoke

本学習で使う予定の新しい`RUN_NAME`をここから一貫して使う。precisionは`all-optim`、worker
thread数はAMD Ryzen 9 9950Xの物理コア数に合わせた16で確定済みであり、候補比較は行わない。

```bash
RUN_NAME=<新しいrun名> \
PROGRESS_APPROVAL=<承認manifestの絶対path> \
  scripts/experiments/progress-legacy9-1024x16x64/run-smoke.sh
```

scriptは`all-optim-t16`を1回実行する。runが`completed`、loss/test loss/clamp metricが有限である
ことを検証し、`gates/<RUN_NAME>/precision.approved.txt`へ固定値を記録して`smoke.done`と
`precision.done`を作る。`smoke/report.json`のclampとthroughputは後続gateへ進む前に確認する。

trainer、量子化checkpoint、raw checkpoint、experiment JSON、reportまで完走した後にgate検証だけが
失敗した場合は、成果物を削除・上書き・再学習せず、同じ`RUN_NAME`、`PROGRESS_APPROVAL`、
`SMOKE_BATCHES`（未指定時は8）を使って次を実行する。

```bash
RUN_NAME=<同じrun名> \
PROGRESS_APPROVAL=<同じ承認manifest> \
FINALIZE_EXISTING_SMOKE=1 \
  scripts/experiments/progress-legacy9-1024x16x64/run-smoke.sh
```

回復経路は、元のcommand、PSV size、単一experiment、`completed`かつ非中断の状態、SB1履歴、
固定構成、`.bin`/`.ckpt`、有限metric、trainer commitを検証し、不足しているgate文書だけを作る。
既存gateが一つでもある場合や固定contractと一致しない場合は停止する。

`approve-smoke.sh`は既存の手順との互換用で、固定値manifestをread-only確認する場合だけ使う。

```bash
RUN_NAME=<同じrun名> \
  scripts/experiments/progress-legacy9-1024x16x64/approve-smoke.sh
```

## 6. T4/T5: resumeとengine end-to-end

```bash
RUN_NAME=<同じrun名> \
PROGRESS_APPROVAL=<承認manifest> \
  scripts/experiments/progress-legacy9-1024x16x64/run-resume-drill.sh

RUN_NAME=<同じrun名> \
PROGRESS_APPROVAL=<承認manifest> \
  scripts/experiments/progress-legacy9-1024x16x64/run-export-test.sh
```

T4はSB1 raw checkpointからSB2へoptimizer stateを含めてresumeする。T5は選択smoke networkを
1024x16x64・9 slotへ変換し、固定YaneuraOuでstartposと7境界上下の14 fixtureを探索する。
YaneuraOu buildではcontainerに存在する`python3`をMakefileの`PYTHON`変数へ明示する。

T5でnetwork変換と14件のfixture JSONL生成後、engine buildまたは探索だけが失敗した場合は、
既存成果物を削除・上書き・再変換せず、同じ値で次を実行する。

```bash
RUN_NAME=<同じrun名> \
PROGRESS_APPROVAL=<同じ承認manifest> \
CONTINUE_EXISTING_EXPORT=1 \
  scripts/experiments/progress-legacy9-1024x16x64/run-export-test.sh
```

回復経路は、固定smoke network、承認済み境界fixture、変換済みnetwork、14行のJSONL、
固定YaneuraOu origin/revisionとtracked差分なしを確認し、engine build以降だけを再開する。
探索時は`BookFile=no_book`を明示し、評価networkとprogressだけを検証対象にする。前回の探索logが
残っている場合は、それを保持したまま次のように新しいbasenameを指定する。

```bash
RUN_NAME=<同じrun名> \
PROGRESS_APPROVAL=<同じ承認manifest> \
CONTINUE_EXISTING_EXPORT=1 \
EXPORT_TRANSCRIPT_NAME=<未使用の名前.log> \
  scripts/experiments/progress-legacy9-1024x16x64/run-export-test.sh
```

指定した探索logまたはexport gateが既にある場合は上書きせず停止する。

## 7. T6: monitor

Vast.aiのport 6001 mapped URLを確認する。credentialはlogやmanifestへ保存しない。

```bash
RUN_NAME=<同じrun名> \
MONITOR_USER=<user> \
MONITOR_PASSWORD=<十分長いpassword> \
MONITOR_PUBLIC_URL=<末尾slashなしのmapped URL> \
  scripts/experiments/progress-legacy9-1024x16x64/run-monitor.sh
```

rendererとHTTP serverは別tmuxで動き、trainerへsignalやwriteを行わない。配信routeは`/`と
`/status.json`だけで、未認証requestが401、localとmapped URLの認証済みreadbackが成功して
`monitor.done`になる。HTMLは最新値の表に加えて、experiment JSONの全履歴からtrain lossを
青の左軸、test lossを赤の右軸として独立した縮尺で同じグラフに描き、test accuracyを別の
グラフに表示する。描画は外部CDNやJavaScriptへ依存しないインラインSVGで、15秒ごとの
snapshot更新時に再生成する。`status.json`は最新値に加えて`history`も含む。

## 8. 初回10 epoch学習

```bash
RUN_NAME=<同じrun名> \
PROGRESS_APPROVAL=<承認manifest> \
MONITOR_USER=<user> \
MONITOR_PASSWORD=<password> \
  scripts/experiments/progress-legacy9-1024x16x64/run-training.sh
```

起動直前に全gate、固定source、RTX 5090 1枚、教師・validation・progressのSHA-256、monitorを
再検証する。成功時だけ`train-<RUN_NAME>` tmuxを作る。

```bash
tmux attach -t train-<RUN_NAME>
tail -f runs/<RUN_NAME>/logs/train.log
curl --user '<user>:<password>' http://127.0.0.1:6001/status.json
```

異常終了、NaN/inf、CUDA/Xid/OOM、I/O error、checksum不一致を検知しても自動resume・自動stopは
しない。log、exit code、checkpointを保持し、ユーザーへ提示する。

## 9. 保存済みcheckpointの比較と手動resume

```bash
python3 scripts/experiments/progress-legacy9-1024x16x64/select-saved-checkpoint.py \
  --run-root runs/<親RUN_NAME>
```

報告だけで自動採用しない。延長承認後、新しいrun名とraw checkpointを指定する。

```bash
RUN_NAME=<新しいresume run名> \
PARENT_RUN_NAME=<親run名> \
RESUME_CHECKPOINT=<親run内のraw.ckpt絶対path> \
TARGET_SB=<440|513|587|660|733> \
PROGRESS_APPROVAL=<同じ承認manifest> \
MONITOR_USER=<user> \
MONITOR_PASSWORD=<password> \
  scripts/experiments/progress-legacy9-1024x16x64/resume-training.sh
```

resume runにも先に同じ`RUN_NAME`でmonitor gateを作る。親runが正常終了し、checkpoint、教師、
validation、progressが親manifestと一致する場合だけ別runとして開始する。

## 10. backup

自動backupはない。必要になった時だけ、ローカルの`rclone.conf`を一時的にinstanceへ置き、
manifest、survey、log、選択checkpoint、変換後networkだけをinstanceからGoogle Driveへ直接copyする。
実行前に転送元・転送先・一時config path・後処理を提示し、別途ユーザー承認を得る。
