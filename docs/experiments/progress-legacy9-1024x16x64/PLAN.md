# 公開教師データによる legacy progress 係数校正と 8 routing / 9-slot LayerStack 学習計画

## 1. 目的

Vast.ai の RTX 5090 1 枚で、SOJO が公開したシャッフル済み教師データを使い、
`1024x16x64` の LayerStack NNUE を 10–20 epoch 学習する。networkは従来互換の9 slotを
保持し、`keinoda/YaneuraOu` の legacy `progress.bin` を基礎にした局面進行度8分割で
slot 0–7だけを選ぶ。slot 8は従来どおり未使用とし、既存エンジンと同じ`0.125`刻みを
維持する。8-slot形式や新しいbinning形式は導入しない。

SOJO 教師は局単位の並びと総手数を保持しないため、SOJO を教師信号にした進行度モデルの
再学習は行わない。既存係数に対し、手数を使わない affine 校正だけで出力分布を広げられるか
事前 survey する。提示された現行 `d77` の既知比率も参照し、検証用 sample で事前承認した
条件を満たせなければ、係数は変更せず、現行 `progress.bin` を使う。

学習中は floodgate 公開局面による held-out `test_loss` を毎 superbatch 計測する。
学習プロセスとは分離した監視プロセスを用意し、`0.0.0.0:6001` のページで進捗、
損失、学習率、GPU、checkpoint、プロセス健全性を確認できる状態にする。

## 2. 対象としないこと

- `ngs436` 以下の個人データは使用しない。
- SOJO 教師データは再シャッフルしない。
- 事前試験が完了するまで本学習を開始しない。
- 学習や監視の異常時に、自動再起動、自動 resume、自動停止を行わない。
- 既存 run、dataset、checkpoint、`progress.bin` を上書きしない。
- この計画の確定前に Vast.ai インスタンスを起動・変更しない。

## 3. 確定している入力

| 項目 | 値 |
|---|---|
| container image | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011` |
| GPU | RTX 5090 × 1 |
| 教師 dataset | `washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique` |
| 教師データ | 30 shard、586,757,977,480 bytes、14,668,949,437 局面 |
| 教師の順序 | シャッフル済みとして扱い、連結後も再シャッフルしない |
| validation dataset | `takaoyamaoka/floodgate.hcpe` |
| validation PSV | 34,276,920 bytes、856,923 局面 |
| network | LayerStack、FT 1024、L1 16、L2 64 |
| LayerStack layout | 9 slot。progress routingではslot 0–7を使用し、slot 8は未使用 |
| bucket routing | progressによる固定8分割、既存エンジンと同じ`0.125`刻み |
| 学習量 | 最低 10 epoch、最大 20 epochを候補とし、途中の検証値で判断 |
| 監視 | 学習と別プロセス、`0.0.0.0:6001` |

公開情報の参照先:

- [NNUE Lab / sh11235](https://nnue-lab.sh11235.com/t/sh11235)
- [tatara 標準レシピと学習結果](https://note.com/ramu_shogi_dev/n/ndb0d9f2f2187)
- [SOJO WCSC36 アピール資料](https://www.apply.computer-shogi.org/wcsc36/appeal/sojo/sojo_WCSC36_appeal.pdf)
- [SOJO 公開教師データ](https://huggingface.co/datasets/washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique)
- [floodgate validation data](https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe)

## 4. 現行実装との差分と実行前ブロッカー

### 4.1 legacy 係数だけで可能な校正

従来の Tatara progress 実装と現行 YaneuraOu は、局面の active KP-absolute 特徴の重み和を
`z` として `p = sigmoid(z)` を計算し、`min(7, floor(p * 8))` で bucket を選ぶ。
現行 Tatara `main` との差分は 4.2 節で扱い、既存の可変 N routing 自体は変更しない。
ローカルで確認した `keinoda/YaneuraOu/source/progress.bin` は 1,003,104 bytes の
legacy `f64[81][1548]` で、SHA-256 は
`d77f47e874558d42fa2d87d173de3aba054eef51bcca9c1fc9f3a8daf93630d8`。

既存エンジンの式とファイル形式を変えずに表現できる単調変換は、全係数の scale と
共通 offset による次の affine 校正に限られる。

```text
z' = a * z + b
p' = sigmoid(z') = sigmoid(a * logit(p) + b)
```

全合法局面の active index 数 `K` は実装上76で固定される。非玉38駒を列挙し、黒玉基準と
反転した白玉基準の2 indexを必ず加えるためである。したがって各係数を
`w'_i = a * w_i + b / 76` とすれば、定数項 `b` を既存形式へ埋め込める。surveyでは
この不変条件も検証し、76以外のrecordは校正に混ぜず入力異常として報告する。

これは SOJO の手数や総手数を使う再学習ではない。既存 `progress.bin` が出す `p` の
分布だけを使い、SOJO の校正 sample 上で `a > 0` と `b` を探索する。ただし2変数では
任意の経験分布関数を表現できないため、8 bucket の完全な等頻度化は保証しない。

既知の参考分布は次のとおり。`更新前 ce88` は比較用として記録するだけで、採否判定は
現行 `d77` とSOJO上の実測を基準にする。ただし、この表を測定したdatasetとsample条件は
未確認であるため、SOJO実測の代用やaffine校正の直接targetにはしない。

| progress bucket | 更新前 ce88 | 現行 d77 |
|---:|---:|---:|
| 0 | 0.15% | 1.64% |
| 1 | 3.60% | 9.29% |
| 2 | 12.07% | 12.63% |
| 3 | 14.34% | 12.51% |
| 4 | 14.83% | 13.46% |
| 5 | 17.29% | 16.99% |
| 6 | 23.47% | 23.88% |
| 7 | 14.25% | 9.60% |

今回の主目的はbucket 0が薄すぎる問題の改善であり、8 bucket全体の均等化は目的にしない。
bucket 0を増やしたときのbucket 1–7への移動、飽和、順位保存を同時に示し、進行度としての
意味を壊していないかを人が判断する。

採用手順:

1. downloadとchecksum確認が完了し、書込み中でないSOJO shardだけをeligibleにする。
   各shardの局面数に比例した、40-byte record単位の重複なしランダム抽出を行う。
2. 初回候補は校正2,000,000局面＋検証2,000,000局面とする。両集合は重複させず、sampling
   algorithm、seed、eligible shard名・bytes・SHA-256、抽出数を先にmanifestへ保存する。
   dataset本体の順序や内容は変更せず、sample用に全教師を再シャッフルしない。
3. download途中の初回surveyは暫定結果とし、全30 shard完了後に全shardを母集団として
   同じ抽出数の独立sampleで再確認する。候補係数の採否は最終survey後に決める。
4. 現行係数の bucket 比率をread-only surveyし、active index数が全合法局面で76である
   ことを確認する。結果は上記 `d77` 参考値と並べるが、測定条件が不明な参考値との一致
   自体は要求しない。
5. 校正sampleだけで`a>0, b`を探索する。単一の自動最適解にせず、bucket 0の増加量と
   bucket 1–7の最大変化量が異なるPareto候補を複数作る。
6. 検証sampleについて、全bucket比率、bucket 0の増加percentage point・倍率、bucket 1–7の
   最大絶対変化、total variation、旧新progressのSpearman順位相関、飽和率を比較表にする。
7. survey結果と候補係数をユーザーへ提示し、明示的に選ばれた場合だけ別名のlegacy
   `progress.bin`を生成する。不採用なら現行SHA-256 `d77f47...30d8`をそのまま使う。

局順と総手数を復元・推測する処理、SOJO をラベルにした `progress-kpabs-train`、
経験 CDF を pseudo-target にした係数再学習は行わない。

なお、分位点 trailer を実装した fork は
[`souyuukou/tatara:feat/num-buckets-256`](https://github.com/souyuukou/tatara/tree/feat/num-buckets-256)
と確認したが、本計画では利用しない。`progress.bin` に trailer を追加せず、YaneuraOu の
binning 実装も変更しない。

### 4.2 9-slot legacy routing と YaneuraOu 形式への変換

従来のTatara progress実装は、network layoutを9 slotのまま保持し、
`min(7, floor(p * 8))`でslot 0–7だけを使っていた。現行Tatara `main`はcommit
`2622e41`で`--num-buckets N`を一般化し、`N=9`では`floor(p * 9)`に変わってslot 8も
学習する。一方、現行YaneuraOuの`progress8kpabs`は固定`0.125`境界のままであり、9
LayerStacks buildでもslot 8を選ばない。そのため現行Tataraの`progress8kpabs + N=9`を
そのまま使うとtrainerとengineのroutingが一致しない。

既存動作を壊さない最小修正として、現行Tataraの可変N routingは変更せず、追加の
`progress8kpabs-legacy9` modeを実装する。このmodeは`--num-buckets 9`だけを受理し、
bucket indexを固定`min(7, floor(p * 8))`で計算する。training、held-out validation、
eval、resume checkpoint、experiment JSONの全経路で同じmodeを記録・使用し、slot 8の
gradientとoptimizer stateが更新されないことをtestする。名称は実装時にCLI helpと
checkpoint互換性を確認して最終固定するが、別名へ暗黙fallbackしない。

現行`net_to_yo`は9 LayerStacksに限ればfeature setとFT/L1/L2次元をheaderから自動検出
するため、`1024x16x64`のweight layout自体には既に対応している。不足しているのは
`--assume-kingrank9`しかなく、9-slot progress netであることを正しく表明できない点である。
converterは次の最小差分に限定する。

1. `--assume-progress8kpabs`を`--assume-kingrank9`と排他的に追加する。
2. progress assertionは入力がexactly 9 slotであることを要求する。
3. 次元検出、量子化、9-slot loop、YaneuraOu共通writer、hash、binary layoutは変更しない。
4. `nnue-train --output-format yaneuraou`と`net_from_yo`は今回の経路で使わず、変更しない。
5. 既存KingRank9の変換結果がbyte一致するregression testを維持する。
6. `1024x16x64 / 9-slot` fixtureを追加し、期待architecture文字列、input全消費、slot数、
   非zero weightの順序と出力再現性を検証する。

YaneuraOu側は既に9 LayerStacks + progress routingを受理する。ローカル
`master@66bef215`の関連実装は`origin/master@771fe811`と同一で、対応build keyは
`YANEURAOU_ENGINE_SFNN_halfkahm2_1024_15_64_ls9`とする。`15`はTataraのL1出力16から
shortcut 1次元を除いたYaneuraOu表記であり、network shapeの変更ではない。実行時は
`LS_BUCKET_MODE=progress8kpabs`と同一legacy `progress.bin`を指定する。

本学習前に、Tatara legacy modeとYaneuraOuが固定局面および境界近傍で同じ0–7を返し、
slot 8を一度も返さないことをend-to-endで確認する。8-slot net、progress trailer、
writer format変更は行わない。

### 4.3 現行 `onstart.sh`

現行 `onstart.sh` が生成する既存run scriptは `1536x16x32`、KingRank9、
160 superbatch 用であり、この実験には使用しない。データ取得と build の部分は再利用
候補だが、この計画書を追加する commit では `onstart.sh` を変更・追跡しない。
実装は 14 節の gate に従い、legacy9 routing と converter の検証後に別 commit で行う。

## 5. 学習量の換算

ユーザー指定と NNUE Lab / ブログの標準 run に合わせ、学習量の基本単位を次で固定する。

```text
batch-size              = 65,536
batches-per-superbatch  = 6,104
1 superbatch            = 400,031,744 局面
1 superbatch            = 0.0272706471 epoch
1 epoch                 = 約 36.669464 superbatch
```

SOJO 全体 14,668,949,437 局面は 1 superbatch の整数倍ではない。データは再シャッフルせず、
Tatara の既存 dataloader がファイル終端から先頭へ循環する通常動作を使う。

| 目標 epoch | superbatch | 実 epoch |
|---:|---:|---:|
| 1 | 37 | 1.009013944 |
| 2 | 73 | 1.990757241 |
| 5 | 183 | 4.990528426 |
| 8 | 293 | 7.990299612 |
| 9 | 330 | 8.999313556 |
| 10 | 367 | 10.008327500 |
| 12 | 440 | 11.999084741 |
| 14 | 513 | 13.989841982 |
| 16 | 587 | 16.007869871 |
| 18 | 660 | 17.998627112 |
| 20 | 733 | 19.989384353 |

最初の本学習は `TARGET_SB=367` で自然終了させる。floodgate test loss を確認して延長する
場合だけ、同じ raw checkpoint から `440 / 513 / 587 / 660 / 733` の順で手動 resume
する。これにより10 epochで必ず判定を挟み、学習プロセスを外部から途中 kill しない。

## 6. 学習率

NNUE Lab の基準 run は `8.75e-4` から始め、Tatara 既定の step schedule
`gamma=0.992, step=1` を160 superbatch使う。SB160で実際に使われる LR は
`8.75e-4 * 0.992^159 = 2.4398544857e-4`、初期値の `0.2788405126` 倍である。

標準の減衰率を変えずにstep scheduleを367 SBへ延ばした場合、SB367のLRは
`8.75e-4 * 0.992^366 = 4.6267929105e-5`、初期値の約5.29%になる。今回はこの10 epoch
終端値を維持しつつ、Tataraが標準でサポートする滑らかな`exponential` scheduleで
SB367まで減衰させる。

```text
--lr 8.75e-4
--lr-schedule exponential
--lr-final 4.6267929105e-5
--lr-final-superbatch 367
```

Tataraの実装は`lr(sb) = initial * (final/initial)^(sb/horizon)`である。SB1から補間が
進むため、1 SBあたりの実効倍率は`0.9920217112`となるが、SB367の終端は標準stepと
一致する。実効値は次のとおり。

| 点 | SB | LR |
|---|---:|---:|
| 開始 | 1 | 8.6801900e-4 |
| 約1 epoch | 37 | 6.5056637e-4 |
| 約2 epoch | 73 | 4.8758909e-4 |
| 約5 epoch | 183 | 2.0201493e-4 |
| 約8 epoch | 293 | 8.3697590e-5 |
| 約9 epoch | 330 | 6.2229528e-5 |
| 約10 epoch | 367 | 4.6267929e-5 |

10 epoch以降の resume では `--lr-final-superbatch 367` を明示し、LRを
`4.6267929105e-5` に保持する。raw checkpointにも horizon は保存されるが、CLIにも固定値を
書いて再現性を二重に確認する。warmup、cosine、one-cycle、WDL taperは追加しない。

## 7. 学習パラメータと基準 run との差分

NNUE Lab の公開 experiment
[`20260715-sfnn1536-kingrank9-tatara`](https://nnue-lab.sh11235.com/t/sh11235/experiments/01JTJ1J6LP022C3207CAC821EB)
とブログ掲載コマンドを現行 Tatara `main` の CLI と照合した。

### 7.1 同じ設定

| 項目 | 今回の値 | 根拠 |
|---|---:|---|
| feature set | `halfka-hm-merged` | 基準 run と同じ |
| batches / superbatch | `6104` | 基準 run と同じ |
| 初期 LR | `8.75e-4` | 基準 run と同じ |
| WDL | `0.3333333` 固定 | 教師スコア 2/3 + 勝敗 1/3、基準 run と同じ |
| loss | `--win-rate-model` | 基準 run と同じ |
| WRM | in scale 340、in offset 270、nnue2score 600、target offset 270、target scale 380 | 基準 run の記録値を明示固定 |
| loss power / asymmetry / boost | power `2.0`、asymmetry `0.0`、boost w1 `0.0` / w2 `0.5` | 基準 run と同じ |
| optimizer | `ranger`、weight decay `0` | 基準 run と同じ |
| FT factorizer | ON | 基準 run と同じ、現行 Tatara でも既定ON |
| GPU最適化 | `--all-optim` | 基準 run と同じ |
| validation | `floodgate.psv`、856,923局面 | 基準 run と同じ |
| dataloader threads | `30` を初期値 | 基準 run と同じ。Vast CPUでpreflight確認 |
| checkpoint | `--save-rate 20 --keep-checkpoints 2` | 基準 run と同じ |
| score drop / clamp | 指定なし | 基準 run と同じ |
| FV_SCALE | `28` | 基準 run と同じ。export/engine試験で明示確認 |

`--all-optim` は `tf32 / ft_fp16 / ft_fp16_out / fp16_opt_state` を有効化する。これは
Tatara の無指定既定値ではなく基準 run に合わせた明示的選択であり、T3で数値健全性を確認する。

### 7.2 異なる設定

| 項目 | NNUE Lab / ブログ | 今回 | 理由 |
|---|---|---|---|
| 教師 | tanuki 約80億局面 | SOJO 14,668,949,437局面 | ユーザー指定、公開SOJOのみ使用 |
| batch size | `16384` | `65536` | ユーザー指定 |
| optimizer更新数 | 約488,285 update/epoch | 約223,830 update/epoch | batch sizeを4倍にした結果 |
| architecture | `1536x16x32` | `1024x16x64` | ユーザー指定 |
| bucket | KingRank9 9 routing | legacy progress 8 routing / 9 slot（slot 8未使用） | 既存配布形式とYaneuraOu互換を維持 |
| progress | 不使用 | `keinoda/YaneuraOu` 基準の legacy `progress.bin` | progress routingに必要 |
| 学習量 | 160 SB、約2 epoch | 初回367 SB、約10 epoch。最大733 SB | ユーザー指定 |
| LR schedule | step、`gamma=0.992` | exponential、SB367で標準stepと同じLR | 標準減衰率を保って滑らかに下げる |
| validation頻度 | 毎SB、約80回/epoch | 毎SB、約36.7回/epoch | 1 SBの局面数が4倍 |
| checkpoint実効間隔 | 20 SB、約0.25 epoch | 20 SB、約0.545 epoch | flagは同じだが1 SBの局面数が4倍 |
| Tatara revision | generator `0.5.0`、`bfed43d-dirty` | official `main@da3ea68d` | current upstream exact commitを使用 |
| export | KingRank9 assertion付き9-slot converter | 9-slot writerは再利用しprogress assertionだけ追加 | 形式変更を避ける最小差分 |

教師、architecture、bucket、学習量は意図した差である。LRは基準runの標準
`gamma=0.992`を367 SBまで延長した終端値を使い、曲線形状だけをexponentialへ変える。
それ以外の学習ハイパーパラメータは基準runと同じにする。

### 7.3 初回10 epochコマンド

未確定なのは `RUN_NAME`、実ファイルpath、校正後 progressを採用するかだけである。

```bash
"$NNUE_TRAIN" \
  --data "$TRAIN_PSV" \
  --feature-set halfka-hm-merged \
  --batch-size 65536 \
  --batches-per-superbatch 6104 \
  --superbatches 367 \
  --lr 8.75e-4 \
  --lr-schedule exponential \
  --lr-final 4.6267929105e-5 \
  --lr-final-superbatch 367 \
  --wdl 0.3333333 \
  --win-rate-model \
  --wrm-in-scaling 340 \
  --wrm-in-offset 270 \
  --wrm-nnue2score 600 \
  --wrm-target-offset 270 \
  --wrm-target-scaling 380 \
  --loss-pow-exp 2.0 \
  --loss-qp-asymmetry 0 \
  --loss-weight-boost-w1 0 \
  --loss-weight-boost-w2 0.5 \
  --optimizer ranger \
  --weight-decay 0 \
  --ft-factorize \
  --all-optim \
  --test-data "$VALIDATION_PSV" \
  --test-positions 856923 \
  --threads 30 \
  --save-rate 20 \
  --keep-checkpoints 2 \
  --monitor-fp16-clamps \
  --monitor-active-features \
  --output "$RUN_ROOT/checkpoints" \
  --net-id "$RUN_NAME" \
  --experiment-name "$RUN_NAME" \
  layerstack \
  --ft-out 1024 \
  --l1 16 \
  --l2 64 \
  --bucket-mode progress8kpabs-legacy9 \
  --num-buckets 9 \
  --progress-coeff "$LEGACY_PROGRESS_BIN"
```

## 8. Vast.ai 上の run 構成

Gitから取得したこの学習専用folder自身を`EXPERIMENT_ROOT`とする。clone / pull先の絶対pathを
コードへ埋め込まず、launcherの所在からrootを解決する。dataset、survey、run、artifactは
必要なものだけをこのfolder直下へ置き、`/workspace/runs`のようなfolder外のrun rootは
作らない。

既存runと衝突しない`RUN_NAME`を確定し、次の単位で保存する。

```text
<EXPERIMENT_ROOT>/
├── PLAN.md
├── data/
│   ├── training/             # 完成済みshardと連結PSV
│   └── validation/           # floodgate PSV
├── progress/
│   ├── baseline/             # d77。上書き禁止
│   └── candidates/           # 承認前候補。別名保存
├── survey/
│   └── <survey-id>/          # seed、eligible shard、結果表、候補manifest
└── runs/<RUN_NAME>/
    ├── config/
    │   ├── command.txt
    │   ├── environment.txt
    │   ├── git-revisions.txt
    │   ├── dataset-manifest.txt
    │   └── progress-manifest.txt
    ├── checkpoints/
    │   └── experiments/<experiment-id>.json
    ├── logs/
    │   └── train.log
    ├── monitor/
    │   ├── report.html
    │   ├── metrics.json
    │   ├── render.log
    │   └── http.log
    └── artifacts/
        ├── selected-network.bin
        └── manifest.txt
```

`environment.txt` は秘密情報や環境変数の全 dump を保存せず、CUDA、driver、GPU、
Rust、Tatara commit、実行引数など再現に必要な非機密情報だけを allow-list で記録する。

### 8.1 環境変数

実装時に環境変数で渡すのは、instanceごとに変わる識別子とpathだけに限定する。

| 変数 | 必須 | 用途 |
|---|---:|---|
| `RUN_NAME` | yes | 新規run directory、tmux、experiment名。既存名との重複をpreflightで拒否 |
| `EXPERIMENT_ROOT` | no | launcherのdirectoryから自動解決。明示値を渡す場合も同じpathとの一致を検証 |
| `TRAIN_PSV` | yes | 公開教師30 shardを順番どおり連結したPSV |
| `VALIDATION_PSV` | yes | floodgate held-out PSV |
| `LEGACY_PROGRESS_BIN` | yes | survey後に確定した別名のlegacy係数path |
| `MONITOR_PORT` | no | 既定`6001`。既存listenerがあれば自動変更せず失敗 |
| `HF_TOKEN` | no | 公開dataset取得のrate limit対策時だけ。値をlog/manifestへ出さない |
| `BACKUP_REMOTE` / `RCLONE_CONFIG` | no | 手動Google Drive backup実行時だけ一時的に渡す。常駐設定にしない |

`BATCH_SIZE=65536`、`BATCHES_PER_SUPERBATCH=6104`、`TARGET_SB=367`、LR、network
dimensions、9-slot / 固定8 routingはこのrunの再現性contractであり、任意の環境変数
上書き対象にしない。
Vast.ai API key、GitHub token、SSH秘密鍵はcontainer内へ渡さない。`HF_TOKEN`を使う場合は
Vast.aiのinstance環境へsecretとして設定し、shell tracingを無効にしてから取得処理だけへ渡す。

## 9. 事前試験

以下を順に完了させる。失敗した場合は原因を特定するまで次へ進まない。

### T0: image / GPU / build preflight

1. `ghcr.io/keinoda/shogi-lab:cuda129-trt1011` で RTX 5090 を認識する。
2. CUDA、cuBLAS、`llc-21` 以上、`clang-21` 以上、Rust toolchain を確認する。
3. Tatara と CUDA kernel を exact commit から clean build する。
4. `nnue-train --help` で使用予定 flag を機械的に確認する。
5. image の不足が判明した場合だけ Docker image の変更案を作り、別 tag で build する。

### T1: 入力の完全性

1. SOJO 30 shard の個数、合計 bytes、各 40-byte record 境界を確認する。
2. 連結 PSV の bytes と局面数を確認する。再シャッフルしない。
3. floodgate PSV の bytes と局面数を確認する。
4. Tatara / YaneuraOu / `progress.bin` の commit、path、bytes、SHA-256を manifest 化する。
5. 入力 `progress.bin` が exactly 1,003,104 bytes で、125,388 個の有限な little-endian
   `f64` だけを含む legacy 形式であることを検証する。
6. SOJO の局順・`game_ply`・総手数を progress 教師ラベルに利用する処理がないことを
   実行コマンドとソース差分で確認する。

### T2: legacy 係数の校正可否 survey

この試験は NNUE 本学習と progress 再学習を開始せず、局面ごとの既存 progress 出力を
集計する read-only 試験とする。

1. checksum確認済みの完成shard一覧をsnapshotし、shard局面数に比例して重複なしの
   校正2,000,000局面と検証2,000,000局面をランダム抽出する。書込み中のpart fileは
   eligibleにせず、sampling seedとalgorithm versionを記録する。
2. 全 sample でactive index数を集計し、全合法局面が76であることを確認する。76以外は
   offsetを無効化して処理を続けず、入力recordまたはTatara/YaneuraOu間の特徴抽出差として
   preflightを失敗させる。
3. 現行係数で `p`、`z=logit(p)`、8 bucket 件数・比率を集計する。
   結果には提示済みの `d77` 参考比率も併記するが、測定条件不明のため一致を合格条件には
   しない。
4. 校正sampleだけで`a>0, b`を探索し、bucket 0の増加量とbucket 1–7の変化量が異なる
   Pareto候補を複数作る。検証sampleは候補選択に使わない。
5. 別名の候補について、bucket 0の増加percentage point・倍率、全bucket比率、bucket 1–7の
   最大絶対変化、total variation、旧新progressのSpearman順位相関、飽和率を表示する。
6. 結果をユーザーへ提示し、明示採用された候補だけを本学習へ渡す。自動採用せず、
   不採用なら現行SHA-256 `d77f47...30d8`を使う。

境界は常に `0.125, 0.250, ..., 0.875` である。経験分位点をファイルへ格納せず、
`PRGQ` trailer、engine側の境界テーブル、pseudo-target による係数学習は使わない。

### T3: CPU/GPU smoke

1. ごく小さい独立 sample と別 output directory で 3 batch / 1 superbatch を実行する。
2. `1024x16x64`、9 slot、固定8 bucket、legacy`0.125`routingの実効設定を起動ログで
   確認し、bucket histogramのslot 8が0件であることを確認する。
3. loss、test loss、LR、active feature、FP16 clamp が有限値であることを確認する。
4. `--all-optim` の有無だけを変えた短い比較で、loss挙動と pos/s を記録する。
5. CPU thread 数は同一 sample で比較し、GPU starvation がなく最も安定する値を選ぶ。

### T4: checkpoint / resume drill

1. 別の試験 run を checkpoint まで正常終了させる。
2. raw checkpoint から同一引数で resume する。
3. superbatch番号、optimizer state、LR、legacy progress SHA-256、experiment JSONの
   継続性を確認する。
4. 既存出力を上書きせず、試験 run の lineage が追えることを確認する。

### T5: export / engine end-to-end

1. 試験checkpointを9-slot、`1024x16x64`のYaneuraOu networkへ変換する。converterには
   `--assume-progress8kpabs`を明示し、KingRank9として偽装しない。
2. 変換 manifest に converter commit、入力 checkpoint、出力 SHA-256、dimensions、
   bucket mode、legacy progress SHA-256を記録する。
3. 既存の progress 計算を使う YaneuraOu で `isready`、固定局面、固定 nodes の探索を
   実行する。分位点対応 branch は要求しない。
4. 同一のlegacy `progress.bin`と`0.125`刻みで、Tatara surveyとYaneuraOuが同じ
   bucket 0–7を返すことを、通常局面と7境界の前後で照合する。
5. Tatara training / validation / evalとYaneuraOuのいずれもslot 8を返さないこと、
   smoke checkpointでslot 8のweight・optimizer stateが初期値から更新されないことを確認する。
6. 既存KingRank9 fixtureのconverter出力が変更前とbyte一致することを確認する。

### T6: monitor acceptance

1. fixture の experiment JSON と train log だけで report を生成する。
2. renderer が学習プロセスへ signal や write を行わないことを確認する。
3. `127.0.0.1:6001` で表示確認後、`0.0.0.0:6001` に bind する。
4. `ss -ltnp`、instance 内 `curl`、Vast.ai の mapped URL の順に readback する。
5. report に path、token、環境変数、SSH情報などの秘密がないことを確認する。

## 10. 本学習の実行

本学習は事前試験とは別の新規 `RUN_NAME` で、次の順に手動開始する。

1. 未確定パラメータを全て固定し、実行コマンドと manifest を先に保存する。
2. 同名 run directory、同名 tmux、port 6001 の既存 listener がないことを確認する。
3. monitor renderer と HTTP server を学習とは別 tmux session / window で起動する。
4. monitor の local / mapped readback 後に trainer を別 tmux sessionで起動する。
5. 最初の superbatch が終わるまで train log、experiment JSON、GPU を同時に観測する。
6. 確定した換算表の 1 / 2 / 5 / 10 / 12 / 14 / 16 / 18 / 20 epoch 相当SBで
   snapshotを残す。

tmux の候補名:

```text
train-<RUN_NAME>
monitor-<RUN_NAME>  # render と http を別 window / PID / log にする
```

## 11. 監視ページ

監視 renderer は `checkpoints/experiments/*.json`、`logs/train.log`、`nvidia-smi`、
OS process情報を読み取り、`monitor/` 内だけへ temp file + atomic renameで書く。
trainerの file、PID、tmux、checkpointには書き込まない。

表示項目:

- 現在 / 最大 superbatch、換算 epoch、進捗率、直近 SB 時間、ETA
- train loss、floodgate test loss、test accuracy、WDL、LR の推移
- best test loss とその superbatch、直近傾向、train-test gap
- pos/s、GPU utilization / temperature / VRAM / power
- CPU、RAM、disk 空き容量
- raw / quantized checkpoint 一覧と最新時刻
- trainer / renderer / HTTP server の PID、tmux、終了コード、最終更新時刻
- experiment JSON の `status`、Tatara commit、dataset / progress manifest識別子
- FP16 clamp と active feature の観測値

更新間隔は 30 秒を候補とする。stale 判定は固定値だけにせず、最初の数 SB から得た
直近 SB 時間を使い、`max(直近SB時間の3倍, 15分)` 更新がなければ警告する。
警告は表示のみで、trainerを操作しない。

`0.0.0.0:6001` は公開面になるため、HTTP server は `monitor/` directoryだけを配信し、
directory listingを無効にする。認証が必要なら Vast.ai 側の公開方法を確認して別途決める。

## 12. 判定・停止・再開

### 即時に人へ通知し、勝手に再開しない条件

- trainer の異常終了
- loss / test loss / LR の NaN または無限大
- CUDA error、Xid、OOM、I/O error
- dataset / checkpoint / experiment JSON の読取・書込エラー
- checkpoint 1本と一時書込みに必要な容量を下回る disk 空き
- progress manifest、legacyファイルサイズ、SHA-256、固定境界が起動時の確定値と一致しない

### 品質の判定点

10 epoch 到達後、12 / 14 / 16 / 18 / 20 epoch の各点で次を比較する。

1. 直近区間の最良 floodgate test loss が更新されたか
2. test loss の平滑化傾向が悪化していないか
3. train-test gap が継続拡大していないか
4. test accuracy と数値健全性に悪化がないか
5. checkpointから変換・起動できるか

「改善が何 epoch 止まれば終了するか」「どの改善幅を有意とするか」は未確定とし、
短い 2 epoch の公開例から推測しない。判定時に最良 checkpointを選ぶが、trainerの停止は
ユーザー確認後に手動で行う。

### resume

resume は raw `.ckpt` だけを使い、元 run と同じ architecture、feature set、legacy
progress SHA-256、固定 `0.125` 境界、optimizer、loss、batch設定を渡す。
延長先が `440 / 513 / 587 / 660 / 733` のいずれでも
`--lr-final-superbatch 367` を明示し、checkpoint内の保存 horizon と一致させる。
自動 resumeは行わず、checkpoint SHA-256と前回終了理由を確認してから手動実行する。

## 13. backup

通常時は`EXPERIMENT_ROOT`直下だけに保存し、定期的な外部backupは自動実行しない。
SOJO/floodgate原本、target build、全中間checkpointは再取得・再生成可能なのでGoogle Driveへ
転送しない。外部backupが必要と判断された時だけ、次を手動対象にする。

- `PLAN.md`、launcher/config、manifests
- surveyのseed・eligible shard一覧・集計表・採用候補`progress.bin`
- experiment JSONとlogs
- 10 epoch以降の判定点または選定済みraw checkpoint
- 選定済みquantized / YaneuraOu networkとmonitor snapshot

手動backup時はローカルの`rclone.conf`をinstanceへ一時転送し、mode `0600`の専用pathを
`rclone --config`で明示する。instanceからGoogle Driveへ直接`rclone copy`し、転送元と転送先の
bytes / SHA-256または`rclone check`を照合する。設定内容をlog、manifest、Google Driveへ
含めない。転送先pathと一時`rclone.conf`の後処理は実行直前に提示し、ユーザー確認後に行う。

実施時点は固定周期ではなく、ユーザーが必要と判断した時、特にinstanceのstop/destroy前と
採用network確定時を候補とする。自動timer、学習プロセスからのbackup起動は実装しない。

## 14. 実装・実行ゲート

networkは`1024x16x64`で確定した。preflight実装は次の順序で行う。

1. `progress8kpabs-legacy9`の追加routingとCPU test
2. training / validation / eval / resume / experiment JSONへのmode伝播test
3. `net_to_yo --assume-progress8kpabs`の最小追加と既存KingRank9 byte regression test
4. `1024x16x64 / 9-slot` converter fixtureとYaneuraOu `ls9` build/load test
5. download済みSOJO shardによる400万局面surveyと候補提示
6. ユーザーによる`progress.bin`候補の採否
7. GPU smoke、resume drill、monitor acceptance
8. 初回10 epoch本学習

各gateが失敗した場合は後段へ進まない。特に5–6が完了するまで本学習を開始せず、
8-slot形式、slot 8の学習、分位点trailerへ自動で切り替えない。
