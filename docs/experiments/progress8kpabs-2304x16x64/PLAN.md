# 公開教師によるprogress8kpabs 2304x16x64学習計画

## 目的

公開教師局面を使い、HalfKA_hm mergedのLayerStack `2304x16x64`を学習する。
局面bucketは通常の`progress8kpabs` 8分割を維持し、相入玉専用bucketなどの
追加ルーティングは使わない。

2304はTataraの既存runtime可変FT幅と`net_to_yo`の対応範囲内であるため、
trainer本体へアーキテクチャ専用コードは追加しない。

## 固定する入力

| 項目 | 値 |
|---|---|
| 教師 | `sashimin/test20260726` |
| revision | `8f461dd8dc4cb90c356392545a41e4e45c8f2418` |
| license | MIT |
| shard | `split_000.bin`から`split_033.bin`までの34 file |
| 合計 | 673,002,105,840 bytes、16,825,052,646局面 |
| record | PackedSfenValue、40 bytes |
| validation | `takaoyamaoka/floodgate.hcpe`固定revision |
| YaneuraOu | private `keinoda/YaneuraOu-private`、engine commit `771fe811f877859d6851ceccfd3e04c16454e689` |
| 既存改造版progress | commit `35752abe3035cb972ecfb98b1ce197028625c250`、SHA-256 `e7ed0eef88868335f9a46c58a121dccb5ad82a5eb1c8ee12de90365ab351e37d` |

Dataset Cardには生成方法・PSV形式・shuffle方法の説明がない。取得前の監査では
`split_000.bin`先頭10万局面をTataraでdecodeし、40-byte境界、score、結果、
非連続な手数順を確認した。実行時は固定revisionに加えて、各shardのbyte数と
LFS SHA-256を`training-shards.tsv`で照合する。

## 教師データの保存方法

34 shard全部と連結PSVを同時に保存しない。

1. shardを番号順に1個だけ取得する。
2. 固定byte数、40-byte境界、SHA-256を検証する。
3. `public-teacher.psv.partial`の末尾へ追記して`fsync`する。
4. 追記後の累積byte数をshard単位のmarkerへatomicに記録する。
5. `split_000.bin`以外の元shardを削除して次へ進む。
6. 全34 shardの確定後、`.partial`を`public-teacher.psv`へrenameする。

中断後はmarkerが示す最後の確定境界を正とし、それより後ろの未確定末尾だけを
切り戻して同じshardから再開する。確定済みprefixや完成PSVは作り直さない。

`split_000.bin`は、全download完了を待たずに400万局面surveyを行うため一時保持する。
progress係数の採用後、専用cleanupを実行してから本学習を開始する。

容量preflightは、完成PSV 673GB、処理中shardの重複分を保守的に40GB、
checkpoints等の余裕100GBとして約813GBを要求する。Vast.aiの`/workspace`は
1000GBを指定する。

## progress係数の確認

採用済みの改造版`progress.bin`を固定入力とし、`split_000.bin`からseed固定で
400万局面を一度だけ選んで分布を確認する。通常手順ではaffine候補の生成、
係数最適化、`progress.bin`の再作成を行わない。

| 集合 | 局面数 | 用途 |
|---|---:|---|
| primary | 2,000,000 | 主分布の確認 |
| confirmation-a | 1,000,000 | 独立sampleでの再確認 |
| confirmation-b | 1,000,000 | 独立sampleでの再確認 |

3集合のbucket割合と境界fixtureを提示する。分布が大きく崩れたと判定する
数値閾値は自動設定しない。ユーザーが結果を確認し、大きな崩れがないと判断した場合は、
同じSHA-256の既存改造版を承認する。大きな崩れがある場合もsurvey scriptは
再調整せず、別途方針を決めるまで停止する。

## 学習設定

| 項目 | 値 |
|---|---|
| feature | `halfka-hm-merged` |
| network | LayerStack `2304x16x64` |
| training buckets | 8 |
| bucket mode | `progress8kpabs` |
| batch size | 65,536 |
| batches / superbatch | 6,104 |
| 初回 | 841 superbatch |
| 学習局面数 | 336,426,696,704 |
| 実epoch | 約19.995581 |
| LR | step、start `8.75e-4`、gamma `0.992`、every 1 SB |
| loss | WRM、WDL `0.3333333`、既存WRM係数を維持 |
| optimizer | Ranger、weight decay 0 |
| precision | `--all-optim` |
| threads | 16 |
| checkpoint | 20 SBごと、rawは直近2個 |
| validation | floodgate 851,968局面を毎SB評価 |
| FV scale | 28 |

841 SBは`65536 × 6104 × 841 / 16,825,052,646`から求めた20 epochへの
最寄りの整数SBである。LRの方式・開始値・gammaは変えない。

## 本学習前gate

同じ`RUN_NAME`と承認済みprogress.binで次を完了させる。

- 2304x16x64、all-optim、16 threadsのGPU smoke
- raw checkpointのSB1からSB2へのresume
- `net_to_yo --assume-progress8kpabs`による8 bucketから9 slotへの変換
- YaneuraOu `halfkahm2_2304_15_64_ls9`でのnetwork読込
- startposと7境界上下14局面の固定node探索
- monitorのlocal・mapped readbackとBasic認証
- survey用`split_000.bin`の明示cleanup

exportでは既存形式を維持し、training bucket 7を未使用の第9 slotへ複製する。
8ekや相入玉専用振り分けは導入しない。

## 完走後

保存済みcheckpointだけを候補にし、毎SBのtrain loss、test loss、test accuracy、
fp16 clamp、active feature監視を確認する。自動でbestを採用しない。

延長が必要な場合は、初回runを変更せず別runへresumeする。目標SBは自動決定せず、
progress.bin、教師SHA-256、precision、threadsを親runと一致させる。

## 自動化しない操作

- progress係数の採用
- 本学習開始
- 学習延長
- checkpoint選択
- 外部backup
- instanceの停止・再起動
