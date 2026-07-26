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
| baseline progress | SHA-256 `d77f47e874558d42fa2d87d173de3aba054eef51bcca9c1fc9f3a8daf93630d8` |

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

## progress係数の決定

教師が変わるため、前回採用値を自動流用しない。`split_000.bin`からseed固定で
400万局面を一度だけ選び、次の3集合を固定する。

| 集合 | 局面数 | 用途 |
|---|---:|---|
| calibration | 2,000,000 | affine係数`a,b`の最適化 |
| selection | 1,000,000 | 候補比較 |
| final-test | 1,000,000 | 採用前の独立確認 |

比較対象は次の3種類とする。

- baseline `d77f47...`
- 前回教師で選んだ`a=1.2980837735881936`、
  `b=-0.5975424282106219`
- 今回のcalibration集合で新たに最適化した候補

新候補は元progress値との単調性を保つ`a > 0`のaffine変換とし、目標分布は
`11,12,13,14,14,13,12,11%`とする。baseline・候補ごとにselectionと
final-testのbucket割合、境界fixture、係数、生成物SHA-256を提示する。
scriptは採用を行わず、ユーザーが候補名を明示して承認manifestを作る。

## 学習設定

| 項目 | 値 |
|---|---|
| feature | `halfka-hm-merged` |
| network | LayerStack `2304x16x64` |
| training buckets | 8 |
| bucket mode | `progress8kpabs` |
| batch size | 65,536 |
| batches / superbatch | 6,104 |
| 初回 | 421 superbatch |
| 学習局面数 | 168,413,364,224 |
| 実epoch | 約10.009678 |
| LR | step、start `8.75e-4`、gamma `0.992`、every 1 SB |
| loss | WRM、WDL `0.3333333`、既存WRM係数を維持 |
| optimizer | Ranger、weight decay 0 |
| precision | `--all-optim` |
| threads | 16 |
| checkpoint | 20 SBごと、rawは直近2個 |
| validation | floodgate 851,968局面を毎SB評価 |
| FV scale | 28 |

421 SBは`65536 × 6104 × 421 / 16,825,052,646`から求める。367 SBでは
約8.73 epochにしかならないため、教師総量に合わせて421 SBとする。
LRの方式・開始値・gammaは変えない。

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

延長が必要な場合は、初回runを変更せず別runへresumeする。候補は約12、14、16、
18、20 epochに対応する505、589、673、757、841 SBとする。progress.bin、
教師SHA-256、precision、threadsは親runと一致させる。

## 自動化しない操作

- progress係数の採用
- 本学習開始
- 学習延長
- checkpoint選択
- 外部backup
- instanceの停止・再起動
