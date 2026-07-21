# 公開教師によるprogress fixed8学習計画

## 目的

RTX 5090 1枚で、公開済み・shuffle済みの
`washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique`だけを使い、
`1024x16x64` LayerStackを学習する。Tataraでは既存の
`progress8kpabs --num-buckets 8`をそのまま使い、YaneuraOuへのexport時だけ
bucket 7を未使用のslot 8へ複製する。新しいbinning形式や9 bucket学習は導入しない。

基準`progress.bin`でbucket 0が薄い問題は、教師の手数・総手数を復元せず、既存出力に
`z' = a*z+b`を適用できる明示候補だけをsurveyする。分布とbucket移動をユーザーへ提示し、
承認された係数だけを使う。候補の自動探索・自動採用は行わない。

## 固定する実験条件

| 項目 | 値 |
|---|---|
| 教師 | 公開30 shard、586,757,977,480 bytes、14,668,949,437局面 |
| 教師順序 | file名順に連結。再shuffleしない |
| validation | floodgate 856,923局面中、full batch 851,968局面を使用 |
| network | LayerStack 1024x16x64 |
| routing | Tatara 8 bucket、固定境界0.125刻み |
| export | YaneuraOu 9 slot、slot 8はbucket 7の複製 |
| batch size | 65,536 |
| batches / superbatch | 6,104 |
| precision | `--all-optim` |
| worker threads | 16 |
| 初回 | 367 superbatch、約10.0083 epoch |
| LR | step、start 0.000875、gamma 0.992、every 1 superbatch |
| loss | WRM、既存Tatara基準値を維持 |
| checkpoint | 20 SBごと、rawは直近2本、量子化binは保持 |
| backup | 自動実行しない |
| recovery | 自動resume・自動再起動・自動stopを行わない |

precisionは`--all-optim`、worker thread数はAMD Ryzen 9 9950Xの物理コア数に合わせて16に固定する。
T3ではこの組み合わせだけをsmoke実行し、選択や再承認は行わない。disk上の教師順序は変えないが、
workerが2以上なのでoptimizerへ届く順序は非決定的であり、bit単位の再現性は持たない。

## 実行gate

| Gate | 内容 | 実行入口 | 完了条件 |
|---|---|---|---|
| T0 | 固定revision、image、RTX 5090、build | `onstart.sh` | `build_tatara.done`、`build_rshogi.done` |
| T1 | 30 shard、連結PSV、validation、SHA-256 | `onstart.sh` | `prepare_data.done` |
| T2 | 重複なし400万局面survey | `run-survey.sh` | `metrics.json`を提示し、`approve-progress.sh`で承認 |
| T3 | all-optim、16 threadの固定smoke | `run-smoke.sh` | run完走・有限metricを検証し、固定値manifestを生成 |
| T4 | raw checkpoint true resume | `run-resume-drill.sh` | SB1からSB2へのlineageとhistoryを検証 |
| T5 | 8→9 export、YaneuraOu load/search | `run-export-test.sh` | startposと14境界fixtureが完走 |
| T6 | 独立monitor、Basic認証、readback | `run-monitor.sh` | local/public readbackと未認証401 |

`run-training.sh`はT0–T6、係数承認、データSHA-256、monitor生存を再検証する。いずれかが欠ければ
本学習を開始しない。

## Surveyの評価対象

- 各splitのbucket 0–7件数と比率
- baselineからのmigration matrix
- 7境界それぞれのcrossing率
- total variation
- 低端・高端の飽和件数
- active KP-absolute indexが全局面76であること
- 各境界の直下・直上に最も近いfixture

samplingはglobal record index上で決定的に行い、重複なしで抽出した後、file offset順に読んで
random I/Oを抑える。calibration・selection・final-testの件数とseedは実行時に明示し、3集合の
合計を4,000,000とする。final-testを見た後に候補を再調整する場合、その結果を未使用testとは
扱わず、新しいsurvey IDでやり直す。

## 本学習後の判定

初回367 SBを完走した後、毎SBのvalidation履歴を確認する。選択対象は「実際に保存されている
量子化checkpointのうちtest lossが最小のもの」とし、毎SBの最小点に対応するfileがない場合は
選ばない。延長する場合だけ、raw checkpointのSHA-256と親runを確認し、別runとして
440 / 513 / 587 / 660 / 733 SBのいずれかへ手動resumeする。

## 対象外

- 個人データ、非公開データ、別datasetの混在
- 教師の再shuffle
- progressの再学習、手数や総手数の推測
- 分位点table、trailer、9 bucket routing
- gate結果に基づく係数の自動採用
- 学習中の自動stop、障害時の自動resume
- 自動Google Drive backup

確定値と未確定値は[DECISIONS.md](DECISIONS.md)、操作手順は[RUNBOOK.md](RUNBOOK.md)を正とする。
