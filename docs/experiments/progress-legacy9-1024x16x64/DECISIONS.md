# 決定台帳

## 確定

| 項目 | 決定 | 根拠・扱い |
|---|---|---|
| source branch | `codex/progress-legacy9-1024x16x64-training` | 学習専用branch |
| Vast起動 | Web UIのOn-start Scriptで自動tmuxを停止し、Git cloneで`onstart.sh`を取得 | CLIからinstanceを作らず、`onstart.sh`本文をVast設定へ貼らない |
| checkout | branch先端の40桁SHAを`TATARA_COMMIT`で固定しdetached checkout | branch移動の影響を受けない |
| upstream | `SH11235/tatara@da3ea68d46a5c1ac0c18c10a57fef52d02788879` | 専用commitがこのcommitを含むことを検証 |
| rshogi | `29245a1d8e4f198aba3fc832a506649221cb2f2c` | converter tool buildを固定。`--no-default-features --features nnue-arch`でbuild |
| YaneuraOu | `771fe811f877859d6851ceccfd3e04c16454e689` | progress.binとengine testを固定 |
| image | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011@sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876` | tagだけでなくindex digestを固定 |
| 教師revision | `5da309f4de4091cfb004eff94da97d49e3268aa2` | 公開30 shardのみ |
| validation revision | `fdd5f602db82d888a87116f087d10dd5ea8313ab` | floodgate固定snapshot |
| 教師順序 | file名順で連結、再shuffleなし | 配布済みshuffleを維持 |
| validation | `--test-positions 851968` | 65536×13 full batches。末尾4,955局面は使わない |
| bucket | 学習8 bucket、export 9 slot、slot 8未使用 | 既存engine形式を維持 |
| architecture | 1024x16x64 | ユーザー指定 |
| 学習量 | 65536 × 6104 × 367 | 約10.0083 epoch |
| LR | step 0.000875、gamma 0.992、step 1 | Tatara標準の減衰規則 |
| precision | `--all-optim` | SH11235の公開運用例を踏まえたユーザー指定 |
| worker threads | 16 | AMD Ryzen 9 9950Xの物理16コアに合わせる |
| survey入力 | 開始時点の取得完了済み公開shardを固定 | 1 shard以上かつ合計400万局面以上。通常は教師読込み1回で最適化まで完了 |
| survey split | calibration 2,000,000、selection 1,000,000、final-test 1,000,000 | 合計400万局面 |
| survey seed | `20260721` | 重複なしglobal index samplingを固定 |
| affine optimizer | calibrationで明示目標比率からのMSEを最小化。省略時は12.5%ずつ | `a > 0`、257×257全域格子＋6回絞込み、PSV再読込みなし |
| 中央厚め比較target | `11,12,13,14,14,13,12,11%` | bucket 0を薄く戻さず、中央を穏やかに厚くする比較条件。採用値ではない |
| survey採用 | 最適化候補とbaselineの3 split結果を提示し、採用はユーザー判断 | 係数候補は生成するが自動採用なし |
| monitor | port 6001、Basic認証、2 routeだけ配信 | directory listingと無認証公開をしない |
| checkpoint選択 | 保存済み`.bin`内の最小test lossを報告 | 自動採用しない |
| backup | 自動backupなし | 必要時だけ一時rclone.confで手動実行 |
| restart | 自動resume・自動stopなし | 異常時は状態を保存して人へ提示 |

## 実測後に決める

| 項目 | 決める時点 | 記録先 |
|---|---|---|
| 使用progress.bin | 最適化候補とbaselineの3 split結果を提示後 | `progress/approved/*.txt` |
| `RUN_NAME` | 各run開始前 | run manifest |
| monitor URLと認証情報 | T6直前 | credentialsはmanifestへ保存しない |
| 10 epoch後の延長 | 367 SB完走後 | 新しいresume run manifest |
| 最終network | 保存済みcheckpoint比較後 | 手動選択artifact |

未確定のrun名や承認対象をscriptの既定値で埋めない。入力がなければ開始前に失敗させる。
