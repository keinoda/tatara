# 決定台帳

| 項目 | 決定 | 根拠 |
|---|---|---|
| project | `progress8kpabs-2304x16x64` | bucket方式とarchitectureを名前で表す |
| branch | `codex/progress8kpabs-2304x16x64-training` | 既存1024学習と分離する |
| base | `codex/progress-legacy9-1024x16x64-training` | 実運用済みのsurvey・gate・monitorを引き継ぐ |
| 8ek | 使用しない | 効果がなく、通常のprogress8kpabs係数調整が本質 |
| 教師 | `sashimin/test20260726` | ユーザー指定の公開教師 |
| 教師revision | `8f461dd8dc4cb90c356392545a41e4e45c8f2418` | 取得内容を固定する |
| 教師規模 | 34 shard、673,002,105,840 bytes | Hugging Faceの固定revision |
| 保存 | shardごとに検証して単一PSV末尾へ追記 | shard全体と連結PSVの二重保持を避ける |
| 中断復旧 | shard境界markerまで切り戻す | 未確定の部分追記だけを再取得可能にする |
| survey入力 | `split_000.bin`を一時保持 | 全download完了前に400万局面を固定できる |
| survey shard削除 | progress承認後、本学習前 | 20GBを解放し、元shardを残さない |
| volume | 1000GB `/workspace` | 約813GBの保守的preflightを満たす |
| architecture | HalfKA_hm merged `2304x16x64` | ユーザー指定、既存runtime可変幅で対応済み |
| training buckets | 8 | `progress8kpabs`の既存動作を維持 |
| export slots | 9 | bucket 7をslot 8へ複製する既存形式 |
| YaneuraOu repository | private `keinoda/YaneuraOu-private` | engineとprogress.binの正本 |
| engine commit | `771fe811f877859d6851ceccfd3e04c16454e689` | 既存の変換・読込試験条件を維持 |
| 既存改造版progress | commit `35752abe3035cb972ecfb98b1ce197028625c250`、SHA-256 `e7ed0eef...` | 採用済み配布評価関数と同一fileを固定 |
| progress survey | 400万局面で既存fileだけを評価 | 通常手順では候補生成・係数最適化を行わない |
| progress再調整 | 分布を提示後、ユーザーが大きな崩れと判断した場合だけ別途計画 | 数値閾値や自動判定を新設しない |
| private認証 | fine-grained tokenのContents readだけを一時利用 | tokenをGit remote・manifest・logへ保存しない |
| batch size | 65,536 | 既存設定を維持 |
| batches / SB | 6,104 | 既存設定を維持 |
| 初回学習量 | 841 SB、約19.995581 epoch | 新教師の局面数に対する約20周 |
| LR | step、`8.75e-4`、gamma `0.992`、step 1 | Tatara標準の既存設定を維持 |
| precision | `--all-optim` | 既存判断を維持 |
| threads | 16 | Ryzen 9 9950Xの物理16 coreに合わせる |
| validation | floodgate 851,968局面 | 既存比較系列を維持 |
| checkpoint | save 20、raw keep 2 | 容量と復旧性を両立する |
| container | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011`固定digest | 既存環境を維持 |
| backup | 自動実装しない | 必要時だけ外部rcloneを手動設定する |
