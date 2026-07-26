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
| baseline progress | SHA-256 `d77f47...` | 比較の基準 |
| 前回候補 | `a=1.2980837735881936`、`b=-0.5975424282106219` | 新教師での比較対象に限る |
| 新候補目標 | `11,12,13,14,14,13,12,11%` | 中央をやや厚くする合意済み目標 |
| 係数採用 | survey提示後に手動承認 | 自動採用しない |
| batch size | 65,536 | 既存設定を維持 |
| batches / SB | 6,104 | 既存設定を維持 |
| 初回学習量 | 421 SB、約10.009678 epoch | 新教師の局面数から再計算 |
| LR | step、`8.75e-4`、gamma `0.992`、step 1 | Tatara標準の既存設定を維持 |
| precision | `--all-optim` | 既存判断を維持 |
| threads | 16 | Ryzen 9 9950Xの物理16 coreに合わせる |
| validation | floodgate 851,968局面 | 既存比較系列を維持 |
| checkpoint | save 20、raw keep 2 | 容量と復旧性を両立する |
| container | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011`固定digest | 既存環境を維持 |
| backup | 自動実装しない | 必要時だけ外部rcloneを手動設定する |
