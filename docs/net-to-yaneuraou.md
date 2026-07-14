# YaneuraOu 用 LayerStack net 変換

`net_to_yo` は tatara の `HalfKaHmMerged` LayerStack `.bin` を YaneuraOu の
SFNN 評価ファイルへ変換する。層次元 (`ft_out` / `l1` / `l2`) と bucket 数は
入力 `.bin` の header (arch 文字列 + `num_buckets` field) から自動検出されるため、
既定の 1536-16-32 以外 (例: `--ft-out 3072 --l1 16 --l2 64` で学習した
3072-16-64) も変換できる。

```bash
# kingrank9 で学習した net (9 bucket、素の YaneuraOu の KingRank9 ルーティング)
cargo run --release -p net-to-yo -- \
  --input /path/to/tatara.bin \
  --output /path/to/eval/nn.bin \
  --assume-kingrank9

# progress8kpabs で学習した net (progress ルーティング実装済み YaneuraOu 向け)
cargo run --release -p net-to-yo -- \
  --input /path/to/tatara.bin \
  --output /path/to/eval/nn.bin \
  --assume-progress8kpabs
```

変換対象は PSQT、Threat、EffectBucket を持たない `HalfKaHmMerged` に限定される。
feature set、追加 block の有無が一致しない入力はエラーになる。層次元と
bucket 数は検出値をそのまま使い、weights 長の整合を検証する。

量子化 `.bin` は bucket routing mode を記録しないため、変換前に学習時の
`--bucket-mode` を確認し、以下のどちらか一方を明示する:

- `--assume-kingrank9`: `--bucket-mode kingrank9` で学習した net。bucket 数は
  9 固定 (YaneuraOu の KingRank9 ルーティングの前提)。素の YaneuraOu で動く。
- `--assume-progress8kpabs`: 既定の `--bucket-mode progress8kpabs` で学習した
  net。**変換先の YaneuraOu が progress bucket ルーティング
  (`Tanuki::Progress`) を実装しており、学習時と同一の `progress.bin` を配備
  していることが前提**。素の YaneuraOu では bucket 選択規則が異なり正しく
  動かない。

いずれの場合も、変換先の YaneuraOu は**入力と同じ層次元でビルドされた engine**
(`nnue_arch_gen.py` で該当次元の architecture header を生成してビルドしたもの)
が必要。YaneuraOu の loader は header の version / hash / arch 文字列の不一致を
警告のみで許容するが、weights の byte layout は次元で決まるため engine 側次元が
違うと正しく読めない。

architecture 文字列は 1536-16-32 / 9 bucket のとき従来と同一の
`Network=SFNN-1536-V2{LayerStack=9}` を書き (byte 互換)、それ以外の次元では
`Network=SFNN-<ft_out>-<l1-1>-<l2>-V2{LayerStack=<N>}` 形式で次元を明示する
(例: `SFNN-3072-15-64-V2`。`<l1-1>` は skip 分離後の main 次元で、YaneuraOu の
`nnue_arch_gen.py` の表記に合わせている)。

YaneuraOu の SFNN loader は FT の bias と weight をそれぞれ signed
LEB128 block として読み、dense 層は bias の i32 LE、続いて canonical row-major の
i8 weight を読む。dense weight は YaneuraOu がロード時に実行用 SIMD layout へ並べ
替えるため、変換ファイルには並べ替え前の順序で格納する (入力次元は 32 の倍数へ
ゼロパディング)。

量子化 scale は両形式とも FT が QA=127、dense weight が QB=64、dense bias が
QA×QB=8128 であり、変換時の scale 変更は行わない。architecture string に
`fv_scale` は含めない。YaneuraOu では読み込み前に `setoption name FV_SCALE value 28`
を指定する。
