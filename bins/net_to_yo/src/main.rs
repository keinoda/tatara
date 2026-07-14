use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::LayerStackWeights;
use nnue_format::layerstack_weights::{
    LEGACY_NNUE_VERSION_BUCKETS9, NNUE_VERSION, QA, QB, write_leb128_tensor_i16,
};
use shogi_features::FeatureSet;

const YO_VERSION: u32 = 0x7af3_2f16;
const YO_TOP_HASH: u32 = 0x3c20_3b32;
const YO_FT_HASH: u32 = 0x5f13_4ab8;
const YO_NETWORK_HASH: u32 = 0x6333_718a;

/// 入力 `.bin` の header (arch 文字列 + num_buckets field) から検出した LayerStack 次元。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NetDims {
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    num_buckets: usize,
}

impl NetDims {
    fn l2_in(&self) -> usize {
        (self.l1_out - 1) * 2
    }
}

/// YaneuraOu 側 header に書く architecture 文字列。
///
/// 既定次元 (1536-16-32, 9 bucket) は従来の固定文字列と byte 一致を維持する。
/// それ以外の次元は `SFNN-<ft_out>-<l1_out-1>-<l2_out>-V2` 形式で次元を明示する
/// (YaneuraOu 側 loader は arch 文字列不一致を警告のみで許容する。
/// `nnue_arch_gen.py` の生成 header と表記を揃えるのは YO 側対応時に確定する)。
fn yo_arch_string(ft_in: usize, dims: &NetDims) -> String {
    let network = if (dims.ft_out, dims.l1_out, dims.l2_out) == (1536, 16, 32) {
        "SFNN-1536-V2".to_string()
    } else {
        format!(
            "SFNN-{}-{}-{}-V2",
            dims.ft_out,
            dims.l1_out - 1,
            dims.l2_out
        )
    };
    format!(
        "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[{}->{}x2],Network={}{{LayerStack={}}}",
        ft_in, dims.ft_out, network, dims.num_buckets
    )
}

#[derive(Parser)]
#[command(about = "Convert a tatara LayerStack net for YaneuraOu SFNN (dims auto-detected)")]
struct Args {
    /// tatara LayerStack quantised .bin
    #[arg(long)]
    input: PathBuf,
    /// YaneuraOu nn.bin
    #[arg(long)]
    output: PathBuf,
    /// Assert that the input was trained with `--bucket-mode kingrank9`
    /// (9 buckets; the stock YaneuraOu KingRank9 routing applies).
    /// Quantised `.bin` files do not record their bucket routing mode.
    #[arg(long, conflicts_with = "assume_progress8kpabs")]
    assume_kingrank9: bool,
    /// Assert that the input was trained with `--bucket-mode progress8kpabs`.
    /// The target YaneuraOu must implement progress bucket routing
    /// (Tanuki::Progress) and be deployed with the same progress.bin that was
    /// used for training.
    #[arg(long)]
    assume_progress8kpabs: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output {
        return Err("input and output must be different paths".into());
    }

    let input = File::open(&args.input)?;
    let mut reader = BufReader::new(input);
    let dims = detect_dims(&mut reader)?;
    eprintln!(
        "detected dims: ft_out={} l1_out={} l2_out={} num_buckets={}",
        dims.ft_out, dims.l1_out, dims.l2_out, dims.num_buckets
    );
    require_bucket_mode_assertion(args.assume_kingrank9, args.assume_progress8kpabs, &dims)?;

    reader.rewind()?;
    let weights = LayerStackWeights::load_quantised(
        &mut reader,
        FeatureSet::HalfKaHmMerged.spec(),
        dims.ft_out,
        dims.l1_out,
        dims.l2_out,
        dims.num_buckets,
    )?;
    reject_trailing_data(&mut reader, dims.num_buckets)?;

    let output = File::create(&args.output)?;
    let mut writer = BufWriter::new(output);
    write_yo(&mut writer, &weights, &dims)?;
    writer.flush()?;
    Ok(())
}

/// 入力 `.bin` の header だけを読み、arch 文字列と `num_buckets` field から
/// LayerStack 次元を検出する。weights 本体は読まない。
///
/// header layout (nnue-format `layerstack_weights.rs` の write 側と対称):
/// `version(u32) + network_hash(u32) + arch_len(u32) + arch_str
///  [+ num_buckets(u32) (現行 version のみ、legacy は暗黙 9)]`
fn detect_dims<R: Read>(reader: &mut R) -> io::Result<NetDims> {
    let mut buf4 = [0u8; 4];

    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unknown NNUE version: {version:#x} (expected {NNUE_VERSION:#x} or legacy {LEGACY_NNUE_VERSION_BUCKETS9:#x})"
            ),
        ));
    }

    // network_hash は次元検出には使わない (load_quantised が検出次元で照合する)
    reader.read_exact(&mut buf4)?;

    reader.read_exact(&mut buf4)?;
    let arch_len = u32::from_le_bytes(buf4) as usize;
    if arch_len == 0 || arch_len > 16384 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("implausible arch string length: {arch_len}"),
        ));
    }
    let mut arch_bytes = vec![0u8; arch_len];
    reader.read_exact(&mut arch_bytes)?;
    let arch = String::from_utf8(arch_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "arch string is not UTF-8"))?;

    let num_buckets = if version == NNUE_VERSION {
        reader.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        if !(1..=1024).contains(&n) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("implausible num_buckets: {n}"),
            ));
        }
        n
    } else {
        9
    };

    let (ft_out, l1_out, l2_out) = parse_arch_dims(&arch)?;
    Ok(NetDims {
        ft_out,
        l1_out,
        l2_out,
        num_buckets,
    })
}

/// arch 文字列から `(ft_out, l1_out, l2_out)` を取り出す。
///
/// arch 文字列は nnue-format `build_arch_str` が単一情報源で、次の形:
/// `Features=<name>(Friend)[<in>-><ft_out>x2],
///  Network=AffineTransform[1<-<l2_out>](ClippedReLU[<l2_out>](
///  AffineTransform[<l2_out><-<l2_in>](SqrClippedReLU[<l2_in>](
///  AffineTransform[<l1_out><-<ft_out*2>](InputSlice[...]))))),fv_scale=<n>`
///
/// AffineTransform は出力側から入力側の順に 3 つ並ぶ。層の連結整合
/// (`l2_in == 2*(l1_out-1)`、最終層入力 == `ft_out*2`) も検証する。
fn parse_arch_dims(arch: &str) -> io::Result<(usize, usize, usize)> {
    for unsupported in ["PSQT=", "Threat=", "EffectBucket="] {
        if arch.contains(unsupported) {
            return invalid_input(format!(
                "unsupported extension token {unsupported} in arch string (only base HalfKaHmMerged nets are convertible)"
            ));
        }
    }

    // ft_out: Features token の "-><ft_out>x2]"
    let x2 = arch
        .find("x2]")
        .ok_or_else(|| invalid_arch("missing 'x2]' in Features token"))?;
    let arrow = arch[..x2]
        .rfind("->")
        .ok_or_else(|| invalid_arch("missing '->' in Features token"))?;
    let ft_out: usize = arch[arrow + 2..x2]
        .parse()
        .map_err(|_| invalid_arch("ft_out is not a number"))?;

    // AffineTransform[<out><-<in>] を出現順に収集
    let mut affines = Vec::new();
    let mut search = 0;
    while let Some(pos) = arch[search..].find("AffineTransform[") {
        let start = search + pos + "AffineTransform[".len();
        let end = start
            + arch[start..]
                .find(']')
                .ok_or_else(|| invalid_arch("unterminated AffineTransform token"))?;
        let body = &arch[start..end];
        let sep = body
            .find("<-")
            .ok_or_else(|| invalid_arch("AffineTransform token without '<-'"))?;
        let out: usize = body[..sep]
            .parse()
            .map_err(|_| invalid_arch("AffineTransform output dim is not a number"))?;
        let inp: usize = body[sep + 2..]
            .parse()
            .map_err(|_| invalid_arch("AffineTransform input dim is not a number"))?;
        affines.push((out, inp));
        search = end;
    }
    let [output_layer, l2_layer, l1_layer]: [(usize, usize); 3] =
        affines.as_slice().try_into().map_err(|_| {
            invalid_arch(format!(
                "expected 3 AffineTransform tokens, found {}",
                affines.len()
            ))
        })?;

    let (out_out, out_in) = output_layer;
    let (l2_out, l2_in) = l2_layer;
    let (l1_out, l1_in) = l1_layer;
    if out_out != 1 || out_in != l2_out {
        return Err(invalid_arch(format!(
            "output layer dims mismatch: [{out_out}<-{out_in}] vs l2_out={l2_out}"
        )));
    }
    if l2_in != (l1_out - 1) * 2 {
        return Err(invalid_arch(format!(
            "l2_in ({l2_in}) != 2*(l1_out-1) (l1_out={l1_out})"
        )));
    }
    if l1_in != ft_out * 2 {
        return Err(invalid_arch(format!(
            "l1 input ({l1_in}) != ft_out*2 (ft_out={ft_out})"
        )));
    }
    Ok((ft_out, l1_out, l2_out))
}

fn invalid_arch(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("failed to parse LayerStack dims from arch string: {message}"),
    )
}

/// bucket routing mode の明示 assertion を要求する。
///
/// 量子化 `.bin` は bucket routing mode を記録しないため、変換前に学習時の
/// `--bucket-mode` を確認して `--assume-kingrank9` / `--assume-progress8kpabs`
/// のどちらか一方を渡す。kingrank9 は 9 bucket 固定 (YaneuraOu KingRank9
/// ルーティングの前提)。progress8kpabs は progress ルーティングを実装した
/// YaneuraOu + 学習時と同一の progress.bin 配備が必要。
fn require_bucket_mode_assertion(
    assume_kingrank9: bool,
    assume_progress8kpabs: bool,
    dims: &NetDims,
) -> io::Result<()> {
    match (assume_kingrank9, assume_progress8kpabs) {
        (false, false) => invalid_input(
            "tatara .bin files do not record bucket routing; pass --assume-kingrank9 or \
             --assume-progress8kpabs after confirming the net's training --bucket-mode",
        ),
        (true, _) if dims.num_buckets != 9 => invalid_input(format!(
            "--assume-kingrank9 requires exactly 9 buckets, but the input has {}",
            dims.num_buckets
        )),
        _ => Ok(()),
    }
}

fn reject_trailing_data<R: Read>(reader: &mut R, num_buckets: usize) -> io::Result<()> {
    let mut byte = [0_u8; 1];
    if reader.read(&mut byte)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tatara input has trailing data after the expected {num_buckets} LayerStacks"),
        ));
    }
    Ok(())
}

fn write_yo<W: Write>(
    writer: &mut W,
    weights: &LayerStackWeights,
    dims: &NetDims,
) -> io::Result<()> {
    validate_weights(weights, dims)?;

    let yo_arch = yo_arch_string(weights.feature_set.ft_in(), dims);
    write_u32(writer, YO_VERSION)?;
    write_u32(writer, YO_TOP_HASH)?;
    write_u32(
        writer,
        u32::try_from(yo_arch.len()).expect("YO architecture string length fits in u32"),
    )?;
    writer.write_all(yo_arch.as_bytes())?;

    write_u32(writer, YO_FT_HASH)?;
    let ft_biases = quantize_i16(&weights.ft_b, QA as f64);
    write_leb128_tensor_i16(writer, &ft_biases)?;
    let ft_weights = quantize_i16(&weights.ft_w, QA as f64);
    write_leb128_tensor_i16(writer, &ft_weights)?;

    let (ft_out, l1_out, l2_out) = (dims.ft_out, dims.l1_out, dims.l2_out);
    let l2_in = dims.l2_in();
    for bucket in 0..dims.num_buckets {
        write_u32(writer, YO_NETWORK_HASH)?;

        // l1f (factorizer 共有項) は save 時に l1 へ merge 済みで load 側は常に 0 を返す。
        // 加算は「未 merge の入力が来ても正しい」防御であって、通常経路では no-op。
        let l1_biases = (0..l1_out)
            .map(|output| weights.l1_b[bucket * l1_out + output] + weights.l1f_b[output]);
        let l1_weights = (0..l1_out).flat_map(|output| {
            (0..ft_out).map(move |input| {
                weights.l1_w[bucket * l1_out * ft_out + output * ft_out + input]
                    + weights.l1f_w[input * l1_out + output]
            })
        });
        write_affine(writer, l1_biases, l1_weights, ft_out, l1_out)?;

        let l2_biases = (0..l2_out).map(|output| weights.l2_b[bucket * l2_out + output]);
        let l2_weights = (0..l2_out).flat_map(|output| {
            (0..l2_in)
                .map(move |input| weights.l2_w[bucket * l2_out * l2_in + output * l2_in + input])
        });
        write_affine(writer, l2_biases, l2_weights, l2_in, l2_out)?;

        let l3_biases = std::iter::once(weights.l3_b[bucket]);
        let l3_weights = (0..l2_out).map(|input| weights.l3_w[bucket * l2_out + input]);
        write_affine(writer, l3_biases, l3_weights, l2_out, 1)?;
    }
    Ok(())
}

fn validate_weights(weights: &LayerStackWeights, dims: &NetDims) -> io::Result<()> {
    let expected_feature_set = FeatureSet::HalfKaHmMerged.spec();
    if weights.feature_set != expected_feature_set {
        return invalid_input("feature set must be HalfKaHmMerged without extensions");
    }
    if weights.num_buckets != dims.num_buckets {
        return invalid_input(format!(
            "LayerStack bucket count mismatch: header {} vs weights {}",
            dims.num_buckets, weights.num_buckets
        ));
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT models are not supported");
    }
    let (ft_out, l1_out, l2_out) = (dims.ft_out, dims.l1_out, dims.l2_out);
    let (num_buckets, l2_in) = (dims.num_buckets, dims.l2_in());
    let lengths = [
        ("ft_b", weights.ft_b.len(), ft_out),
        (
            "ft_w",
            weights.ft_w.len(),
            expected_feature_set.ft_in() * ft_out,
        ),
        ("l1_b", weights.l1_b.len(), num_buckets * l1_out),
        ("l1_w", weights.l1_w.len(), num_buckets * l1_out * ft_out),
        ("l1f_b", weights.l1f_b.len(), l1_out),
        ("l1f_w", weights.l1f_w.len(), ft_out * l1_out),
        ("l2_b", weights.l2_b.len(), num_buckets * l2_out),
        ("l2_w", weights.l2_w.len(), num_buckets * l2_out * l2_in),
        ("l3_b", weights.l3_b.len(), num_buckets),
        ("l3_w", weights.l3_w.len(), num_buckets * l2_out),
    ];
    for (name, actual, expected) in lengths {
        if actual != expected {
            return invalid_input(format!(
                "{name} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn write_affine<W, B, V>(
    writer: &mut W,
    biases: B,
    weights: V,
    input_dimensions: usize,
    output_dimensions: usize,
) -> io::Result<()>
where
    W: Write,
    B: IntoIterator<Item = f32>,
    V: IntoIterator<Item = f32>,
{
    for bias in biases {
        writer.write_all(&quantize_i32(bias, (QA * QB) as f64).to_le_bytes())?;
    }

    let padded_input = input_dimensions.div_ceil(32) * 32;
    let mut weights = weights.into_iter();
    for _ in 0..output_dimensions {
        for input in 0..padded_input {
            let value = if input < input_dimensions {
                weights.next().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "affine weight iterator is short",
                    )
                })?
            } else {
                0.0
            };
            writer.write_all(&[quantize_i8(value, QB as f64) as u8])?;
        }
    }
    if weights.next().is_some() {
        return invalid_input("affine weight iterator has extra values");
    }
    Ok(())
}

fn quantize_i16(values: &[f32], scale: f64) -> Vec<i16> {
    values
        .iter()
        .map(|&value| {
            (value as f64 * scale)
                .round()
                .clamp(i16::MIN as f64, i16::MAX as f64) as i16
        })
        .collect()
}

fn quantize_i32(value: f32, scale: f64) -> i32 {
    (value as f64 * scale)
        .round()
        .clamp(i32::MIN as f64, i32::MAX as f64) as i32
}

fn quantize_i8(value: f32, scale: f64) -> i8 {
    (value as f64 * scale)
        .round()
        .clamp(i8::MIN as f64, i8::MAX as f64) as i8
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnue_format::layerstack_weights::build_arch_str;

    const LEGACY_YO_ARCH: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1536x2],Network=SFNN-1536-V2{LayerStack=9}";

    fn dims(ft_out: usize, l1_out: usize, l2_out: usize, num_buckets: usize) -> NetDims {
        NetDims {
            ft_out,
            l1_out,
            l2_out,
            num_buckets,
        }
    }

    #[test]
    fn yo_arch_string_for_default_dims_matches_legacy_constant() {
        assert_eq!(
            yo_arch_string(73305, &dims(1536, 16, 32, 9)),
            LEGACY_YO_ARCH
        );
    }

    #[test]
    fn yo_arch_string_encodes_non_default_dims() {
        assert_eq!(
            yo_arch_string(73305, &dims(3072, 16, 64, 9)),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->3072x2],Network=SFNN-3072-15-64-V2{LayerStack=9}"
        );
        assert_eq!(
            yo_arch_string(73305, &dims(2048, 16, 64, 8)),
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->2048x2],Network=SFNN-2048-15-64-V2{LayerStack=8}"
        );
    }

    #[test]
    fn header_matches_yaneuraou_constants_and_architecture() {
        let yo_arch = yo_arch_string(73305, &dims(1536, 16, 32, 9));
        let mut output = Vec::new();
        write_u32(&mut output, YO_VERSION).unwrap();
        write_u32(&mut output, YO_TOP_HASH).unwrap();
        write_u32(&mut output, yo_arch.len() as u32).unwrap();
        output.extend_from_slice(yo_arch.as_bytes());
        write_u32(&mut output, YO_FT_HASH).unwrap();

        assert_eq!(&output[0..4], &0x7af3_2f16_u32.to_le_bytes());
        assert_eq!(&output[4..8], &1_008_745_266_u32.to_le_bytes());
        assert_eq!(&output[12..12 + yo_arch.len()], yo_arch.as_bytes());
        assert_eq!(
            &output[12 + yo_arch.len()..],
            &0x5f13_4ab8_u32.to_le_bytes()
        );
    }

    /// arch 文字列の生成 (nnue-format `build_arch_str`) と本 bin のパースの round-trip。
    #[test]
    fn parse_arch_dims_round_trips_build_arch_str() {
        for (ft_out, l1_out, l2_out) in
            [(1536, 16, 32), (3072, 16, 64), (2048, 16, 64), (768, 8, 32)]
        {
            let arch = build_arch_str(
                "HalfKaHmMerged",
                73305,
                ft_out,
                l1_out,
                (l1_out - 1) * 2,
                l2_out,
                28,
                None,
                None,
                None,
            );
            assert_eq!(
                parse_arch_dims(&arch).unwrap(),
                (ft_out, l1_out, l2_out),
                "arch = {arch}"
            );
        }
    }

    #[test]
    fn parse_arch_dims_rejects_extension_tokens() {
        let arch = build_arch_str(
            "HalfKaHmMerged",
            73305,
            1536,
            16,
            30,
            32,
            28,
            Some(9),
            None,
            None,
        );
        let error = parse_arch_dims(&arch).unwrap_err();
        assert!(error.to_string().contains("PSQT="), "error = {error}");
    }

    #[test]
    fn detect_dims_reads_header_and_num_buckets_field() {
        let arch = build_arch_str(
            "HalfKaHmMerged",
            73305,
            3072,
            16,
            30,
            64,
            28,
            None,
            None,
            None,
        );
        let mut header = Vec::new();
        header.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        header.extend_from_slice(&0xdead_beef_u32.to_le_bytes()); // network_hash (検出では未使用)
        header.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        header.extend_from_slice(arch.as_bytes());
        header.extend_from_slice(&7_u32.to_le_bytes()); // num_buckets

        let detected = detect_dims(&mut &header[..]).unwrap();
        assert_eq!(detected, dims(3072, 16, 64, 7));
    }

    #[test]
    fn detect_dims_defaults_legacy_version_to_nine_buckets() {
        let arch = build_arch_str(
            "HalfKaHmMerged",
            73305,
            1536,
            16,
            30,
            32,
            28,
            None,
            None,
            None,
        );
        let mut header = Vec::new();
        header.extend_from_slice(&LEGACY_NNUE_VERSION_BUCKETS9.to_le_bytes());
        header.extend_from_slice(&0xdead_beef_u32.to_le_bytes());
        header.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        header.extend_from_slice(arch.as_bytes());

        let detected = detect_dims(&mut &header[..]).unwrap();
        assert_eq!(detected, dims(1536, 16, 32, 9));
    }

    #[test]
    fn affine_file_weights_are_canonical_row_major_with_zero_padding() {
        let mut output = Vec::new();
        write_affine(
            &mut output,
            [1.0, -1.0],
            [
                1.0 / 64.0,
                2.0 / 64.0,
                3.0 / 64.0,
                -1.0 / 64.0,
                -2.0 / 64.0,
                -3.0 / 64.0,
            ],
            3,
            2,
        )
        .unwrap();

        assert_eq!(
            i32::from_le_bytes(output[0..4].try_into().unwrap()),
            QA * QB
        );
        assert_eq!(
            i32::from_le_bytes(output[4..8].try_into().unwrap()),
            -(QA * QB)
        );
        assert_eq!(&output[8..11], &[1, 2, 3]);
        assert!(output[11..40].iter().all(|&byte| byte == 0));
        assert_eq!(&output[40..43], &[255, 254, 253]);
        assert!(output[43..72].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn trailing_input_is_rejected() {
        let error = reject_trailing_data(&mut &b"x"[..], 9).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        reject_trailing_data(&mut &b""[..], 9).unwrap();
    }

    #[test]
    fn bucket_mode_requires_an_explicit_assertion() {
        let nine = dims(1536, 16, 32, 9);
        let error = require_bucket_mode_assertion(false, false, &nine).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--assume-kingrank9"));
        assert!(error.to_string().contains("--assume-progress8kpabs"));
        require_bucket_mode_assertion(true, false, &nine).unwrap();
        require_bucket_mode_assertion(false, true, &nine).unwrap();
    }

    #[test]
    fn kingrank9_assertion_requires_nine_buckets() {
        let eight = dims(2048, 16, 64, 8);
        let error = require_bucket_mode_assertion(true, false, &eight).unwrap_err();
        assert!(error.to_string().contains("9 buckets"), "error = {error}");
        // progress8kpabs は 9 以外の bucket 数も許容する
        require_bucket_mode_assertion(false, true, &eight).unwrap();
    }
}
