use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use clap::Parser;
use nnue_format::layerstack_weights::{LEGACY_NNUE_VERSION_BUCKETS9, NNUE_VERSION};
use nnue_format::{LayerStackWeights, YANEURAOU_LAYER_STACKS, save_yaneuraou};
use shogi_features::FeatureSet;

/// YaneuraOu SFNN が格納する LayerStack 数。routing 規則は binary に含まれない。
const YO_LAYER_STACKS: usize = YANEURAOU_LAYER_STACKS;
/// 既存YaneuraOuのprogress routingが実際に選ぶbucket数。
const PROGRESS_INPUT_BUCKETS: usize = 8;

/// 変換対象の SFNN 次元上限。実在アーキは十分収まり、壊れた arch 文字列 (0 次元 /
/// 巨大値) が overflow や過大 allocation を起こす前に弾くための健全性ガード。
const MAX_FT_OUT: usize = 8192;
const MAX_HIDDEN_DIM: usize = 4096;

/// tatara `.bin` header から読み取った変換対象アーキ。
#[derive(Debug)]
struct DetectedArch {
    feature_set: FeatureSet,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
}

/// headerとarchitecture文字列から検出した入力形式。
#[derive(Debug)]
struct DetectedInput {
    arch: DetectedArch,
    num_buckets: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutingAssumption {
    KingRank9,
    Progress8KpAbs,
}

#[derive(Parser)]
#[command(about = "Convert a tatara LayerStack net to a YaneuraOu SFNN evaluation file")]
struct Args {
    /// tatara LayerStack quantised .bin
    #[arg(long)]
    input: PathBuf,
    /// YaneuraOu nn.bin
    #[arg(long)]
    output: PathBuf,
    /// 入力が`--bucket-mode kingrank9`で学習されたことを明示する。
    /// 量子化`.bin`はbucket routing modeを記録しない。
    #[arg(long, conflicts_with = "assume_progress8kpabs")]
    assume_kingrank9: bool,
    /// 入力が`--bucket-mode progress8kpabs --num-buckets 8`で学習されたことを
    /// 明示する。変換時にbucket 7を未使用の第9slotへ複製する。
    #[arg(long, conflicts_with = "assume_kingrank9")]
    assume_progress8kpabs: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.input == args.output {
        return Err("input and output must be different paths".into());
    }
    let routing = require_routing_assertion(args.assume_kingrank9, args.assume_progress8kpabs)?;

    let detect_input = File::open(&args.input)?;
    let detected = detect_arch(&mut BufReader::new(detect_input))?;
    validate_input_buckets(routing, detected.num_buckets)?;

    let input = File::open(&args.input)?;
    let mut reader = BufReader::new(input);
    let weights = LayerStackWeights::load_quantised(
        &mut reader,
        detected.arch.feature_set.spec(),
        detected.arch.ft_out,
        detected.arch.l1_out,
        detected.arch.l2_out,
        detected.num_buckets,
    )?;
    reject_trailing_data(&mut reader, detected.num_buckets)?;
    let weights = prepare_yaneuraou_weights(weights, &detected.arch, routing)?;

    let output = File::create(&args.output)?;
    let mut writer = BufWriter::new(output);
    save_yaneuraou(&mut writer, &weights)?;
    writer.flush()?;
    Ok(())
}

fn require_routing_assertion(
    assume_kingrank9: bool,
    assume_progress8kpabs: bool,
) -> io::Result<RoutingAssumption> {
    match (assume_kingrank9, assume_progress8kpabs) {
        (true, false) => Ok(RoutingAssumption::KingRank9),
        (false, true) => Ok(RoutingAssumption::Progress8KpAbs),
        (false, false) => invalid_input(
            "tatara .bin files do not record bucket routing; pass exactly one of --assume-kingrank9 or --assume-progress8kpabs after confirming the training configuration",
        ),
        (true, true) => {
            invalid_input("--assume-kingrank9 and --assume-progress8kpabs are mutually exclusive")
        }
    }
}

fn validate_input_buckets(routing: RoutingAssumption, num_buckets: usize) -> io::Result<()> {
    let expected = match routing {
        RoutingAssumption::KingRank9 => YO_LAYER_STACKS,
        RoutingAssumption::Progress8KpAbs => PROGRESS_INPUT_BUCKETS,
    };
    if num_buckets != expected {
        return invalid_input(format!(
            "{routing:?} conversion requires {expected} input buckets, but the input has {num_buckets}"
        ));
    }
    Ok(())
}

fn prepare_yaneuraou_weights(
    weights: LayerStackWeights,
    arch: &DetectedArch,
    routing: RoutingAssumption,
) -> io::Result<LayerStackWeights> {
    match routing {
        RoutingAssumption::KingRank9 => Ok(weights),
        RoutingAssumption::Progress8KpAbs => pad_progress8_with_unused_ninth(weights, arch),
    }
}

/// 固定8分割のweightを9-slot形式へ変換する。既存エンジンはslot 0..=7しか
/// 選ばないためslot 8は未使用だが、壊れたzero networkを置かず終盤側のslot 7を複製する。
fn pad_progress8_with_unused_ninth(
    mut weights: LayerStackWeights,
    arch: &DetectedArch,
) -> io::Result<LayerStackWeights> {
    if weights.num_buckets != PROGRESS_INPUT_BUCKETS {
        return invalid_input(format!(
            "progress padding requires {PROGRESS_INPUT_BUCKETS} input buckets, got {}",
            weights.num_buckets
        ));
    }
    if YO_LAYER_STACKS != PROGRESS_INPUT_BUCKETS + 1 {
        return invalid_input(
            "YaneuraOu output must have exactly one unused slot after 8 progress buckets",
        );
    }
    if weights.psqt_w.is_some() {
        return invalid_input("PSQT progress nets are not representable in YaneuraOu SFNN");
    }

    let l2_in = (arch.l1_out - 1) * 2;
    append_last_bucket(&mut weights.l1_w, arch.l1_out * arch.ft_out, "l1_w")?;
    append_last_bucket(&mut weights.l1_b, arch.l1_out, "l1_b")?;
    append_last_bucket(&mut weights.l2_w, arch.l2_out * l2_in, "l2_w")?;
    append_last_bucket(&mut weights.l2_b, arch.l2_out, "l2_b")?;
    append_last_bucket(&mut weights.l3_w, arch.l2_out, "l3_w")?;
    append_last_bucket(&mut weights.l3_b, 1, "l3_b")?;
    weights.num_buckets = YO_LAYER_STACKS;
    Ok(weights)
}

fn append_last_bucket(
    values: &mut Vec<f32>,
    elements_per_bucket: usize,
    name: &str,
) -> io::Result<()> {
    let expected = PROGRESS_INPUT_BUCKETS
        .checked_mul(elements_per_bucket)
        .ok_or_else(|| invalid_input_err(format!("{name} size overflow")))?;
    if values.len() != expected {
        return invalid_input(format!(
            "{name} length {} does not match {PROGRESS_INPUT_BUCKETS} buckets x {elements_per_bucket} elements",
            values.len()
        ));
    }
    let start = expected - elements_per_bucket;
    let last = values[start..expected].to_vec();
    values.extend_from_slice(&last);
    Ok(())
}

/// `.bin` header (version + network_hash + arch_str + num_buckets) を読み、変換
/// 可能な SFNN アーキかを判定する。PSQT / threat / effect bucket /
/// 未知 feature は YaneuraOu SFNN に受け皿が無いため明示的に reject する。
fn detect_arch<R: Read>(reader: &mut R) -> io::Result<DetectedInput> {
    let version = read_u32(reader)?;
    if version != NNUE_VERSION && version != LEGACY_NNUE_VERSION_BUCKETS9 {
        return invalid_input(format!(
            "unknown tatara NNUE version: {version:#x} (expected {NNUE_VERSION:#x} or legacy {LEGACY_NNUE_VERSION_BUCKETS9:#x})"
        ));
    }
    let _network_hash = read_u32(reader)?;

    let arch_len = read_u32(reader)? as usize;
    if arch_len == 0 || arch_len > 16_384 {
        return invalid_input(format!("invalid arch string length: {arch_len}"));
    }
    let mut arch_bytes = vec![0_u8; arch_len];
    reader.read_exact(&mut arch_bytes)?;
    let arch_str = std::str::from_utf8(&arch_bytes)
        .map_err(|error| invalid_input_err(format!("arch string is not UTF-8: {error}")))?;

    // num_buckets は現行 version のみ header に持ち、legacy は暗黙 9。
    let num_buckets = if version == LEGACY_NNUE_VERSION_BUCKETS9 {
        YO_LAYER_STACKS
    } else {
        read_u32(reader)? as usize
    };
    let arch = parse_arch_str(arch_str)?;
    Ok(DetectedInput { arch, num_buckets })
}

/// tatara `build_arch_str` が生成する arch 文字列から feature set と隠れ層次元を
/// 取り出す。書式は
/// `Features=<name>(Friend)[<in>-><ft>x2],...,Network=AffineTransform[1<-<l2_out>](...
/// SqrClippedReLU[<l2_in>](AffineTransform[<l1_out>-<ft*2>]...`。
fn parse_arch_str(arch_str: &str) -> io::Result<DetectedArch> {
    for unsupported in ["PSQT=", "Threat=", "EffectBucket="] {
        if arch_str.contains(unsupported) {
            let token = unsupported.trim_end_matches('=');
            return invalid_input(format!(
                "{token} models are not representable in YaneuraOu SFNN and cannot be converted"
            ));
        }
    }

    let features = between(arch_str, "Features=", "(Friend)[").ok_or_else(|| {
        invalid_input_err("arch string has no `Features=<name>(Friend)[` token".to_string())
    })?;
    let feature_set = FeatureSet::ALL
        .into_iter()
        .find(|fs| fs.spec().arch_feature_name() == features)
        .ok_or_else(|| invalid_input_err(format!("unknown feature set `{features}`")))?;

    let ft_out = between(arch_str, "->", "x2")
        .ok_or_else(|| invalid_input_err("arch string has no `-><ft>x2` token".to_string()))
        .and_then(parse_usize)?;
    // YaneuraOu は kTransformedFeatureDimensions % kMaxSimdWidth(32) == 0 を要求する。
    if ft_out == 0 || ft_out > MAX_FT_OUT || ft_out % 32 != 0 {
        return invalid_input(format!(
            "unsupported FT output dimension {ft_out} (expected a positive multiple of 32 up to {MAX_FT_OUT})"
        ));
    }

    let l2_out = between(arch_str, "AffineTransform[1<-", "]")
        .ok_or_else(|| invalid_input_err("arch string has no output affine token".to_string()))
        .and_then(parse_usize)?;
    if l2_out == 0 || l2_out > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 output dimension {l2_out} (expected 1..={MAX_HIDDEN_DIM})"
        ));
    }

    let l2_in = between(arch_str, "SqrClippedReLU[", "]")
        .ok_or_else(|| invalid_input_err("arch string has no SqrClippedReLU token".to_string()))
        .and_then(parse_usize)?;
    if l2_in == 0 || l2_in % 2 != 0 || l2_in > MAX_HIDDEN_DIM {
        return invalid_input(format!(
            "unsupported L2 input dimension {l2_in} (expected a positive even value up to {MAX_HIDDEN_DIM})"
        ));
    }
    // L1 出力のうち skip 1 dim を除いた `l1_out - 1` を 2 乗連結して L2 入力にする
    // ため、`l2_in = (l1_out - 1) * 2`。
    let l1_out = l2_in / 2 + 1;

    Ok(DetectedArch {
        feature_set,
        ft_out,
        l1_out,
        l2_out,
    })
}

fn between<'a>(haystack: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let after = haystack.split_once(start)?.1;
    Some(after.split_once(end)?.0)
}

fn parse_usize(value: &str) -> io::Result<usize> {
    value
        .parse::<usize>()
        .map_err(|error| invalid_input_err(format!("expected integer, got `{value}`: {error}")))
}

fn reject_trailing_data<R: Read>(reader: &mut R, expected_buckets: usize) -> io::Result<()> {
    let mut byte = [0_u8; 1];
    if reader.read(&mut byte)? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tatara input has trailing data after the expected {expected_buckets} LayerStacks"
            ),
        ));
    }
    Ok(())
}

fn invalid_input<T>(message: impl Into<String>) -> io::Result<T> {
    Err(invalid_input_err(message.into()))
}

fn invalid_input_err(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use nnue_format::layerstack_weights::build_arch_str;

    fn detected(
        feature_set: FeatureSet,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
    ) -> DetectedArch {
        DetectedArch {
            feature_set,
            ft_out,
            l1_out,
            l2_out,
        }
    }

    #[test]
    fn parse_arch_str_roundtrips_build_arch_str_over_dims_and_features() {
        let configs = [
            (FeatureSet::HalfKaHmMerged, 1536_usize, 16_usize, 32_usize),
            (FeatureSet::HalfKaHmMerged, 512, 16, 32),
            (FeatureSet::HalfKaHmMerged, 1024, 8, 16),
            (FeatureSet::HalfKaHmMerged, 1024, 16, 64),
            (FeatureSet::HalfKp, 1536, 16, 32),
            (FeatureSet::HalfKaSplit, 768, 16, 32),
            (FeatureSet::HalfKaMerged, 1536, 16, 32),
            (FeatureSet::HalfKaHmSplit, 1536, 16, 32),
        ];
        for (fs, ft_out, l1_out, l2_out) in configs {
            let spec = fs.spec();
            let l2_in = (l1_out - 1) * 2;
            let arch_str = build_arch_str(
                spec.arch_feature_name(),
                spec.ft_in(),
                ft_out,
                l1_out,
                l2_in,
                l2_out,
                Some(28),
                None,
                None,
                None,
            );
            let parsed = parse_arch_str(&arch_str).expect("parses");
            assert_eq!(parsed.feature_set, fs);
            assert_eq!(parsed.ft_out, ft_out);
            assert_eq!(parsed.l1_out, l1_out);
            assert_eq!(parsed.l2_out, l2_out);
        }
    }

    #[test]
    fn parse_arch_str_rejects_psqt_threat_effect() {
        use nnue_format::layerstack_weights::{EffectBucketArch, ThreatArch};

        let name = FeatureSet::HalfKaHmMerged.spec().arch_feature_name();
        let cases = [
            ("PSQT", Some(9), None, None),
            (
                "Threat",
                None,
                Some(ThreatArch {
                    dims: 128,
                    profile_id: 0,
                }),
                None,
            ),
            (
                "EffectBucket",
                None,
                None,
                Some(EffectBucketArch {
                    nb: 4,
                    king_bucketed: false,
                }),
            ),
        ];
        for (token, psqt, threat, effect) in cases {
            let arch_str = build_arch_str(
                name,
                73305,
                1536,
                16,
                30,
                32,
                Some(28),
                psqt,
                threat,
                effect,
            );
            let error = parse_arch_str(&arch_str).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(
                error.to_string().contains(token),
                "expected {token} in error, got: {error}"
            );
        }
    }

    #[test]
    fn parse_arch_str_rejects_degenerate_dimensions() {
        let good = build_arch_str(
            FeatureSet::HalfKaHmMerged.spec().arch_feature_name(),
            73305,
            1536,
            16,
            30,
            32,
            Some(28),
            None,
            None,
            None,
        );
        // ft_out = 0 / ft_out が上限超過 (32 の倍数だが MAX_FT_OUT 超) / l2_out = 0。
        for (bad, needle) in [
            (good.replace("->1536x2", "->0x2"), "FT output"),
            (good.replace("->1536x2", "->32768x2"), "FT output"),
            (
                good.replace("AffineTransform[1<-32]", "AffineTransform[1<-0]"),
                "L2 output",
            ),
        ] {
            let error = parse_arch_str(&bad).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert!(error.to_string().contains(needle), "got: {error}");
        }
    }

    fn header_bytes(version: u32, arch_str: &str, num_buckets: Option<u32>) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // network_hash (detect は無視)
        bytes.extend_from_slice(&(arch_str.len() as u32).to_le_bytes());
        bytes.extend_from_slice(arch_str.as_bytes());
        if let Some(n) = num_buckets {
            bytes.extend_from_slice(&n.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn detect_arch_reads_current_bucket_count_and_legacy_implicit9() {
        let arch_str = build_arch_str(
            FeatureSet::HalfKaHmMerged.spec().arch_feature_name(),
            73305,
            1536,
            16,
            30,
            32,
            Some(28),
            None,
            None,
            None,
        );

        let current = header_bytes(NNUE_VERSION, &arch_str, Some(8));
        let detected = detect_arch(&mut std::io::Cursor::new(current)).expect("current 8");
        assert_eq!(detected.num_buckets, 8);
        assert_eq!(detected.arch.feature_set, FeatureSet::HalfKaHmMerged);

        let legacy = header_bytes(LEGACY_NNUE_VERSION_BUCKETS9, &arch_str, None);
        let detected = detect_arch(&mut std::io::Cursor::new(legacy)).expect("legacy implicit 9");
        assert_eq!(detected.num_buckets, 9);
        assert_eq!(detected.arch.feature_set, FeatureSet::HalfKaHmMerged);
        assert_eq!(detected.arch.ft_out, 1536);
        assert_eq!(detected.arch.l1_out, 16);
        assert_eq!(detected.arch.l2_out, 32);
    }

    #[test]
    fn trailing_input_is_rejected() {
        let error = reject_trailing_data(&mut &b"x"[..], 8).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("8 LayerStacks"));
        reject_trailing_data(&mut &b""[..], 9).unwrap();
    }

    #[test]
    fn routing_mode_requires_exactly_one_explicit_assertion() {
        let error = require_routing_assertion(false, false).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("--assume-kingrank9"));
        assert!(error.to_string().contains("--assume-progress8kpabs"));

        assert_eq!(
            require_routing_assertion(true, false).unwrap(),
            RoutingAssumption::KingRank9
        );
        assert_eq!(
            require_routing_assertion(false, true).unwrap(),
            RoutingAssumption::Progress8KpAbs
        );

        let error = require_routing_assertion(true, true).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn routing_assertion_flags_conflict_at_cli_parse() {
        let error = Args::try_parse_from([
            "net_to_yo",
            "--input",
            "input.bin",
            "--output",
            "nn.bin",
            "--assume-kingrank9",
            "--assume-progress8kpabs",
        ])
        .err()
        .expect("routing assertions must conflict");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn routing_assertion_requires_the_matching_input_bucket_count() {
        validate_input_buckets(RoutingAssumption::KingRank9, 9).unwrap();
        validate_input_buckets(RoutingAssumption::Progress8KpAbs, 8).unwrap();

        let error = validate_input_buckets(RoutingAssumption::KingRank9, 8).unwrap_err();
        assert!(error.to_string().contains("requires 9 input buckets"));
        let error = validate_input_buckets(RoutingAssumption::Progress8KpAbs, 9).unwrap_err();
        assert!(error.to_string().contains("requires 8 input buckets"));
    }

    fn assert_last_bucket_was_duplicated(before: &[f32], after: &[f32], per_bucket: usize) {
        assert_eq!(after.len(), before.len() + per_bucket);
        assert_eq!(&after[..before.len()], before);
        assert_eq!(&after[before.len()..], &before[before.len() - per_bucket..]);
    }

    #[test]
    fn progress8_padding_preserves_buckets_and_duplicates_bucket7_into_slot8() {
        let arch = detected(FeatureSet::HalfKaHmMerged, 32, 3, 2);
        let mut weights = LayerStackWeights::zeroed(
            arch.feature_set.spec(),
            arch.ft_out,
            arch.l1_out,
            arch.l2_out,
            PROGRESS_INPUT_BUCKETS,
        );
        for values in [
            &mut weights.l1_w,
            &mut weights.l1_b,
            &mut weights.l2_w,
            &mut weights.l2_b,
            &mut weights.l3_w,
            &mut weights.l3_b,
        ] {
            for (index, value) in values.iter_mut().enumerate() {
                *value = index as f32 + 0.25;
            }
        }
        let before = weights.clone();
        let padded = pad_progress8_with_unused_ninth(weights, &arch).expect("pad progress8");

        assert_eq!(padded.num_buckets, YO_LAYER_STACKS);
        assert_eq!(padded.ft_w, before.ft_w);
        assert_eq!(padded.ft_b, before.ft_b);
        assert_eq!(padded.l1f_w, before.l1f_w);
        assert_eq!(padded.l1f_b, before.l1f_b);
        assert_last_bucket_was_duplicated(&before.l1_w, &padded.l1_w, arch.l1_out * arch.ft_out);
        assert_last_bucket_was_duplicated(&before.l1_b, &padded.l1_b, arch.l1_out);
        assert_last_bucket_was_duplicated(
            &before.l2_w,
            &padded.l2_w,
            arch.l2_out * (arch.l1_out - 1) * 2,
        );
        assert_last_bucket_was_duplicated(&before.l2_b, &padded.l2_b, arch.l2_out);
        assert_last_bucket_was_duplicated(&before.l3_w, &padded.l3_w, arch.l2_out);
        assert_last_bucket_was_duplicated(&before.l3_b, &padded.l3_b, 1);
    }

    /// zeroed weights から合成した tatara `.bin` を返す。
    fn synthetic_bin(
        feature_set: FeatureSet,
        ft_out: usize,
        l1_out: usize,
        l2_out: usize,
        num_buckets: usize,
    ) -> Vec<u8> {
        let weights =
            LayerStackWeights::zeroed(feature_set.spec(), ft_out, l1_out, l2_out, num_buckets);
        let mut bytes = Vec::new();
        weights
            .save_quantised(&mut bytes, Some(nnue_format::layerstack_weights::FV_SCALE))
            .expect("save synthetic .bin");
        bytes
    }

    /// 合成 `.bin` を `detect_arch` → `load_quantised` → `write_yo` のフル経路に
    /// 通し、YaneuraOu 出力バイト列を返す。detect が期待アーキと一致することも確認。
    fn convert(bytes: &[u8], expect: &DetectedArch, routing: RoutingAssumption) -> Vec<u8> {
        let detected = detect_arch(&mut std::io::Cursor::new(bytes)).expect("detect");
        assert_eq!(detected.arch.feature_set, expect.feature_set);
        assert_eq!(detected.arch.ft_out, expect.ft_out);
        assert_eq!(detected.arch.l1_out, expect.l1_out);
        assert_eq!(detected.arch.l2_out, expect.l2_out);
        validate_input_buckets(routing, detected.num_buckets).expect("routing matches input");

        let mut load_reader = std::io::Cursor::new(bytes);
        let weights = LayerStackWeights::load_quantised(
            &mut load_reader,
            detected.arch.feature_set.spec(),
            detected.arch.ft_out,
            detected.arch.l1_out,
            detected.arch.l2_out,
            detected.num_buckets,
        )
        .expect("load_quantised");
        reject_trailing_data(&mut load_reader, detected.num_buckets).expect("no trailing data");
        let weights = prepare_yaneuraou_weights(weights, &detected.arch, routing)
            .expect("prepare YaneuraOu weights");

        let mut out = Vec::new();
        save_yaneuraou(&mut out, &weights).expect("save_yaneuraou");
        out
    }

    #[test]
    fn full_pipeline_produces_valid_yo_header_across_feature_sets_and_dims() {
        // 検証対象は header と affine のパディング済み次元追随なので、FT 出力は
        // 小さめ (128 の倍数) にして全 feature set を高速に網羅する。
        let configs = [
            (
                FeatureSet::HalfKaHmMerged,
                256_usize,
                16_usize,
                32_usize,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->256x2],Network=SFNN_HALFKAHM2_256_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKp,
                128,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKP(Friend)[125388->128x2],Network=SFNN_HALFKP_128_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaSplit,
                128,
                8,
                16,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA1(Friend)[138510->128x2],Network=SFNN_HALFKA1_128_7_16_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaMerged,
                128,
                16,
                32,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA2(Friend)[131949->128x2],Network=SFNN_HALFKA2_128_15_32_K3K3{LayerStack=9}",
            ),
            (
                FeatureSet::HalfKaHmSplit,
                256,
                7,
                16,
                "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm1(Friend)[76950->256x2],Network=SFNN_HALFKAHM1_256_6_16_K3K3{LayerStack=9}",
            ),
        ];
        for (fs, ft_out, l1_out, l2_out, expected_arch) in configs {
            let expect = detected(fs, ft_out, l1_out, l2_out);
            let bytes = synthetic_bin(fs, ft_out, l1_out, l2_out, YO_LAYER_STACKS);
            let out = convert(&bytes, &expect, RoutingAssumption::KingRank9);

            assert_eq!(
                u32::from_le_bytes(out[0..4].try_into().unwrap()),
                0x7af3_2f16
            );
            assert_eq!(
                u32::from_le_bytes(out[4..8].try_into().unwrap()),
                0x3c20_3b32
            );
            let arch_len = u32::from_le_bytes(out[8..12].try_into().unwrap()) as usize;
            let arch_str = std::str::from_utf8(&out[12..12 + arch_len]).unwrap();
            assert_eq!(arch_str, expected_arch);
            let ft_hash_at = 12 + arch_len;
            assert_eq!(
                u32::from_le_bytes(out[ft_hash_at..ft_hash_at + 4].try_into().unwrap()),
                0x5f13_4ab8
            );
        }
    }

    #[test]
    fn progress8_1024x16x64_full_pipeline_outputs_a_nine_stack_yaneuraou_file() {
        let expect = detected(FeatureSet::HalfKaHmMerged, 1024, 16, 64);
        let bytes = synthetic_bin(
            expect.feature_set,
            expect.ft_out,
            expect.l1_out,
            expect.l2_out,
            PROGRESS_INPUT_BUCKETS,
        );
        let out = convert(&bytes, &expect, RoutingAssumption::Progress8KpAbs);

        let arch_len = u32::from_le_bytes(out[8..12].try_into().unwrap()) as usize;
        let arch_str = std::str::from_utf8(&out[12..12 + arch_len]).unwrap();
        assert_eq!(
            arch_str,
            "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm2(Friend)[73305->1024x2],Network=SFNN_HALFKAHM2_1024_15_64_K3K3{LayerStack=9}"
        );
    }

    #[test]
    fn kingrank9_full_pipeline_matches_the_existing_direct_writer() {
        let expect = detected(FeatureSet::HalfKaHmMerged, 128, 16, 32);
        let bytes = synthetic_bin(
            expect.feature_set,
            expect.ft_out,
            expect.l1_out,
            expect.l2_out,
            YO_LAYER_STACKS,
        );

        let mut reader = std::io::Cursor::new(&bytes);
        let weights = LayerStackWeights::load_quantised(
            &mut reader,
            expect.feature_set.spec(),
            expect.ft_out,
            expect.l1_out,
            expect.l2_out,
            YO_LAYER_STACKS,
        )
        .expect("load direct fixture");
        reject_trailing_data(&mut reader, YO_LAYER_STACKS).expect("consume direct fixture");
        let mut direct = Vec::new();
        save_yaneuraou(&mut direct, &weights).expect("direct writer");

        let converted = convert(&bytes, &expect, RoutingAssumption::KingRank9);
        assert_eq!(converted, direct);
    }
}
