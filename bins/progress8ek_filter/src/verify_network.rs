//! progress fixed8の基準networkとprogress8ek 9-slot networkの非対象parameterを比較する。

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use clap::Parser;
use nnue_format::LayerStackWeights;
use serde::Serialize;
use shogi_features::FeatureSet;

const BASE_BUCKETS: usize = 8;
const CANDIDATE_BUCKETS: usize = 9;

#[derive(Debug, Parser)]
#[command(name = "progress8ek-verify-network")]
#[command(about = "Verify that a progress8ek network changes only slot 8")]
struct Args {
    /// 基準となる8-bucket Tatara量子化network。
    #[arg(long)]
    base: PathBuf,

    /// 比較する9-slot Tatara量子化network。
    #[arg(long)]
    candidate: PathBuf,

    #[arg(long, default_value_t = 1024)]
    ft_out: usize,

    #[arg(long, default_value_t = 16)]
    l1: usize,

    #[arg(long, default_value_t = 64)]
    l2: usize,

    /// slot 8の初期値として使ったfixed8側のslot。
    #[arg(long)]
    source_slot: usize,

    /// slot 8が基準source slotから一つ以上変化していることも要求する。
    #[arg(long)]
    require_slot8_difference: bool,
}

#[derive(Debug, Serialize)]
struct TensorReport {
    name: &'static str,
    elements_per_slot: usize,
    preserved_elements: usize,
    slot8_differences_from_source_slot: usize,
}

#[derive(Debug, Serialize)]
struct Report {
    base: PathBuf,
    candidate: PathBuf,
    shared_parameters_bit_identical: bool,
    slots_0_through_7_bit_identical: bool,
    source_slot: usize,
    slot8_has_difference_from_source_slot: bool,
    tensors: Vec<TensorReport>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let base = load(&args.base, &args, BASE_BUCKETS)?;
    let candidate = load(&args.candidate, &args, CANDIDATE_BUCKETS)?;
    check_exact("ft_w", &base.ft_w, &candidate.ft_w)?;
    check_exact("ft_b", &base.ft_b, &candidate.ft_b)?;
    check_exact("l1f_w", &base.l1f_w, &candidate.l1f_w)?;
    check_exact("l1f_b", &base.l1f_b, &candidate.l1f_b)?;

    let mut tensors = Vec::new();
    for (name, base_values, candidate_values) in [
        ("l1_w", &base.l1_w, &candidate.l1_w),
        ("l1_b", &base.l1_b, &candidate.l1_b),
        ("l2_w", &base.l2_w, &candidate.l2_w),
        ("l2_b", &base.l2_b, &candidate.l2_b),
        ("l3_w", &base.l3_w, &candidate.l3_w),
        ("l3_b", &base.l3_b, &candidate.l3_b),
    ] {
        tensors.push(check_bucketed(
            name,
            base_values,
            candidate_values,
            args.source_slot,
        )?);
    }
    let slot8_has_difference = tensors
        .iter()
        .any(|tensor| tensor.slot8_differences_from_source_slot > 0);
    if args.require_slot8_difference && !slot8_has_difference {
        return Err(format!("slot 8が基準slot {}から変化していません", args.source_slot).into());
    }
    let report = Report {
        base: args.base,
        candidate: args.candidate,
        shared_parameters_bit_identical: true,
        slots_0_through_7_bit_identical: true,
        source_slot: args.source_slot,
        slot8_has_difference_from_source_slot: slot8_has_difference,
        tensors,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn load(
    path: &Path,
    args: &Args,
    buckets: usize,
) -> Result<LayerStackWeights, Box<dyn std::error::Error>> {
    let mut input = BufReader::new(File::open(path)?);
    LayerStackWeights::load_quantised(
        &mut input,
        FeatureSet::HalfKaHmMerged.spec(),
        args.ft_out,
        args.l1,
        args.l2,
        buckets,
    )
    .map_err(Into::into)
}

fn check_exact(name: &str, base: &[f32], candidate: &[f32]) -> Result<(), String> {
    if base.len() != candidate.len() {
        return Err(format!(
            "{name}の長さが異なります: base={} candidate={}",
            base.len(),
            candidate.len()
        ));
    }
    if let Some((index, (base, candidate))) = base
        .iter()
        .zip(candidate)
        .enumerate()
        .find(|(_, (base, candidate))| base.to_bits() != candidate.to_bits())
    {
        return Err(format!(
            "共有parameter {name}[{index}]が変化しています: base_bits={:#010x} candidate_bits={:#010x}",
            base.to_bits(),
            candidate.to_bits()
        ));
    }
    Ok(())
}

fn check_bucketed(
    name: &'static str,
    base: &[f32],
    candidate: &[f32],
    source_slot: usize,
) -> Result<TensorReport, String> {
    if source_slot >= BASE_BUCKETS {
        return Err(format!(
            "source slot must be in 0..{BASE_BUCKETS}, got {source_slot}"
        ));
    }
    if !base.len().is_multiple_of(BASE_BUCKETS)
        || !candidate.len().is_multiple_of(CANDIDATE_BUCKETS)
    {
        return Err(format!("{name}のbucket shapeが不正です"));
    }
    let per_slot = base.len() / BASE_BUCKETS;
    if candidate.len() / CANDIDATE_BUCKETS != per_slot {
        return Err(format!(
            "{name}のslot shapeが異なります: base={} candidate={}",
            per_slot,
            candidate.len() / CANDIDATE_BUCKETS
        ));
    }
    if let Some((index, (base, candidate))) = base
        .iter()
        .zip(&candidate[..base.len()])
        .enumerate()
        .find(|(_, (base, candidate))| base.to_bits() != candidate.to_bits())
    {
        return Err(format!(
            "非対象parameter {name}[slot={}, element={}]が変化しています: base_bits={:#010x} candidate_bits={:#010x}",
            index / per_slot,
            index % per_slot,
            base.to_bits(),
            candidate.to_bits()
        ));
    }
    let source = &base[source_slot * per_slot..(source_slot + 1) * per_slot];
    let candidate_slot8 = &candidate[8 * per_slot..9 * per_slot];
    let slot8_differences = source
        .iter()
        .zip(candidate_slot8)
        .filter(|(base, candidate)| base.to_bits() != candidate.to_bits())
        .count();
    Ok(TensorReport {
        name,
        elements_per_slot: per_slot,
        preserved_elements: base.len(),
        slot8_differences_from_source_slot: slot8_differences,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketed_check_accepts_only_slot8_difference() {
        let base: Vec<f32> = (0..24).map(|value| value as f32).collect();
        let mut candidate = base.clone();
        candidate.extend_from_slice(&base[18..21]);
        candidate[24] += 1.0;
        let report = check_bucketed("test", &base, &candidate, 6).unwrap();
        assert_eq!(report.elements_per_slot, 3);
        assert_eq!(report.slot8_differences_from_source_slot, 1);
    }

    #[test]
    fn bucketed_check_rejects_existing_slot_difference() {
        let base = vec![0.0_f32; 16];
        let mut candidate = vec![0.0_f32; 18];
        candidate[15] = 1.0;
        let error = check_bucketed("test", &base, &candidate, 6).unwrap_err();
        assert!(error.contains("slot=7"));
    }

    #[test]
    fn exact_check_compares_float_bits() {
        let positive_zero = [0.0_f32];
        let negative_zero = [-0.0_f32];
        assert!(check_exact("test", &positive_zero, &negative_zero).is_err());
    }
}
