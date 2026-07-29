//! `progress-bucket-survey`: PSV上のprogress8kpabs分布を再現可能に調査する。
//!
//! 従来の先頭/stride読みを残しつつ、`--output-dir`指定時は複数PSVを一つの
//! 母集団として扱い、seed固定の重複なし無作為抽出を行う。affine候補の明示比較に加え、
//! 一度読み込んだsampleのbaseline logitから、uniformまたは明示したbucket比率へ近づける
//! `a,b`を`a > 0`制約付きで決定的に最適化できる。生成候補の採用は行わない。

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use nnue_train::dataloader::{PruneBands, PruneConfig};
use serde::Serialize;
use shogi_features::ShogiProgressKPAbs;
use shogi_features::progress_kpabs::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;
use shogi_format::PackedSfenValue;

const PSV_RECORD_BYTES: u64 = size_of::<PackedSfenValue>() as u64;
const EXPECTED_ACTIVE_INDICES: usize = 76;
const SAMPLE_PLAN_MAGIC: &[u8; 4] = b"TSP1";

#[derive(Parser, Debug)]
#[command(name = "progress-bucket-survey")]
#[command(about = "Survey and optimize progress8kpabs with deterministic one-pass sampling")]
struct Args {
    /// PSV data files. Comma-separated paths are accepted.
    #[arg(long)]
    data: String,

    /// Baseline legacy progress.bin.
    #[arg(long)]
    progress: PathBuf,

    /// Legacy sequential mode sample count, or random mode total when --split is omitted.
    #[arg(long, default_value_t = 50_000)]
    samples: usize,

    /// Legacy sequential mode stride.
    #[arg(long, default_value_t = 1)]
    stride: u64,

    /// Legacy sequential mode starting record offset per file.
    #[arg(long, default_value_t = 0)]
    offset: u64,

    /// Print a per-file histogram in legacy sequential mode.
    #[arg(long)]
    per_pack: bool,

    /// Number of fixed-width progress buckets.
    #[arg(long, default_value_t = 8)]
    num_buckets: usize,

    /// Random survey output directory. Existing non-empty directories are rejected.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Random sampling seed. Required with --output-dir.
    #[arg(long)]
    seed: Option<u64>,

    /// Named random split as NAME:COUNT. Repeat for calibration/selection/test.
    #[arg(long = "split")]
    splits: Vec<String>,

    /// Explicit affine candidate NAME:A:B. Repeat as needed. No candidate is auto-adopted.
    #[arg(long = "candidate")]
    candidates: Vec<String>,

    /// Optimize one monotone affine candidate toward a uniform bucket distribution.
    #[arg(long)]
    optimize_affine: bool,

    /// Split used only for affine optimization. Other splits remain evaluation-only.
    #[arg(long, default_value = "calibration")]
    optimize_split: String,

    /// Name of the generated optimized affine candidate.
    #[arg(long, default_value = "optimized-uniform")]
    optimized_candidate_name: String,

    /// Target bucket percentages as a comma-separated list summing to 100.
    /// When omitted, every bucket receives the uniform target.
    #[arg(long)]
    optimizer_target_percentages: Option<String>,

    /// Grid points per axis for each deterministic affine optimization round.
    #[arg(long, default_value_t = 257)]
    optimizer_grid_points: usize,

    /// Number of local refinement rounds after the full-range grid search.
    #[arg(long, default_value_t = 6)]
    optimizer_refinements: usize,

    /// Diagnostic saturation threshold p<=eps or p>=1-eps.
    #[arg(long, default_value_t = 1.0e-6)]
    saturation_epsilon: f64,

    /// 評価値依存の保持帯。random surveyで学習時の間引き後分布を再現する。
    #[arg(long)]
    prune_bands: Option<PruneBands>,

    /// 間引き判定へ渡すseed。
    #[arg(long, default_value_t = 0)]
    prune_seed: u64,

    /// 間引き判定へ渡すepoch。
    #[arg(long, default_value_t = 0)]
    prune_epoch: u64,
}

#[derive(Debug, Clone, Serialize)]
struct DataFileInfo {
    path: PathBuf,
    bytes: u64,
    records: u64,
    global_start: u64,
}

#[derive(Debug, Clone)]
struct SplitSpec {
    name: String,
    count: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    name: String,
    a: f64,
    b: f64,
    weights: Vec<f32>,
}

#[derive(Clone, Copy)]
struct ObservedSample {
    global_index: u64,
    split_index: usize,
    baseline_logit: f32,
    psv: PackedSfenValue,
}

#[derive(Debug, Clone, Copy)]
struct TargetScore {
    mean_squared_error: f64,
    max_abs_deviation: f64,
    boundary_fit_mse: f64,
}

#[derive(Debug)]
struct AffineOptimization {
    a: f64,
    b: f64,
    score: TargetScore,
    histogram: Vec<u64>,
    evaluations: u64,
    logit_min: f64,
    logit_max: f64,
}

#[derive(Debug, Clone, Copy)]
struct PlannedSample {
    global_index: u64,
    split_index: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct FixtureMeta {
    global_index: u64,
    progress: f32,
}

#[derive(Clone, Copy)]
struct Fixture {
    meta: FixtureMeta,
    psv: PackedSfenValue,
}

#[derive(Clone, Default)]
struct BoundaryPair {
    below: Option<Fixture>,
    above: Option<Fixture>,
}

#[derive(Debug, Clone)]
struct SplitAccumulator {
    positions: u64,
    histogram: Vec<u64>,
    migration_from_baseline: Vec<u64>,
    boundary_crossings: Vec<u64>,
    saturation_low: u64,
    saturation_high: u64,
    active_min: usize,
    active_max: usize,
}

impl SplitAccumulator {
    fn new(num_buckets: usize) -> Self {
        Self {
            positions: 0,
            histogram: vec![0; num_buckets],
            migration_from_baseline: vec![0; num_buckets * num_buckets],
            boundary_crossings: vec![0; num_buckets.saturating_sub(1)],
            saturation_low: 0,
            saturation_high: 0,
            active_min: usize::MAX,
            active_max: 0,
        }
    }

    fn record(
        &mut self,
        progress: f32,
        bucket: usize,
        baseline_bucket: usize,
        active: usize,
        epsilon: f64,
    ) {
        self.positions += 1;
        self.histogram[bucket] += 1;
        let n = self.histogram.len();
        self.migration_from_baseline[baseline_bucket * n + bucket] += 1;
        for boundary in 1..n {
            if (baseline_bucket < boundary) != (bucket < boundary) {
                self.boundary_crossings[boundary - 1] += 1;
            }
        }
        if f64::from(progress) <= epsilon {
            self.saturation_low += 1;
        }
        if f64::from(progress) >= 1.0 - epsilon {
            self.saturation_high += 1;
        }
        self.active_min = self.active_min.min(active);
        self.active_max = self.active_max.max(active);
    }
}

#[derive(Debug, Serialize)]
struct SplitReport {
    positions: u64,
    histogram: Vec<u64>,
    percentages: Vec<f64>,
    migration_from_baseline: Vec<Vec<u64>>,
    boundary_crossing_rates: Vec<f64>,
    total_variation_from_baseline: f64,
    saturation_low: u64,
    saturation_high: u64,
    active_index_min: usize,
    active_index_max: usize,
    uniformity_mean_squared_error: f64,
    uniformity_max_abs_deviation_percentage_points: f64,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    name: String,
    a: f64,
    b: f64,
    generated_progress_bin: Option<PathBuf>,
    boundary_fixture_psv: PathBuf,
    boundary_fixture_complete: bool,
    splits: BTreeMap<String, SplitReport>,
}

#[derive(Debug, Serialize)]
struct AffineOptimizationReport {
    source_split: String,
    method: &'static str,
    objective: &'static str,
    tie_breakers: &'static str,
    parameterization: &'static str,
    search_domain: &'static str,
    monotonicity_constraint: &'static str,
    target_percentages: Vec<f64>,
    target_percentage_per_bucket: Option<f64>,
    grid_points_per_axis: usize,
    refinement_rounds: usize,
    evaluations: u64,
    calibration_logit_min: f64,
    calibration_logit_max: f64,
    candidate_name: String,
    a: f64,
    b: f64,
    theoretical_calibration_histogram: Vec<u64>,
    theoretical_target_mean_squared_error: f64,
    theoretical_target_max_abs_deviation_percentage_points: f64,
    theoretical_uniformity_mean_squared_error: f64,
    theoretical_max_abs_deviation_percentage_points: f64,
    teacher_data_passes: u32,
}

#[derive(Debug, Serialize)]
struct SurveyReport {
    schema_version: u32,
    algorithm: &'static str,
    seed: u64,
    num_buckets: usize,
    saturation_epsilon: f64,
    expected_active_indices: usize,
    data_files: Vec<DataFileInfo>,
    sample_plan: PathBuf,
    splits: BTreeMap<String, usize>,
    prune_bands: Option<String>,
    prune_seed: Option<u64>,
    prune_epoch: Option<u64>,
    candidates: Vec<CandidateReport>,
    affine_optimization: Option<AffineOptimizationReport>,
    automatic_adoption: bool,
}

fn parse_data_paths(value: &str) -> Result<Vec<PathBuf>, String> {
    let paths: Vec<_> = value
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if paths.is_empty() {
        return Err("--data is required (comma-separated PSV files)".to_string());
    }
    Ok(paths)
}

fn parse_splits(values: &[String], fallback: usize) -> Result<Vec<SplitSpec>, String> {
    if values.is_empty() {
        if fallback == 0 {
            return Err("--samples must be >= 1".to_string());
        }
        return Ok(vec![SplitSpec {
            name: "survey".to_string(),
            count: fallback,
        }]);
    }
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let (name, count) = value
            .split_once(':')
            .ok_or_else(|| format!("invalid --split '{value}', expected NAME:COUNT"))?;
        validate_name(name)?;
        let count = count
            .parse::<usize>()
            .map_err(|_| format!("invalid --split count in '{value}'"))?;
        if count == 0 {
            return Err(format!("--split count must be >= 1: '{value}'"));
        }
        if out.iter().any(|s: &SplitSpec| s.name == name) {
            return Err(format!("duplicate --split name: {name}"));
        }
        out.push(SplitSpec {
            name: name.to_string(),
            count,
        });
    }
    Ok(out)
}

fn parse_optimizer_target_percentages(
    value: Option<&str>,
    num_buckets: usize,
) -> Result<Vec<f64>, String> {
    if value.is_none() {
        return Ok(vec![100.0 / num_buckets as f64; num_buckets]);
    }
    let value = value.expect("checked above");
    let percentages: Vec<f64> = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            part.parse::<f64>().map_err(|_| {
                format!("invalid --optimizer-target-percentages value '{part}' in '{value}'")
            })
        })
        .collect::<Result<_, _>>()?;
    if percentages.len() != num_buckets {
        return Err(format!(
            "--optimizer-target-percentages requires exactly {num_buckets} values (got {})",
            percentages.len()
        ));
    }
    if percentages
        .iter()
        .any(|percentage| !percentage.is_finite() || *percentage <= 0.0)
    {
        return Err(
            "--optimizer-target-percentages requires finite values greater than zero".to_string(),
        );
    }
    let sum: f64 = percentages.iter().sum();
    if (sum - 100.0).abs() > 1.0e-6 {
        return Err(format!(
            "--optimizer-target-percentages must sum to 100 (got {sum})"
        ));
    }
    Ok(percentages)
}

fn target_is_uniform(percentages: &[f64]) -> bool {
    let target = 100.0 / percentages.len() as f64;
    percentages
        .iter()
        .all(|percentage| (*percentage - target).abs() <= 1.0e-12)
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    {
        return Err(format!(
            "name must contain only ASCII letters, digits, '.', '_' or '-': '{name}'"
        ));
    }
    Ok(())
}

fn load_progress_weights(path: &Path) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read progress '{}': {e}", path.display()))?;
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
    if bytes.len() != expected {
        return Err(format!(
            "progress.bin size mismatch: got {}, expected {expected}",
            bytes.len()
        ));
    }
    let weights: Vec<f64> = bytes
        .chunks_exact(size_of::<f64>())
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("checked chunk")))
        .collect();
    if let Some((index, value)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "non-finite progress weight at index {index}: {value}"
        ));
    }
    Ok(weights)
}

fn parse_candidates(values: &[String], baseline: &[f64]) -> Result<Vec<Candidate>, String> {
    let mut out = vec![build_candidate("baseline", 1.0, 0.0, baseline)];
    for value in values {
        let fields: Vec<_> = value.split(':').collect();
        if fields.len() != 3 {
            return Err(format!("invalid --candidate '{value}', expected NAME:A:B"));
        }
        let name = fields[0];
        validate_name(name)?;
        if name == "baseline" || out.iter().any(|c| c.name == name) {
            return Err(format!("duplicate/reserved candidate name: {name}"));
        }
        let a = fields[1]
            .parse::<f64>()
            .map_err(|_| format!("invalid candidate a in '{value}'"))?;
        let b = fields[2]
            .parse::<f64>()
            .map_err(|_| format!("invalid candidate b in '{value}'"))?;
        if !(a.is_finite() && a > 0.0 && b.is_finite()) {
            return Err(format!(
                "candidate requires finite a>0 and finite b: '{value}'"
            ));
        }
        out.push(build_candidate(name, a, b, baseline));
    }
    Ok(out)
}

fn build_candidate(name: &str, a: f64, b: f64, baseline: &[f64]) -> Candidate {
    let bias_per_active = b / EXPECTED_ACTIVE_INDICES as f64;
    let weights = baseline
        .iter()
        // Engineもbaseline読込時にf64をf32へ変換するため、その実効値へaffineを適用する。
        .map(|&weight| (a * f64::from(weight as f32) + bias_per_active) as f32)
        .collect();
    Candidate {
        name: name.to_string(),
        a,
        b,
        weights,
    }
}

fn inspect_data_files(paths: &[PathBuf]) -> io::Result<Vec<DataFileInfo>> {
    let mut global_start = 0u64;
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = std::fs::metadata(path)?.len();
        if bytes == 0 || bytes % PSV_RECORD_BYTES != 0 {
            return Err(io::Error::other(format!(
                "PSV file must be non-empty and {PSV_RECORD_BYTES}-byte aligned: {} ({bytes} bytes)",
                path.display()
            )));
        }
        let records = bytes / PSV_RECORD_BYTES;
        out.push(DataFileInfo {
            path: path.clone(),
            bytes,
            records,
            global_start,
        });
        global_start = global_start
            .checked_add(records)
            .ok_or_else(|| io::Error::other("total record count overflow"))?;
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, upper: u64) -> u64 {
        debug_assert!(upper > 0);
        let threshold = upper.wrapping_neg() % upper;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % upper;
            }
        }
    }
}

fn build_sample_plan(
    total_records: u64,
    splits: &[SplitSpec],
    seed: u64,
) -> Result<Vec<PlannedSample>, String> {
    let requested: usize = splits.iter().map(|s| s.count).sum();
    if requested as u128 > total_records as u128 {
        return Err(format!(
            "requested {requested} unique samples from only {total_records} records"
        ));
    }
    let mut rng = SplitMix64(seed);
    let mut selected: HashMap<u64, usize> = HashMap::with_capacity(requested);
    while selected.len() < requested {
        let global_index = rng.below(total_records);
        let sequence = selected.len();
        selected.entry(global_index).or_insert(sequence);
    }

    let mut cumulative = Vec::with_capacity(splits.len());
    let mut sum = 0usize;
    for split in splits {
        sum += split.count;
        cumulative.push(sum);
    }
    let mut plan: Vec<_> = selected
        .into_iter()
        .map(|(global_index, sequence)| PlannedSample {
            global_index,
            split_index: cumulative.partition_point(|&end| end <= sequence),
        })
        .collect();
    plan.sort_unstable_by_key(|sample| sample.global_index);
    Ok(plan)
}

fn write_sample_plan(path: &Path, seed: u64, plan: &[PlannedSample]) -> io::Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(SAMPLE_PLAN_MAGIC)?;
    writer.write_all(&1u32.to_le_bytes())?;
    writer.write_all(&seed.to_le_bytes())?;
    writer.write_all(&(plan.len() as u64).to_le_bytes())?;
    for sample in plan {
        writer.write_all(&sample.global_index.to_le_bytes())?;
        writer.write_all(&(sample.split_index as u16).to_le_bytes())?;
    }
    writer.flush()
}

fn sigmoid(sum: f32) -> f32 {
    (1.0 / (1.0 + (-sum).exp())).clamp(0.0, 1.0)
}

fn progress_for_indices(weights: &[f32], indices: &[usize]) -> f32 {
    sigmoid(logit_for_indices(weights, indices))
}

fn logit_for_indices(weights: &[f32], indices: &[usize]) -> f32 {
    let mut sum = 0.0f32;
    for &index in indices {
        sum += weights[index];
    }
    sum
}

fn progress_bucket(progress: f32, num_buckets: usize) -> usize {
    ((progress * num_buckets as f32).floor() as usize).min(num_buckets - 1)
}

fn probability_logit(probability: f64) -> f64 {
    (probability / (1.0 - probability)).ln()
}

fn quantile_sorted(sorted: &[f32], probability: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&probability));
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    let lower_value = f64::from(sorted[lower]);
    let upper_value = f64::from(sorted[upper]);
    lower_value + fraction * (upper_value - lower_value)
}

fn histogram_for_affine(sorted_logits: &[f32], a: f64, b: f64, num_buckets: usize) -> Vec<u64> {
    let mut histogram = Vec::with_capacity(num_buckets);
    let mut previous = 0usize;
    for boundary in 1..num_buckets {
        let transformed_boundary = probability_logit(boundary as f64 / num_buckets as f64);
        let baseline_boundary = (transformed_boundary - b) / a;
        let next = sorted_logits.partition_point(|&value| f64::from(value) < baseline_boundary);
        histogram.push((next - previous) as u64);
        previous = next;
    }
    histogram.push((sorted_logits.len() - previous) as u64);
    histogram
}

fn distribution_score(histogram: &[u64], target_percentages: &[f64]) -> (f64, f64) {
    debug_assert_eq!(histogram.len(), target_percentages.len());
    let total = histogram.iter().sum::<u64>() as f64;
    let mut squared_error = 0.0;
    let mut max_abs_deviation = 0.0f64;
    for (&count, &target_percentage) in histogram.iter().zip(target_percentages) {
        let deviation = count as f64 / total - target_percentage / 100.0;
        squared_error += deviation * deviation;
        max_abs_deviation = max_abs_deviation.max(deviation.abs());
    }
    (squared_error / histogram.len() as f64, max_abs_deviation)
}

fn target_score(
    sorted_logits: &[f32],
    target_quantiles: &[f64],
    target_percentages: &[f64],
    a: f64,
    b: f64,
    num_buckets: usize,
) -> TargetScore {
    let histogram = histogram_for_affine(sorted_logits, a, b, num_buckets);
    let (mean_squared_error, max_abs_deviation) =
        distribution_score(&histogram, target_percentages);

    let boundary_fit_mse = target_quantiles
        .iter()
        .enumerate()
        .map(|(index, &baseline_quantile)| {
            let target_logit = probability_logit((index + 1) as f64 / num_buckets as f64);
            let residual = a * baseline_quantile + b - target_logit;
            residual * residual
        })
        .sum::<f64>()
        / target_quantiles.len().max(1) as f64;

    TargetScore {
        mean_squared_error,
        max_abs_deviation,
        boundary_fit_mse,
    }
}

fn score_is_better(
    candidate: TargetScore,
    candidate_a: f64,
    candidate_b: f64,
    current: TargetScore,
    current_a: f64,
    current_b: f64,
) -> bool {
    candidate
        .mean_squared_error
        .total_cmp(&current.mean_squared_error)
        .then_with(|| {
            candidate
                .max_abs_deviation
                .total_cmp(&current.max_abs_deviation)
        })
        .then_with(|| {
            candidate
                .boundary_fit_mse
                .total_cmp(&current.boundary_fit_mse)
        })
        .then_with(|| candidate_a.total_cmp(&current_a))
        .then_with(|| candidate_b.total_cmp(&current_b))
        .is_lt()
}

fn optimize_affine_for_target(
    sorted_logits: &[f32],
    num_buckets: usize,
    target_percentages: &[f64],
    grid_points: usize,
    refinement_rounds: usize,
) -> Result<AffineOptimization, String> {
    if sorted_logits.len() < num_buckets {
        return Err(format!(
            "optimization split needs at least {num_buckets} positions"
        ));
    }
    if grid_points < 3 || grid_points.is_multiple_of(2) {
        return Err("--optimizer-grid-points must be an odd integer >= 3".to_string());
    }
    if sorted_logits.iter().any(|value| !value.is_finite()) {
        return Err("optimization split contains a non-finite baseline logit".to_string());
    }
    if target_percentages.len() != num_buckets
        || target_percentages
            .iter()
            .any(|percentage| !percentage.is_finite() || *percentage <= 0.0)
        || (target_percentages.iter().sum::<f64>() - 100.0).abs() > 1.0e-6
    {
        return Err(format!(
            "optimizer target requires {num_buckets} finite positive percentages summing to 100"
        ));
    }

    let logit_min = f64::from(sorted_logits[0]);
    let logit_max = f64::from(sorted_logits[sorted_logits.len() - 1]);
    if logit_min >= logit_max {
        return Err("optimization split baseline logits have no variation".to_string());
    }

    let mut cumulative_target = 0.0;
    let target_quantiles: Vec<_> = target_percentages
        .iter()
        .take(num_buckets - 1)
        .map(|percentage| {
            cumulative_target += percentage / 100.0;
            quantile_sorted(sorted_logits, cumulative_target)
        })
        .collect();
    let first_target_logit = probability_logit(1.0 / num_buckets as f64);
    let last_target_logit = probability_logit((num_buckets - 1) as f64 / num_buckets as f64);

    let mut first_low = logit_min;
    let mut first_high = logit_max;
    let mut last_low = logit_min;
    let mut last_high = logit_max;
    let mut best: Option<(TargetScore, f64, f64)> = None;
    let mut evaluations = 0u64;

    for _ in 0..=refinement_rounds {
        let first_step = (first_high - first_low) / (grid_points - 1) as f64;
        let last_step = (last_high - last_low) / (grid_points - 1) as f64;
        let mut round_best: Option<(TargetScore, f64, f64)> = None;

        for first_index in 0..grid_points {
            let first_boundary = first_low + first_step * first_index as f64;
            for last_index in 0..grid_points {
                let last_boundary = last_low + last_step * last_index as f64;
                if first_boundary >= last_boundary {
                    continue;
                }
                let a = (last_target_logit - first_target_logit) / (last_boundary - first_boundary);
                let b = first_target_logit - a * first_boundary;
                if !(a.is_finite() && a > 0.0 && b.is_finite()) {
                    continue;
                }
                let score = target_score(
                    sorted_logits,
                    &target_quantiles,
                    target_percentages,
                    a,
                    b,
                    num_buckets,
                );
                evaluations += 1;
                if round_best.is_none_or(|(current, current_a, current_b)| {
                    score_is_better(score, a, b, current, current_a, current_b)
                }) {
                    round_best = Some((score, a, b));
                }
            }
        }

        let Some((round_score, round_a, round_b)) = round_best else {
            return Err("affine optimizer found no valid a>0 candidate".to_string());
        };
        if best.is_none_or(|(current, current_a, current_b)| {
            score_is_better(round_score, round_a, round_b, current, current_a, current_b)
        }) {
            best = Some((round_score, round_a, round_b));
        }

        let best_first = (first_target_logit - round_b) / round_a;
        let best_last = (last_target_logit - round_b) / round_a;
        first_low = (best_first - first_step).max(first_low);
        first_high = (best_first + first_step).min(first_high);
        last_low = (best_last - last_step).max(last_low);
        last_high = (best_last + last_step).min(last_high);
    }

    let (score, a, b) = best.expect("at least one valid grid point");
    Ok(AffineOptimization {
        a,
        b,
        score,
        histogram: histogram_for_affine(sorted_logits, a, b, num_buckets),
        evaluations,
        logit_min,
        logit_max,
    })
}

fn update_boundaries(
    boundaries: &mut [BoundaryPair],
    progress: f32,
    global_index: u64,
    psv: PackedSfenValue,
    num_buckets: usize,
) {
    for (index, pair) in boundaries.iter_mut().enumerate() {
        let boundary = (index + 1) as f32 / num_buckets as f32;
        let fixture = Fixture {
            meta: FixtureMeta {
                global_index,
                progress,
            },
            psv,
        };
        if progress < boundary {
            if pair
                .below
                .is_none_or(|old| boundary - progress < boundary - old.meta.progress)
            {
                pair.below = Some(fixture);
            }
        } else if pair
            .above
            .is_none_or(|old| progress - boundary < old.meta.progress - boundary)
        {
            pair.above = Some(fixture);
        }
    }
}

fn write_candidate_progress(path: &Path, candidate: &Candidate) -> io::Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    for &weight in &candidate.weights {
        writer.write_all(&f64::from(weight).to_le_bytes())?;
    }
    writer.flush()
}

#[derive(Serialize)]
struct BoundaryFixtureReport {
    boundary: f64,
    below: Option<FixtureMeta>,
    above: Option<FixtureMeta>,
    below_record_index: Option<usize>,
    above_record_index: Option<usize>,
}

fn write_boundary_fixtures(
    output_dir: &Path,
    candidate_name: &str,
    boundaries: &[BoundaryPair],
    num_buckets: usize,
) -> Result<(PathBuf, bool), Box<dyn std::error::Error>> {
    let psv_path = output_dir.join(format!("boundary-{candidate_name}.psv"));
    let json_path = output_dir.join(format!("boundary-{candidate_name}.json"));
    let mut psv_writer = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&psv_path)?,
    );
    let mut reports = Vec::with_capacity(boundaries.len());
    let mut next_record_index = 0usize;
    let mut complete = true;
    for (index, pair) in boundaries.iter().enumerate() {
        let below_record_index = if let Some(below) = pair.below {
            let record_index = next_record_index;
            next_record_index += 1;
            psv_writer.write_all(below.psv.as_bytes())?;
            Some(record_index)
        } else {
            complete = false;
            None
        };
        let above_record_index = if let Some(above) = pair.above {
            let record_index = next_record_index;
            next_record_index += 1;
            psv_writer.write_all(above.psv.as_bytes())?;
            Some(record_index)
        } else {
            complete = false;
            None
        };
        reports.push(BoundaryFixtureReport {
            boundary: (index + 1) as f64 / num_buckets as f64,
            below: pair.below.map(|fixture| fixture.meta),
            above: pair.above.map(|fixture| fixture.meta),
            below_record_index,
            above_record_index,
        });
    }
    psv_writer.flush()?;
    let json = serde_json::to_vec_pretty(&reports)?;
    let mut json_writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(json_path)?;
    json_writer.write_all(&json)?;
    json_writer.write_all(b"\n")?;
    Ok((psv_path, complete))
}

fn percentages(histogram: &[u64]) -> Vec<f64> {
    let total: u64 = histogram.iter().sum();
    if total == 0 {
        return vec![0.0; histogram.len()];
    }
    histogram
        .iter()
        .map(|&count| 100.0 * count as f64 / total as f64)
        .collect()
}

fn total_variation(histogram: &[u64], baseline: &[u64]) -> f64 {
    let total: u64 = histogram.iter().sum();
    let baseline_total: u64 = baseline.iter().sum();
    if total == 0 || baseline_total == 0 {
        return 0.0;
    }
    0.5 * histogram
        .iter()
        .zip(baseline)
        .map(|(&a, &b)| (a as f64 / total as f64 - b as f64 / baseline_total as f64).abs())
        .sum::<f64>()
}

fn build_split_report(acc: &SplitAccumulator, baseline: &SplitAccumulator) -> SplitReport {
    let n = acc.histogram.len();
    let migration = acc
        .migration_from_baseline
        .chunks_exact(n)
        .map(<[u64]>::to_vec)
        .collect();
    let positions = acc.positions.max(1) as f64;
    let target = 1.0 / n as f64;
    let deviations: Vec<_> = acc
        .histogram
        .iter()
        .map(|&count| count as f64 / positions - target)
        .collect();
    SplitReport {
        positions: acc.positions,
        histogram: acc.histogram.clone(),
        percentages: percentages(&acc.histogram),
        migration_from_baseline: migration,
        boundary_crossing_rates: acc
            .boundary_crossings
            .iter()
            .map(|&count| count as f64 / positions)
            .collect(),
        total_variation_from_baseline: total_variation(&acc.histogram, &baseline.histogram),
        saturation_low: acc.saturation_low,
        saturation_high: acc.saturation_high,
        active_index_min: acc.active_min,
        active_index_max: acc.active_max,
        uniformity_mean_squared_error: deviations
            .iter()
            .map(|deviation| deviation * deviation)
            .sum::<f64>()
            / n as f64,
        uniformity_max_abs_deviation_percentage_points: 100.0
            * deviations
                .iter()
                .map(|deviation| deviation.abs())
                .fold(0.0f64, f64::max),
    }
}

fn ensure_output_dir(path: &Path) -> io::Result<()> {
    if path.exists() {
        if std::fs::read_dir(path)?.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("output directory is not empty: {}", path.display()),
            ));
        }
    } else {
        std::fs::create_dir(path)?;
    }
    Ok(())
}

fn run_random_survey(
    args: &Args,
    data_paths: &[PathBuf],
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let seed = args.seed.ok_or("--seed is required with --output-dir")?;
    if !(args.saturation_epsilon.is_finite()
        && args.saturation_epsilon > 0.0
        && args.saturation_epsilon < 0.5)
    {
        return Err("--saturation-epsilon must be finite and in (0, 0.5)".into());
    }
    let optimizer_target_percentages = parse_optimizer_target_percentages(
        args.optimizer_target_percentages.as_deref(),
        args.num_buckets,
    )?;
    ensure_output_dir(output_dir)?;
    let files = inspect_data_files(data_paths)?;
    let total_records: u64 = files.iter().map(|file| file.records).sum();
    let splits = parse_splits(&args.splits, args.samples)?;
    let plan = build_sample_plan(total_records, &splits, seed)?;
    let plan_path = output_dir.join("sample-plan.bin");
    write_sample_plan(&plan_path, seed, &plan)?;

    let baseline_weights = load_progress_weights(&args.progress)?;
    let mut candidates = parse_candidates(&args.candidates, &baseline_weights)?;
    let prune = args
        .prune_bands
        .clone()
        .map(|bands| PruneConfig::new(bands, 0.0, args.prune_seed))
        .transpose()?;
    let mut current_file_index = usize::MAX;
    let mut current_file: Option<File> = None;
    let mut active_indices = Vec::with_capacity(EXPECTED_ACTIVE_INDICES);
    let mut observations = Vec::with_capacity(plan.len());
    for sample in &plan {
        let file_index =
            files.partition_point(|file| file.global_start + file.records <= sample.global_index);
        let info = files
            .get(file_index)
            .ok_or("sample index did not map to a data file")?;
        if current_file_index != file_index {
            current_file = Some(File::open(&info.path)?);
            current_file_index = file_index;
        }
        let local_index = sample.global_index - info.global_start;
        let file = current_file.as_mut().expect("opened above");
        file.seek(SeekFrom::Start(local_index * PSV_RECORD_BYTES))?;
        let mut psv = PackedSfenValue::default();
        file.read_exact(psv.as_bytes_mut())?;
        if prune.as_ref().is_some_and(|config| {
            !config
                .sample(psv.score(), sample.global_index, args.prune_epoch)
                .0
        }) {
            continue;
        }
        ShogiProgressKPAbs::collect_active_indices(&psv, &mut active_indices);
        if active_indices.len() != EXPECTED_ACTIVE_INDICES {
            return Err(format!(
                "active index count mismatch: expected {EXPECTED_ACTIVE_INDICES}, observed {} at global index {}",
                active_indices.len(), sample.global_index
            )
            .into());
        }
        observations.push(ObservedSample {
            global_index: sample.global_index,
            split_index: sample.split_index,
            baseline_logit: logit_for_indices(&candidates[0].weights, &active_indices),
            psv,
        });
    }
    drop(plan);

    let affine_optimization = if args.optimize_affine {
        validate_name(&args.optimize_split)?;
        validate_name(&args.optimized_candidate_name)?;
        if candidates
            .iter()
            .any(|candidate| candidate.name == args.optimized_candidate_name)
        {
            return Err(format!(
                "duplicate/reserved optimized candidate name: {}",
                args.optimized_candidate_name
            )
            .into());
        }
        let optimization_split_index = splits
            .iter()
            .position(|split| split.name == args.optimize_split)
            .ok_or_else(|| {
                format!(
                    "--optimize-split '{}' is not present in --split",
                    args.optimize_split
                )
            })?;
        let mut sorted_logits: Vec<_> = observations
            .iter()
            .filter(|sample| sample.split_index == optimization_split_index)
            .map(|sample| sample.baseline_logit)
            .collect();
        sorted_logits.sort_unstable_by(f32::total_cmp);
        let optimized = optimize_affine_for_target(
            &sorted_logits,
            args.num_buckets,
            &optimizer_target_percentages,
            args.optimizer_grid_points,
            args.optimizer_refinements,
        )?;
        let uniform_target = vec![100.0 / args.num_buckets as f64; args.num_buckets];
        let (uniform_mse, uniform_max_abs_deviation) =
            distribution_score(&optimized.histogram, &uniform_target);
        let uniform_target_requested = target_is_uniform(&optimizer_target_percentages);
        candidates.push(build_candidate(
            &args.optimized_candidate_name,
            optimized.a,
            optimized.b,
            &baseline_weights,
        ));
        Some(AffineOptimizationReport {
            source_split: args.optimize_split.clone(),
            method: "deterministic-full-range-grid-with-local-refinement-v1",
            objective: if uniform_target_requested {
                "mean-squared-error-of-bucket-fractions-from-uniform"
            } else {
                "mean-squared-error-of-bucket-fractions-from-explicit-target"
            },
            tie_breakers: "max-absolute-bucket-deviation,boundary-logit-fit-mse,a,b",
            parameterization: "baseline-logit-boundaries-for-first-and-last-bucket",
            search_domain: "both-extreme-boundaries-within-calibration-logit-min-max",
            monotonicity_constraint: "a>0",
            target_percentages: optimizer_target_percentages.clone(),
            target_percentage_per_bucket: uniform_target_requested
                .then_some(100.0 / args.num_buckets as f64),
            grid_points_per_axis: args.optimizer_grid_points,
            refinement_rounds: args.optimizer_refinements,
            evaluations: optimized.evaluations,
            calibration_logit_min: optimized.logit_min,
            calibration_logit_max: optimized.logit_max,
            candidate_name: args.optimized_candidate_name.clone(),
            a: optimized.a,
            b: optimized.b,
            theoretical_calibration_histogram: optimized.histogram,
            theoretical_target_mean_squared_error: optimized.score.mean_squared_error,
            theoretical_target_max_abs_deviation_percentage_points: 100.0
                * optimized.score.max_abs_deviation,
            theoretical_uniformity_mean_squared_error: uniform_mse,
            theoretical_max_abs_deviation_percentage_points: 100.0 * uniform_max_abs_deviation,
            teacher_data_passes: 1,
        })
    } else {
        None
    };

    let mut accumulators: Vec<Vec<SplitAccumulator>> = candidates
        .iter()
        .map(|_| {
            splits
                .iter()
                .map(|_| SplitAccumulator::new(args.num_buckets))
                .collect()
        })
        .collect();
    let mut boundaries =
        vec![vec![BoundaryPair::default(); args.num_buckets.saturating_sub(1)]; candidates.len()];

    for sample in &observations {
        ShogiProgressKPAbs::collect_active_indices(&sample.psv, &mut active_indices);
        let candidate_progress: Vec<f32> = candidates
            .iter()
            .map(|candidate| progress_for_indices(&candidate.weights, &active_indices))
            .collect();
        let baseline_bucket = progress_bucket(candidate_progress[0], args.num_buckets);
        for (candidate_index, &progress) in candidate_progress.iter().enumerate() {
            let bucket = progress_bucket(progress, args.num_buckets);
            accumulators[candidate_index][sample.split_index].record(
                progress,
                bucket,
                baseline_bucket,
                active_indices.len(),
                args.saturation_epsilon,
            );
            if sample.split_index == 0 {
                update_boundaries(
                    &mut boundaries[candidate_index],
                    progress,
                    sample.global_index,
                    sample.psv,
                    args.num_buckets,
                );
            }
        }
    }

    for candidate_acc in &accumulators {
        for split_acc in candidate_acc {
            if split_acc.active_min != EXPECTED_ACTIVE_INDICES
                || split_acc.active_max != EXPECTED_ACTIVE_INDICES
            {
                return Err(format!(
                    "active index count mismatch: expected {EXPECTED_ACTIVE_INDICES}, observed {}..={}",
                    split_acc.active_min, split_acc.active_max
                )
                .into());
            }
        }
    }

    let mut candidate_reports = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.iter().enumerate() {
        let generated_progress_bin = if candidate_index == 0 {
            None
        } else {
            let path = output_dir.join(format!("progress-{}.bin", candidate.name));
            write_candidate_progress(&path, candidate)?;
            Some(path)
        };
        let (fixture_path, fixture_complete) = write_boundary_fixtures(
            output_dir,
            &candidate.name,
            &boundaries[candidate_index],
            args.num_buckets,
        )?;
        let mut split_reports = BTreeMap::new();
        for (split_index, split) in splits.iter().enumerate() {
            split_reports.insert(
                split.name.clone(),
                build_split_report(
                    &accumulators[candidate_index][split_index],
                    &accumulators[0][split_index],
                ),
            );
        }
        candidate_reports.push(CandidateReport {
            name: candidate.name.clone(),
            a: candidate.a,
            b: candidate.b,
            generated_progress_bin,
            boundary_fixture_psv: fixture_path,
            boundary_fixture_complete: fixture_complete,
            splits: split_reports,
        });
    }

    let report = SurveyReport {
        schema_version: 2,
        algorithm: "splitmix64-rejection-unique-global-index-v1",
        seed,
        num_buckets: args.num_buckets,
        saturation_epsilon: args.saturation_epsilon,
        expected_active_indices: EXPECTED_ACTIVE_INDICES,
        data_files: files,
        sample_plan: plan_path,
        splits: splits
            .iter()
            .map(|split| (split.name.clone(), split.count))
            .collect(),
        prune_bands: args.prune_bands.as_ref().map(ToString::to_string),
        prune_seed: args.prune_bands.as_ref().map(|_| args.prune_seed),
        prune_epoch: args.prune_bands.as_ref().map(|_| args.prune_epoch),
        candidates: candidate_reports,
        affine_optimization,
        automatic_adoption: false,
    };
    let metrics_path = output_dir.join("metrics.json");
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&metrics_path)?;
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    println!("survey metrics: {}", metrics_path.display());
    println!("no progress candidate was automatically adopted");
    Ok(())
}

fn read_samples(
    path: &Path,
    offset: u64,
    stride: u64,
    max: usize,
) -> io::Result<Vec<PackedSfenValue>> {
    let mut file = File::open(path)?;
    let total_records = file.metadata()?.len() / PSV_RECORD_BYTES;
    let mut out = Vec::new();
    if offset >= total_records {
        return Ok(out);
    }
    file.seek(SeekFrom::Start(offset * PSV_RECORD_BYTES))?;
    while out.len() < max {
        let mut psv = PackedSfenValue::default();
        match file.read_exact(psv.as_bytes_mut()) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error),
        }
        out.push(psv);
        if stride > 1 {
            let skip =
                i64::try_from((stride - 1).saturating_mul(PSV_RECORD_BYTES)).unwrap_or(i64::MAX);
            file.seek(SeekFrom::Current(skip))?;
        }
    }
    Ok(out)
}

fn print_hist(label: &str, hist: &[u64]) {
    let total: u64 = hist.iter().sum();
    println!("\n== {label} ==");
    for (index, &count) in hist.iter().enumerate() {
        let pct = if total == 0 {
            0.0
        } else {
            100.0 * count as f64 / total as f64
        };
        println!("bucket {index}: {count:>10}  ({pct:>6.2}%)");
    }
    println!("total {total}");
}

fn run_legacy(args: &Args, paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    if args.stride == 0 {
        return Err("--stride must be >= 1".into());
    }
    let kpabs = ShogiProgressKPAbs::load_from_bin(&args.progress)?;
    let mut total_hist = vec![0u64; args.num_buckets];
    let mut remaining = args.samples;
    for path in paths {
        if remaining == 0 {
            break;
        }
        let samples = read_samples(path, args.offset, args.stride, remaining)?;
        remaining -= samples.len();
        let mut pack_hist = vec![0u64; args.num_buckets];
        for psv in &samples {
            pack_hist[kpabs.bucket(psv, args.num_buckets) as usize] += 1;
        }
        for (index, count) in pack_hist.iter().enumerate() {
            total_hist[index] += count;
        }
        println!("loaded {} positions from {}", samples.len(), path.display());
        if args.per_pack {
            print_hist(&format!("per-pack: {}", path.display()), &pack_hist);
        }
    }
    if total_hist.iter().sum::<u64>() == 0 {
        return Err("no positions read from --data files".into());
    }
    print_hist("progress-kpabs bucket distribution", &total_hist);
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=9).contains(&args.num_buckets) {
        return Err(format!("--num-buckets must be in [1, 9] (got {})", args.num_buckets).into());
    }
    if args.optimize_affine {
        if args.num_buckets < 2 {
            return Err("--optimize-affine requires --num-buckets >= 2".into());
        }
        if args.optimizer_grid_points < 3 || args.optimizer_grid_points.is_multiple_of(2) {
            return Err("--optimizer-grid-points must be an odd integer >= 3".into());
        }
    } else if args.optimizer_target_percentages.is_some() {
        return Err("--optimizer-target-percentages requires --optimize-affine".into());
    }
    if args.prune_bands.is_none() && (args.prune_seed != 0 || args.prune_epoch != 0) {
        return Err("--prune-seed/--prune-epoch require --prune-bands".into());
    }
    let paths = parse_data_paths(&args.data)?;
    if let Some(output_dir) = &args.output_dir {
        run_random_survey(&args, &paths, output_dir)
    } else {
        if args.seed.is_some()
            || !args.splits.is_empty()
            || !args.candidates.is_empty()
            || args.optimize_affine
            || args.optimizer_target_percentages.is_some()
            || args.prune_bands.is_some()
        {
            return Err("--seed/--split/--candidate/--optimize-affine/--optimizer-target-percentages/--prune-bands require --output-dir".into());
        }
        run_legacy(&args, &paths)
    }
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sample_plan_is_unique_sorted_and_deterministic() {
        let splits = vec![
            SplitSpec {
                name: "calibration".to_string(),
                count: 20,
            },
            SplitSpec {
                name: "test".to_string(),
                count: 10,
            },
        ];
        let first = build_sample_plan(1_000, &splits, 42).expect("plan");
        let second = build_sample_plan(1_000, &splits, 42).expect("plan");
        assert_eq!(first.len(), 30);
        assert!(
            first
                .windows(2)
                .all(|w| w[0].global_index < w[1].global_index)
        );
        assert!(
            first
                .iter()
                .zip(&second)
                .all(|(a, b)| a.global_index == b.global_index && a.split_index == b.split_index)
        );
        assert_eq!(first.iter().filter(|s| s.split_index == 0).count(), 20);
        assert_eq!(first.iter().filter(|s| s.split_index == 1).count(), 10);
    }

    #[test]
    fn sample_plan_rejects_population_overflow() {
        let splits = vec![SplitSpec {
            name: "x".to_string(),
            count: 11,
        }];
        assert!(build_sample_plan(10, &splits, 1).is_err());
    }

    #[test]
    fn parses_explicit_affine_candidates_without_default_search() {
        let baseline = vec![1.0, -2.0];
        let candidates =
            parse_candidates(&["wide:0.8:-0.25".to_string()], &baseline).expect("candidate");
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].name, "baseline");
        assert_eq!(candidates[1].name, "wide");
        assert_eq!(candidates[1].a, 0.8);
        assert_eq!(candidates[1].b, -0.25);
    }

    #[test]
    fn optimizer_target_percentages_default_to_uniform_and_validate_explicit_target() {
        assert_eq!(
            parse_optimizer_target_percentages(None, 8).expect("uniform target"),
            vec![12.5; 8]
        );
        assert_eq!(
            parse_optimizer_target_percentages(Some("11,12,13,14,14,13,12,11"), 8,)
                .expect("center target"),
            vec![11.0, 12.0, 13.0, 14.0, 14.0, 13.0, 12.0, 11.0]
        );
        assert!(
            parse_optimizer_target_percentages(Some("12.5,12.5"), 8)
                .expect_err("wrong bucket count")
                .contains("exactly 8")
        );
        assert!(
            parse_optimizer_target_percentages(Some("10,10,10,10,10,10,10,10"), 8)
                .expect_err("wrong sum")
                .contains("sum to 100")
        );
    }

    #[test]
    fn migration_and_crossing_are_recorded() {
        let mut acc = SplitAccumulator::new(8);
        acc.record(0.10, 0, 2, 76, 1.0e-6);
        assert_eq!(acc.migration_from_baseline[2 * 8], 1);
        assert_eq!(&acc.boundary_crossings[..2], &[1, 1]);
        assert_eq!(acc.active_min, 76);
        assert_eq!(acc.active_max, 76);
    }

    #[test]
    fn bucket_clamps_progress_one_to_last_bucket() {
        assert_eq!(progress_bucket(0.0, 8), 0);
        assert_eq!(progress_bucket(0.125, 8), 1);
        assert_eq!(progress_bucket(1.0, 8), 7);
    }

    #[test]
    fn uniform_affine_optimizer_balances_a_synthetic_distribution() {
        let expected_a = 1.7;
        let expected_b = -0.35;
        let mut logits = Vec::with_capacity(8_000);
        for index in 0..8_000 {
            let probability = (index as f64 + 0.5) / 8_000.0;
            logits.push(((probability_logit(probability) - expected_b) / expected_a) as f32);
        }
        logits.sort_unstable_by(f32::total_cmp);

        let optimized =
            optimize_affine_for_target(&logits, 8, &[12.5; 8], 65, 4).expect("optimization");
        assert!(optimized.a > 0.0);
        assert_eq!(optimized.histogram.iter().sum::<u64>(), 8_000);
        assert!(
            optimized
                .histogram
                .iter()
                .all(|&count| count.abs_diff(1_000) <= 2),
            "histogram={:?}, a={}, b={}",
            optimized.histogram,
            optimized.a,
            optimized.b
        );
        assert!(optimized.score.mean_squared_error <= 1.0e-7);
    }

    #[test]
    fn affine_optimizer_matches_a_center_weighted_target() {
        let expected_a = 1.25;
        let expected_b = -0.45;
        let target_percentages = [11.0, 12.0, 13.0, 14.0, 14.0, 13.0, 12.0, 11.0];
        let target_counts = [880usize, 960, 1_040, 1_120, 1_120, 1_040, 960, 880];
        let mut logits = Vec::with_capacity(8_000);
        for (bucket, &count) in target_counts.iter().enumerate() {
            for index in 0..count {
                let within_bucket = (index as f64 + 0.5) / count as f64;
                let probability = (bucket as f64 + within_bucket) / 8.0;
                logits.push(((probability_logit(probability) - expected_b) / expected_a) as f32);
            }
        }
        logits.sort_unstable_by(f32::total_cmp);

        let optimized = optimize_affine_for_target(&logits, 8, &target_percentages, 65, 4)
            .expect("center target optimization");
        assert!(optimized.a > 0.0);
        assert!(
            optimized
                .histogram
                .iter()
                .zip(target_counts)
                .all(|(&actual, expected)| actual.abs_diff(expected as u64) <= 2),
            "histogram={:?}, a={}, b={}",
            optimized.histogram,
            optimized.a,
            optimized.b
        );
        assert!(optimized.score.mean_squared_error <= 1.0e-7);
    }

    #[test]
    fn optimized_affine_mapping_is_monotone_by_construction() {
        let logits = vec![-4.0, -2.0, -0.5, 0.0, 0.75, 1.5, 3.0, 5.0];
        let optimized =
            optimize_affine_for_target(&logits, 8, &[12.5; 8], 33, 3).expect("optimization");
        let transformed: Vec<_> = logits
            .iter()
            .map(|&logit| optimized.a * f64::from(logit) + optimized.b)
            .collect();
        assert!(optimized.a > 0.0);
        assert!(transformed.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn affine_optimizer_rejects_even_grid_size() {
        let logits = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let error =
            optimize_affine_for_target(&logits, 8, &[12.5; 8], 32, 2).expect_err("even grid");
        assert!(error.contains("odd integer"));
    }

    #[test]
    fn random_survey_optimizes_after_one_teacher_read() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let temporary = std::env::temp_dir().join(format!(
            "tatara-progress-survey-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&temporary).expect("temporary directory");
        let progress = temporary.join("progress.bin");
        let mut progress_writer = BufWriter::new(File::create(&progress).expect("progress file"));
        for index in 0..SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
            let weight = ((index % 997) as f64 - 498.0) / 2_000.0;
            progress_writer
                .write_all(&weight.to_le_bytes())
                .expect("progress weight");
        }
        progress_writer.flush().expect("progress flush");

        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../crates/shogi-format/tests/data/sample.psv");
        let output = temporary.join("survey");
        let args = Args {
            data: data.display().to_string(),
            progress,
            samples: 100,
            stride: 1,
            offset: 0,
            per_pack: false,
            num_buckets: 8,
            output_dir: Some(output.clone()),
            seed: Some(123),
            splits: vec![
                "calibration:60".to_string(),
                "selection:20".to_string(),
                "final-test:20".to_string(),
            ],
            candidates: Vec::new(),
            optimize_affine: true,
            optimize_split: "calibration".to_string(),
            optimized_candidate_name: "optimized-center-gentle".to_string(),
            optimizer_target_percentages: Some("11,12,13,14,14,13,12,11".to_string()),
            optimizer_grid_points: 33,
            optimizer_refinements: 2,
            saturation_epsilon: 1.0e-6,
            prune_bands: Some("inf:1.0".parse().expect("prune bands")),
            prune_seed: 7,
            prune_epoch: 3,
        };

        run_random_survey(&args, &[data], &output).expect("one-pass optimized survey");
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("metrics.json")).expect("metrics"))
                .expect("metrics JSON");
        assert_eq!(report["schema_version"], 2);
        assert_eq!(report["automatic_adoption"], false);
        assert_eq!(report["prune_bands"], "inf:1");
        assert_eq!(report["prune_seed"], 7);
        assert_eq!(report["prune_epoch"], 3);
        assert_eq!(report["affine_optimization"]["teacher_data_passes"], 1);
        assert_eq!(
            report["affine_optimization"]["target_percentages"],
            serde_json::json!([11.0, 12.0, 13.0, 14.0, 14.0, 13.0, 12.0, 11.0])
        );
        assert_eq!(
            report["affine_optimization"]["target_percentage_per_bucket"],
            serde_json::Value::Null
        );
        assert!(report["affine_optimization"]["a"].as_f64().unwrap() > 0.0);
        assert_eq!(report["candidates"][0]["name"], "baseline");
        assert_eq!(report["candidates"][1]["name"], "optimized-center-gentle");
        assert_eq!(
            std::fs::metadata(output.join("progress-optimized-center-gentle.bin"))
                .expect("optimized progress")
                .len(),
            (SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>()) as u64
        );

        std::fs::remove_dir_all(&temporary).expect("remove test temporary directory");
    }
}
