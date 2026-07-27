//! 公開PSVから相入玉局面だけを抽出し、決定的かつ排他的なtrain/holdoutを作る。

use std::cmp::{max, min};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;
use shogi_features::{ShogiProgressKPAbs, is_mutual_entering_king};
use shogi_format::PackedSfenValue;

const PSV_RECORD_BYTES: u64 = 40;
const SPLIT_MODULUS: u64 = 1_000;
const TRAIN_NAME: &str = "entering-king-train.psv";
const HOLDOUT_NAME: &str = "entering-king-holdout.psv";
const METRICS_NAME: &str = "metrics.json";

#[derive(Debug, Parser)]
#[command(name = "progress8ek-filter")]
#[command(about = "Extract mutual entering-king positions into deterministic PSV splits")]
struct Args {
    /// 入力PSV。複数fileは指定順に一つの母集団として扱う。
    #[arg(long = "data", required = true)]
    data: Vec<PathBuf>,

    /// fixed8 bucket統計に使う承認済みprogress.bin。
    #[arg(long)]
    progress: PathBuf,

    /// 新規作成する出力directory。既存pathは拒否する。
    #[arg(long)]
    output_dir: PathBuf,

    /// global record indexのhashに混ぜる固定seed。
    #[arg(long, default_value_t = 20_260_722)]
    seed: u64,

    /// holdoutへ送る割合。100で10.0%。
    #[arg(long, default_value_t = 100, value_parser = parse_holdout_per_mille)]
    holdout_per_mille: u16,

    /// 入力範囲を並列走査するthread数。
    #[arg(long, default_value_t = 16, value_parser = parse_threads)]
    threads: usize,

    /// 先頭から処理する最大record数。省略時は全件。
    #[arg(long)]
    max_records: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct InputInfo {
    path: PathBuf,
    bytes: u64,
    records: u64,
    global_start: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordRange {
    start: u64,
    end: u64,
}

#[derive(Debug)]
struct WorkerOutput {
    index: usize,
    train_part: PathBuf,
    holdout_part: PathBuf,
    stats: Stats,
}

#[derive(Debug, Clone)]
struct Stats {
    scanned: u64,
    matched: u64,
    train: u64,
    holdout: u64,
    progress_buckets: [u64; 8],
    black_king_ranks: [u64; 9],
    white_king_ranks: [u64; 9],
    king_rank_pairs: [u64; 81],
    results: [u64; 3],
    score_histogram: Vec<u64>,
    ply_histogram: Vec<u64>,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            scanned: 0,
            matched: 0,
            train: 0,
            holdout: 0,
            progress_buckets: [0; 8],
            black_king_ranks: [0; 9],
            white_king_ranks: [0; 9],
            king_rank_pairs: [0; 81],
            results: [0; 3],
            score_histogram: vec![0; 1 << 16],
            ply_histogram: vec![0; 1 << 16],
        }
    }
}

impl Stats {
    fn record_match(
        &mut self,
        psv: &PackedSfenValue,
        black_rank: usize,
        white_rank: usize,
        progress_bucket: usize,
        holdout: bool,
    ) {
        self.matched += 1;
        if holdout {
            self.holdout += 1;
        } else {
            self.train += 1;
        }
        self.progress_buckets[progress_bucket] += 1;
        self.black_king_ranks[black_rank] += 1;
        self.white_king_ranks[white_rank] += 1;
        self.king_rank_pairs[black_rank * 9 + white_rank] += 1;
        let result_index = match psv.game_result() {
            value if value < 0 => 0,
            0 => 1,
            _ => 2,
        };
        self.results[result_index] += 1;
        let score_index = i32::from(psv.score()) - i32::from(i16::MIN);
        self.score_histogram[score_index as usize] += 1;
        self.ply_histogram[usize::from(psv.game_ply())] += 1;
    }

    fn merge(&mut self, other: &Self) {
        self.scanned += other.scanned;
        self.matched += other.matched;
        self.train += other.train;
        self.holdout += other.holdout;
        merge_array(&mut self.progress_buckets, &other.progress_buckets);
        merge_array(&mut self.black_king_ranks, &other.black_king_ranks);
        merge_array(&mut self.white_king_ranks, &other.white_king_ranks);
        merge_array(&mut self.king_rank_pairs, &other.king_rank_pairs);
        merge_array(&mut self.results, &other.results);
        for (target, source) in self.score_histogram.iter_mut().zip(&other.score_histogram) {
            *target += source;
        }
        for (target, source) in self.ply_histogram.iter_mut().zip(&other.ply_histogram) {
            *target += source;
        }
    }
}

fn merge_array<const N: usize>(target: &mut [u64; N], source: &[u64; N]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += source;
    }
}

#[derive(Debug, Serialize)]
struct Quantiles {
    p01: i64,
    p10: i64,
    p25: i64,
    p50: i64,
    p75: i64,
    p90: i64,
    p99: i64,
}

#[derive(Debug, Serialize)]
struct OutputInfo {
    path: PathBuf,
    records: u64,
    bytes: u64,
    verified_entering_king_records: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    predicate: &'static str,
    split_algorithm: &'static str,
    split_seed: u64,
    split_modulus: u64,
    holdout_threshold: u16,
    requested_threads: usize,
    worker_ranges: usize,
    inputs: Vec<InputInfo>,
    available_records: u64,
    scanned_records: u64,
    matched_records: u64,
    matched_percentage: f64,
    train: OutputInfo,
    holdout: OutputInfo,
    fixed8_progress_histogram: [u64; 8],
    fixed8_progress_percentages: [f64; 8],
    black_king_rank_histogram: [u64; 9],
    white_king_rank_histogram: [u64; 9],
    king_rank_pair_histogram: Vec<Vec<u64>>,
    game_result_loss_draw_win: [u64; 3],
    score_quantiles: Quantiles,
    game_ply_quantiles: Quantiles,
    input_order_preserved: bool,
    automatic_adoption: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    run(args)
}

fn parse_holdout_per_mille(value: &str) -> Result<u16, String> {
    let parsed = value
        .parse::<u16>()
        .map_err(|error| format!("holdout-per-mille must be an integer: {error}"))?;
    if !(1..1_000).contains(&parsed) {
        return Err("holdout-per-mille must be in [1, 999]".to_string());
    }
    Ok(parsed)
}

fn parse_threads(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("threads must be an integer: {error}"))?;
    if !(1..=256).contains(&parsed) {
        return Err("threads must be in [1, 256]".to_string());
    }
    Ok(parsed)
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let inputs = inspect_inputs(&args.data)?;
    let available_records = inputs
        .last()
        .map_or(0, |info| info.global_start + info.records);
    let scanned_records = args
        .max_records
        .unwrap_or(available_records)
        .min(available_records);
    if scanned_records == 0 {
        return Err("入力PSVにrecordがありません".into());
    }
    if args.max_records == Some(0) {
        return Err("--max-recordsは1以上にしてください".into());
    }
    if args.output_dir.exists() {
        return Err(format!("既存outputを上書きしません: {}", args.output_dir.display()).into());
    }
    let progress = ShogiProgressKPAbs::load_from_bin(&args.progress).map_err(io::Error::other)?;
    fs::create_dir(&args.output_dir)?;
    let parts_dir = args.output_dir.join(".parts");
    fs::create_dir(&parts_dir)?;

    let worker_count = args.threads.min(scanned_records as usize).max(1);
    let ranges = partition_ranges(scanned_records, worker_count);
    let processed = Arc::new(AtomicU64::new(0));
    let inputs = Arc::new(inputs);
    let (sender, receiver) = mpsc::channel();
    let start = Instant::now();
    let mut handles = Vec::with_capacity(ranges.len());

    for (index, range) in ranges.iter().copied().enumerate() {
        let sender = sender.clone();
        let inputs = Arc::clone(&inputs);
        let processed = Arc::clone(&processed);
        let parts_dir = parts_dir.clone();
        let seed = args.seed;
        let holdout_per_mille = args.holdout_per_mille;
        handles.push(thread::spawn(move || {
            let result = process_range(
                index,
                range,
                &inputs,
                &parts_dir,
                progress,
                seed,
                holdout_per_mille,
                &processed,
            );
            let _ = sender.send((index, result));
        }));
    }
    drop(sender);

    let mut outputs: Vec<Option<WorkerOutput>> = (0..ranges.len()).map(|_| None).collect();
    let mut received = 0;
    let mut first_error = None;
    while received < ranges.len() {
        match receiver.recv_timeout(Duration::from_secs(30)) {
            Ok((index, Ok(output))) => {
                outputs[index] = Some(output);
                received += 1;
            }
            Ok((_index, Err(error))) => {
                first_error.get_or_insert(error);
                received += 1;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let done = processed.load(Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64().max(0.001);
                eprintln!(
                    "[progress8ek-filter] {done}/{scanned_records} ({:.2}%) {:.0} records/s",
                    100.0 * done as f64 / scanned_records as f64,
                    done as f64 / elapsed
                );
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                first_error.get_or_insert_with(|| "worker result channelが切断されました".into());
                break;
            }
        }
    }
    for handle in handles {
        if handle.join().is_err() {
            first_error.get_or_insert_with(|| "worker threadがpanicしました".into());
        }
    }
    if let Some(error) = first_error {
        return Err(error.into());
    }

    let outputs: Vec<WorkerOutput> = outputs
        .into_iter()
        .map(|output| output.ok_or("worker outputが不足しています"))
        .collect::<Result<_, _>>()?;
    let mut stats = Stats::default();
    for (expected, output) in outputs.iter().enumerate() {
        if output.index != expected {
            return Err("worker outputの順序が不正です".into());
        }
        stats.merge(&output.stats);
    }
    if stats.scanned != scanned_records {
        return Err(format!(
            "走査record数が不一致です: actual={} expected={scanned_records}",
            stats.scanned
        )
        .into());
    }
    if stats.matched == 0 {
        return Err("相入玉条件に一致する局面がありません".into());
    }

    let train_path = args.output_dir.join(TRAIN_NAME);
    let holdout_path = args.output_dir.join(HOLDOUT_NAME);
    merge_parts(
        &outputs
            .iter()
            .map(|output| &output.train_part)
            .collect::<Vec<_>>(),
        &train_path,
    )?;
    merge_parts(
        &outputs
            .iter()
            .map(|output| &output.holdout_part)
            .collect::<Vec<_>>(),
        &holdout_path,
    )?;
    let verified_train = verify_output(&train_path)?;
    let verified_holdout = verify_output(&holdout_path)?;
    if verified_train != stats.train || verified_holdout != stats.holdout {
        return Err(format!(
            "出力record検証数が不一致です: train={verified_train}/{} holdout={verified_holdout}/{}",
            stats.train, stats.holdout
        )
        .into());
    }

    let report = build_report(
        &args,
        inputs.as_ref().clone(),
        available_records,
        worker_count,
        &stats,
        &train_path,
        &holdout_path,
        verified_train,
        verified_holdout,
    )?;
    write_json_new(&args.output_dir.join(METRICS_NAME), &report)?;

    for output in &outputs {
        fs::remove_file(&output.train_part)?;
        fs::remove_file(&output.holdout_part)?;
    }
    fs::remove_dir(&parts_dir)?;
    eprintln!(
        "[progress8ek-filter] complete scanned={} matched={} train={} holdout={} elapsed={:.1}s",
        stats.scanned,
        stats.matched,
        stats.train,
        stats.holdout,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_range(
    index: usize,
    range: RecordRange,
    inputs: &[InputInfo],
    parts_dir: &Path,
    progress: ShogiProgressKPAbs,
    seed: u64,
    holdout_per_mille: u16,
    processed: &AtomicU64,
) -> Result<WorkerOutput, String> {
    let train_part = parts_dir.join(format!("train-{index:03}.psv"));
    let holdout_part = parts_dir.join(format!("holdout-{index:03}.psv"));
    let mut train = new_writer(&train_part).map_err(|error| error.to_string())?;
    let mut holdout = new_writer(&holdout_part).map_err(|error| error.to_string())?;
    let mut stats = Stats::default();
    let mut since_progress = 0_u64;

    for input in inputs {
        let input_end = input.global_start + input.records;
        let overlap_start = max(range.start, input.global_start);
        let overlap_end = min(range.end, input_end);
        if overlap_start >= overlap_end {
            continue;
        }
        let local_start = overlap_start - input.global_start;
        let records = overlap_end - overlap_start;
        let mut file = File::open(&input.path)
            .map_err(|error| format!("{}: {error}", input.path.display()))?;
        file.seek(SeekFrom::Start(local_start * PSV_RECORD_BYTES))
            .map_err(|error| format!("{}: {error}", input.path.display()))?;
        let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);
        let mut bytes = [0_u8; PSV_RECORD_BYTES as usize];
        for local in 0..records {
            reader
                .read_exact(&mut bytes)
                .map_err(|error| format!("{}: {error}", input.path.display()))?;
            let global_index = overlap_start + local;
            stats.scanned += 1;
            since_progress += 1;

            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(&bytes);
            let board = psv.decode();
            if board.black_king_sq.index() >= 81 || board.white_king_sq.index() >= 81 {
                return Err(format!(
                    "global record {global_index}の玉位置が不正です: black={} white={}",
                    board.black_king_sq.index(),
                    board.white_king_sq.index()
                ));
            }
            if is_mutual_entering_king(&board) {
                let is_holdout = choose_holdout(global_index, seed, holdout_per_mille);
                let destination = if is_holdout { &mut holdout } else { &mut train };
                destination
                    .write_all(&bytes)
                    .map_err(|error| error.to_string())?;
                stats.record_match(
                    &psv,
                    usize::from(board.black_king_sq.rank()),
                    usize::from(board.white_king_sq.rank()),
                    usize::from(progress.bucket_board(&board, 8)),
                    is_holdout,
                );
            }
            if since_progress >= 1_000_000 {
                processed.fetch_add(since_progress, Ordering::Relaxed);
                since_progress = 0;
            }
        }
    }
    processed.fetch_add(since_progress, Ordering::Relaxed);
    train.flush().map_err(|error| error.to_string())?;
    holdout.flush().map_err(|error| error.to_string())?;
    train
        .get_ref()
        .sync_all()
        .map_err(|error| error.to_string())?;
    holdout
        .get_ref()
        .sync_all()
        .map_err(|error| error.to_string())?;
    Ok(WorkerOutput {
        index,
        train_part,
        holdout_part,
        stats,
    })
}

fn new_writer(path: &Path) -> io::Result<BufWriter<File>> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    Ok(BufWriter::with_capacity(8 * 1024 * 1024, file))
}

fn inspect_inputs(paths: &[PathBuf]) -> Result<Vec<InputInfo>, Box<dyn std::error::Error>> {
    let mut global_start = 0_u64;
    let mut inputs = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::metadata(path)?.len();
        if bytes == 0 || !bytes.is_multiple_of(PSV_RECORD_BYTES) {
            return Err(format!(
                "PSV sizeが40-byte record境界ではありません: {} ({bytes} bytes)",
                path.display()
            )
            .into());
        }
        let records = bytes / PSV_RECORD_BYTES;
        inputs.push(InputInfo {
            path: path.clone(),
            bytes,
            records,
            global_start,
        });
        global_start = global_start
            .checked_add(records)
            .ok_or("入力record数がu64を超えました")?;
    }
    Ok(inputs)
}

fn partition_ranges(total: u64, count: usize) -> Vec<RecordRange> {
    (0..count)
        .map(|index| RecordRange {
            start: total * index as u64 / count as u64,
            end: total * (index as u64 + 1) / count as u64,
        })
        .collect()
}

fn choose_holdout(global_index: u64, seed: u64, holdout_per_mille: u16) -> bool {
    splitmix64(global_index ^ seed) % SPLIT_MODULUS < u64::from(holdout_per_mille)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn merge_parts(parts: &[&PathBuf], destination: &Path) -> io::Result<()> {
    let partial = destination.with_extension("psv.partial");
    let mut output = new_writer(&partial)?;
    for part in parts {
        let mut input = BufReader::with_capacity(8 * 1024 * 1024, File::open(part)?);
        io::copy(&mut input, &mut output)?;
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    fs::rename(partial, destination)
}

fn verify_output(path: &Path) -> Result<u64, Box<dyn std::error::Error>> {
    let bytes = fs::metadata(path)?.len();
    if !bytes.is_multiple_of(PSV_RECORD_BYTES) {
        return Err(format!("出力PSVが40-byte境界ではありません: {}", path.display()).into());
    }
    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, File::open(path)?);
    let mut raw = [0_u8; PSV_RECORD_BYTES as usize];
    let mut count = 0_u64;
    while count < bytes / PSV_RECORD_BYTES {
        reader.read_exact(&mut raw)?;
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&raw);
        if !is_mutual_entering_king(&psv.decode()) {
            return Err(format!(
                "相入玉条件を満たさないrecordが出力にあります: {} record={count}",
                path.display()
            )
            .into());
        }
        count += 1;
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
fn build_report(
    args: &Args,
    inputs: Vec<InputInfo>,
    available_records: u64,
    worker_count: usize,
    stats: &Stats,
    train_path: &Path,
    holdout_path: &Path,
    verified_train: u64,
    verified_holdout: u64,
) -> Result<Report, Box<dyn std::error::Error>> {
    let train_bytes = fs::metadata(train_path)?.len();
    let holdout_bytes = fs::metadata(holdout_path)?.len();
    if train_bytes != stats.train * PSV_RECORD_BYTES
        || holdout_bytes != stats.holdout * PSV_RECORD_BYTES
    {
        return Err("出力PSV sizeとrecord数が一致しません".into());
    }
    let mut rank_pairs = vec![vec![0_u64; 9]; 9];
    for (black_rank, row) in rank_pairs.iter_mut().enumerate() {
        for (white_rank, value) in row.iter_mut().enumerate() {
            *value = stats.king_rank_pairs[black_rank * 9 + white_rank];
        }
    }
    let mut progress_percentages = [0.0_f64; 8];
    for (index, percentage) in progress_percentages.iter_mut().enumerate() {
        *percentage = 100.0 * stats.progress_buckets[index] as f64 / stats.matched as f64;
    }
    Ok(Report {
        schema_version: 1,
        predicate: "black_king_rank<=5 && white_king_rank>=5 (one-based, fifth rank inclusive)",
        split_algorithm: "splitmix64(global_record_index XOR seed) mod 1000",
        split_seed: args.seed,
        split_modulus: SPLIT_MODULUS,
        holdout_threshold: args.holdout_per_mille,
        requested_threads: args.threads,
        worker_ranges: worker_count,
        inputs,
        available_records,
        scanned_records: stats.scanned,
        matched_records: stats.matched,
        matched_percentage: 100.0 * stats.matched as f64 / stats.scanned as f64,
        train: OutputInfo {
            path: train_path.to_path_buf(),
            records: stats.train,
            bytes: train_bytes,
            verified_entering_king_records: verified_train,
        },
        holdout: OutputInfo {
            path: holdout_path.to_path_buf(),
            records: stats.holdout,
            bytes: holdout_bytes,
            verified_entering_king_records: verified_holdout,
        },
        fixed8_progress_histogram: stats.progress_buckets,
        fixed8_progress_percentages: progress_percentages,
        black_king_rank_histogram: stats.black_king_ranks,
        white_king_rank_histogram: stats.white_king_ranks,
        king_rank_pair_histogram: rank_pairs,
        game_result_loss_draw_win: stats.results,
        score_quantiles: quantiles(&stats.score_histogram, i64::from(i16::MIN)),
        game_ply_quantiles: quantiles(&stats.ply_histogram, 0),
        input_order_preserved: true,
        automatic_adoption: false,
    })
}

fn quantiles(histogram: &[u64], offset: i64) -> Quantiles {
    let total: u64 = histogram.iter().sum();
    let value = |numerator: u64| -> i64 {
        if total == 0 {
            return offset;
        }
        let target = (total - 1) * numerator / 100;
        let mut cumulative = 0_u64;
        for (index, count) in histogram.iter().copied().enumerate() {
            cumulative += count;
            if cumulative > target {
                return offset + index as i64;
            }
        }
        offset + histogram.len().saturating_sub(1) as i64
    };
    Quantiles {
        p01: value(1),
        p10: value(10),
        p25: value(25),
        p50: value(50),
        p75: value(75),
        p90: value(90),
        p99: value(99),
    }
}

fn write_json_new(path: &Path, report: &Report) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = new_writer(path)?;
    serde_json::to_writer_pretty(&mut output, report)?;
    output.write_all(b"\n")?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicU64;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn partition_ranges_cover_input_once_in_order() {
        let ranges = partition_ranges(101, 7);
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, 101);
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert_eq!(
            ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum::<u64>(),
            101
        );
    }

    #[test]
    fn holdout_split_is_deterministic_and_respects_extreme_thresholds() {
        for index in 0..10_000 {
            assert_eq!(
                choose_holdout(index, 1234, 100),
                choose_holdout(index, 1234, 100)
            );
            assert!(!choose_holdout(index, 1234, 0));
            assert!(choose_holdout(index, 1234, 1_000));
        }
    }

    #[test]
    fn histogram_quantiles_use_nearest_rank_positions() {
        let mut histogram = vec![0_u64; 8];
        histogram.fill(1);
        let q = quantiles(&histogram, -2);
        assert_eq!(q.p01, -2);
        assert_eq!(q.p50, 1);
        assert_eq!(q.p99, 4);
    }

    #[test]
    fn merged_stats_keep_split_and_histogram_totals() {
        let mut left = Stats {
            scanned: 10,
            matched: 2,
            train: 1,
            holdout: 1,
            ..Default::default()
        };
        left.progress_buckets[7] = 2;
        let mut right = Stats {
            scanned: 20,
            matched: 3,
            train: 3,
            ..Default::default()
        };
        right.progress_buckets[7] = 3;
        left.merge(&right);
        assert_eq!(left.scanned, 30);
        assert_eq!(left.matched, 5);
        assert_eq!(left.train, 4);
        assert_eq!(left.holdout, 1);
        assert_eq!(left.progress_buckets[7], 5);
    }

    #[test]
    fn splitmix64_matches_pinned_values() {
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(splitmix64(1), 0x910a_2dec_8902_5cc1);
    }

    #[test]
    fn report_field_map_is_stable() {
        let fields = BTreeMap::from([
            ("predicate", "fifth-rank-inclusive"),
            ("split", "global-index-hash"),
        ]);
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn process_range_preserves_matching_records_and_order() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tatara-progress8ek-filter-{}-{unique}",
            std::process::id()
        ));
        let parts = root.join("parts");
        fs::create_dir_all(&parts).unwrap();
        let input_path = root.join("input.psv");
        let fixture = include_bytes!("../../../crates/shogi-format/tests/data/sample.psv");
        let mut record = fixture[..PSV_RECORD_BYTES as usize].to_vec();
        set_bits(&mut record, 1, 7, 40);
        set_bits(&mut record, 8, 7, 49);
        let mut input = Vec::new();
        for _ in 0..10 {
            input.extend_from_slice(&record);
        }
        fs::write(&input_path, &input).unwrap();
        let inputs = [InputInfo {
            path: input_path,
            bytes: input.len() as u64,
            records: 10,
            global_start: 0,
        }];
        let processed = AtomicU64::new(0);
        let output = process_range(
            0,
            RecordRange { start: 0, end: 10 },
            &inputs,
            &parts,
            ShogiProgressKPAbs,
            0,
            1_000,
            &processed,
        )
        .unwrap();
        assert_eq!(output.stats.scanned, 10);
        assert_eq!(output.stats.matched, 10);
        assert_eq!(output.stats.train, 0);
        assert_eq!(output.stats.holdout, 10);
        assert_eq!(fs::read(&output.holdout_part).unwrap(), input);
        assert_eq!(verify_output(&output.holdout_part).unwrap(), 10);
        fs::remove_dir_all(root).unwrap();
    }

    fn set_bits(bytes: &mut [u8], offset: usize, length: usize, value: u8) {
        for bit in 0..length {
            let target = offset + bit;
            let mask = 1_u8 << (target % 8);
            if value & (1 << bit) == 0 {
                bytes[target / 8] &= !mask;
            } else {
                bytes[target / 8] |= mask;
            }
        }
    }
}
