//! 公開PSVを相入玉模様とそれ以外へ排他的に全件分割する。

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
use shogi_features::is_mutual_entering_king;
use shogi_format::PackedSfenValue;

const PSV_RECORD_BYTES: u64 = 40;
const ORDINARY_NAME: &str = "ordinary.psv";
const ENTERING_KING_NAME: &str = "entering-king.psv";
const METRICS_NAME: &str = "metrics.json";

#[derive(Debug, Parser)]
#[command(name = "progress8ek-partition")]
#[command(about = "Partition every PSV record by the mutual entering-king predicate")]
struct Args {
    /// 入力PSV。複数fileは指定順に一つの母集団として扱う。
    #[arg(long = "data", required = true)]
    data: Vec<PathBuf>,

    /// 新規作成する出力directory。既存pathは拒否する。
    #[arg(long)]
    output_dir: PathBuf,

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

#[derive(Debug, Clone, Copy, Default)]
struct Stats {
    scanned: u64,
    ordinary: u64,
    entering_king: u64,
}

impl Stats {
    fn merge(&mut self, other: Self) {
        self.scanned += other.scanned;
        self.ordinary += other.ordinary;
        self.entering_king += other.entering_king;
    }
}

#[derive(Debug)]
struct WorkerOutput {
    index: usize,
    ordinary_part: PathBuf,
    entering_king_part: PathBuf,
    stats: Stats,
}

#[derive(Debug, Serialize)]
struct OutputInfo {
    path: PathBuf,
    records: u64,
    bytes: u64,
    verified_records: u64,
    expected_predicate: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    predicate: &'static str,
    requested_threads: usize,
    worker_ranges: usize,
    inputs: Vec<InputInfo>,
    available_records: u64,
    scanned_records: u64,
    ordinary: OutputInfo,
    entering_king: OutputInfo,
    exhaustive: bool,
    mutually_exclusive: bool,
    input_order_preserved: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run(Args::parse())
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
        .map_or(0, |input| input.global_start + input.records);
    if args.max_records == Some(0) {
        return Err("--max-recordsは1以上にしてください".into());
    }
    let scanned_records = args
        .max_records
        .unwrap_or(available_records)
        .min(available_records);
    if scanned_records == 0 {
        return Err("入力PSVにrecordがありません".into());
    }
    if args.output_dir.exists() {
        return Err(format!("既存outputを上書きしません: {}", args.output_dir.display()).into());
    }

    fs::create_dir(&args.output_dir)?;
    let parts_dir = args.output_dir.join(".parts");
    fs::create_dir(&parts_dir)?;

    let worker_count = args.threads.min(scanned_records as usize).max(1);
    let ranges = partition_ranges(scanned_records, worker_count);
    let processed = Arc::new(AtomicU64::new(0));
    let inputs = Arc::new(inputs);
    let (sender, receiver) = mpsc::channel();
    let started = Instant::now();
    let mut handles = Vec::with_capacity(ranges.len());

    for (index, range) in ranges.iter().copied().enumerate() {
        let sender = sender.clone();
        let inputs = Arc::clone(&inputs);
        let processed = Arc::clone(&processed);
        let parts_dir = parts_dir.clone();
        handles.push(thread::spawn(move || {
            let result = process_range(index, range, &inputs, &parts_dir, &processed);
            let _ = sender.send((index, result));
        }));
    }
    drop(sender);

    let outputs = collect_worker_outputs(
        receiver,
        handles,
        ranges.len(),
        &processed,
        scanned_records,
        "partition",
        started,
    )?;
    let mut stats = Stats::default();
    for (expected_index, output) in outputs.iter().enumerate() {
        if output.index != expected_index {
            return Err("worker outputの順序が不正です".into());
        }
        stats.merge(output.stats);
    }
    if stats.scanned != scanned_records || stats.ordinary + stats.entering_king != scanned_records {
        return Err(format!(
            "排他分割のrecord数が不一致です: scanned={} ordinary={} entering_king={} expected={scanned_records}",
            stats.scanned, stats.ordinary, stats.entering_king
        )
        .into());
    }

    let ordinary_path = args.output_dir.join(ORDINARY_NAME);
    let entering_king_path = args.output_dir.join(ENTERING_KING_NAME);
    merge_parts(
        &outputs
            .iter()
            .map(|output| &output.ordinary_part)
            .collect::<Vec<_>>(),
        &ordinary_path,
    )?;
    merge_parts(
        &outputs
            .iter()
            .map(|output| &output.entering_king_part)
            .collect::<Vec<_>>(),
        &entering_king_path,
    )?;

    let verified_ordinary = verify_output(&ordinary_path, false, args.threads)?;
    let verified_entering_king = verify_output(&entering_king_path, true, args.threads)?;
    if verified_ordinary != stats.ordinary || verified_entering_king != stats.entering_king {
        return Err(format!(
            "出力record検証数が不一致です: ordinary={verified_ordinary}/{} entering_king={verified_entering_king}/{}",
            stats.ordinary, stats.entering_king
        )
        .into());
    }

    let ordinary = output_info(&ordinary_path, stats.ordinary, verified_ordinary, false)?;
    let entering_king = output_info(
        &entering_king_path,
        stats.entering_king,
        verified_entering_king,
        true,
    )?;
    let report = Report {
        schema_version: 1,
        predicate: "black_king_rank<=5 && white_king_rank>=5 (one-based, fifth rank inclusive)",
        requested_threads: args.threads,
        worker_ranges: worker_count,
        inputs: inputs.as_ref().clone(),
        available_records,
        scanned_records,
        ordinary,
        entering_king,
        exhaustive: scanned_records == available_records,
        mutually_exclusive: true,
        input_order_preserved: true,
    };
    write_json_new(&args.output_dir.join(METRICS_NAME), &report)?;

    for output in &outputs {
        fs::remove_file(&output.ordinary_part)?;
        fs::remove_file(&output.entering_king_part)?;
    }
    fs::remove_dir(&parts_dir)?;
    eprintln!(
        "[progress8ek-partition] complete scanned={} ordinary={} entering_king={} elapsed={:.1}s",
        stats.scanned,
        stats.ordinary,
        stats.entering_king,
        started.elapsed().as_secs_f64()
    );
    Ok(())
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

fn process_range(
    index: usize,
    range: RecordRange,
    inputs: &[InputInfo],
    parts_dir: &Path,
    processed: &AtomicU64,
) -> Result<WorkerOutput, String> {
    let ordinary_part = parts_dir.join(format!("ordinary-{index:03}.psv"));
    let entering_king_part = parts_dir.join(format!("entering-king-{index:03}.psv"));
    let mut ordinary = new_writer(&ordinary_part).map_err(|error| error.to_string())?;
    let mut entering_king = new_writer(&entering_king_part).map_err(|error| error.to_string())?;
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
            let predicate = decode_predicate(&bytes, &input.path, global_index)?;
            let destination = if predicate {
                stats.entering_king += 1;
                &mut entering_king
            } else {
                stats.ordinary += 1;
                &mut ordinary
            };
            destination
                .write_all(&bytes)
                .map_err(|error| error.to_string())?;
            stats.scanned += 1;
            since_progress += 1;
            if since_progress >= 1_000_000 {
                processed.fetch_add(since_progress, Ordering::Relaxed);
                since_progress = 0;
            }
        }
    }
    processed.fetch_add(since_progress, Ordering::Relaxed);
    sync_writer(ordinary).map_err(|error| error.to_string())?;
    sync_writer(entering_king).map_err(|error| error.to_string())?;
    Ok(WorkerOutput {
        index,
        ordinary_part,
        entering_king_part,
        stats,
    })
}

fn decode_predicate(
    bytes: &[u8; PSV_RECORD_BYTES as usize],
    path: &Path,
    index: u64,
) -> Result<bool, String> {
    let mut psv = PackedSfenValue::default();
    psv.as_bytes_mut().copy_from_slice(bytes);
    let board = psv.decode();
    if board.black_king_sq.index() >= 81 || board.white_king_sq.index() >= 81 {
        return Err(format!(
            "{} record={index}の玉位置が不正です: black={} white={}",
            path.display(),
            board.black_king_sq.index(),
            board.white_king_sq.index()
        ));
    }
    Ok(is_mutual_entering_king(&board))
}

fn collect_worker_outputs(
    receiver: mpsc::Receiver<(usize, Result<WorkerOutput, String>)>,
    handles: Vec<thread::JoinHandle<()>>,
    count: usize,
    processed: &AtomicU64,
    total: u64,
    label: &str,
    started: Instant,
) -> Result<Vec<WorkerOutput>, Box<dyn std::error::Error>> {
    let mut outputs: Vec<Option<WorkerOutput>> = (0..count).map(|_| None).collect();
    let mut received = 0;
    let mut first_error = None;
    while received < count {
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
                let elapsed = started.elapsed().as_secs_f64().max(0.001);
                eprintln!(
                    "[progress8ek-partition:{label}] {done}/{total} ({:.2}%) {:.0} records/s",
                    100.0 * done as f64 / total as f64,
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
    outputs
        .into_iter()
        .map(|output| output.ok_or_else(|| "worker outputが不足しています".into()))
        .collect()
}

fn new_writer(path: &Path) -> io::Result<BufWriter<File>> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    Ok(BufWriter::with_capacity(8 * 1024 * 1024, file))
}

fn sync_writer(mut writer: BufWriter<File>) -> io::Result<()> {
    writer.flush()?;
    writer.get_ref().sync_all()
}

fn merge_parts(parts: &[&PathBuf], destination: &Path) -> io::Result<()> {
    let partial = destination.with_extension("psv.partial");
    let mut output = new_writer(&partial)?;
    for part in parts {
        let mut input = BufReader::with_capacity(8 * 1024 * 1024, File::open(part)?);
        io::copy(&mut input, &mut output)?;
    }
    sync_writer(output)?;
    fs::rename(partial, destination)
}

fn verify_output(
    path: &Path,
    expected_predicate: bool,
    threads: usize,
) -> Result<u64, Box<dyn std::error::Error>> {
    let bytes = fs::metadata(path)?.len();
    if !bytes.is_multiple_of(PSV_RECORD_BYTES) {
        return Err(format!("出力PSVが40-byte境界ではありません: {}", path.display()).into());
    }
    let records = bytes / PSV_RECORD_BYTES;
    if records == 0 {
        return Ok(0);
    }
    let worker_count = threads.min(records as usize).max(1);
    let ranges = partition_ranges(records, worker_count);
    let processed = Arc::new(AtomicU64::new(0));
    let (sender, receiver) = mpsc::channel();
    let started = Instant::now();
    let mut handles = Vec::with_capacity(worker_count);
    for (index, range) in ranges.into_iter().enumerate() {
        let sender = sender.clone();
        let processed = Arc::clone(&processed);
        let path = path.to_path_buf();
        handles.push(thread::spawn(move || {
            let result = verify_range(&path, range, expected_predicate, &processed);
            let _ = sender.send((index, result));
        }));
    }
    drop(sender);

    let mut counts = vec![0_u64; worker_count];
    let mut received = 0;
    let mut first_error = None;
    while received < worker_count {
        match receiver.recv_timeout(Duration::from_secs(30)) {
            Ok((index, Ok(count))) => {
                counts[index] = count;
                received += 1;
            }
            Ok((_index, Err(error))) => {
                first_error.get_or_insert(error);
                received += 1;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let done = processed.load(Ordering::Relaxed);
                let elapsed = started.elapsed().as_secs_f64().max(0.001);
                eprintln!(
                    "[progress8ek-partition:verify] {} {done}/{records} ({:.2}%) {:.0} records/s",
                    path.display(),
                    100.0 * done as f64 / records as f64,
                    done as f64 / elapsed
                );
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                first_error.get_or_insert_with(|| "verify result channelが切断されました".into());
                break;
            }
        }
    }
    for handle in handles {
        if handle.join().is_err() {
            first_error.get_or_insert_with(|| "verify worker threadがpanicしました".into());
        }
    }
    if let Some(error) = first_error {
        return Err(error.into());
    }
    Ok(counts.into_iter().sum())
}

fn verify_range(
    path: &Path,
    range: RecordRange,
    expected_predicate: bool,
    processed: &AtomicU64,
) -> Result<u64, String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(range.start * PSV_RECORD_BYTES))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut bytes = [0_u8; PSV_RECORD_BYTES as usize];
    let mut since_progress = 0_u64;
    for index in range.start..range.end {
        reader
            .read_exact(&mut bytes)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        let actual = decode_predicate(&bytes, path, index)?;
        if actual != expected_predicate {
            return Err(format!(
                "出力の相入玉述語が不一致です: {} record={index} actual={actual} expected={expected_predicate}",
                path.display()
            ));
        }
        since_progress += 1;
        if since_progress >= 1_000_000 {
            processed.fetch_add(since_progress, Ordering::Relaxed);
            since_progress = 0;
        }
    }
    processed.fetch_add(since_progress, Ordering::Relaxed);
    Ok(range.end - range.start)
}

fn output_info(
    path: &Path,
    records: u64,
    verified_records: u64,
    expected_predicate: bool,
) -> Result<OutputInfo, Box<dyn std::error::Error>> {
    let bytes = fs::metadata(path)?.len();
    if bytes != records * PSV_RECORD_BYTES {
        return Err(format!(
            "出力PSV sizeとrecord数が一致しません: {} bytes={bytes} records={records}",
            path.display()
        )
        .into());
    }
    Ok(OutputInfo {
        path: path.to_path_buf(),
        records,
        bytes,
        verified_records,
        expected_predicate,
    })
}

fn write_json_new(path: &Path, report: &Report) -> Result<(), Box<dyn std::error::Error>> {
    let mut output = new_writer(path)?;
    serde_json::to_writer_pretty(&mut output, report)?;
    output.write_all(b"\n")?;
    sync_writer(output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn ranges_cover_input_once_in_order() {
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
    fn run_partitions_every_record_and_preserves_each_output_order() {
        let root = temporary_directory("partition");
        let first_input = root.join("input-1.psv");
        let second_input = root.join("input-2.psv");
        let output_dir = root.join("output");
        fs::create_dir(&root).unwrap();

        let fixture = include_bytes!("../../../crates/shogi-format/tests/data/sample.psv");
        let mut entering = fixture[..PSV_RECORD_BYTES as usize].to_vec();
        set_bits(&mut entering, 1, 7, 40);
        set_bits(&mut entering, 8, 7, 49);
        let mut ordinary = entering.clone();
        set_bits(&mut ordinary, 1, 7, 41);
        assert!(predicate(&entering));
        assert!(!predicate(&ordinary));

        fs::write(&first_input, [ordinary.clone(), entering.clone()].concat()).unwrap();
        fs::write(&second_input, [ordinary.clone(), entering.clone()].concat()).unwrap();
        run(Args {
            data: vec![first_input, second_input],
            output_dir: output_dir.clone(),
            threads: 3,
            max_records: None,
        })
        .unwrap();

        assert_eq!(
            fs::read(output_dir.join(ORDINARY_NAME)).unwrap(),
            [ordinary.clone(), ordinary].concat()
        );
        assert_eq!(
            fs::read(output_dir.join(ENTERING_KING_NAME)).unwrap(),
            [entering.clone(), entering].concat()
        );
        let metrics: serde_json::Value =
            serde_json::from_slice(&fs::read(output_dir.join(METRICS_NAME)).unwrap()).unwrap();
        assert_eq!(metrics["scanned_records"], 4);
        assert_eq!(metrics["ordinary"]["verified_records"], 2);
        assert_eq!(metrics["entering_king"]["verified_records"], 2);
        assert_eq!(metrics["exhaustive"], true);
        assert_eq!(metrics["mutually_exclusive"], true);
        assert!(!output_dir.join(".parts").exists());
        fs::remove_dir_all(root).unwrap();
    }

    fn predicate(record: &[u8]) -> bool {
        let bytes: &[u8; PSV_RECORD_BYTES as usize] = record.try_into().unwrap();
        decode_predicate(bytes, Path::new("fixture"), 0).unwrap()
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "tatara-progress8ek-{label}-{}-{unique}",
            std::process::id()
        ))
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
