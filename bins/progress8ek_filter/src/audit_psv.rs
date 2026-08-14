//! PSV全件のprogress有効index数を監査する。

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;

use clap::Parser;
use serde::Serialize;
use shogi_features::ShogiProgressKPAbs;
use shogi_format::PackedSfenValue;

const PSV_RECORD_BYTES: u64 = 40;
const EXPECTED_ACTIVE_INDICES: usize = 76;

#[derive(Debug, Parser)]
#[command(name = "progress8ek-audit-psv")]
#[command(about = "Audit every PSV record for the 76-active-index progress invariant")]
struct Args {
    /// 監査する単一PSV。
    #[arg(long)]
    data: PathBuf,

    /// 新規作成するJSON監査結果。
    #[arg(long)]
    output: PathBuf,

    /// 入力範囲を並列走査するthread数。
    #[arg(long, default_value_t = 16, value_parser = parse_threads)]
    threads: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecordRange {
    start: u64,
    end: u64,
}

#[derive(Debug, Default)]
struct Stats {
    scanned: u64,
    eligible: u64,
    ineligible: u64,
    zero_records: u64,
    active_index_histogram: BTreeMap<usize, u64>,
    first_ineligible_index: Option<u64>,
    last_ineligible_index: Option<u64>,
}

impl Stats {
    fn record(
        &mut self,
        global_index: u64,
        bytes: &[u8; PSV_RECORD_BYTES as usize],
        active: usize,
    ) {
        self.scanned += 1;
        *self.active_index_histogram.entry(active).or_default() += 1;
        if bytes.iter().all(|&byte| byte == 0) {
            self.zero_records += 1;
        }
        if active == EXPECTED_ACTIVE_INDICES {
            self.eligible += 1;
        } else {
            self.ineligible += 1;
            self.first_ineligible_index = Some(
                self.first_ineligible_index
                    .map_or(global_index, |current| current.min(global_index)),
            );
            self.last_ineligible_index = Some(
                self.last_ineligible_index
                    .map_or(global_index, |current| current.max(global_index)),
            );
        }
    }

    fn merge(&mut self, other: Self) {
        self.scanned += other.scanned;
        self.eligible += other.eligible;
        self.ineligible += other.ineligible;
        self.zero_records += other.zero_records;
        for (active, count) in other.active_index_histogram {
            *self.active_index_histogram.entry(active).or_default() += count;
        }
        if let Some(index) = other.first_ineligible_index {
            self.first_ineligible_index = Some(
                self.first_ineligible_index
                    .map_or(index, |current| current.min(index)),
            );
        }
        if let Some(index) = other.last_ineligible_index {
            self.last_ineligible_index = Some(
                self.last_ineligible_index
                    .map_or(index, |current| current.max(index)),
            );
        }
    }
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    data: PathBuf,
    bytes: u64,
    records: u64,
    requested_threads: usize,
    worker_ranges: usize,
    expected_active_indices: usize,
    scanned_records: u64,
    eligible_records: u64,
    ineligible_records: u64,
    zero_records: u64,
    active_index_histogram: BTreeMap<usize, u64>,
    first_ineligible_index: Option<u64>,
    last_ineligible_index: Option<u64>,
    exhaustive: bool,
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
    let bytes = std::fs::metadata(&args.data)?.len();
    if bytes == 0 || bytes % PSV_RECORD_BYTES != 0 {
        return Err(format!(
            "PSV file must be non-empty and {PSV_RECORD_BYTES}-byte aligned: {} ({bytes} bytes)",
            args.data.display()
        )
        .into());
    }
    if args.output.exists() {
        return Err(format!("output already exists: {}", args.output.display()).into());
    }
    let records = bytes / PSV_RECORD_BYTES;
    let ranges = partition_ranges(records, args.threads);
    let mut handles = Vec::with_capacity(ranges.len());
    for range in ranges.iter().copied() {
        let data = args.data.clone();
        handles.push(thread::spawn(move || scan_range(&data, range)));
    }

    let mut stats = Stats::default();
    for handle in handles {
        let worker = handle.join().map_err(|_| "PSV audit worker panicked")??;
        stats.merge(worker);
    }
    let report = Report {
        schema_version: 1,
        data: args.data,
        bytes,
        records,
        requested_threads: args.threads,
        worker_ranges: ranges.len(),
        expected_active_indices: EXPECTED_ACTIVE_INDICES,
        scanned_records: stats.scanned,
        eligible_records: stats.eligible,
        ineligible_records: stats.ineligible,
        zero_records: stats.zero_records,
        active_index_histogram: stats.active_index_histogram,
        first_ineligible_index: stats.first_ineligible_index,
        last_ineligible_index: stats.last_ineligible_index,
        exhaustive: stats.scanned == records && stats.eligible + stats.ineligible == records,
    };
    write_json_new(&args.output, &report)?;
    Ok(())
}

fn partition_ranges(records: u64, threads: usize) -> Vec<RecordRange> {
    let workers = threads.min(records as usize);
    (0..workers)
        .map(|worker| RecordRange {
            start: records * worker as u64 / workers as u64,
            end: records * (worker + 1) as u64 / workers as u64,
        })
        .collect()
}

fn scan_range(path: &Path, range: RecordRange) -> Result<Stats, String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    file.seek(SeekFrom::Start(range.start * PSV_RECORD_BYTES))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut bytes = [0_u8; PSV_RECORD_BYTES as usize];
    let mut psv = PackedSfenValue::default();
    let mut active_indices = Vec::with_capacity(EXPECTED_ACTIVE_INDICES);
    let mut stats = Stats::default();
    for global_index in range.start..range.end {
        reader
            .read_exact(&mut bytes)
            .map_err(|error| format!("{} record {global_index}: {error}", path.display()))?;
        psv.as_bytes_mut().copy_from_slice(&bytes);
        ShogiProgressKPAbs::collect_active_indices(&psv, &mut active_indices);
        stats.record(global_index, &bytes, active_indices.len());
    }
    Ok(stats)
}

fn write_json_new(path: &Path, report: &Report) -> Result<(), Box<dyn std::error::Error>> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut output, report)?;
    output.write_all(b"\n")?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cover_every_record_once() {
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
    fn zero_record_is_ineligible_and_counted() {
        let bytes = [0_u8; PSV_RECORD_BYTES as usize];
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&bytes);
        let mut active_indices = Vec::new();
        ShogiProgressKPAbs::collect_active_indices(&psv, &mut active_indices);
        let mut stats = Stats::default();
        stats.record(42, &bytes, active_indices.len());
        assert_eq!(active_indices.len(), 108);
        assert_eq!(stats.scanned, 1);
        assert_eq!(stats.eligible, 0);
        assert_eq!(stats.ineligible, 1);
        assert_eq!(stats.zero_records, 1);
        assert_eq!(stats.first_ineligible_index, Some(42));
        assert_eq!(stats.last_ineligible_index, Some(42));
    }

    #[test]
    fn ordinary_sample_record_has_expected_active_count() {
        let fixture = include_bytes!("../../../crates/shogi-format/tests/data/sample.psv");
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut()
            .copy_from_slice(&fixture[..PSV_RECORD_BYTES as usize]);
        let mut active_indices = Vec::new();
        ShogiProgressKPAbs::collect_active_indices(&psv, &mut active_indices);
        assert_eq!(active_indices.len(), EXPECTED_ACTIVE_INDICES);
    }
}
