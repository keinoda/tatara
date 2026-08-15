#!/usr/bin/env python3
"""学習runをread-onlyで監視し、atomicなHTML/JSON snapshotを生成する。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def newest_experiment(run_root: Path) -> tuple[Path | None, dict[str, Any] | None, str | None]:
    paths = sorted(
        (run_root / "checkpoints" / "experiments").glob("*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        return None, None, None
    path = paths[-1]
    try:
        with path.open(encoding="utf-8") as stream:
            return path, json.load(stream), None
    except (OSError, json.JSONDecodeError) as error:
        return path, None, str(error)


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def next_milestone_estimate(
    experiment: dict[str, Any] | None,
    history: list[dict[str, Any]],
    *,
    milestone_interval: int,
    experiment_age_seconds: float | None,
    trainer_alive: bool,
    now: float,
) -> dict[str, Any] | None:
    if milestone_interval <= 0 or experiment is None:
        return None

    params = experiment.get("params") or {}
    total_value = finite_number(params.get("superbatches"))
    if total_value is None or total_value < 1:
        return None
    total_superbatches = int(total_value)

    valid_history = [
        item for item in history if finite_number(item.get("superbatch")) is not None
    ]
    if valid_history:
        current_superbatch = int(finite_number(valid_history[-1]["superbatch"]) or 0)
    else:
        start_value = finite_number(params.get("start_superbatch"))
        current_superbatch = max(0, int(start_value or 1) - 1)

    if current_superbatch >= total_superbatches or experiment.get("status") == "completed":
        return None
    target_superbatch = min(
        ((current_superbatch // milestone_interval) + 1) * milestone_interval,
        total_superbatches,
    )
    remaining_superbatches = target_superbatch - current_superbatch

    result: dict[str, Any] = {
        "interval_superbatches": milestone_interval,
        "target_superbatch": target_superbatch,
        "remaining_superbatches": remaining_superbatches,
        "estimated_seconds_remaining": None,
        "estimated_at_utc": None,
        "seconds_per_superbatch": None,
        "basis_completed_superbatches": len(valid_history),
    }
    elapsed = finite_number(
        (experiment.get("results") or {}).get("training_time_seconds")
    )
    if elapsed is None or elapsed <= 0 or not valid_history:
        return result

    seconds_per_superbatch = elapsed / len(valid_history)
    partial_elapsed = 0.0
    if trainer_alive and experiment_age_seconds is not None:
        partial_elapsed = max(0.0, experiment_age_seconds)
    remaining_seconds = max(
        0.0,
        remaining_superbatches * seconds_per_superbatch - partial_elapsed,
    )
    result["estimated_seconds_remaining"] = round(remaining_seconds)
    result["estimated_at_utc"] = datetime.fromtimestamp(
        now + remaining_seconds, tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["seconds_per_superbatch"] = round(seconds_per_superbatch, 3)
    return result


def history_points(history: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    points = []
    for item in history:
        superbatch = finite_number(item.get("superbatch"))
        value = finite_number(item.get(key))
        if superbatch is not None and value is not None:
            points.append((superbatch, value))
    return points


def chart_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if count < 2:
        return [low]
    step = (high - low) / (count - 1)
    return [low + step * index for index in range(count)]


def chart_range(values: list[float], *, bounded_ratio: bool) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0.0:
        padding = max(abs(low) * 0.05, 1e-6)
    else:
        padding = span * 0.08
    low -= padding
    high += padding
    if bounded_ratio:
        low = max(0.0, low)
        high = min(1.0, high)
    if low == high:
        low = max(0.0, low - 1e-6) if bounded_ratio else low - 1e-6
        high = min(1.0, high + 1e-6) if bounded_ratio else high + 1e-6
    return low, high


def format_y_tick(value: float, *, percent: bool) -> str:
    if percent:
        return f"{value * 100:.1f}%"
    if abs(value) < 0.1:
        return f"{value:.4f}"
    return f"{value:.3f}"


def render_line_chart(
    history: list[dict[str, Any]],
    *,
    chart_id: str,
    title: str,
    series: tuple[tuple[str, str, str], ...],
    percent: bool = False,
    independent_series_scales: bool = False,
) -> str:
    plotted = [
        (label, color, history_points(history, key)) for label, key, color in series
    ]
    plotted = [(label, color, points) for label, color, points in plotted if points]
    if not plotted:
        return (
            f'<section class="chart-card" data-chart="{html.escape(chart_id)}">'
            f"<h2>{html.escape(title)}</h2>"
            '<p class="empty-chart">履歴データを待っています。</p></section>'
        )

    width = 900.0
    height = 320.0
    left = 110.0
    use_independent_scales = independent_series_scales and len(plotted) == 2
    right = 110.0 if use_independent_scales else 24.0
    top = 22.0
    bottom = 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = [x for _, _, points in plotted for x, _ in points]
    x_low = min(x_values)
    x_high = max(x_values)
    x_ticks = chart_ticks(x_low, x_high) if x_low != x_high else [x_low]
    if x_low == x_high:
        x_low -= 0.5
        x_high += 0.5
    if use_independent_scales:
        y_ranges = [
            chart_range([y for _, y in points], bounded_ratio=percent)
            for _, _, points in plotted
        ]
    else:
        y_values = [y for _, _, points in plotted for _, y in points]
        shared_range = chart_range(y_values, bounded_ratio=percent)
        y_ranges = [shared_range] * len(plotted)

    def x_position(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_width

    def y_position(value: float, value_range: tuple[float, float]) -> float:
        y_low, y_high = value_range
        return top + (y_high - value) / (y_high - y_low) * plot_height

    svg = [
        f'<svg class="chart-svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="{html.escape(title)}">',
        f'<rect class="plot-background" x="{left:.1f}" y="{top:.1f}" '
        f'width="{plot_width:.1f}" height="{plot_height:.1f}" />',
    ]
    left_ticks = chart_ticks(*y_ranges[0])
    right_ticks = chart_ticks(*y_ranges[1]) if use_independent_scales else []
    for index, tick in enumerate(left_ticks):
        y = y_position(tick, y_ranges[0])
        svg.append(
            f'<line class="grid-line" x1="{left:.1f}" y1="{y:.2f}" '
            f'x2="{width - right:.1f}" y2="{y:.2f}" />'
        )
        left_style = f' style="fill:{plotted[0][1]}"' if use_independent_scales else ""
        svg.append(
            f'<text class="axis-tick" data-axis="left"{left_style} '
            f'x="{left - 12:.1f}" y="{y + 4:.2f}" '
            f'text-anchor="end">{format_y_tick(tick, percent=percent)}</text>'
        )
        if use_independent_scales:
            right_tick = right_ticks[index]
            svg.append(
                f'<text class="axis-tick" data-axis="right" '
                f'style="fill:{plotted[1][1]}" x="{width - right + 12:.1f}" '
                f'y="{y + 4:.2f}" text-anchor="start">'
                f'{format_y_tick(right_tick, percent=percent)}</text>'
            )
    for tick in x_ticks:
        x = x_position(tick)
        svg.append(
            f'<line class="grid-line vertical" x1="{x:.2f}" y1="{top:.1f}" '
            f'x2="{x:.2f}" y2="{height - bottom:.1f}" />'
        )
        svg.append(
            f'<text class="axis-tick" x="{x:.2f}" y="{height - 18:.1f}" '
            f'text-anchor="middle">{tick:.0f}</text>'
        )
    svg.append(
        f'<text class="axis-title" x="{left + plot_width / 2:.2f}" '
        f'y="{height - 2:.1f}" text-anchor="middle">superbatch</text>'
    )
    for index, (label, color, points) in enumerate(plotted):
        coordinates = " ".join(
            f"{x_position(x):.2f},{y_position(y, y_ranges[index]):.2f}"
            for x, y in points
        )
        if len(points) > 1:
            svg.append(
                f'<polyline class="data-line" points="{coordinates}" '
                f'stroke="{color}" aria-label="{html.escape(label)}" />'
            )
        last_x, last_y = points[-1]
        svg.append(
            f'<circle class="latest-point" cx="{x_position(last_x):.2f}" '
            f'cy="{y_position(last_y, y_ranges[index]):.2f}" r="5" fill="{color}" />'
        )
    svg.append("</svg>")

    legend_parts = []
    for index, (label, color, _) in enumerate(plotted):
        if use_independent_scales:
            label = f"{label}（{'左軸' if index == 0 else '右軸'}）"
        legend_parts.append(
            f'<span><i style="--series-color:{color}"></i>{html.escape(label)}</span>'
        )
    legend = "".join(legend_parts)
    return (
        f'<section class="chart-card" data-chart="{html.escape(chart_id)}">'
        f"<h2>{html.escape(title)}</h2>"
        f'<div class="legend">{legend}</div>{"".join(svg)}</section>'
    )


def build_status(
    run_name: str,
    run_root: Path,
    stale_seconds: int,
    milestone_interval: int = 0,
) -> dict[str, Any]:
    now = time.time()
    experiment_path, experiment, read_error = newest_experiment(run_root)
    experiment_age = None
    if experiment_path is not None:
        experiment_age = max(0.0, now - experiment_path.stat().st_mtime)
    trainer_pid = read_text(run_root / "state" / "trainer.pid")
    trainer_alive = False
    if trainer_pid and trainer_pid.isdigit():
        try:
            os.kill(int(trainer_pid), 0)
            trainer_alive = True
        except OSError:
            trainer_alive = False

    history = experiment.get("history", []) if experiment else []
    latest = history[-1] if history else None
    next_milestone = next_milestone_estimate(
        experiment,
        history,
        milestone_interval=milestone_interval,
        experiment_age_seconds=experiment_age,
        trainer_alive=trainer_alive,
        now=now,
    )
    return {
        "schema_version": 1,
        "generated_at_epoch": now,
        "run_name": run_name,
        "run_root": str(run_root),
        "trainer_pid": trainer_pid,
        "trainer_alive": trainer_alive,
        "trainer_exit_code": read_text(run_root / "state" / "trainer.exit-code"),
        "experiment_path": str(experiment_path) if experiment_path else None,
        "experiment_read_error": read_error,
        "experiment_age_seconds": experiment_age,
        "stale": experiment_age is not None and experiment_age > stale_seconds,
        "status": experiment.get("status") if experiment else "waiting",
        "commit": experiment.get("commit") if experiment else None,
        "history": history,
        "latest": latest,
        "results": experiment.get("results") if experiment else None,
        "checkpoints": experiment.get("checkpoints", []) if experiment else [],
        "next_milestone": next_milestone,
    }


def render_html(status: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape("-" if value is None else str(value))

    latest = status.get("latest") or {}
    results = status.get("results") or {}
    milestone = status.get("next_milestone") or {}

    def duration(value: Any) -> str | None:
        seconds_value = finite_number(value)
        if seconds_value is None:
            return None
        seconds = max(0, round(seconds_value))
        days, remainder = divmod(seconds, 86_400)
        hours, remainder = divmod(remainder, 3_600)
        minutes, seconds = divmod(remainder, 60)
        clock = f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{days}日 {clock}" if days else clock

    rows = [
        ("run", status.get("run_name")),
        ("status", status.get("status")),
        ("trainer alive", status.get("trainer_alive")),
        ("stale", status.get("stale")),
        ("superbatch", latest.get("superbatch")),
        ("loss", latest.get("loss")),
        ("test loss", latest.get("test_loss")),
        ("test accuracy", latest.get("test_accuracy")),
        ("positions/s", results.get("mean_pos_per_sec")),
        ("best test loss", results.get("best_test_loss")),
        ("best test SB", results.get("best_test_loss_superbatch")),
    ]
    if milestone:
        milestone_interval = milestone["interval_superbatches"]
        rows.extend(
            [
                (
                    f"next {milestone_interval} SB boundary",
                    milestone.get("target_superbatch"),
                ),
                ("SB to boundary", milestone.get("remaining_superbatches")),
                (
                    "EST remaining",
                    duration(milestone.get("estimated_seconds_remaining")),
                ),
                ("EST (UTC)", milestone.get("estimated_at_utc")),
            ]
        )
    rows.extend(
        [
            ("commit", status.get("commit")),
            ("experiment age (s)", status.get("experiment_age_seconds")),
            ("exit code", status.get("trainer_exit_code")),
        ]
    )
    table = "\n".join(f"<tr><th>{html.escape(label)}</th><td>{esc(value)}</td></tr>" for label, value in rows)
    history = status.get("history") or []
    loss_chart = render_line_chart(
        history,
        chart_id="loss",
        title="train / test loss（独立縮尺）",
        series=(
            ("train loss", "loss", "#2563eb"),
            ("test loss", "test_loss", "#dc2626"),
        ),
        independent_series_scales=True,
    )
    accuracy_chart = render_line_chart(
        history,
        chart_id="test-accuracy",
        title="test accuracy",
        series=(("test accuracy", "test_accuracy", "#059669"),),
        percent=True,
    )
    return f"""<!doctype html>
<html lang=\"ja\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">
<meta http-equiv=\"refresh\" content=\"15\"><title>{esc(status.get('run_name'))}</title>
<style>:root{{--page:#f3f4f6;--surface:#fff;--text:#111827;--muted:#4b5563;--border:#d1d5db;
--grid:#e5e7eb;--header:#f9fafb}}*{{box-sizing:border-box}}body{{font-family:system-ui,sans-serif;
margin:0;background:var(--page);color:var(--text)}}main{{max-width:76rem;margin:0 auto;padding:2rem}}
.status-line{{color:var(--muted)}}.summary,.chart-card{{background:var(--surface);border:1px solid var(--border);
border-radius:.8rem;box-shadow:0 1px 3px rgb(0 0 0/.08)}}.summary{{overflow:hidden;margin:1.5rem 0}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid var(--border);padding:.55rem .7rem;
text-align:left}}tr:last-child th,tr:last-child td{{border-bottom:0}}th{{width:14rem;background:var(--header)}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,34rem),1fr));gap:1rem}}
.chart-card{{padding:1rem}}.chart-card h2{{font-size:1.05rem;margin:0 0 .6rem}}.legend{{display:flex;
flex-wrap:wrap;gap:1rem;color:var(--muted);font-size:.88rem}}.legend span{{display:inline-flex;align-items:center;
gap:.35rem}}.legend i{{display:inline-block;width:1.4rem;border-top:3px solid var(--series-color)}}
.chart-svg{{display:block;width:100%;height:auto;margin-top:.3rem}}.plot-background{{fill:var(--surface)}}
.grid-line{{stroke:var(--grid);stroke-width:1;vector-effect:non-scaling-stroke}}.grid-line.vertical{{stroke-dasharray:3 5}}
.axis-tick,.axis-title{{fill:var(--muted);font-size:12px}}.data-line{{fill:none;stroke-width:2.5;
stroke-linejoin:round;stroke-linecap:round;vector-effect:non-scaling-stroke}}.latest-point{{stroke:var(--surface);
stroke-width:2;vector-effect:non-scaling-stroke}}.empty-chart{{color:var(--muted);min-height:12rem;
display:grid;place-items:center}}.warn{{color:#b91c1c;font-weight:700}}a{{color:#1d4ed8}}
@media (max-width:40rem){{main{{padding:1rem}}th{{width:9rem}}th,td{{font-size:.88rem}}
.axis-tick,.axis-title{{font-size:28px}}}}
@media (prefers-color-scheme:dark){{:root{{--page:#111827;--surface:#1f2937;--text:#f9fafb;
--muted:#d1d5db;--border:#4b5563;--grid:#374151;--header:#273244}}a{{color:#93c5fd}}}}
</style></head><body><main>
<h1>{esc(status.get('run_name'))}</h1>
<p class=\"status-line {'warn' if status.get('stale') else ''}\">15秒ごとに更新。stale={esc(status.get('stale'))}</p>
<div class=\"summary\"><table>{table}</table></div>
<div class=\"charts\">{loss_chart}{accuracy_chart}</div>
<p><a href=\"/status.json\">status.json</a></p></main></body></html>\n"""


def render_once(
    run_name: str,
    run_root: Path,
    output_dir: Path,
    stale_seconds: int,
    milestone_interval: int = 0,
) -> None:
    status = build_status(run_name, run_root, stale_seconds, milestone_interval)
    atomic_write(output_dir / "status.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output_dir / "index.html", render_html(status))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--milestone-interval", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1 or args.stale_seconds < 1 or args.milestone_interval < 0:
        parser.error(
            "--interval and --stale-seconds must be >= 1; "
            "--milestone-interval must be >= 0"
        )
    while True:
        render_once(
            args.run_name,
            args.run_root,
            args.output_dir,
            args.stale_seconds,
            args.milestone_interval,
        )
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
