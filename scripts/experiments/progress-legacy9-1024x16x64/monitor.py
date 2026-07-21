#!/usr/bin/env python3
"""学習runをread-onlyで監視し、atomicなHTML/JSON snapshotを生成する。"""

from __future__ import annotations

import argparse
import html
import json
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


def build_status(run_name: str, run_root: Path, stale_seconds: int) -> dict[str, Any]:
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
        "latest": latest,
        "results": experiment.get("results") if experiment else None,
        "checkpoints": experiment.get("checkpoints", []) if experiment else [],
    }


def render_html(status: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape("-" if value is None else str(value))

    latest = status.get("latest") or {}
    results = status.get("results") or {}
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
        ("commit", status.get("commit")),
        ("experiment age (s)", status.get("experiment_age_seconds")),
        ("exit code", status.get("trainer_exit_code")),
    ]
    table = "\n".join(f"<tr><th>{html.escape(label)}</th><td>{esc(value)}</td></tr>" for label, value in rows)
    return f"""<!doctype html>
<html lang=\"ja\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">
<meta http-equiv=\"refresh\" content=\"15\"><title>{esc(status.get('run_name'))}</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:60rem}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}th{{width:14rem;background:#eee}}
.warn{{color:#a00;font-weight:700}}</style></head><body>
<h1>{esc(status.get('run_name'))}</h1>
<p class=\"{'warn' if status.get('stale') else ''}\">15秒ごとに更新。stale={esc(status.get('stale'))}</p>
<table>{table}</table><p><a href=\"/status.json\">status.json</a></p></body></html>\n"""


def render_once(run_name: str, run_root: Path, output_dir: Path, stale_seconds: int) -> None:
    status = build_status(run_name, run_root, stale_seconds)
    atomic_write(output_dir / "status.json", json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    atomic_write(output_dir / "index.html", render_html(status))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--stale-seconds", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1 or args.stale_seconds < 1:
        parser.error("--interval and --stale-seconds must be >= 1")
    while True:
        render_once(args.run_name, args.run_root, args.output_dir, args.stale_seconds)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
