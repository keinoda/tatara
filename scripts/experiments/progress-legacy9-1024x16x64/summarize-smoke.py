#!/usr/bin/env python3
"""smoke各runのexperiment.jsonとclamp logを一つの比較表へまとめる。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


CLAMP_RE = re.compile(r"\[fp16-clamp\].*?ratio=([0-9.eE+-]+)")


def load_run(path: Path) -> dict[str, object]:
    experiments = sorted((path / "checkpoints" / "experiments").glob("*.json"))
    if len(experiments) != 1:
        raise SystemExit(f"ERROR: expected one experiment JSON under {path}, got {len(experiments)}")
    with experiments[0].open(encoding="utf-8") as stream:
        experiment = json.load(stream)
    if experiment.get("status") != "completed":
        raise SystemExit(
            f"ERROR: smoke experiment is not completed: "
            f"{experiments[0]} status={experiment.get('status')!r}"
        )
    if (experiment.get("results") or {}).get("interrupted") is not False:
        raise SystemExit(f"ERROR: smoke experiment is interrupted: {experiments[0]}")
    history = experiment.get("history", [])
    if len(history) != 1:
        raise SystemExit(f"ERROR: smoke history must contain exactly one SB: {experiments[0]}")
    log_path = path / "train.log"
    clamp_ratios: list[float] = []
    if log_path.exists():
        for match in CLAMP_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
            clamp_ratios.append(float(match.group(1)))
    return {
        "experiment": str(experiments[0]),
        "status": experiment.get("status"),
        "precision": "all-optim" if experiment["params"].get("tf32") else "fp32",
        "threads": experiment["params"]["threads"],
        "mean_pos_per_sec": experiment["results"]["mean_pos_per_sec"],
        "loss": history[0]["loss"],
        "test_loss": history[0].get("test_loss"),
        "test_accuracy": history[0].get("test_accuracy"),
        "max_fp16_clamp_ratio": max(clamp_ratios, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "automatic_selection": False,
        "runs": [load_run(path) for path in args.runs],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
