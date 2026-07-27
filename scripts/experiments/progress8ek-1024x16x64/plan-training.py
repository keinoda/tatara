#!/usr/bin/env python3
"""相入玉train件数から指定epoch数の学習単位を決定する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

BATCH_SIZE = 65_536
TARGET_EPOCHS = 10.0
DESIRED_SUPERBATCHES = 100
MAX_VALIDATION_POSITIONS = 851_968


def build_plan(
    train_records: int,
    holdout_records: int,
    *,
    target_epochs: float = TARGET_EPOCHS,
    desired_superbatches: int = DESIRED_SUPERBATCHES,
    lr_schedule: str = "one-cycle",
    lr_gamma: float = 0.992,
    save_rate: int | None = None,
    batch_rounding: str = "ceil",
) -> dict[str, int | float | str]:
    if train_records < BATCH_SIZE:
        raise ValueError(f"train records must be at least one batch ({BATCH_SIZE})")
    if holdout_records < BATCH_SIZE:
        raise ValueError(f"holdout records must be at least one batch ({BATCH_SIZE})")
    if not math.isfinite(target_epochs) or target_epochs <= 0:
        raise ValueError("target epochs must be finite and positive")
    if desired_superbatches <= 0:
        raise ValueError("desired superbatches must be positive")
    if lr_schedule not in {"one-cycle", "step"}:
        raise ValueError(f"unsupported LR schedule: {lr_schedule}")
    if not math.isfinite(lr_gamma) or lr_gamma <= 0:
        raise ValueError("LR gamma must be finite and positive")
    if batch_rounding not in {"ceil", "nearest"}:
        raise ValueError(f"unsupported batch rounding: {batch_rounding}")
    target_batches = math.ceil(target_epochs * train_records / BATCH_SIZE)
    superbatches = min(desired_superbatches, target_batches)
    ratio = target_batches / superbatches
    if batch_rounding == "nearest":
        batches_per_superbatch = max(1, math.floor(ratio + 0.5))
    else:
        batches_per_superbatch = math.ceil(ratio)
    total_batches = superbatches * batches_per_superbatch
    total_positions = total_batches * BATCH_SIZE
    validation_positions = min(
        MAX_VALIDATION_POSITIONS,
        holdout_records // BATCH_SIZE * BATCH_SIZE,
    )
    return {
        "batch_size": BATCH_SIZE,
        "target_epochs": target_epochs,
        "target_batches": target_batches,
        "superbatches": superbatches,
        "batches_per_superbatch": batches_per_superbatch,
        "total_batches": total_batches,
        "total_positions": total_positions,
        "actual_epochs": total_positions / train_records,
        "validation_positions": validation_positions,
        "save_rate": save_rate if save_rate is not None else max(1, superbatches // 10),
        "lr_schedule": lr_schedule,
        "lr": 8.75e-4,
        "lr_gamma": lr_gamma,
        "lr_step": 1,
        "batch_rounding": batch_rounding,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-epochs", type=float, default=TARGET_EPOCHS)
    parser.add_argument("--superbatches", type=int, default=DESIRED_SUPERBATCHES)
    parser.add_argument("--lr-schedule", choices=("one-cycle", "step"), default="one-cycle")
    parser.add_argument("--lr-gamma", type=float, default=0.992)
    parser.add_argument("--save-rate", type=int)
    parser.add_argument("--batch-rounding", choices=("ceil", "nearest"), default="ceil")
    args = parser.parse_args()
    with args.metrics.open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    plan = build_plan(
        int(metrics["train"]["records"]),
        int(metrics["holdout"]["records"]),
        target_epochs=args.target_epochs,
        desired_superbatches=args.superbatches,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        save_rate=args.save_rate,
        batch_rounding=args.batch_rounding,
    )
    encoded = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    if args.output.exists():
        raise SystemExit(f"ERROR: existing output is not overwritten: {args.output}")
    args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
