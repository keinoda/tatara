#!/usr/bin/env python3
"""毎SBのtest lossと実在する保存checkpointを突き合わせ、保存済み最良を報告する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    experiments = sorted((args.run_root / "checkpoints" / "experiments").glob("*.json"))
    if not experiments:
        raise SystemExit("ERROR: experiment.json is missing")
    with experiments[-1].open(encoding="utf-8") as stream:
        doc = json.load(stream)
    test_losses = {
        int(entry["superbatch"]): float(entry["test_loss"])
        for entry in doc.get("history", [])
        if entry.get("test_loss") is not None
    }
    candidates = []
    for path in (args.run_root / "checkpoints").glob("*.bin"):
        match = re.search(r"-(\d+)\.bin$", path.name)
        if not match:
            continue
        sb = int(match.group(1))
        ckpt = path.with_suffix(".ckpt")
        if sb in test_losses:
            candidates.append({"superbatch": sb, "test_loss": test_losses[sb], "bin": str(path), "ckpt": str(ckpt) if ckpt.exists() else None})
    if not candidates:
        raise SystemExit("ERROR: no saved checkpoint has a recorded test_loss")
    candidates.sort(key=lambda item: (item["test_loss"], item["superbatch"]))
    report = {"policy": "minimum test_loss among saved quantized checkpoints", "automatic_adoption": False, "best": candidates[0], "candidates": candidates}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
