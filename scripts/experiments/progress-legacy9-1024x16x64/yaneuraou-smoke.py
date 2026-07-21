#!/usr/bin/env python3
"""YaneuraOuへ対話的にUSI commandを送り、各局面のbestmoveを待つ。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import subprocess
import threading
import time


def reader_thread(stream: object, output: queue.Queue[str | None], transcript: object) -> None:
    for line in stream:  # type: ignore[union-attr]
        transcript.write(line)  # type: ignore[union-attr]
        transcript.flush()  # type: ignore[union-attr]
        output.put(line.rstrip("\n"))
    output.put(None)


def wait_for(output: queue.Queue[str | None], predicate: object, timeout: float, label: str) -> list[str]:
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"timeout waiting for {label}")
        try:
            line = output.get(timeout=remaining)
        except queue.Empty as error:
            raise RuntimeError(f"timeout waiting for {label}") from error
        if line is None:
            raise RuntimeError(f"engine exited while waiting for {label}")
        lines.append(line)
        if predicate(line):  # type: ignore[operator]
            return lines


def send(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--fixtures-jsonl", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.nodes < 1:
        parser.error("--nodes must be >= 1")

    fixtures: list[str] = []
    with args.fixtures_jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                fixtures.append(json.loads(line)["sfen"])
    if len(fixtures) != 14:
        raise SystemExit(f"ERROR: expected 14 boundary fixtures, got {len(fixtures)}")

    with args.transcript.open("x", encoding="utf-8") as transcript:
        process = subprocess.Popen(
            [str(args.engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        output: queue.Queue[str | None] = queue.Queue()
        thread = threading.Thread(target=reader_thread, args=(process.stdout, output, transcript), daemon=True)
        thread.start()
        try:
            send(process, "usi")
            usi_lines = wait_for(output, lambda line: line == "usiok", args.timeout, "usiok")
            options = "\n".join(usi_lines)
            for required in ("option name EvalDir", "option name LS_PROGRESS_COEFF", "option name LS_BUCKET_MODE"):
                if required not in options:
                    raise RuntimeError(f"required USI option is missing: {required}")
            send(process, f"setoption name EvalDir value {args.eval_dir}")
            send(process, "setoption name FV_SCALE value 28")
            send(process, f"setoption name LS_PROGRESS_COEFF value {args.progress}")
            send(process, "setoption name LS_BUCKET_MODE value progress8kpabs")
            send(process, "isready")
            ready_lines = wait_for(output, lambda line: line == "readyok", args.timeout, "readyok")
            ready_text = "\n".join(ready_lines)
            bad = ("falling back", "Error!", "failed to read", "mismatch")
            if any(token.lower() in ready_text.lower() for token in bad):
                raise RuntimeError(f"engine load reported an error or fallback:\n{ready_text}")
            if f"loading progress file : {args.progress}" not in ready_text:
                raise RuntimeError("selected external progress file was not confirmed by the engine")

            positions = [None, *fixtures]
            for index, sfen in enumerate(positions):
                send(process, "position startpos" if sfen is None else f"position sfen {sfen}")
                send(process, f"go nodes {args.nodes}")
                wait_for(output, lambda line: line.startswith("bestmove "), args.timeout, f"bestmove {index}")
            send(process, "quit")
            return_code = process.wait(timeout=args.timeout)
            if return_code != 0:
                raise RuntimeError(f"engine exited with {return_code}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
