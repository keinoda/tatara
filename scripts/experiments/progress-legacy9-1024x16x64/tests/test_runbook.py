#!/usr/bin/env python3
"""Vast学習runbookのGPU不要な回帰test。"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"moduleをloadできません: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunbookTests(unittest.TestCase):
    def test_generated_browser_settings_contain_only_clone_bootstrap(self) -> None:
        branch = "codex/progress-legacy9-1024x16x64-training"
        remote = subprocess.run(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        ).stdout.split()[0]
        environment = os.environ.copy()
        environment.update({"TATARA_COMMIT": remote})
        generated = subprocess.run(
            [str(SCRIPT_DIR / "print-vast-browser-settings.sh")],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env=environment,
        ).stdout
        bootstrap = generated.split("On-start Script:\n", 1)[1]
        subprocess.run(["bash", "-n", "-c", bootstrap], check=True)
        self.assertLess(
            bootstrap.index("touch /root/.no_auto_tmux"), bootstrap.index("git clone")
        )
        self.assertIn("checkout --detach \"$TATARA_COMMIT\"", bootstrap)
        self.assertIn('bash "$target/onstart.sh"', bootstrap)
        self.assertNotIn("cargo build", bootstrap)
        self.assertNotIn("hf download", bootstrap)
        self.assertIn(
            "ghcr.io/keinoda/shogi-lab:cuda129-trt1011@sha256:",
            generated,
        )
        self.assertIn(f"-p 6001:6001 -e TATARA_COMMIT={remote}", generated)
        self.assertNotIn("vastai create instance", generated)
        self.assertNotIn("/tmp/", generated)

    def test_training_command_keeps_fixed_contract(self) -> None:
        shell = f"""
source {SCRIPT_DIR / 'lib.sh'!s}
COMMAND_DATA=/tmp/train.psv
COMMAND_OUTPUT=/tmp/output
COMMAND_NET_ID=test-run
COMMAND_SUPERBATCHES=367
COMMAND_BATCHES_PER_SB=6104
COMMAND_BATCH_SIZE=65536
COMMAND_THREADS=16
COMMAND_PROGRESS=/tmp/progress.bin
COMMAND_VALIDATION=/tmp/validation.psv
COMMAND_PRECISION=all-optim
COMMAND_RESUME=
build_training_command
printf '%s\n' "${{TRAINING_COMMAND[@]}}"
"""
        completed = subprocess.run(
            ["bash", "-c", shell],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
        args = completed.stdout.splitlines()

        def value_after(flag: str) -> str:
            return args[args.index(flag) + 1]

        self.assertEqual(value_after("--batch-size"), "65536")
        self.assertEqual(value_after("--batches-per-superbatch"), "6104")
        self.assertEqual(value_after("--superbatches"), "367")
        self.assertEqual(value_after("--test-positions"), "851968")
        self.assertEqual(value_after("--lr-schedule"), "step")
        self.assertEqual(value_after("--lr"), "8.75e-4")
        self.assertEqual(value_after("--lr-gamma"), "0.992")
        self.assertEqual(value_after("--lr-step"), "1")
        self.assertEqual(value_after("--ft-out"), "1024")
        self.assertEqual(value_after("--l1"), "16")
        self.assertEqual(value_after("--l2"), "64")
        self.assertEqual(value_after("--bucket-mode"), "progress8kpabs")
        self.assertEqual(value_after("--num-buckets"), "8")
        self.assertIn("--all-optim", args)
        self.assertNotIn("one-cycle", args)

    def test_monitor_renders_atomic_snapshot(self) -> None:
        monitor = load_module("tatara_monitor", SCRIPT_DIR / "monitor.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            experiment_dir = run_root / "checkpoints" / "experiments"
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "run.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "commit": "a" * 40,
                        "history": [
                            {
                                "superbatch": 2,
                                "loss": 0.12,
                                "test_loss": 0.13,
                                "test_accuracy": 0.7,
                            }
                        ],
                        "results": {
                            "mean_pos_per_sec": 1234,
                            "best_test_loss": 0.13,
                            "best_test_loss_superbatch": 2,
                        },
                        "checkpoints": ["run-2.ckpt"],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "monitor"
            monitor.render_once("run-a", run_root, output, 180)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertEqual(status["latest"]["superbatch"], 2)
            self.assertEqual(status["results"]["best_test_loss"], 0.13)
            self.assertIn("run-a", html)
            self.assertNotIn("MONITOR_PASSWORD", html)

    def test_monitor_server_requires_basic_auth_and_has_only_two_routes(self) -> None:
        server_module = load_module("tatara_monitor_server", SCRIPT_DIR / "monitor_server.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.html").write_text("ok", encoding="utf-8")
            (root / "status.json").write_text('{"ok":true}\n', encoding="utf-8")
            token = base64.b64encode(b"user:password").decode("ascii")
            server_module.MonitorHandler.root = root
            server_module.MonitorHandler.expected_authorization = f"Basic {token}"
            server = server_module.ThreadingHTTPServer(
                ("127.0.0.1", 0), server_module.MonitorHandler
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_address[1]
            try:
                with self.assertRaises(urllib.error.HTTPError) as unauthenticated:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                self.assertEqual(unauthenticated.exception.code, 401)

                request = urllib.request.Request(f"http://127.0.0.1:{port}/status.json")
                request.add_header("Authorization", f"Basic {token}")
                with urllib.request.urlopen(request, timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.read(), b'{"ok":true}\n')

                missing = urllib.request.Request(f"http://127.0.0.1:{port}/other")
                missing.add_header("Authorization", f"Basic {token}")
                with self.assertRaises(urllib.error.HTTPError) as not_found:
                    urllib.request.urlopen(missing, timeout=2)
                self.assertEqual(not_found.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_smoke_summary_reads_current_experiment_schema(self) -> None:
        summary = load_module("tatara_smoke_summary", SCRIPT_DIR / "summarize-smoke.py")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            experiment_dir = run / "checkpoints" / "experiments"
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "smoke.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "params": {"tf32": True, "threads": 30},
                        "results": {"mean_pos_per_sec": 5000},
                        "history": [
                            {
                                "superbatch": 1,
                                "loss": 0.1,
                                "test_loss": 0.2,
                                "test_accuracy": 0.8,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (run / "train.log").write_text(
                "[fp16-clamp] layer=ft ratio=1.5e-5\n", encoding="utf-8"
            )
            report = summary.load_run(run)
            self.assertEqual(report["precision"], "all-optim")
            self.assertEqual(report["threads"], 30)
            self.assertAlmostEqual(report["max_fp16_clamp_ratio"], 1.5e-5)

    def test_saved_checkpoint_selection_ignores_unsaved_best_superbatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            experiment_dir = run / "checkpoints" / "experiments"
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "run.json").write_text(
                json.dumps(
                    {
                        "history": [
                            {"superbatch": 1, "test_loss": 0.1},
                            {"superbatch": 20, "test_loss": 0.2},
                            {"superbatch": 40, "test_loss": 0.15},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            for superbatch in (20, 40):
                (run / "checkpoints" / f"run-{superbatch}.bin").write_bytes(b"bin")
            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT_DIR / "select-saved-checkpoint.py"),
                    "--run-root",
                    str(run),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["best"]["superbatch"], 40)
            self.assertFalse(report["automatic_adoption"])

    def test_revision_pins_match_onstart_and_runbook_library(self) -> None:
        onstart = (REPO_ROOT / "onstart.sh").read_text(encoding="utf-8")
        library = (SCRIPT_DIR / "lib.sh").read_text(encoding="utf-8")
        for revision in (
            "da3ea68d46a5c1ac0c18c10a57fef52d02788879",
            "29245a1d8e4f198aba3fc832a506649221cb2f2c",
            "771fe811f877859d6851ceccfd3e04c16454e689",
            "5da309f4de4091cfb004eff94da97d49e3268aa2",
            "fdd5f602db82d888a87116f087d10dd5ea8313ab",
            "sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876",
        ):
            self.assertIn(revision, onstart)
            self.assertIn(revision, library)


if __name__ == "__main__":
    unittest.main()
