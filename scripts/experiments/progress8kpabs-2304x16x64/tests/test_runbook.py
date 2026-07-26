#!/usr/bin/env python3
"""Vast学習runbookのGPU不要な回帰test。"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
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
        branch = "codex/progress8kpabs-2304x16x64-training"
        remote = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            fake_git = Path(temporary) / "git"
            fake_git.write_text(
                f"""#!/usr/bin/env bash
printf '%s\\trefs/heads/{branch}\\n' '{remote}'
""",
                encoding="utf-8",
            )
            fake_git.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "TATARA_COMMIT": remote,
                    "PATH": f"{temporary}:{environment['PATH']}",
                }
            )
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
        self.assertIn(
            "-e YANEURAOU_GITHUB_TOKEN=REPLACE_WITH_FINE_GRAINED_PAT",
            generated,
        )
        self.assertNotIn("vastai create instance", generated)
        self.assertNotIn("/tmp/", generated)

    def test_training_command_keeps_fixed_contract(self) -> None:
        shell = f"""
source {SCRIPT_DIR / 'lib.sh'!s}
COMMAND_DATA=/tmp/train.psv
COMMAND_OUTPUT=/tmp/output
COMMAND_NET_ID=test-run
COMMAND_SUPERBATCHES=841
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
        self.assertEqual(value_after("--superbatches"), "841")
        self.assertEqual(value_after("--test-positions"), "851968")
        self.assertEqual(value_after("--lr-schedule"), "step")
        self.assertEqual(value_after("--lr"), "8.75e-4")
        self.assertEqual(value_after("--lr-gamma"), "0.992")
        self.assertEqual(value_after("--lr-step"), "1")
        self.assertEqual(value_after("--ft-out"), "2304")
        self.assertEqual(value_after("--l1"), "16")
        self.assertEqual(value_after("--l2"), "64")
        self.assertEqual(value_after("--bucket-mode"), "progress8kpabs")
        self.assertEqual(value_after("--num-buckets"), "8")
        self.assertIn("--all-optim", args)
        self.assertNotIn("one-cycle", args)

    def test_training_shards_are_pinned_and_appended_incrementally(self) -> None:
        shard_spec = SCRIPT_DIR / "training-shards.tsv"
        rows = [
            line.split("\t")
            for line in shard_spec.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(len(rows), 34)
        self.assertEqual([row[0] for row in rows], [f"split_{i:03}.bin" for i in range(34)])
        self.assertEqual(sum(int(row[1]) for row in rows), 673_002_105_840)
        for name, size, digest in rows:
            self.assertEqual(int(size) % 40, 0, name)
            self.assertEqual(len(digest), 64, name)
            int(digest, 16)

        onstart = (REPO_ROOT / "onstart.sh").read_text(encoding="utf-8")
        self.assertIn('hf download "$TRAIN_DATASET" "$shard_name"', onstart)
        self.assertIn('open(output, "ab", buffering=0)', onstart)
        self.assertIn('truncate --size "$committed_bytes" "$TRAIN_PARTIAL_PSV"', onstart)
        self.assertIn('if [[ "$shard" == "$TRAIN_SURVEY_SHARD" ]]', onstart)
        self.assertIn('rm -- "$shard"', onstart)
        self.assertNotIn("--include 'split_*.bin'", onstart)
        self.assertNotIn('shards=("$TRAIN_SHARD_DIR"/split_*.bin)', onstart)

    def test_inline_append_body_resumes_at_committed_shard_boundary(self) -> None:
        onstart = (REPO_ROOT / "onstart.sh").read_text(encoding="utf-8")
        body = onstart.split(
            "read -r -d '' download_training_body <<'STEP' || true\n", 1
        )[1].split("\nSTEP\nstart_step download_training", 1)[0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = root / "fixtures"
            fixtures.mkdir()
            first = bytes(range(40)) * 2
            second = bytes(reversed(range(40))) * 3
            (fixtures / "split_000.bin").write_bytes(first)
            (fixtures / "split_001.bin").write_bytes(second)
            spec = root / "training-shards.tsv"
            spec.write_text(
                "\n".join(
                    (
                        f"split_000.bin\t{len(first)}\t{hashlib.sha256(first).hexdigest()}",
                        f"split_001.bin\t{len(second)}\t{hashlib.sha256(second).hexdigest()}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_hf = bin_dir / "hf"
            fake_hf.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
shard_name="$3"
shift 3
while (( $# > 0 )); do
  if [[ "$1" == --local-dir ]]; then
    local_dir="$2"
    shift 2
  else
    shift
  fi
done
mkdir -p "$local_dir"
cp "$HF_FIXTURES/$shard_name" "$local_dir/$shard_name"
""",
                encoding="utf-8",
            )
            fake_hf.chmod(0o755)
            fake_stat = bin_dir / "stat"
            fake_stat.write_text(
                """#!/usr/bin/env bash
if [[ "$1" == -c && "$2" == %s ]]; then
  exec /usr/bin/stat -f %z "$3"
fi
exec /usr/bin/stat "$@"
""",
                encoding="utf-8",
            )
            fake_stat.chmod(0o755)
            fake_truncate = bin_dir / "truncate"
            fake_truncate.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
[[ "$1" == --size ]]
/usr/bin/python3 -c 'import sys; open(sys.argv[2], "r+b").truncate(int(sys.argv[1]))' "$2" "$3"
""",
                encoding="utf-8",
            )
            fake_truncate.chmod(0o755)

            training_dir = root / "data/training"
            shard_dir = training_dir / "shards"
            state_dir = root / "state"
            manifest_dir = root / "manifests"
            shard_dir.mkdir(parents=True)
            state_dir.mkdir()
            manifest_dir.mkdir()
            partial = training_dir / "public-teacher.psv.partial"
            partial.write_bytes(first + b"x" * 40)
            survey_shard = shard_dir / "split_000.bin"
            survey_shard.write_bytes(first)
            first_sha = hashlib.sha256(first).hexdigest()
            (state_dir / "split_000.bin.done").write_text(
                f"split_000.bin\t{len(first)}\t{first_sha}\t{len(first)}\n",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{bin_dir}:{environment['PATH']}",
                    "HF_FIXTURES": str(fixtures),
                    "TRAIN_DATASET": "fixture/dataset",
                    "TRAIN_DATASET_REVISION": "a" * 40,
                    "TRAIN_SHARD_DIR": str(shard_dir),
                    "TRAIN_PSV": str(training_dir / "public-teacher.psv"),
                    "TRAIN_PARTIAL_PSV": str(partial),
                    "TRAIN_APPEND_STATE_DIR": str(state_dir),
                    "TRAIN_SHARD_SPEC": str(spec),
                    "TRAIN_SURVEY_SHARD": str(survey_shard),
                    "TRAIN_EXPECTED_BYTES": str(len(first) + len(second)),
                    "TRAIN_EXPECTED_SHARDS": "2",
                    "PSV_RECORD_BYTES": "40",
                    "MANIFEST_DIR": str(manifest_dir),
                    "LC_ALL": "C",
                }
            )
            completed = subprocess.run(
                ["bash", "-c", body],
                check=False,
                text=True,
                capture_output=True,
                env=environment,
                cwd=root,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            combined = training_dir / "public-teacher.psv"
            self.assertEqual(combined.read_bytes(), first + second)
            self.assertTrue(survey_shard.exists())
            self.assertFalse((shard_dir / "split_001.bin").exists())
            self.assertTrue((state_dir / "split_001.bin.done").exists())
            self.assertIn(
                "中断した未確定末尾を切り戻します", completed.stdout
            )
            append_rows = (
                manifest_dir / "training-inline-append.tsv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(append_rows), 2)

    def test_training_requires_survey_shard_cleanup(self) -> None:
        training = (SCRIPT_DIR / "run-training.sh").read_text(encoding="utf-8")
        cleanup = (SCRIPT_DIR / "cleanup-survey-shard.sh").read_text(encoding="utf-8")
        self.assertIn('survey-shard-cleanup.txt', training)
        self.assertIn('remaining_shards=("$TRAIN_SHARD_DIR"/split_*.bin)', training)
        self.assertIn("CONFIRM_REMOVE_SURVEY_SHARD", cleanup)
        self.assertIn("require_progress_approval", cleanup)
        self.assertIn('rm -- "$TRAIN_SURVEY_SHARD"', cleanup)

    def test_survey_accepts_completed_shard_snapshot_before_full_download(self) -> None:
        script = (SCRIPT_DIR / "run-survey.sh").read_text(encoding="utf-8")
        plan = (
            REPO_ROOT / "docs/experiments/progress8kpabs-2304x16x64/PLAN.md"
        ).read_text(encoding="utf-8")
        runbook = (
            REPO_ROOT / "docs/experiments/progress8kpabs-2304x16x64/RUNBOOK.md"
        ).read_text(encoding="utf-8")
        self.assertNotIn('STATE_DIR/prepare_data.done', script)
        self.assertIn('(( ${#shards[@]} >= 1 ))', script)
        self.assertIn('total_shard_positions >= 4000000', script)
        self.assertIn('input-shards.txt', script)
        self.assertIn('input_shards_sha256', script)
        self.assertNotIn('--optimize-affine', script)
        self.assertNotIn('--candidate', script)
        self.assertIn('automatic_recalibration=false', script)
        self.assertIn('通常surveyではprogress係数を再最適化しません', script)
        self.assertIn('teacher_data_passes=1', script)
        self.assertIn("全download完了を待たず", plan)
        self.assertIn("全34 shardの完了前", runbook)
        self.assertIn("400万局面を一度だけ", plan)
        self.assertIn("400万局面を固定", runbook)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied_script_dir = (
                root / "scripts/experiments/progress8kpabs-2304x16x64"
            )
            copied_script_dir.mkdir(parents=True)
            for name in ("lib.sh", "run-survey.sh"):
                shutil.copy2(SCRIPT_DIR / name, copied_script_dir / name)

            survey = root / "target/release/progress-bucket-survey"
            survey.parent.mkdir(parents=True)
            survey.write_text(
                """#!/usr/bin/env bash
set -Eeuo pipefail
while (( $# > 0 )); do
  if [[ "$1" == --output-dir ]]; then
    output_dir="$2"
    shift 2
  else
    shift
  fi
done
: "${output_dir:?--output-dir is required}"
mkdir -p "$output_dir"
printf '{}\n' >"$output_dir/metrics.json"
printf 'sample-plan\n' >"$output_dir/sample-plan.bin"
""",
                encoding="utf-8",
            )
            survey.chmod(0o755)

            reference = root / "progress/reference/progress.bin"
            reference.parent.mkdir(parents=True)
            with reference.open("wb") as stream:
                stream.truncate(1_003_104)
            shard = root / "data/training/shards/split_000.bin"
            shard.parent.mkdir(parents=True)
            with shard.open("wb") as stream:
                stream.truncate(4_000_000 * 40)

            environment = os.environ.copy()
            environment.update(
                {
                    "SURVEY_ID": "partial-one-shard",
                    "SURVEY_SEED": "20260721",
                    "PRIMARY_SAMPLES": "2000000",
                    "CONFIRMATION_A_SAMPLES": "1000000",
                    "CONFIRMATION_B_SAMPLES": "1000000",
                    "BASHPID": "12345",
                }
            )
            completed = subprocess.run(
                [str(copied_script_dir / "run-survey.sh")],
                check=False,
                text=True,
                capture_output=True,
                cwd=root,
                env=environment,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            output = root / "survey/partial-one-shard"
            input_snapshot = (output / "input-shards.txt").read_text(
                encoding="utf-8"
            )
            manifest = (output / "manifest.txt").read_text(encoding="utf-8")
            self.assertIn("completed_shards=1", input_snapshot)
            self.assertIn("total_positions=4000000", input_snapshot)
            self.assertIn(str(shard), input_snapshot)
            self.assertIn("input_shards_sha256=", manifest)
            self.assertIn("affine_optimization=disabled", manifest)
            self.assertIn("automatic_recalibration=false", manifest)
            self.assertIn("reference_progress_sha256=", manifest)
            self.assertIn("automatic_adoption=false", manifest)

            second_shard = root / "data/training/shards/split_001.bin"
            with second_shard.open("wb") as stream:
                stream.truncate(4_000_000 * 40)
            environment.update(
                {
                    "SURVEY_ID": "reused-one-shard",
                    "SURVEY_INPUT_MANIFEST": str(output / "input-shards.txt"),
                }
            )
            reused = subprocess.run(
                [str(copied_script_dir / "run-survey.sh")],
                check=False,
                text=True,
                capture_output=True,
                cwd=root,
                env=environment,
            )
            self.assertEqual(
                reused.returncode,
                0,
                f"stdout:\n{reused.stdout}\nstderr:\n{reused.stderr}",
            )
            reused_snapshot = (
                root / "survey/reused-one-shard/input-shards.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("completed_shards=1", reused_snapshot)
            source_manifest = os.path.realpath(output / "input-shards.txt")
            self.assertIn(f"source_manifest={source_manifest}", reused_snapshot)
            self.assertNotIn(str(second_shard), reused_snapshot)

            environment.update(
                {
                    "SURVEY_ID": "optimizer-must-stop",
                    "OPTIMIZER_TARGET_PERCENTAGES": "11,12,13,14,14,13,12,11",
                    "OPTIMIZED_CANDIDATE_NAME": "optimized-center-gentle",
                }
            )
            optimizer_attempt = subprocess.run(
                [str(copied_script_dir / "run-survey.sh")],
                check=False,
                text=True,
                capture_output=True,
                cwd=root,
                env=environment,
            )
            self.assertNotEqual(optimizer_attempt.returncode, 0)
            self.assertIn(
                "通常surveyではprogress係数を再最適化しません",
                optimizer_attempt.stderr,
            )
            self.assertFalse((root / "survey/optimizer-must-stop").exists())

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
                                "superbatch": 1,
                                "loss": 0.15,
                                "test_loss": 0.14,
                                "test_accuracy": 0.68,
                            },
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
            self.assertEqual(len(status["history"]), 2)
            self.assertEqual(status["results"]["best_test_loss"], 0.13)
            self.assertIn("run-a", html)
            self.assertIn('data-chart="loss"', html)
            self.assertIn('data-chart="test-accuracy"', html)
            self.assertIn("train loss", html)
            self.assertIn("test loss", html)
            self.assertIn("train loss（左軸）", html)
            self.assertIn("test loss（右軸）", html)
            self.assertEqual(html.count("<svg"), 2)
            self.assertIn("<polyline", html)
            loss_chart = html.split('data-chart="loss"', 1)[1].split(
                "</section>", 1
            )[0]
            self.assertIn('data-axis="left"', loss_chart)
            self.assertIn('data-axis="right"', loss_chart)
            self.assertIn("0.118", loss_chart)
            self.assertIn("0.129", loss_chart)
            self.assertNotIn("<script", html)
            self.assertNotIn("MONITOR_PASSWORD", html)

    def test_monitor_chart_ignores_non_finite_values(self) -> None:
        monitor = load_module("tatara_monitor_non_finite", SCRIPT_DIR / "monitor.py")
        chart = monitor.render_line_chart(
            [
                {"superbatch": 1, "loss": 0.2},
                {"superbatch": 2, "loss": float("nan")},
                {"superbatch": 3, "loss": 0.1},
            ],
            chart_id="loss",
            title="loss",
            series=(("train loss", "loss", "#2563eb"),),
        )
        self.assertIn("<polyline", chart)
        self.assertNotIn("nan", chart.lower())

        one_point = monitor.render_line_chart(
            [{"superbatch": 1, "test_accuracy": 0.7}],
            chart_id="test-accuracy",
            title="test accuracy",
            series=(("test accuracy", "test_accuracy", "#059669"),),
            percent=True,
        )
        self.assertIn("<circle", one_point)
        self.assertNotIn("<polyline", one_point)

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
                        "status": "completed",
                        "params": {"tf32": True, "threads": 30},
                        "results": {"mean_pos_per_sec": 5000, "interrupted": False},
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
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["precision"], "all-optim")
            self.assertEqual(report["threads"], 30)
            self.assertAlmostEqual(report["max_fp16_clamp_ratio"], 1.5e-5)

    def test_smoke_scripts_use_completed_status_and_explicit_recovery(self) -> None:
        smoke = (SCRIPT_DIR / "run-smoke.sh").read_text(encoding="utf-8")
        resume = (SCRIPT_DIR / "run-resume-drill.sh").read_text(encoding="utf-8")
        runbook = (
            REPO_ROOT / "docs/experiments/progress8kpabs-2304x16x64/RUNBOOK.md"
        ).read_text(encoding="utf-8")
        self.assertIn('run.get("status") != "completed"', smoke)
        self.assertIn('doc.get("status") != "completed"', resume)
        self.assertNotIn('get("status") != "complete"', smoke)
        self.assertNotIn('get("status") != "complete"', resume)
        self.assertIn('FINALIZE_EXISTING_SMOKE="${FINALIZE_EXISTING_SMOKE:-0}"', smoke)
        self.assertIn('finalization_mode="existing-completed-smoke"', smoke)
        self.assertIn("FINALIZE_EXISTING_SMOKE=1", runbook)

    def test_export_uses_python3_and_explicit_partial_recovery(self) -> None:
        export = (SCRIPT_DIR / "run-export-test.sh").read_text(encoding="utf-8")
        smoke = (SCRIPT_DIR / "yaneuraou-smoke.py").read_text(encoding="utf-8")
        runbook = (
            REPO_ROOT / "docs/experiments/progress8kpabs-2304x16x64/RUNBOOK.md"
        ).read_text(encoding="utf-8")
        self.assertIn("PYTHON=python3 normal", export)
        self.assertIn(
            'CONTINUE_EXISTING_EXPORT="${CONTINUE_EXISTING_EXPORT:-0}"', export
        )
        self.assertIn('finalization_mode="existing-conversion-artifacts"', export)
        self.assertIn("EXPORT_TRANSCRIPT_NAME", export)
        self.assertIn(
            'readonly YANEURAOU_REPO="https://github.com/keinoda/YaneuraOu-private.git"',
            (SCRIPT_DIR / "lib.sh").read_text(encoding="utf-8"),
        )
        self.assertIn('GIT_ASKPASS="$askpass"', export)
        self.assertNotIn("https://github.com/keinoda/YaneuraOu.git", export)
        self.assertIn('send(process, "setoption name BookFile value no_book")', smoke)
        self.assertIn("CONTINUE_EXISTING_EXPORT=1", runbook)

    def test_modified_progress_is_fixed_without_automatic_recalibration(self) -> None:
        onstart = (REPO_ROOT / "onstart.sh").read_text(encoding="utf-8")
        survey = (SCRIPT_DIR / "run-survey.sh").read_text(encoding="utf-8")
        approve = (SCRIPT_DIR / "approve-progress.sh").read_text(encoding="utf-8")
        self.assertIn(
            'readonly PROGRESS_SOURCE_COMMIT="35752abe3035cb972ecfb98b1ce197028625c250"',
            onstart,
        )
        self.assertIn(
            'readonly REFERENCE_PROGRESS_SHA256="e7ed0eef88868335f9a46c58a121dccb5ad82a5eb1c8ee12de90365ab351e37d"',
            onstart,
        )
        self.assertIn(
            "api.github.com/repos/keinoda/YaneuraOu-private/contents/source/progress.bin",
            onstart,
        )
        self.assertNotIn("raw.githubusercontent.com/keinoda/YaneuraOu/", onstart)
        self.assertIn("--progress \"$REFERENCE_PROGRESS\"", survey)
        self.assertNotIn("--optimize-affine", survey)
        self.assertIn('CANDIDATE_NAME:-}" == "baseline"', approve)

    def test_resume_requires_target_after_initial_twenty_epochs(self) -> None:
        resume = (SCRIPT_DIR / "resume-training.sh").read_text(encoding="utf-8")
        self.assertIn("(( TARGET_SB > 841 ))", resume)
        self.assertNotIn("505|589|673|757|841", resume)

    def test_yaneuraou_smoke_disables_book_and_gets_all_bestmoves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = root / "fake-engine.py"
            engine.write_text(
                """#!/usr/bin/env python3
import sys

book_disabled = False
progress = None
for raw in sys.stdin:
    command = raw.rstrip("\\n")
    if command == "usi":
        print("option name EvalDir type string default eval")
        print("option name LS_PROGRESS_COEFF type string default <internal>")
        print("option name LS_BUCKET_MODE type combo default progress8kpabs")
        print("option name BookFile type combo default standard_book.db var no_book")
        print("usiok", flush=True)
    elif command.startswith("setoption name LS_PROGRESS_COEFF value "):
        progress = command.split(" value ", 1)[1]
    elif command == "setoption name BookFile value no_book":
        book_disabled = True
    elif command == "isready":
        if not book_disabled:
            print("info string Error! : attempted to load a book")
        print(f"info string loading progress file : {progress}")
        print("readyok", flush=True)
    elif command.startswith("go nodes "):
        print("bestmove 7g7f", flush=True)
    elif command == "quit":
        break
""",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            fixtures = root / "fixtures.jsonl"
            fixtures.write_text(
                "".join(json.dumps({"sfen": f"fixture-{index}"}) + "\n" for index in range(14)),
                encoding="utf-8",
            )
            transcript = root / "transcript.log"
            completed = subprocess.run(
                [
                    "python3",
                    str(SCRIPT_DIR / "yaneuraou-smoke.py"),
                    "--engine",
                    str(engine),
                    "--eval-dir",
                    str(root / "eval"),
                    "--progress",
                    str(root / "progress.bin"),
                    "--fixtures-jsonl",
                    str(fixtures),
                    "--transcript",
                    str(transcript),
                    "--nodes",
                    "1",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            output = transcript.read_text(encoding="utf-8")
            self.assertEqual(output.count("bestmove "), 15)
            self.assertNotIn("attempted to load a book", output)

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
            "8f461dd8dc4cb90c356392545a41e4e45c8f2418",
            "fdd5f602db82d888a87116f087d10dd5ea8313ab",
            "sha256:f84acfc2e3b147f5dacaf473061723ea5662eb2bddc648f3283ab2b7cd63b876",
        ):
            self.assertIn(revision, onstart)
            self.assertIn(revision, library)


if __name__ == "__main__":
    unittest.main()
