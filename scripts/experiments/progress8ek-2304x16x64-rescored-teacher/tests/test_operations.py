#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]


class OperationsTests(unittest.TestCase):
    def test_source_manifest_fixes_wcsc36_shards(self) -> None:
        manifest = SCRIPT_DIR / "source-shards.tsv"
        rows = [
            line.split("\t")
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(len(row) == 6 for row in rows))

        source = "penguinkumimanu/Knowledge_distilled_dataset_by_ponkotsuWCSC36"
        revision = "526bd42a59cdd961ef0c42e6068499625811ffee"
        expected_filenames = [
            f"ponkotsu_WCSC36_{index:03d}.bin" for index in range(1, 30)
        ] + ["dlsuisho_uniqueponkotsu_WCSC36"]

        self.assertEqual([row[2] for row in rows], expected_filenames)
        self.assertTrue(all(row[0] == source for row in rows))
        self.assertTrue(all(row[1] == revision for row in rows))
        self.assertTrue(all(row[5] == "rescored" for row in rows))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", row[4]) for row in rows))

        sizes = [int(row[3]) for row in rows]
        self.assertTrue(all(size % 40 == 0 for size in sizes))
        self.assertEqual(sum(sizes), 586_757_977_480)
        self.assertEqual(sum(sizes) // 40, 14_668_949_437)

    def test_browser_settings_pin_remote_commit_and_omit_dataset_download(self) -> None:
        branch = "codex/progress8ek-rescored-teacher-operations"
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
        self.assertIn("1x RTX 5070 / AMD Ryzen 9 9950X", generated)
        self.assertIn("利用者が指定。/workspaceの実容量はon-startで記録", generated)
        self.assertIn("-p 6001:6001", generated)
        self.assertIn(f'readonly commit="{remote}"', bootstrap)
        self.assertIn("checkout --detach \"$commit\"", bootstrap)
        self.assertIn('bash "$target/onstart.sh"', bootstrap)
        self.assertNotIn("VAST_VOLUME_GB", generated)
        self.assertNotIn("hf download", generated)
        self.assertNotIn("YANEURAOU_GITHUB_TOKEN", generated)
        self.assertNotIn("vastai create instance", generated)

    def test_onstart_builds_tools_without_dataset_identity(self) -> None:
        onstart = (REPO_ROOT / "onstart.sh").read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(REPO_ROOT / "onstart.sh")], check=True)
        self.assertIn("-p progress8ek-filter", onstart)
        self.assertIn("progress8ek-partition --help", onstart)
        self.assertIn("progress8ek-audit-psv --help", onstart)
        self.assertIn("progress8ek_finetune_updates_only_slot8", onstart)
        self.assertIn("dataset_download=deferred", onstart)
        self.assertNotIn("hf download", onstart)
        self.assertNotIn("TRAIN_DATASET", onstart)
        self.assertNotIn("download_training", onstart)

    def test_teacher_partition_produces_exactly_two_predicate_outputs(self) -> None:
        script = (SCRIPT_DIR / "run-teacher-partition.sh").read_text(encoding="utf-8")
        subprocess.run(
            ["bash", "-n", str(SCRIPT_DIR / "run-teacher-partition.sh")], check=True
        )
        self.assertIn('"$PARTITION_BIN"', script)
        self.assertIn('"$PARTITION_ORDINARY_PSV"', script)
        self.assertIn('ordinary_psv=%s', script)
        self.assertIn('entering_king_psv=%s', script)
        self.assertIn('predicate_verification=all_records', script)
        self.assertIn('--threads 16', script)
        self.assertNotIn('holdout', script)

    def test_invalid_ordinary_removal_requires_exhaustive_active_audits(self) -> None:
        script_path = SCRIPT_DIR / "run-remove-invalid-ordinary.sh"
        script = script_path.read_text(encoding="utf-8")
        subprocess.run(["bash", "-n", str(script_path)], check=True)
        self.assertIn('ordinary["expected_active_indices"] == 76', script)
        self.assertIn('ordinary["ineligible_records"]', script)
        self.assertIn('ordinary["zero_records"]', script)
        self.assertIn('last - first + 1 == removed', script)
        self.assertIn('entering["ineligible_records"] == 0', script)
        self.assertIn('iflag=count_bytes', script)
        self.assertIn('iflag=skip_bytes', script)
        self.assertIn('full_prefix_and_suffix_cmp', script)
        self.assertIn('"$PREPARED_DATA_MANIFEST"', script)

    def test_progress_adjustment_reuses_previous_affine_protocol(self) -> None:
        script = (SCRIPT_DIR / "run-progress-affine-survey.sh").read_text(
            encoding="utf-8"
        )
        subprocess.run(
            ["bash", "-n", str(SCRIPT_DIR / "run-progress-affine-survey.sh")],
            check=True,
        )
        self.assertIn('--optimize-affine', script)
        self.assertIn('--split calibration:2000000', script)
        self.assertIn('--split selection:1000000', script)
        self.assertIn('--split final-test:1000000', script)
        self.assertIn('11,12,13,14,14,13,12,11', script)
        self.assertIn('1.2980837735881936:-0.5975424282106219', script)
        self.assertIn('mkdir -p "$(dirname "$SURVEY_DIR")"', script)
        self.assertIn('"$PREPARED_DATA_MANIFEST"', script)
        self.assertIn('wcsc36-ordinary-valid76-affine', script)
        self.assertNotIn('--optimize-scale', script)

    def test_base_training_command_matches_requested_settings(self) -> None:
        printer = SCRIPT_DIR / "print-base-training-command.sh"
        subprocess.run(["bash", "-n", str(printer)], check=True)
        progress = "/approved/progress.bin"
        output = "/runs/nagisa-v5-base/checkpoints"
        printed = subprocess.run(
            [str(printer), progress, output],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        ).stdout
        command = shlex.split(printed)
        self.assertEqual(
            command,
            [
                str(REPO_ROOT / "target/release/nnue-train"),
                "--win-rate-model",
                "--batch-size",
                "65536",
                "--batches-per-superbatch",
                "10943",
                "--superbatches",
                "800",
                "--lr",
                "8.75e-4",
                "--lr-gamma",
                "0.995",
                "--lr-step",
                "1",
                "--weight-decay",
                "0.0",
                "--wdl",
                "0.0",
                "--scale",
                "290",
                "--save-rate",
                "100",
                "--threads",
                "16",
                "--all-optim",
                "--output",
                output,
                "--net-id",
                "nagisa-v5",
                "--data",
                str(REPO_ROOT / "data/training/ordinary-valid76.psv"),
                "--test-data",
                str(REPO_ROOT / "data/validation/floodgate.psv"),
                "--test-positions",
                "851968",
                "layerstack",
                "--ft-out",
                "2304",
                "--l1",
                "16",
                "--l2",
                "64",
                "--bucket-mode",
                "progress8kpabs",
                "--num-buckets",
                "8",
                "--progress-coeff",
                progress,
            ],
        )
        self.assertNotIn("--lr-schedule", command)
        self.assertNotIn("--optimizer", command)

        presented = 65_536 * 10_943 * 800
        epochs = presented / 14_342_411_752
        self.assertEqual(presented, 573_728_358_400)
        self.assertGreaterEqual(epochs, 40.0)
        self.assertLess(epochs, 40.01)

    def _bucket8_command(self, *, tail: int | None = None) -> list[str]:
        printer = SCRIPT_DIR / "print-bucket8-training-command.sh"
        subprocess.run(["bash", "-n", str(printer)], check=True)
        environment = os.environ.copy()
        if tail is not None:
            environment["BUCKET8_VALIDATION_TAIL_POSITIONS"] = str(tail)
        printed = subprocess.run(
            [str(printer), "/base/nagisa-v5-600.bin", "/approved/progress.bin"],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
            env=environment,
        ).stdout
        return shlex.split(printed)

    def test_bucket8_training_command_matches_approved_settings(self) -> None:
        command = self._bucket8_command()
        self.assertEqual(
            command,
            [
                str(REPO_ROOT / "target/release/nnue-train"),
                "--init-from",
                "/base/nagisa-v5-600.bin",
                "--win-rate-model",
                "--batch-size",
                "65536",
                "--batches-per-superbatch",
                "250",
                "--superbatches",
                "800",
                "--lr",
                "8.75e-4",
                "--lr-gamma",
                "0.995",
                "--lr-step",
                "1",
                "--weight-decay",
                "0.0",
                "--wdl",
                "0.33",
                "--scale",
                "290",
                "--save-rate",
                "100",
                "--threads",
                "16",
                "--all-optim",
                "--output",
                str(REPO_ROOT / "runs/nagisa-v5-bucket8/checkpoints"),
                "--net-id",
                "nagisa-v5",
                "--data",
                str(REPO_ROOT / "data/training/entering-king.psv"),
                "--test-tail-positions",
                "851968",
                "--test-positions",
                "851968",
                "layerstack",
                "--ft-out",
                "2304",
                "--l1",
                "16",
                "--l2",
                "64",
                "--bucket-mode",
                "progress8ek",
                "--num-buckets",
                "9",
                "--progress-coeff",
                "/approved/progress.bin",
                "--progress8ek-finetune",
                "--progress8ek-source-slot",
                "7",
            ],
        )
        self.assertNotIn("--resume", command)
        self.assertNotIn("--lr-schedule", command)
        self.assertNotIn("--keep-checkpoints", command)
        self.assertNotIn("--test-data", command)

    def test_bucket8_training_command_without_validation_tail(self) -> None:
        command = self._bucket8_command(tail=0)
        self.assertNotIn("--test-tail-positions", command)
        self.assertNotIn("--test-positions", command)
        self.assertIn("--progress8ek-finetune", command)

    def test_bucket8_volume_is_minimal_ceil_over_entering_king_file(self) -> None:
        entering_king = 326_531_157
        tail = 851_968
        batch_size, batches, superbatches = 65_536, 250, 800
        presented = batch_size * batches * superbatches
        self.assertEqual(presented, 13_107_200_000)
        self.assertGreaterEqual(presented, entering_king * 40)
        self.assertLess(batch_size * (batches - 1) * superbatches, entering_king * 40)
        self.assertEqual(tail % batch_size, 0)
        self.assertAlmostEqual(presented / entering_king, 40.1407, places=4)
        self.assertAlmostEqual(presented / (entering_king - tail), 40.2457, places=4)

    def test_bucket8_gate_scripts_fail_closed(self) -> None:
        scripts = {
            name: (SCRIPT_DIR / name).read_text(encoding="utf-8")
            for name in (
                "run-bucket8-cuda-gate.sh",
                "run-bucket8-preflight-smoke.sh",
                "prepare-bucket8-run.sh",
                "switch-monitor-to-bucket8.sh",
                "start-bucket8-training.sh",
            )
        }
        for name in scripts:
            subprocess.run(["bash", "-n", str(SCRIPT_DIR / name)], check=True)

        gate = scripts["run-bucket8-cuda-gate.sh"]
        self.assertIn("progress8ek_finetune_updates_only_slot8", gate)
        self.assertIn("既存CUDA gateを上書きしません", gate)

        smoke = scripts["run-bucket8-preflight-smoke.sh"]
        self.assertIn("build_bucket8_smoke_command", smoke)
        self.assertIn("--require-slot8-difference", smoke)
        self.assertIn("--ft-out 2304", smoke)
        self.assertIn("--assume-progress8ek", smoke)
        self.assertIn("shared_parameters_bit_identical", smoke)
        self.assertIn("slots_0_through_7_bit_identical", smoke)
        self.assertIn("require_no_trainer_process", smoke)
        self.assertIn('"$CUDA_GATE_DIR/done"', smoke)

        prepare = scripts["prepare-bucket8-run.sh"]
        self.assertIn("FULL_CI_LOG", prepare)
        self.assertIn('"$SMOKE_ROOT/done"', prepare)
        self.assertIn('require_file_sha256 "$ENTERING_KING_PSV" "$ENTERING_KING_SHA256"', prepare)
        self.assertIn("trainer_started=false", prepare)
        self.assertIn("既存runを上書きしません", prepare)
        self.assertNotIn("tmux new-session", prepare)

        switch = scripts["switch-monitor-to-bucket8.sh"]
        self.assertIn('"$previous_root/state/training.ended"', switch)
        self.assertIn("TRAINING_PHASE=bucket8", switch)
        self.assertNotIn("nnue-train --", switch)

        start = scripts["start-bucket8-training.sh"]
        self.assertIn('"$GATE_DIR/monitor.done"', start)
        self.assertIn("query-compute-apps", start)
        self.assertIn("command_sha256", start)
        self.assertIn("base_network_sha256", start)
        self.assertIn('tmux new-session -d -s "$BUCKET8_TRAINER_SESSION"', start)

    def test_browser_settings_reject_remote_commit_mismatch(self) -> None:
        branch = "codex/progress8ek-rescored-teacher-operations"
        with tempfile.TemporaryDirectory() as temporary:
            fake_git = Path(temporary) / "git"
            fake_git.write_text(
                f"""#!/usr/bin/env bash
printf '%s\\trefs/heads/{branch}\\n' '{"b" * 40}'
""",
                encoding="utf-8",
            )
            fake_git.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "TATARA_COMMIT": "a" * 40,
                    "PATH": f"{temporary}:{environment['PATH']}",
                }
            )
            rejected = subprocess.run(
                [str(SCRIPT_DIR / "print-vast-browser-settings.sh")],
                check=False,
                text=True,
                capture_output=True,
                cwd=REPO_ROOT,
                env=environment,
            )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("originの専用branch先端", rejected.stderr)

    def test_phase_names_and_monitor_paths_are_fixed(self) -> None:
        command = f"""
source {SCRIPT_DIR / 'lib.sh'!s}
printf '%s\\n' "$(run_name_for_phase base)" "$(run_name_for_phase bucket8)"
"""
        output = subprocess.run(
            ["bash", "-c", command],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        ).stdout.splitlines()
        self.assertEqual(output, ["nagisa-v5-base", "nagisa-v5-bucket8"])

        monitor = (SCRIPT_DIR / "run-monitor.sh").read_text(encoding="utf-8")
        self.assertIn("MONITOR_IMPLEMENTATION_DIR/monitor.py", monitor)
        self.assertIn("MONITOR_IMPLEMENTATION_DIR/monitor_server.py", monitor)
        self.assertIn("--bind 0.0.0.0 --port %q", monitor)
        self.assertIn("unauth_status", monitor)
        self.assertIn('"$MONITOR_PUBLIC_URL/status.json"', monitor)
        self.assertIn(
            '--milestone-interval "$MONITOR_MILESTONE_INTERVAL"', monitor
        )
        self.assertNotIn("seq 1 30", monitor)

    def test_floodgate_validation_reuses_pinned_yamaoka_dataset(self) -> None:
        script = SCRIPT_DIR / "run-prepare-floodgate-validation.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        lib = (SCRIPT_DIR / "lib.sh").read_text(encoding="utf-8")
        preparation = script.read_text(encoding="utf-8")

        self.assertIn(
            'readonly VALIDATION_DATASET="takaoyamaoka/floodgate.hcpe"', lib
        )
        self.assertIn(
            'readonly VALIDATION_DATASET_REVISION="fdd5f602db82d888a87116f087d10dd5ea8313ab"',
            lib,
        )
        self.assertIn("readonly VALIDATION_FILE_POSITIONS=856923", lib)
        self.assertIn("readonly VALIDATION_EFFECTIVE_POSITIONS=851968", lib)
        self.assertIn(
            'readonly VALIDATION_PSV_SHA256="22e11b82fa4ac7d75e82806480b5bfdd7ba29d773bcfb91ed5b3dfe7a43d66b8"',
            lib,
        )
        self.assertIn("readonly MONITOR_MILESTONE_INTERVAL=100", lib)
        self.assertIn('hf download "$VALIDATION_DATASET"', preparation)
        self.assertIn('checkout --detach "$RSHOGI_COMMIT"', preparation)
        self.assertIn("--bin hcpe_to_psv", preparation)


if __name__ == "__main__":
    unittest.main()
