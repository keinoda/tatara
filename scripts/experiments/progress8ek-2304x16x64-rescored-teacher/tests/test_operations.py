#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]


class OperationsTests(unittest.TestCase):
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
        self.assertIn("1x RTX 5090 / AMD Ryzen 9 9950X", generated)
        self.assertIn("任意の容量を /workspace にmount", generated)
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
        self.assertIn("progress8ek_finetune_updates_only_slot8", onstart)
        self.assertIn("dataset_download=deferred", onstart)
        self.assertNotIn("hf download", onstart)
        self.assertNotIn("TRAIN_DATASET", onstart)
        self.assertNotIn("download_training", onstart)

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
        self.assertNotIn("seq 1 30", monitor)


if __name__ == "__main__":
    unittest.main()
