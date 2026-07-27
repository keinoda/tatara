from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "plan-training.py"
SPEC = importlib.util.spec_from_file_location("plan_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlanTrainingTests(unittest.TestCase):
    def test_plan_covers_at_least_ten_epochs_with_one_cycle(self) -> None:
        plan = MODULE.build_plan(73_000_000, 8_000_000)
        self.assertGreaterEqual(plan["actual_epochs"], 10.0)
        self.assertLess(plan["actual_epochs"], 10.1)
        self.assertEqual(plan["superbatches"], 100)
        self.assertEqual(plan["lr_schedule"], "one-cycle")
        self.assertEqual(plan["lr"], 8.75e-4)

    def test_small_dataset_reduces_superbatch_count(self) -> None:
        plan = MODULE.build_plan(65_536, 65_536)
        self.assertEqual(plan["target_batches"], 10)
        self.assertEqual(plan["superbatches"], 10)
        self.assertEqual(plan["batches_per_superbatch"], 1)

    def test_rejects_dataset_smaller_than_one_batch(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.build_plan(65_535, 65_536)
        with self.assertRaises(ValueError):
            MODULE.build_plan(65_536, 65_535)

    def test_400_superbatch_step_plan_targets_twenty_epochs(self) -> None:
        plan = MODULE.build_plan(
            337_073_603,
            37_452_623,
            target_epochs=20.0,
            desired_superbatches=400,
            lr_schedule="step",
            lr_gamma=0.992,
            save_rate=20,
            batch_rounding="nearest",
        )
        self.assertEqual(plan["superbatches"], 400)
        self.assertEqual(plan["batches_per_superbatch"], 257)
        self.assertAlmostEqual(plan["actual_epochs"], 19.987032, places=5)
        self.assertEqual(plan["save_rate"], 20)
        self.assertEqual(plan["lr_schedule"], "step")
        self.assertEqual(plan["lr_gamma"], 0.992)


if __name__ == "__main__":
    unittest.main()
