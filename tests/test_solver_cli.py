#!/usr/bin/env python3
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = (
    ROOT / "scripts" / "solve" / "solve_b300_exact.py",
    ROOT / "scripts" / "solve" / "solve_b300_exact_batch.py",
)


class SolverCliTest(unittest.TestCase):
    def test_invalid_ranges_fail_before_binary_access(self):
        cases = (
            (["0"], "n must be in 2..27"),
            (["1"], "n must be in 2..27"),
            (["28"], "n must be in 2..27"),
            (["2", "--target-mib", "0"], "--target-mib must be at least 1"),
            (["2", "--max-window", "0"], "--max-window must be at least 1"),
            (["2", "--gpus", "9"], "--gpus must be in 0..8"),
            (["2", "--gpus", "-1"], "--gpus must be in 0..8"),
            (["2", "--max-runs", "-1"], "--max-runs must be nonnegative"),
        )
        for runner in RUNNERS:
            for args, message in cases:
                with self.subTest(runner=runner.name, args=args):
                    proc = subprocess.run(
                        [sys.executable, str(runner), *args],
                        cwd=ROOT, text=True, capture_output=True,
                    )
                    self.assertEqual(proc.returncode, 2, proc.stderr + proc.stdout)
                    self.assertIn(message, proc.stderr)
                    self.assertNotIn("binary not found", proc.stderr)

    def test_b300_build_scripts_reject_out_of_range_n_before_nvcc_lookup(self):
        for script in ("b300-hbm32-batch.sh", "b300-hbm32.sh"):
            with self.subTest(script=script):
                env = os.environ.copy()
                env["N"] = "28"
                proc = subprocess.run(
                    [str(ROOT / "scripts" / "build" / script)],
                    cwd=ROOT, env=env, text=True, capture_output=True,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr + proc.stdout)
                self.assertIn("N must be in 2..27", proc.stderr)
                self.assertNotIn("nvcc", proc.stderr.lower())

    def test_exact_wrapper_rejects_out_of_range_n_before_gpu_probe(self):
        for n in ("1", "28"):
            with self.subTest(n=n):
                proc = subprocess.run(
                    [str(ROOT / "scripts" / "run" / "b300x8-exact.sh"), n],
                    cwd=ROOT, text=True, capture_output=True,
                )
                self.assertEqual(proc.returncode, 2, proc.stderr + proc.stdout)
                self.assertIn("N must be in 2..27", proc.stderr)
                self.assertNotIn("nvidia-smi not found", proc.stderr)

    def test_exact_wrapper_rejects_uncertified_row7_before_gpu_probe(self):
        env = os.environ.copy()
        env["ROW7_TENSOR"] = "1"
        proc = subprocess.run(
            [str(ROOT / "scripts" / "run" / "b300x8-exact.sh"), "27"],
            cwd=ROOT, env=env, text=True, capture_output=True,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr + proc.stdout)
        self.assertIn("ROW7_TENSOR is benchmark-only", proc.stderr)
        self.assertIn("clean-clone reproducible certificate chain", proc.stderr)
        self.assertNotIn("nvidia-smi not found", proc.stderr)



if __name__ == "__main__":
    unittest.main()
