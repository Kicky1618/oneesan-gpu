#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))

from solver_output import SolverResult, parse_result_line


class SolverOutputTest(unittest.TestCase):
    def test_production_style_line(self):
        line = (
            "backend=gridfp-b300-hbm32-factorized-batch n=27 residue=123 "
            "modulus=4294967291 residue_index=0 residues_total=20 gpus=8 "
            "active_max_s=1.2e-05 wall_s=3.125e-04"
        )
        self.assertEqual(parse_result_line(line), SolverResult(27, 123, 4294967291, 0.0003125))

    def test_whitespace_and_plain_decimal_wall(self):
        self.assertEqual(
            parse_result_line("  n=1 residue=0 modulus=2 wall_s=0.001  \n"),
            SolverResult(1, 0, 2, 0.001),
        )

    def test_diagnostic_line_is_ignored(self):
        self.assertIsNone(parse_result_line("prepare_s=1.2 active_sum_s=3.4"))
        self.assertIsNone(parse_result_line("gpu_phase_wall_s=1.2 table_wall_s=0.1"))

    def test_partial_or_duplicate_result_is_rejected(self):
        for line in (
            "n=1 residue=1 modulus=2",
            "n=1 residue=1 modulus=2 residue=1 wall_s=0.1",
            "n=1 residue=1 residue=2 modulus=3 wall_s=0.1",
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_result_line(line)

    def test_similar_diagnostic_key_is_not_a_result_field(self):
        with self.assertRaisesRegex(ValueError, "missing=.*residue"):
            parse_result_line("n=1 prefix-residue=1 modulus=2 wall_s=0.1")

    def test_noncanonical_integers_are_rejected(self):
        for line in (
            "n=1 residue=01 modulus=2 wall_s=0.1",
            "n=1 residue=+1 modulus=2 wall_s=0.1",
            "n=1 residue=-1 modulus=2 wall_s=0.1",
            "n=1 residue=1 modulus=01 wall_s=0.1",
            "n=1 residue=1 modulus=+2 wall_s=0.1",
            "n=1 residue=1 modulus=1 wall_s=0.1",
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_result_line(line)

    def test_invalid_wall_times_are_rejected(self):
        for wall in ("nan", "inf", "-1", "+1", "01", "1e999", ".5"):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                parse_result_line(f"n=1 residue=1 modulus=2 wall_s={wall}")

    def test_missing_or_invalid_n_is_rejected(self):
        for line in (
            "residue=1 modulus=2 wall_s=0.1",
            "n=0 residue=1 modulus=2 wall_s=0.1",
            "n=01 residue=1 modulus=2 wall_s=0.1",
            "n=-1 residue=1 modulus=2 wall_s=0.1",
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_result_line(line)

    def test_n_only_diagnostic_is_ignored(self):
        self.assertIsNone(parse_result_line("backend=plan n=27 gpus=8"))

    def test_field_order_is_irrelevant(self):
        # The C++ output order is covered separately by the source contract test;
        # parsing is key-based so harmless diagnostic reordering does not break resume.
        self.assertEqual(
            parse_result_line("wall_s=1e-06 modulus=7 residue=6 n=1"),
            SolverResult(1, 6, 7, 1e-6),
        )


if __name__ == "__main__":
    unittest.main()
