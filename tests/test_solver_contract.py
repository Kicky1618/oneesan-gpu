#!/usr/bin/env python3
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "cuda" / "b300" / "oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
SINGLE = ROOT / "scripts" / "solve" / "solve_b300_exact.py"
BATCH = ROOT / "scripts" / "solve" / "solve_b300_exact_batch.py"
WRAPPER = ROOT / "scripts" / "run" / "b300x8-exact.sh"


class SolverContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.single = SINGLE.read_text()
        cls.batch = BATCH.read_text()
        cls.wrapper = WRAPPER.read_text()

    def test_production_cli_argument_positions(self):
        patterns = (
            r"int\s+n\s*=\s*argc>1\?std::atoi\(argv\[1\]\)",
            r"int\s+target_mib\s*=\s*argc>2\?std::atoi\(argv\[2\]\)",
            r"int\s+max_window\s*=\s*argc>3\?std::atoi\(argv\[3\]\)",
            r"int\s+requested\s*=\s*argc>4\?std::atoi\(argv\[4\]\)",
            r"for\s*\(int\s+i=5\s*;\s*i<argc\s*;\s*\+\+i\)",
        )
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                self.assertRegex(self.source, pattern)

    def test_runners_use_the_same_cli_prefix(self):
        expected = 'str(binary), str(n), str(args.target_mib), str(args.max_window), str(args.gpus)'
        self.assertIn(expected, self.single)
        self.assertIn(expected, self.batch)

    def test_production_result_contains_required_tokens_in_order(self):
        # Scope to the final stdout expression rather than accepting diagnostic stderr.
        line = next(
            ln for ln in self.source.splitlines()
            if 'std::cout<<"backend=gridfp-b300-hbm32-factorized-batch' in ln
        )
        positions = [line.index(token) for token in ('n=', 'residue=', 'modulus=', 'wall_s=')]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(line.count('n='), 1)
        self.assertEqual(line.count('residue='), 1)
        self.assertEqual(line.count('modulus='), 1)
        self.assertEqual(line.count('wall_s='), 1)

    def test_exact_wrapper_verifies_build_source_closure(self):
        # Exact admission must reject a stale binary when any dependency in its
        # recorded source closure changes. Unrelated checkout files are not part
        # of the build provenance and therefore do not invalidate a resume.
        self.assertIn("--verify-sources", self.wrapper)
        self.assertIn('--root "$ONEESAN_ROOT"', self.wrapper)
        self.assertIn('--expect-compile-arg="-DTARGET_W=$((N + 1))"', self.wrapper)

    def test_runners_reject_result_for_wrong_n(self):
        self.assertIn("if got_n != n:", self.single)
        self.assertIn("if got_n != n:", self.batch)

    def test_runners_share_one_result_parser(self):
        self.assertNotIn("RESULT_RE", self.single)
        self.assertNotIn("RESULT_RE", self.batch)
        self.assertIn("from solver_output import parse_result_line", self.single)
        self.assertIn("from solver_output import parse_result_line", self.batch)


if __name__ == "__main__":
    unittest.main()
