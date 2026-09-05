#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "scripts" / "tools" / "row8_gridfp_structural_cert.py"
spec = importlib.util.spec_from_file_location("row8cert", TOOL)
row8cert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(row8cert)


class Row8StructuralCertBoundTest(unittest.TestCase):
    def test_required_prime_counts_use_parity_fiber_bound(self):
        expected = {10: 3, 22: 6, 28: 7}
        for width, count in expected.items():
            with self.subTest(width=width):
                primes, product, bound = row8cert.required_primes(width)
                self.assertEqual(len(primes), count)
                self.assertGreater(product, bound["difference_abs"])
                if len(primes) > 1:
                    prev = 1
                    for p in primes[:-1]:
                        prev *= p
                    self.assertLessEqual(prev, bound["difference_abs"])

    def test_w28_bound_is_216_free_baseline_bits(self):
        _norms, bound = row8cert.bounds(28)
        self.assertEqual(bound["processed_edges"], 440)
        self.assertEqual(bound["baseline_parity_free_bits"], 216)
        self.assertEqual(bound["baseline_abs"], 1 << 216)
        self.assertLessEqual(bound["structural_abs"].bit_length(), 215)
        self.assertEqual(bound["difference_abs"].bit_length(), 217)

    def test_parity_bound_saves_half_the_w28_comparisons(self):
        primes, _product, _bound = row8cert.required_primes(28)
        self.assertEqual(len(primes), 7)
        # The former all-edge-subset bound needed 14 near-32-bit primes.
        self.assertLess(len(primes), 14)


if __name__ == "__main__":
    unittest.main()
