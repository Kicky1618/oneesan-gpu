#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))

from solve_b300_exact import PRIMES, primes_for_bound, simple_path_upper_bound


def is_prime_32(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


class CrtPrimeTableTest(unittest.TestCase):
    def test_prime_table_is_strict_distinct_prime_sequence(self):
        self.assertEqual(len(PRIMES), len(set(PRIMES)))
        self.assertTrue(all(a > b for a, b in zip(PRIMES, PRIMES[1:])))
        self.assertTrue(all(is_prime_32(p) for p in PRIMES))
        self.assertTrue(
            all(math.gcd(PRIMES[i], PRIMES[j]) == 1
                for i in range(len(PRIMES)) for j in range(i))
        )

    def test_n1_to_n27_use_minimal_sufficient_prefix(self):
        for n in range(1, 28):
            bound, _ = simple_path_upper_bound(n)
            prefix = primes_for_bound(bound)
            product = math.prod(prefix)
            self.assertGreater(product, bound, f"n={n}")
            if len(prefix) > 1:
                self.assertLessEqual(product // prefix[-1], bound, f"n={n}")
            else:
                self.assertLessEqual(1, bound, f"n={n}")

    def test_full_table_capacity_covers_n27_bound(self):
        bound, _ = simple_path_upper_bound(27)
        self.assertGreater(math.prod(PRIMES), bound)


if __name__ == "__main__":
    unittest.main()
