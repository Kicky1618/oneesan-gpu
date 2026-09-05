#!/usr/bin/env python3
import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from solve_b300_exact import (
    _strip_compatible,
    checkerboard_strip_count,
    simple_path_upper_bound,
)
from verify_exact_result import _columns_compatible, independent_path_upper_bound
from path_bound import _strip_orbits


def brute_no_checkerboard(height: int, width: int) -> int:
    cells = height * width
    total = 0
    for mask in range(1 << cells):
        ok = True
        for r in range(height - 1):
            for c in range(width - 1):
                a = (mask >> (r * width + c)) & 1
                b = (mask >> ((r + 1) * width + c)) & 1
                x = (mask >> (r * width + c + 1)) & 1
                y = (mask >> ((r + 1) * width + c + 1)) & 1
                if a != b and x != y and a != x:
                    ok = False
                    break
            if not ok:
                break
        total += ok
    return total


def composition_products(n: int, strip_counts: dict[int, int]):
    if n == 0:
        yield 1, []
        return
    for h, count in strip_counts.items():
        if h <= n:
            for rest, parts in composition_products(n - h, strip_counts):
                yield count * rest, [h] + parts


class PathBoundTest(unittest.TestCase):
    def test_orbits_partition_columns_and_match_burnside_count(self):
        for h in range(1, 10):
            reps, orbit_of, sizes = _strip_orbits(h)
            # Identity + reflection + complement-reflection fixed points;
            # nonempty columns never equal their own bit complement.
            expected = ((1 << h) + (1 << ((h + 1) // 2))
                        + ((1 << (h // 2)) if h % 2 == 0 else 0)) // 4
            self.assertEqual(len(reps), expected)
            self.assertEqual(sum(sizes), 1 << h)
            mask = (1 << h) - 1
            for x, index in enumerate(orbit_of):
                reflected = int(f"{x:0{h}b}"[::-1], 2)
                self.assertEqual(orbit_of[x ^ mask], index)
                self.assertEqual(orbit_of[reflected], index)
                self.assertEqual(reps[index], min(x, x ^ mask, reflected, reflected ^ mask))
            self.assertEqual(sizes, [orbit_of.count(i) for i in range(len(reps))])

    def test_quotient_step_for_nonuniform_orbit_values(self):
        # Compare an entire arbitrary invariant vector, not just the final
        # count from uniform initialization; missing edge multiplicities fail.
        for h in range(1, 8):
            reps, orbit_of, _ = _strip_orbits(h)
            values = [(i * 47 + 13) % 101 for i in range(len(reps))]
            dense_next = [sum(values[orbit_of[y]] for y in range(1 << h)
                              if _columns_compatible(x, y, h)) for x in range(1 << h)]
            for x in range(1 << h):
                self.assertEqual(dense_next[x], dense_next[reps[orbit_of[x]]])

    def test_local_compatibility_rejects_exactly_two_checkerboards(self):
        rejected = []
        for a, b, c, d in itertools.product((0, 1), repeat=4):
            left = a | (b << 1)
            right = c | (d << 1)
            if not _strip_compatible(left, right, 2):
                rejected.append((a, b, c, d))
        self.assertEqual(sorted(rejected), [(0, 1, 1, 0), (1, 0, 0, 1)])

    def test_strip_dp_matches_direct_matrix_enumeration(self):
        for h, w in [(1, 1), (1, 4), (2, 2), (2, 4), (3, 3), (4, 3)]:
            self.assertEqual(
                checkerboard_strip_count(h, w),
                brute_no_checkerboard(h, w),
                (h, w),
            )

    def test_partition_dp_is_globally_minimal_for_small_n(self):
        for n in range(1, 8):
            hmax = min(4, n)
            counts = {h: checkerboard_strip_count(h, n) for h in range(1, hmax + 1)}
            expected_value, _ = min(composition_products(n, counts), key=lambda x: x[0])
            actual_value, actual_parts = simple_path_upper_bound(n, max_strip_height=4)
            self.assertEqual(actual_value, expected_value, n)
            self.assertEqual(sum(actual_parts), n)
            self.assertTrue(all(1 <= h <= hmax for h in actual_parts))

    def test_full_height_bound_equals_unstripped_count_for_small_n(self):
        for n in range(1, 6):
            bound, parts = simple_path_upper_bound(n, max_strip_height=n)
            full = checkerboard_strip_count(n, n)
            self.assertLessEqual(bound, full)
            # Since a one-strip partition is available, the optimized product
            # cannot exceed the exact no-checkerboard matrix count.  In these
            # small cases it should choose that full-height strip.
            self.assertEqual(bound, full)
            self.assertEqual(parts, [n])

    def test_independent_verifier_compatibility_matches_production_exhaustively(self):
        for h in range(1, 10):
            for x in range(1 << h):
                for y in range(1 << h):
                    self.assertEqual(
                        _strip_compatible(x, y, h),
                        _columns_compatible(x, y, h),
                        (h, x, y),
                    )

    def test_independent_verifier_bound_matches_production_n1_to_n10(self):
        for n in range(1, 11):
            self.assertEqual(
                independent_path_upper_bound(n),
                simple_path_upper_bound(n),
                n,
            )

    def test_formal_n27_constants_match_production(self):
        formal = (ROOT / "formal" / "OneesanFormal" / "ProductionBound27.lean").read_text()

        def lean_nat(name: str) -> int:
            import re

            match = re.search(
                rf"def\s+{re.escape(name)}\s*:\s*Nat\s*:=\s*([0-9]+)",
                formal,
            )
            self.assertIsNotNone(match, name)
            return int(match.group(1))

        formal_strip = lean_nat("stripCount9x27")
        formal_bound = lean_nat("pathBound27")
        production_strip = checkerboard_strip_count(9, 27)
        production_bound, parts = simple_path_upper_bound(27)

        self.assertEqual(parts, [9, 9, 9])
        self.assertEqual(formal_strip, production_strip)
        self.assertEqual(formal_bound, production_bound)
        self.assertEqual(formal_bound, formal_strip**3)

    def test_n27_bound_fingerprint(self):
        bound, parts = simple_path_upper_bound(27)
        self.assertEqual(parts, [9, 9, 9])
        self.assertEqual(bound.bit_length(), 633)
        self.assertEqual(
            bound,
            31732427633797389964407887052573851640105323179333844527763421102211579188310190597412900756550874123129342585261840138964010190312364508107168927006232277067783279927977876779029754107293528,
        )


if __name__ == "__main__":
    unittest.main()
