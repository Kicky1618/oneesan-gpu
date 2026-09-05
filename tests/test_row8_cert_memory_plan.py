#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'tools'))

from row8_cert_memory_plan import plan


class Row8CertMemoryPlanTest(unittest.TestCase):
    def test_w22_matches_production_state_counts(self):
        d = plan(22, 1, reserve_mib=128, safety_mib=1024)
        self.assertEqual(d['main_states'], 728_997_192)
        self.assertEqual(d['blocked_states'], 258_215_664)
        self.assertEqual(d['auth_per_gpu_bytes'], (728_997_192 + 258_215_664) * 4)
        self.assertEqual(len(d['forced_windows']), 2)
        self.assertEqual(d['forced_windows'][0]['max_scratch_bytes'], 66_521_760)
        self.assertEqual(d['forced_windows'][1]['max_scratch_bytes'], 102_397_624)

    def test_w28_b300_plan_is_stable(self):
        d = plan(28, 8, reserve_mib=8192, safety_mib=1024)
        self.assertEqual(d['main_states'], 385_719_506_620)
        self.assertEqual(d['blocked_states'], 135_015_505_407)
        self.assertEqual(d['main_chunk'], 48_214_938_328)
        self.assertEqual(d['blocked_chunk'], 16_876_938_176)
        self.assertEqual(d['auth_per_gpu_bytes'], 260_367_506_016)
        self.assertEqual(d['factor_table_bytes'], 319_060_548)
        self.assertEqual(d['forced_windows'][0]['groups'], 16_384)
        self.assertEqual(d['forced_windows'][1]['groups'], 8_192)
        self.assertEqual(d['forced_windows'][0]['max_scratch_bytes'], 10_357_746_288)
        self.assertEqual(d['forced_windows'][1]['max_scratch_bytes'], 15_859_230_032)
        self.assertEqual(d['minimum_total_bytes'], 286_209_473_012)
        # 288 GB decimal B300 HBM leaves only about 1.7 GiB over this guarded plan.
        b300_bytes = 288_000_000_000
        self.assertGreater(b300_bytes, d['minimum_total_bytes'])
        self.assertLess(b300_bytes - d['minimum_total_bytes'], 2 * (1 << 30))

    def test_w10_certificate_fits_small_smoke_target(self):
        d = plan(10, 1, reserve_mib=128, safety_mib=128)
        self.assertLessEqual(d['max_forced_scratch_mib'], 128)


if __name__ == '__main__':
    unittest.main()
