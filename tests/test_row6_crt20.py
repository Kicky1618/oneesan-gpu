#!/usr/bin/env python3
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from gen_row6_crt20 import generate
from verify_row6_crt20 import CRT_HEADER, verify_all


class Row6Crt20CertificateTest(unittest.TestCase):

    def test_generator_reproduces_header_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as td:
            regenerated = Path(td) / "row6_automaton_crt20_generated.hpp"
            generate(regenerated)
            with CRT_HEADER.open("rb") as f:
                expected = hashlib.file_digest(f, "sha256").hexdigest()
            with regenerated.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            self.assertEqual(actual, expected)

    def test_generated_tables_match_exact_rational_certificate(self):
        verify_all()


if __name__ == "__main__":
    unittest.main()
