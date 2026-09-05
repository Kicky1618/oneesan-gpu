#!/usr/bin/env python3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "tools" / "validate_zdd.py"


class ValidateZddTest(unittest.TestCase):
    def test_deep_chain_is_iterative(self):
        depth = 1500
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "deep.zdd"
            lines = ["ONEESAN_ZDD_V1", f"variables {depth}", f"root {depth + 1}"]
            for level in range(1, depth + 1):
                lines.append(f"var {level} {level - 1} 0 1")
            child = 1
            for level in range(1, depth + 1):
                nid = level + 1
                lines.append(f"node {nid} {level} 0 {child}")
                child = nid
            lines.append("end")
            path.write_text("\n".join(lines) + "\n")
            proc = subprocess.run(
                [sys.executable, str(VALIDATOR), str(path)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("exact_cardinality=1", proc.stdout)
            self.assertIn("valid=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
