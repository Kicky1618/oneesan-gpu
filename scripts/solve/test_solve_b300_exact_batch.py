#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import solve_b300_exact_batch as batch


class ExactBatchSafetyTest(unittest.TestCase):
    def test_checkpoint_requires_matching_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cp = root / "checkpoint.json"
            fp = {"schema": batch.CHECKPOINT_SCHEMA, "binary_sha256": "a" * 64}
            batch.save_checkpoint(cp, 27, fp, {13: {"residue": 7, "wall_s": 1.0}})
            self.assertEqual(batch.load_checkpoint(cp, 27, fp)[13]["residue"], 7)
            bad = {"schema": batch.CHECKPOINT_SCHEMA, "binary_sha256": "b" * 64}
            with self.assertRaises(SystemExit):
                batch.load_checkpoint(cp, 27, bad)

    def test_legacy_checkpoint_without_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "checkpoint.json"
            cp.write_text(json.dumps({"n": 27, "residues": {}}))
            fp = {"schema": batch.CHECKPOINT_SCHEMA, "binary_sha256": "a" * 64}
            with self.assertRaises(SystemExit):
                batch.load_checkpoint(cp, 27, fp)

    def test_invalid_cached_residue_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cp = Path(td) / "checkpoint.json"
            fp = {"schema": batch.CHECKPOINT_SCHEMA, "binary_sha256": "a" * 64}
            cp.write_text(json.dumps({
                "n": 27,
                "solver_fingerprint": fp,
                "residues": {"13": {"residue": 13, "wall_s": 0.0}},
            }))
            with self.assertRaises(SystemExit):
                batch.load_checkpoint(cp, 27, fp)

    def test_finish_rejects_candidate_above_rigorous_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fp = {"schema": batch.CHECKPOINT_SCHEMA, "binary_sha256": "a" * 64}
            # Product 13 exceeds bound 10, but canonical CRT representative 12
            # cannot be the true value of anything proven <=10.
            residues = {13: {"residue": 12, "wall_s": 0.0}}
            with self.assertRaises(RuntimeError):
                batch.finish(Path(td), 1, 10, [13], residues, fp)
            self.assertFalse((Path(td) / "exact.txt").exists())

    def test_finish_accepts_candidate_within_bound(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fp = {"schema": batch.CHECKPOINT_SCHEMA, "binary_sha256": "c" * 64}
            residues = {13: {"residue": 7, "wall_s": 2.5}}
            rc = batch.finish(root, 1, 10, [13], residues, fp)
            self.assertEqual(rc, 0)
            text = (root / "exact.txt").read_text()
            self.assertIn("exact=7\n", text)
            self.assertIn("solver_binary_sha256=" + "c" * 64, text)


if __name__ == "__main__":
    unittest.main()
