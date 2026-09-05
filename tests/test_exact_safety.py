#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))

from solve_b300_exact_batch import finish
from build_provenance import CHECKSUM_FIELD as PROVENANCE_CHECKSUM_FIELD, checksum as provenance_checksum, create_provenance

from exact_safety import (
    acquire_workdir_lock,
    checkpoint_checksum,
    admission_certificate_identity,
    load_checkpoint,
    save_checkpoint,
    solver_identity,
    validate_exact_reconstruction,
    validate_residue,
    validate_wall_s,
)


class ExactSafetyTest(unittest.TestCase):
    def test_residue_range(self):
        validate_residue(17, 0)
        validate_residue(17, 16)
        with self.assertRaises(ValueError):
            validate_residue(17, 17)
        with self.assertRaises(ValueError):
            validate_residue(17, -1)



    def test_wall_time_must_be_finite_nonnegative(self):
        self.assertEqual(validate_wall_s(0.25), 0.25)
        for bad in (-1.0, float("inf"), float("-inf"), float("nan"), True, "1.0"):
            with self.assertRaises(ValueError):
                validate_wall_s(bad)  # type: ignore[arg-type]

    def test_batch_finish_rejects_crt_value_above_bound(self):
        # CRT(3 mod 5, 2 mod 7) = 23. The modulus product is already 35,
        # but a rigorous path bound of 20 makes 23 an impossible exact result.
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            residues = {5: {"residue": 3}, 7: {"residue": 2}}
            with self.assertRaisesRegex(ValueError, "violates rigorous path bound"):
                finish(work, 1, 20, [5, 7], residues)
            self.assertFalse((work / "exact.txt").exists())

    def test_workdir_lock_rejects_second_solver(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = acquire_workdir_lock(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    acquire_workdir_lock(root)
            finally:
                first.close()

    def test_solver_identity_rejects_uncertified_row7_exact_table(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            source = root / "solver.cu"
            compiler = root / "fake-nvcc"
            table = root / "row7_exact_compact_u128.bin"
            binary.write_bytes(b"solver-v1")
            source.write_text("int x;\n")
            compiler.write_text("#!/bin/sh\necho 'fake nvcc 1.0'\n")
            compiler.chmod(0o755)
            table.write_bytes(b"uncertified-row7-table")
            create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-DTARGET_W=28"],
                auxiliary_dependencies=[("row7-exact-compact", table)],
            )
            with self.assertRaisesRegex(ValueError, "not admitted for exact CRT results"):
                solver_identity(binary, root, expected_compile_args=["-DTARGET_W=28"])

    def test_solver_identity_requires_row8_admission_certificate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            source = root / "solver.cu"
            compiler = root / "fake-nvcc"
            cache = root / "row8_structural_int_v1.bin"
            cert = root / "admission.json"
            binary.write_bytes(b"solver-v1")
            source.write_text("int x;\n")
            compiler.write_text("#!/bin/sh\necho 'fake nvcc 1.0'\n")
            compiler.chmod(0o755)
            cache.write_bytes(b"structural-cache")
            cert.write_text('{"schema":"oneesan-row8-gridfp-structural-v2"}\n')
            create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-DTARGET_W=10"],
                auxiliary_dependencies=[("row8-structural-int-v1", cache)],
            )
            with self.assertRaisesRegex(ValueError, "requires a verified width-specific"):
                solver_identity(binary, root, expected_compile_args=["-DTARGET_W=10"])
            old_cert = root / "old-admission.json"
            old_cert.write_text('{"schema":"oneesan-row8-gridfp-structural-v1"}\n')
            with self.assertRaisesRegex(ValueError, "requires a verified width-specific"):
                solver_identity(
                    binary, root, expected_compile_args=["-DTARGET_W=10"],
                    admission_certificate=admission_certificate_identity(old_cert),
                )
            admission = admission_certificate_identity(cert)
            identity = solver_identity(
                binary, root, expected_compile_args=["-DTARGET_W=10"],
                admission_certificate=admission,
            )
            self.assertEqual(identity["binary_name"], "solver")

    def test_solver_identity_without_build_provenance_does_not_claim_runtime_head(self):
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            self.assertEqual(identity["git_commit"], "unknown")

    def test_solver_identity_uses_verified_build_commit_from_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            source = root / "solver.cu"
            binary.write_bytes(b"solver")
            source.write_text("int main() {}\n")
            sidecar = create_provenance(
                root=root, binary=binary, source=source, compiler=Path(sys.executable),
                compile_args=["-DTARGET_W=28"],
            )
            data = json.loads(sidecar.read_text())
            data["git_commit"] = "build-commit-deadbeef"
            data[PROVENANCE_CHECKSUM_FIELD] = provenance_checksum(data)
            sidecar.write_text(json.dumps(data))
            identity = solver_identity(binary, ROOT)
            self.assertEqual(identity["git_commit"], "build-commit-deadbeef")

    def test_solver_identity_rejects_malformed_present_sidecar(self):
        with tempfile.TemporaryDirectory() as td:
            binary = Path(td) / "solver"
            binary.write_bytes(b"solver")
            Path(str(binary) + ".provenance.json").write_text("{}")
            with self.assertRaises(ValueError):
                solver_identity(binary, ROOT)

    def test_checkpoint_is_bound_to_binary_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver-v1")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(
                checkpoint, n=9, identity=identity,
                residues={17: {"residue": 3}},
            )
            self.assertEqual(
                load_checkpoint(checkpoint, n=9, identity=identity)[17]["residue"], 3
            )
            binary.write_bytes(b"solver-v2")
            changed = solver_identity(binary, ROOT)
            with self.assertRaisesRegex(ValueError, "solver binary changed"):
                load_checkpoint(checkpoint, n=9, identity=changed)

    def test_checkpoint_is_bound_to_admission_certificate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver-v1")
            identity = solver_identity(binary, ROOT)
            cert = root / "admission.json"
            cert.write_text('{"schema":"test-admission-v1","value":1}\n')
            admission1 = admission_certificate_identity(cert)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(
                checkpoint, n=9, identity=identity, residues={17: {"residue": 3}},
                admission_certificate=admission1,
            )
            self.assertEqual(
                load_checkpoint(
                    checkpoint, n=9, identity=identity, admission_certificate=admission1
                )[17]["residue"],
                3,
            )
            cert.write_text('{"schema":"test-admission-v1","value":2}\n')
            admission2 = admission_certificate_identity(cert)
            with self.assertRaisesRegex(ValueError, "admission certificate changed"):
                load_checkpoint(
                    checkpoint, n=9, identity=identity, admission_certificate=admission2
                )
            with self.assertRaisesRegex(ValueError, "admission certificate changed"):
                load_checkpoint(checkpoint, n=9, identity=identity)


    def test_legacy_checkpoint_without_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text(json.dumps({"n": 9, "residues": {"17": {"residue": 3}}}))
            with self.assertRaisesRegex(ValueError, "no compatible solver fingerprint/integrity format"):
                load_checkpoint(checkpoint, n=9, identity=identity)

    def test_valid_range_checkpoint_tamper_is_rejected_by_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=9, identity=identity, residues={17: {"residue": 3}})
            data = json.loads(checkpoint.read_text())
            data["residues"]["17"]["residue"] = 4  # still canonical, so range checks alone cannot catch it
            checkpoint.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "SHA-256 integrity checksum mismatch"):
                load_checkpoint(checkpoint, n=9, identity=identity)

    def test_semantic_residue_validation_runs_after_valid_checksum(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=9, identity=identity, residues={17: {"residue": 3}})
            data = json.loads(checkpoint.read_text())
            data["residues"]["17"]["residue"] = 17
            data["checkpoint_sha256"] = checkpoint_checksum(data)
            checkpoint.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "outside canonical range"):
                load_checkpoint(checkpoint, n=9, identity=identity)

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=9, identity=identity, residues={17: {"residue": 3}})
            text = checkpoint.read_text()
            text = text.replace('  "n": 9,\n', '  "n": 9,\n  "n": 9,\n', 1)
            checkpoint.write_text(text)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_checkpoint(checkpoint, n=9, identity=identity)

    def test_truncated_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=9, identity=identity, residues={17: {"residue": 3}})
            checkpoint.write_text(checkpoint.read_text()[:-13])
            with self.assertRaisesRegex(ValueError, "malformed exact checkpoint"):
                load_checkpoint(checkpoint, n=9, identity=identity)

    def test_checkpoint_rejects_json_type_coercions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=9, identity=identity, residues={17: {"residue": 3}})
            for bad in ("3", True):
                data = json.loads(checkpoint.read_text())
                data["residues"]["17"]["residue"] = bad
                data["checkpoint_sha256"] = checkpoint_checksum(data)
                checkpoint.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "residue must be a JSON integer"):
                    load_checkpoint(checkpoint, n=9, identity=identity)

    def test_malformed_format_version_type_is_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver")
            identity = solver_identity(binary, ROOT)
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text(json.dumps({
                "format_version": {"bad": 3},
                "n": 9,
                "solver": identity,
                "residues": {},
                "checkpoint_sha256": "0" * 64,
            }))
            with self.assertRaisesRegex(ValueError, "format_version and n must be JSON integers"):
                load_checkpoint(checkpoint, n=9, identity=identity)

    def test_reconstruction_must_respect_bound_and_congruences(self):
        validate_exact_reconstruction(23, 35, 24, [(5, 3), (7, 2)])
        with self.assertRaisesRegex(ValueError, "violates rigorous path bound"):
            validate_exact_reconstruction(23, 35, 20, [(5, 3), (7, 2)])
        with self.assertRaisesRegex(ValueError, "final congruence check"):
            validate_exact_reconstruction(23, 35, 24, [(5, 3), (7, 1)])


if __name__ == "__main__":
    unittest.main()
