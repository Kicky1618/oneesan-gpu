#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from build_provenance import checksum as build_provenance_checksum, create_provenance
from exact_safety import (
    admission_certificate_identity,
    result_manifest_checksum,
    save_checkpoint,
    solver_identity,
    write_exact_result,
)
from solve_b300_exact import PRIMES, simple_path_upper_bound
from verify_exact_result import verify


class ExactManifestTest(unittest.TestCase):
    def make_result(self, root: Path, *, with_provenance: bool = False):
        binary = root / "solver"
        binary.write_bytes(b"solver-v1")
        if with_provenance:
            header = root / "dep.hpp"
            source = root / "solver.cu"
            compiler = root / "fake-nvcc"
            header.write_text("#pragma once\n#define X 1\n")
            source.write_text('#include "dep.hpp"\nint x = X;\n')
            compiler.write_text("#!/bin/sh\necho 'fake nvcc 1.0'\n")
            compiler.chmod(0o755)
            create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-O3", "-DTARGET_W=2"],
            )
        identity = solver_identity(binary, ROOT)
        p = PRIMES[0]
        bound, partition = simple_path_upper_bound(1)
        self.assertEqual(bound, 2)
        residues = {p: {"residue": 2, "wall_s": 1.25}}
        checkpoint = root / "checkpoint.json"
        save_checkpoint(checkpoint, n=1, identity=identity, residues=residues)
        exact, manifest = write_exact_result(
            root,
            n=1,
            exact=2,
            path_bound=bound,
            modulus_product=p,
            used_moduli=[p],
            residues=residues,
            identity=identity,
            checkpoint_path=checkpoint,
            total_wall_s=1.25,
            strip_partition=partition,
            binary_path=binary,
        )
        return binary, checkpoint, exact, manifest

    def make_row8_admitted_result(self, root: Path):
        binary = root / "solver"
        binary.write_bytes(b"solver-row8-test")
        identity = solver_identity(binary, ROOT)
        n = 9
        bound, partition = simple_path_upper_bound(n)
        used = PRIMES[:3]
        exact_value = 0
        residues = {p: {"residue": 0, "wall_s": 0.25} for p in used}
        product = 1
        for p in used:
            product *= p
        self.assertGreater(product, bound)
        admission_path = ROOT / "formal/certificates/row8_gridfp_structural_w10.json"
        admission = admission_certificate_identity(admission_path)
        checkpoint = root / "checkpoint.json"
        save_checkpoint(
            checkpoint, n=n, identity=identity, residues=residues,
            admission_certificate=admission,
        )
        exact, manifest = write_exact_result(
            root, n=n, exact=exact_value, path_bound=bound, modulus_product=product,
            used_moduli=used, residues=residues, identity=identity,
            checkpoint_path=checkpoint, total_wall_s=0.75,
            strip_partition=partition, binary_path=binary,
            admission_certificate_path=admission_path,
            admission_certificate=admission,
        )
        return binary, checkpoint, exact, manifest

    def test_manifest_round_trip_verifies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, exact, manifest = self.make_result(root)
            data = verify(manifest, binary)
            self.assertEqual(data["exact_decimal"], "2")
            self.assertIn("manifest_file=exact_manifest.json", exact.read_text())

    def test_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, _, manifest = self.make_result(root)
            data = json.loads(manifest.read_text())
            data["exact_decimal"] = "1"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "manifest.*checksum mismatch"):
                verify(manifest, binary)

    def test_semantic_tamper_with_recomputed_manifest_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, _, manifest = self.make_result(root)
            data = json.loads(manifest.read_text())
            data["congruences"][0]["residue"] = 1
            data["manifest_sha256"] = result_manifest_checksum(data)
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "CRT exact mismatch"):
                verify(manifest, binary)

    def test_checkpoint_tamper_after_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, checkpoint, _, manifest = self.make_result(root)
            checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA-256 mismatch"):
                verify(manifest, binary)

    def test_exact_txt_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, exact, manifest = self.make_result(root)
            exact.write_text(exact.read_text().replace("exact=2", "exact=1"))
            with self.assertRaisesRegex(ValueError, "exact.txt SHA-256 mismatch"):
                verify(manifest, binary)

    def test_checkpoint_semantic_mismatch_is_rejected_even_with_rehashed_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, checkpoint, _, manifest = self.make_result(root)
            from exact_safety import attach_checkpoint_checksum, binary_sha256
            cp = json.loads(checkpoint.read_text())
            cp["residues"][str(PRIMES[0])]["wall_s"] = 2.5
            cp.pop("checkpoint_sha256")
            cp = attach_checkpoint_checksum(cp)
            checkpoint.write_text(json.dumps(cp, sort_keys=True) + "\n")
            data = json.loads(manifest.read_text())
            data["checkpoint_sha256"] = binary_sha256(checkpoint)
            data["manifest_sha256"] = result_manifest_checksum(data)
            manifest.write_text(json.dumps(data, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "checkpoint/manifest wall_s mismatch"):
                verify(manifest, binary)

    def test_manifest_embeds_build_provenance_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, _, manifest = self.make_result(root, with_provenance=True)
            data = verify(manifest, binary)
            self.assertEqual(data["build_provenance_file"], "solver_build_provenance.json")
            self.assertTrue((root / "solver_build_provenance.json").is_file())

    def test_verifier_rejects_historical_manifest_with_uncertified_row7_table(self):
        from exact_safety import binary_sha256
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, _, manifest = self.make_result(root, with_provenance=True)
            provenance = root / "solver_build_provenance.json"
            pdata = json.loads(provenance.read_text())
            dep = root / "dep.hpp"
            pdata["auxiliary_dependencies"] = [{
                "role": "row7-exact-compact",
                "path": dep.relative_to(root).as_posix(),
                "sha256": binary_sha256(dep),
                "size": dep.stat().st_size,
            }]
            pdata["provenance_sha256"] = build_provenance_checksum(pdata)
            provenance.write_text(json.dumps(pdata, indent=2, sort_keys=True) + "\n")
            mdata = json.loads(manifest.read_text())
            mdata["build_provenance_sha256"] = binary_sha256(provenance)
            mdata["manifest_sha256"] = result_manifest_checksum(mdata)
            manifest.write_text(json.dumps(mdata, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "not admitted for exact CRT results"):
                verify(manifest, binary)

    def test_embedded_build_provenance_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, _, manifest = self.make_result(root, with_provenance=True)
            provenance = root / "solver_build_provenance.json"
            provenance.write_bytes(provenance.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "build provenance SHA-256 mismatch"):
                verify(manifest, binary)

    def test_verify_sources_requires_row6_generation_evidence_in_v2(self):
        build_dir = ROOT / "build"
        build_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_dir) as td:
            work = Path(td)
            binary = work / "solver"
            source = work / "solver.cu"
            compiler = work / "fake-nvcc"
            binary.write_bytes(b"solver-v1")
            source.write_text(
                '#include "../../src/cuda/b300/row6_automaton_crt20_generated.hpp"\nint x;\n'
            )
            compiler.write_text("#!/bin/sh\necho 'fake nvcc 1.0'\n")
            compiler.chmod(0o755)
            create_provenance(
                root=ROOT, binary=binary, source=source, compiler=compiler,
                compile_args=["-DTARGET_W=2"],
            )
            identity = solver_identity(binary, ROOT)
            p = PRIMES[0]
            bound, partition = simple_path_upper_bound(1)
            residues = {p: {"residue": 2, "wall_s": 1.25}}
            checkpoint = work / "checkpoint.json"
            save_checkpoint(checkpoint, n=1, identity=identity, residues=residues)
            _, manifest = write_exact_result(
                work, n=1, exact=2, path_bound=bound, modulus_product=p,
                used_moduli=[p], residues=residues, identity=identity,
                checkpoint_path=checkpoint, total_wall_s=1.25,
                strip_partition=partition, binary_path=binary,
            )
            with self.assertRaisesRegex(ValueError, "missing canonical auxiliary generation evidence"):
                verify(manifest, binary, verify_sources=True)

    def test_verify_sources_checks_auxiliary_provenance_dependencies(self):
        build_dir = ROOT / "build"
        build_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build_dir) as td:
            work = Path(td)
            binary = work / "solver"
            source = work / "solver.cu"
            compiler = work / "fake-nvcc"
            certificate = work / "certificate.bin"
            binary.write_bytes(b"solver-v1")
            source.write_text("int x;\n")
            compiler.write_text("#!/bin/sh\necho 'fake nvcc 1.0'\n")
            compiler.chmod(0o755)
            certificate.write_bytes(b"certificate-v1")
            create_provenance(
                root=ROOT, binary=binary, source=source, compiler=compiler,
                compile_args=["-DTARGET_W=2"],
                auxiliary_dependencies=[("exact-rational-certificate", certificate)],
            )
            identity = solver_identity(binary, ROOT)
            p = PRIMES[0]
            bound, partition = simple_path_upper_bound(1)
            residues = {p: {"residue": 2, "wall_s": 1.25}}
            checkpoint = work / "checkpoint.json"
            save_checkpoint(checkpoint, n=1, identity=identity, residues=residues)
            _, manifest = write_exact_result(
                work, n=1, exact=2, path_bound=bound, modulus_product=p,
                used_moduli=[p], residues=residues, identity=identity,
                checkpoint_path=checkpoint, total_wall_s=1.25,
                strip_partition=partition, binary_path=binary,
            )
            verify(manifest, binary, verify_sources=True)
            certificate.write_bytes(b"certificate-v2")
            with self.assertRaisesRegex(ValueError, "auxiliary source hash/size mismatch"):
                verify(manifest, binary, verify_sources=True)

    def test_finalization_rejects_uncertified_row7_exact_table(self):
        from exact_safety import binary_sha256
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
                compile_args=["-DTARGET_W=2"],
                auxiliary_dependencies=[("row7-exact-compact", table)],
            )
            identity = {
                "binary_name": binary.name,
                "binary_sha256": binary_sha256(binary),
                "git_commit": "unknown",
            }
            p = PRIMES[0]
            bound, partition = simple_path_upper_bound(1)
            residues = {p: {"residue": 2, "wall_s": 1.25}}
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=1, identity=identity, residues=residues)
            with self.assertRaisesRegex(ValueError, "not admitted for exact CRT results"):
                write_exact_result(
                    root, n=1, exact=2, path_bound=bound, modulus_product=p,
                    used_moduli=[p], residues=residues, identity=identity,
                    checkpoint_path=checkpoint, total_wall_s=1.25,
                    strip_partition=partition, binary_path=binary,
                )
            self.assertFalse((root / "exact.txt").exists())
            self.assertFalse((root / "exact_manifest.json").exists())

    def test_finalization_rejects_build_provenance_for_wrong_target_width(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "solver"
            binary.write_bytes(b"solver-v1")
            source = root / "solver.cu"
            compiler = root / "fake-nvcc"
            source.write_text("int x;\n")
            compiler.write_text("#!/bin/sh\necho 'fake nvcc 1.0'\n")
            compiler.chmod(0o755)
            create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-DTARGET_W=28"],
            )
            identity = solver_identity(binary, ROOT)
            p = PRIMES[0]
            bound, partition = simple_path_upper_bound(1)
            residues = {p: {"residue": 2, "wall_s": 1.25}}
            checkpoint = root / "checkpoint.json"
            save_checkpoint(checkpoint, n=1, identity=identity, residues=residues)
            with self.assertRaisesRegex(ValueError, "missing required compile args"):
                write_exact_result(
                    root, n=1, exact=2, path_bound=bound, modulus_product=p,
                    used_moduli=[p], residues=residues, identity=identity,
                    checkpoint_path=checkpoint, total_wall_s=1.25,
                    strip_partition=partition, binary_path=binary,
                )

    def test_row8_admission_certificate_is_embedded_and_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, checkpoint, _, manifest = self.make_row8_admitted_result(root)
            data = verify(manifest, binary)
            self.assertEqual(
                data["admission_certificate_schema"],
                "oneesan-row8-gridfp-structural-v2",
            )
            self.assertEqual(data["admission_certificate_file"], "admission_certificate.json")
            cp = json.loads(checkpoint.read_text())
            self.assertEqual(
                cp["admission_certificate"]["sha256"],
                data["admission_certificate_sha256"],
            )

    def test_row8_admission_certificate_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary, _, _, manifest = self.make_row8_admitted_result(root)
            admission_copy = root / "admission_certificate.json"
            admission_copy.write_bytes(admission_copy.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "admission certificate SHA-256 mismatch"):
                verify(manifest, binary)



if __name__ == "__main__":
    unittest.main()
