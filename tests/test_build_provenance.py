#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))

from build_provenance import FORMAT_V1, checksum, create_provenance, load_provenance


class BuildProvenanceTest(unittest.TestCase):
    def make_tree(self, root: Path):
        header = root / "dep.hpp"
        source = root / "solver.cu"
        binary = root / "solver.bin"
        compiler = root / "fake-nvcc"
        header.write_text("#pragma once\n#define ANSWER 42\n")
        source.write_text('#include "dep.hpp"\nint x = ANSWER;\n')
        binary.write_bytes(b"compiled-solver")
        compiler.write_text("#!/bin/sh\necho 'fake nvcc release 99.1'\n")
        compiler.chmod(0o755)
        return source, header, binary, compiler

    def test_round_trip_and_source_closure_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, _, binary, compiler = self.make_tree(root)
            provenance = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-O3", "-arch=sm_100"],
            )
            data = load_provenance(
                provenance, binary=binary, root=root, verify_sources=True,
            )
            self.assertEqual(data["binary_file"], binary.name)
            self.assertEqual([x["path"] for x in data["dependencies"]], ["dep.hpp", "solver.cu"])

    def test_source_change_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, header, binary, compiler = self.make_tree(root)
            provenance = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-O3"],
            )
            header.write_text("#pragma once\n#define ANSWER 43\n")
            # Runtime identity is the binary, so a changed checkout does not make
            # the old binary/provenance pair invalid for checkpoint resume.
            load_provenance(provenance, binary=binary, verify_sources=False)
            with self.assertRaisesRegex(ValueError, "source hash/size mismatch"):
                load_provenance(provenance, binary=binary, root=root, verify_sources=True)


    def test_auxiliary_dependencies_are_recorded_and_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, _, binary, compiler = self.make_tree(root)
            generator = root / "generator.py"
            certificate = root / "certificate.bin"
            generator.write_text("print('generate')\n")
            certificate.write_bytes(b"exact-certificate")
            provenance = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-O3"],
                auxiliary_dependencies=[
                    ("generated-table-generator", generator),
                    ("exact-rational-certificate", certificate),
                ],
            )
            data = load_provenance(provenance, binary=binary, root=root, verify_sources=True)
            self.assertEqual(data["format"], "ONEESAN_BUILD_PROVENANCE_V2")
            self.assertEqual(
                [(x["role"], x["path"]) for x in data["auxiliary_dependencies"]],
                [
                    ("exact-rational-certificate", "certificate.bin"),
                    ("generated-table-generator", "generator.py"),
                ],
            )
            certificate.write_bytes(b"changed-certificate")
            load_provenance(provenance, binary=binary, verify_sources=False)
            with self.assertRaisesRegex(ValueError, "auxiliary source hash/size mismatch"):
                load_provenance(provenance, binary=binary, root=root, verify_sources=True)

    def test_auxiliary_dependency_roles_must_be_unique_and_repo_local(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside_td:
            root = Path(td)
            source, _, binary, compiler = self.make_tree(root)
            a = root / "a.cert"
            b = root / "b.cert"
            outside = Path(outside_td) / "outside.cert"
            a.write_bytes(b"a")
            b.write_bytes(b"b")
            outside.write_bytes(b"outside")
            with self.assertRaisesRegex(ValueError, "roles must be unique"):
                create_provenance(
                    root=root, binary=binary, source=source, compiler=compiler,
                    compile_args=[], auxiliary_dependencies=[("certificate", a), ("certificate", b)],
                )
            with self.assertRaisesRegex(ValueError, "outside repository"):
                create_provenance(
                    root=root, binary=binary, source=source, compiler=compiler,
                    compile_args=[], auxiliary_dependencies=[("certificate", outside)],
                )

    def test_legacy_v1_provenance_remains_readable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, header, binary, compiler = self.make_tree(root)
            modern = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-O3"],
            )
            import json
            data = json.loads(modern.read_text())
            data["format"] = FORMAT_V1
            data.pop("auxiliary_dependencies")
            data["provenance_sha256"] = checksum(data)
            modern.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            loaded = load_provenance(modern, binary=binary, root=root, verify_sources=True)
            self.assertEqual(loaded["format"], FORMAT_V1)
            self.assertEqual([x["path"] for x in loaded["dependencies"]], [header.name, source.name])

    def test_binary_change_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, _, binary, compiler = self.make_tree(root)
            provenance = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=[],
            )
            binary.write_bytes(b"different-binary")
            with self.assertRaisesRegex(ValueError, "binary hash/size mismatch"):
                load_provenance(provenance, binary=binary)

    def test_provenance_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, _, binary, compiler = self.make_tree(root)
            provenance = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=[],
            )
            raw = provenance.read_text()
            provenance.write_text(raw.replace("-O", "-X", 1) if "-O" in raw else raw.replace("solver.cu", "solvfr.cu", 1))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_provenance(provenance, binary=binary)

    def test_required_compile_arg_is_enforced(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, _, binary, compiler = self.make_tree(root)
            provenance = create_provenance(
                root=root, binary=binary, source=source, compiler=compiler,
                compile_args=["-O3", "-DTARGET_W=28"],
            )
            load_provenance(provenance, binary=binary, expected_compile_args=["-DTARGET_W=28"])
            with self.assertRaisesRegex(ValueError, "missing required compile args"):
                load_provenance(provenance, binary=binary, expected_compile_args=["-DTARGET_W=27"])


if __name__ == "__main__":
    unittest.main()
