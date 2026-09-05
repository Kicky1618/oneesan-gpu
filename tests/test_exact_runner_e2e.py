#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))
from build_provenance import create_provenance
RUNNER = ROOT / "scripts" / "solve" / "solve_b300_exact_batch.py"
SINGLE_RUNNER = ROOT / "scripts" / "solve" / "solve_b300_exact.py"
VERIFIER = ROOT / "scripts" / "tools" / "verify_exact_result.py"


class ExactRunnerE2ETest(unittest.TestCase):
    def test_fake_single_gpu_runner_to_verified_manifest_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "fake_single_solver.py"
            solver.write_text(
                """#!/usr/bin/env python3
import os
import sys
from pathlib import Path
KNOWN = {4: 8512}
n = int(sys.argv[1])
p = int(sys.argv[5])
count_file = Path(os.environ["FAKE_COUNT_FILE"])
count = int(count_file.read_text()) if count_file.exists() else 0
count_file.write_text(str(count + 1))
print(f"n={n} residue={KNOWN[n] % p} modulus={p} wall_s=0.001")
"""
            )
            solver.chmod(0o755)
            work = root / "work"
            count_file = root / "invocations.txt"
            env = os.environ.copy()
            env["FAKE_COUNT_FILE"] = str(count_file)
            cmd = [
                sys.executable, str(SINGLE_RUNNER), "4",
                "--binary", str(solver), "--work-dir", str(work),
                "--target-mib", "1", "--max-window", "1", "--gpus", "1",
            ]
            first = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            self.assertIn("exact=8512", first.stdout)
            self.assertEqual(count_file.read_text(), "1")
            manifest = work / "exact_manifest.json"
            verified = subprocess.run(
                [sys.executable, str(VERIFIER), str(manifest), "--binary", str(solver)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr + verified.stdout)
            second = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            self.assertEqual(count_file.read_text(), "1")

    def test_fake_gpu_batch_to_verified_exact_manifest_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "fake_batch_solver.py"
            solver.write_text(
                """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

KNOWN = {4: 8512}
n = int(sys.argv[1])
mods = [int(x) for x in sys.argv[5:]]
count_file = os.environ.get("FAKE_COUNT_FILE")
if count_file:
    p = Path(count_file)
    count = int(p.read_text()) if p.exists() else 0
    p.write_text(str(count + 1))
for modulus in mods:
    residue = KNOWN[n] % modulus
    print(f"n={n} residue={residue} modulus={modulus} wall_s=0.001", flush=True)
"""
            )
            solver.chmod(0o755)
            work = root / "work"
            count_file = root / "invocations.txt"
            env = os.environ.copy()
            env["FAKE_COUNT_FILE"] = str(count_file)
            cmd = [
                sys.executable,
                str(RUNNER),
                "4",
                "--binary", str(solver),
                "--work-dir", str(work),
                "--target-mib", "1",
                "--max-window", "1",
                "--gpus", "1",
            ]

            first = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr + first.stdout)
            self.assertIn("exact=8512", first.stdout)
            self.assertEqual(count_file.read_text(), "1")
            self.assertTrue((work / "checkpoint.json").is_file())
            self.assertTrue((work / "exact.txt").is_file())
            manifest = work / "exact_manifest.json"
            self.assertTrue(manifest.is_file())

            verified = subprocess.run(
                [sys.executable, str(VERIFIER), str(manifest), "--binary", str(solver)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr + verified.stdout)
            self.assertIn("exact=8512", verified.stdout)
            self.assertIn("valid=1", verified.stdout)

            # A second invocation must finalize entirely from the verified checkpoint;
            # it must not run the fake GPU binary again.
            second = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            self.assertIn("exact=8512", second.stdout)
            self.assertEqual(count_file.read_text(), "1")

    def test_partial_batch_failure_checkpoints_only_completed_residues_then_resumes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "flaky_batch_solver.py"
            solver.write_text(
                """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

KNOWN = {6: 575780564}
n = int(sys.argv[1])
mods = [int(x) for x in sys.argv[5:]]
state = Path(os.environ["FAKE_STATE_FILE"])
invocation = int(state.read_text()) if state.exists() else 0
state.write_text(str(invocation + 1))
if invocation == 0:
    p = mods[0]
    print(f"n={n} residue={KNOWN[n] % p} modulus={p} wall_s=0.001", flush=True)
    raise SystemExit(5)
for p in mods:
    print(f"n={n} residue={KNOWN[n] % p} modulus={p} wall_s=0.001", flush=True)
"""
            )
            solver.chmod(0o755)
            work = root / "work"
            state = root / "state.txt"
            env = os.environ.copy()
            env["FAKE_STATE_FILE"] = str(state)
            cmd = [
                sys.executable, str(RUNNER), "6",
                "--binary", str(solver),
                "--work-dir", str(work),
                "--target-mib", "1", "--max-window", "1", "--gpus", "1",
            ]

            first = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertNotEqual(first.returncode, 0)
            self.assertIn("completed this invocation=1/2", first.stderr + first.stdout)
            self.assertTrue((work / "checkpoint.json").is_file())
            self.assertFalse((work / "exact.txt").exists())
            self.assertFalse((work / "exact_manifest.json").exists())

            second = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr + second.stdout)
            self.assertIn("exact=575780564", second.stdout)
            self.assertEqual(state.read_text(), "2")
            manifest = work / "exact_manifest.json"
            verified = subprocess.run(
                [sys.executable, str(VERIFIER), str(manifest), "--binary", str(solver)],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr + verified.stdout)
            self.assertIn("valid=1", verified.stdout)

    def test_batch_duplicate_modulus_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "duplicate_batch_solver.py"
            solver.write_text(
                """#!/usr/bin/env python3
import sys
KNOWN = 575780564
n = int(sys.argv[1])
mods = [int(x) for x in sys.argv[5:]]
p = mods[0]
print(f"n={n} residue={KNOWN % p} modulus={p} wall_s=0.001", flush=True)
print(f"n={n} residue={(KNOWN + 1) % p} modulus={p} wall_s=0.002", flush=True)
"""
            )
            solver.chmod(0o755)
            work = root / "work"
            cmd = [sys.executable, str(RUNNER), "6", "--binary", str(solver),
                   "--work-dir", str(work), "--target-mib", "1",
                   "--max-window", "1", "--gpus", "1"]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("duplicate modulus", proc.stderr + proc.stdout)
            self.assertFalse((work / "exact.txt").exists())

    def test_batch_nonfinite_wall_time_is_rejected_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "bad_wall_batch_solver.py"
            solver.write_text(
                """#!/usr/bin/env python3
import sys
n = int(sys.argv[1])
p = int(sys.argv[5])
print(f"n={n} residue=0 modulus={p} wall_s=1e999", flush=True)
"""
            )
            solver.chmod(0o755)
            work = root / "work"
            cmd = [sys.executable, str(RUNNER), "6", "--binary", str(solver),
                   "--work-dir", str(work), "--target-mib", "1",
                   "--max-window", "1", "--gpus", "1"]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            output = proc.stderr + proc.stdout
            self.assertIn("wall_s must be finite and nonnegative", output)
            self.assertNotIn("Traceback", output)
            self.assertFalse((work / "checkpoint.json").exists())

    def test_batch_wrong_n_is_rejected_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "wrong_n_batch_solver.py"
            solver.write_text(
                """#!/usr/bin/env python3
import sys
n = int(sys.argv[1])
p = int(sys.argv[5])
print(f"n={n + 1} residue=0 modulus={p} wall_s=0.001", flush=True)
"""
            )
            solver.chmod(0o755)
            work = root / "work"
            cmd = [sys.executable, str(RUNNER), "6", "--binary", str(solver),
                   "--work-dir", str(work), "--target-mib", "1",
                   "--max-window", "1", "--gpus", "1"]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("returned n=7, expected n=6", proc.stderr + proc.stdout)
            self.assertFalse((work / "checkpoint.json").exists())

    def test_wrong_target_width_sidecar_is_rejected_before_solver_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "wrong_width_solver.py"
            marker = root / "invoked.txt"
            solver.write_text(
                """#!/usr/bin/env python3
import os
from pathlib import Path
Path(os.environ["FAKE_INVOKED_FILE"]).write_text("1")
raise SystemExit(99)
"""
            )
            solver.chmod(0o755)
            create_provenance(
                root=root, binary=solver, source=solver, compiler=Path(sys.executable),
                compile_args=["-DTARGET_W=999"],
            )
            work = root / "work"
            env = os.environ.copy()
            env["FAKE_INVOKED_FILE"] = str(marker)
            cmd = [sys.executable, str(RUNNER), "6", "--binary", str(solver),
                   "--work-dir", str(work), "--target-mib", "1",
                   "--max-window", "1", "--gpus", "1"]
            proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            output = proc.stderr + proc.stdout
            self.assertIn("missing required compile args", output)
            self.assertIn("-DTARGET_W=7", output)
            self.assertNotIn("Traceback", output)
            self.assertFalse(marker.exists(), "solver must not run before TARGET_W provenance validation")
            self.assertFalse((work / "checkpoint.json").exists())


    def test_malformed_build_sidecar_fails_cleanly_before_solver_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            solver = root / "fake_solver.py"
            solver.write_text("#!/usr/bin/env python3\nraise SystemExit(99)\n")
            solver.chmod(0o755)
            Path(str(solver) + ".provenance.json").write_text("{}")
            work = root / "work"
            cmd = [sys.executable, str(RUNNER), "6", "--binary", str(solver),
                   "--work-dir", str(work), "--target-mib", "1",
                   "--max-window", "1", "--gpus", "1"]
            proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            output = proc.stderr + proc.stdout
            self.assertIn("unsupported build provenance format", output)
            self.assertNotIn("Traceback", output)
            self.assertFalse((work / "checkpoint.json").exists())


if __name__ == "__main__":
    unittest.main()
