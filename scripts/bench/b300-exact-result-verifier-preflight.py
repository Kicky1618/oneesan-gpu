#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLVE_DIR = ROOT / "scripts" / "solve"
sys.path.insert(0, str(SOLVE_DIR))
from solve_b300_exact import crt_pair, primes_for_bound, simple_path_upper_bound

VERIFIER = SOLVE_DIR / "verify_b300_exact_result.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args: list[str], ok: bool, needle: str) -> subprocess.CompletedProcess[str]:
    p = subprocess.run([sys.executable, str(VERIFIER), *args], text=True, capture_output=True)
    text = p.stdout + p.stderr
    if ok and p.returncode != 0:
        raise AssertionError(f"verifier failed rc={p.returncode}: {text}")
    if not ok and p.returncode == 0:
        raise AssertionError(f"verifier unexpectedly succeeded: {text}")
    if needle not in text:
        raise AssertionError(f"missing marker {needle!r}: {text}")
    return p


with tempfile.TemporaryDirectory() as td:
    t = Path(td)
    binary = t / "solver.bin"
    binary.write_bytes(b"synthetic-b300-verifier\x00")
    bsha = sha(binary)
    psha = "a" * 64

    n = 1
    bound, _ = simple_path_upper_bound(n)
    prefix = primes_for_bound(bound)
    assert len(prefix) == 1
    prime = prefix[0]
    residue = 1
    assert residue <= bound < prime
    x, modulus = crt_pair(0, 1, residue, prime)
    assert x == residue and modulus > bound

    checkpoint = t / "checkpoint.json"
    checkpoint.write_text(json.dumps({
        "n": n,
        "solver_fingerprint": {
            "schema": 3,
            "binary_sha256": bsha,
            "profile_sha256": psha,
        },
        "residues": {
            str(prime): {"residue": residue, "wall_s": 1.25},
        },
    }, indent=2, sort_keys=True) + "\n")
    csha = sha(checkpoint)

    exact = t / "exact.txt"
    exact.write_text(
        f"n={n}\n"
        f"exact={x}\n"
        f"bound_bits={bound.bit_length()}\n"
        f"modulus_bits={modulus.bit_length()}\n"
        "primes_used=1\n"
        "solver_wall_s_sum=1.250000000\n"
        "checkpoint_schema=3\n"
        f"checkpoint_sha256={csha}\n"
        f"solver_binary_sha256={bsha}\n"
        f"solver_profile_sha256={psha}\n"
    )
    cert = t / "exact.verify.json"
    common = [
        str(n),
        "--checkpoint", str(checkpoint),
        "--exact", str(exact),
        "--binary", str(binary),
        "--profile-sha256", psha,
        "--certificate", str(cert),
    ]
    run(common, True, "B300_EXACT_VERIFY_OK")
    d = json.loads(cert.read_text())
    assert d["verified"] is True and d["exact"] == x and d["primes_used"] == 1
    assert d["checkpoint_sha256"] == csha
    assert d["exact_txt_sha256"] == sha(exact)
    assert d["solver_binary_sha256"] == bsha
    assert d["solver_profile_sha256"] == psha
    assert d["residues_used"] == [{"modulus": prime, "residue": residue, "wall_s": 1.25}]

    # exact.txt content tampering is detected independently of the checkpoint.
    exact_good = exact.read_text()
    exact.write_text(exact_good.replace("exact=1\n", "exact=2\n"))
    run(common, False, "exact.txt mismatch exact")
    exact.write_text(exact_good)

    # Checkpoint tampering changes the digest bound into exact.txt before any
    # certificate can be accepted.
    checkpoint_good = checkpoint.read_text()
    obj = json.loads(checkpoint_good)
    obj["residues"][str(prime)]["wall_s"] = 2.0
    checkpoint.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    run(common, False, "exact.txt mismatch solver_wall_s_sum")
    checkpoint.write_text(checkpoint_good)

    # Binary identity is independently checked against the checkpoint.
    binary.write_bytes(binary.read_bytes() + b"tamper")
    run(common, False, "binary SHA mismatch")

print("b300-exact-result-verifier-preflight OK synthetic_complete_crt=1 certificate=1 exact_tamper_rejected=1 checkpoint_tamper_rejected=1 binary_tamper_rejected=1 gpu_work=0")
