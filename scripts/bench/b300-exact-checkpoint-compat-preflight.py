#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOD_PATH = ROOT / "scripts" / "solve" / "solve_b300_exact_batch.py"
sys.path.insert(0, str(MOD_PATH.parent))
spec = importlib.util.spec_from_file_location("solve_b300_exact_batch", MOD_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checkpoint(path: Path, fp: dict, residue: int = 7) -> None:
    path.write_text(
        json.dumps(
            {
                "n": 27,
                "solver_fingerprint": fp,
                "residues": {"4294967291": {"residue": residue, "wall_s": 1.25}},
            },
            sort_keys=True,
        )
        + "\n"
    )


def expect_fail(fn, needle: str) -> None:
    try:
        fn()
    except SystemExit as e:
        if needle not in str(e):
            raise AssertionError(f"wrong failure: {e!r}") from e
    else:
        raise AssertionError("expected SystemExit")


with tempfile.TemporaryDirectory() as td:
    t = Path(td)
    binary = t / "solver.bin"
    binary.write_bytes(b"b300-checkpoint-compat\x00")
    bsha = sha(binary)

    # Legacy/direct exact-run schema 2 remains accepted without mutation.
    cp2 = t / "schema2.json"
    fp2 = {"schema": 2, "binary_sha256": bsha}
    write_checkpoint(cp2, fp2)
    active2 = dict(fp2)
    residues2 = mod.load_checkpoint(cp2, 27, active2)
    assert active2 == fp2
    assert residues2[4294967291]["residue"] == 7

    # The single-pass race uses schema 3 and pins profile SHA in addition to the
    # selected binary.  Exact continuation must adopt and preserve it.
    cp3 = t / "schema3.json"
    profile_sha = "a" * 64
    fp3 = {"schema": 3, "binary_sha256": bsha, "profile_sha256": profile_sha}
    write_checkpoint(cp3, fp3)
    active3 = dict(fp2)
    residues3 = mod.load_checkpoint(cp3, 27, active3)
    assert active3 == fp3
    assert residues3[4294967291]["residue"] == 7
    mod.save_checkpoint(cp3, 27, active3, residues3)
    saved = json.loads(cp3.read_text())
    assert saved["solver_fingerprint"] == fp3, "schema3 fingerprint was downgraded"

    # A different solver binary must never inherit a race checkpoint.
    cp_bad_bin = t / "bad-bin.json"
    bad_bin_fp = {"schema": 3, "binary_sha256": "b" * 64, "profile_sha256": profile_sha}
    write_checkpoint(cp_bad_bin, bad_bin_fp)
    expect_fail(lambda: mod.load_checkpoint(cp_bad_bin, 27, dict(fp2)), "no compatible solver fingerprint")

    # Schema 3 is only accepted when the profile digest is structurally valid.
    cp_bad_profile = t / "bad-profile.json"
    bad_profile_fp = {"schema": 3, "binary_sha256": bsha, "profile_sha256": "short"}
    write_checkpoint(cp_bad_profile, bad_profile_fp)
    expect_fail(lambda: mod.load_checkpoint(cp_bad_profile, 27, dict(fp2)), "no compatible solver fingerprint")

    # n remains part of checkpoint identity.
    cp_bad_n = t / "bad-n.json"
    write_checkpoint(cp_bad_n, fp3)
    d = json.loads(cp_bad_n.read_text())
    d["n"] = 26
    cp_bad_n.write_text(json.dumps(d) + "\n")
    expect_fail(lambda: mod.load_checkpoint(cp_bad_n, 27, dict(fp2)), "belongs to n=26")

print("b300-exact-checkpoint-compat-preflight OK schema2=1 race_schema3=1 schema3_preserved=1 binary_mismatch_rejected=1 malformed_profile_rejected=1 n_mismatch_rejected=1 gpu_work=0")
