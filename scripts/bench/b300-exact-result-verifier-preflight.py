#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLVE_DIR = ROOT / "scripts" / "solve"
sys.path.insert(0, str(SOLVE_DIR))
from solve_b300_exact import (
    PRIMES as PROD_PRIMES,
    crt_pair as prod_crt_pair,
    primes_for_bound as prod_primes_for_bound,
    simple_path_upper_bound as prod_simple_path_upper_bound,
)
from verify_b300_exact_result import (
    VERIFIER_MATH_VERSION,
    crt_pair as verify_crt_pair,
    primes_for_bound as verify_primes_for_bound,
    simple_path_upper_bound as verify_simple_path_upper_bound,
)

VERIFIER = SOLVE_DIR / "verify_b300_exact_result.py"
WRAPPER = ROOT / "scripts" / "run" / "b300x8-grand-verify-exact.sh"


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


def run_wrapper(
    selected: Path,
    certificate: Path,
    shim_dir: Path,
    *,
    ok: bool,
    needle: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SELECTED_ENV"] = str(selected)
    env["CERTIFICATE"] = str(certificate)
    env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"
    p = subprocess.run(["bash", str(WRAPPER), "27"], text=True, capture_output=True, env=env)
    text = p.stdout + p.stderr
    if ok and p.returncode != 0:
        raise AssertionError(f"wrapper failed rc={p.returncode}: {text}")
    if not ok and p.returncode == 0:
        raise AssertionError(f"wrapper unexpectedly succeeded: {text}")
    if needle not in text:
        raise AssertionError(f"wrapper missing marker {needle!r}: {text}")
    return p


# The verifier must not import the production exact solver's CRT/bound code.
verifier_source = VERIFIER.read_text()
assert "solve_b300_exact" not in verifier_source
assert VERIFIER_MATH_VERSION == "independent-v1"

# Cross-check the independently implemented bound and prime-prefix logic against
# the production implementation on small complete instances. This compares
# results without sharing code inside the verifier itself.
for n_check in range(1, 10):
    prod_bound, prod_parts = prod_simple_path_upper_bound(n_check)
    verify_bound, verify_parts = verify_simple_path_upper_bound(n_check)
    assert (verify_bound, verify_parts) == (prod_bound, prod_parts), n_check
    assert verify_primes_for_bound(verify_bound) == prod_primes_for_bound(prod_bound), n_check

# Cross-check CRT accumulation using different modular-inverse implementations:
# production uses Python's modular inverse; verifier uses extended Euclid.
prod_x = verify_x = 0
prod_modulus = verify_modulus = 1
for i, prime in enumerate(PROD_PRIMES[:8], 1):
    residue = (0x9E3779B1 * i + 0x12345) % prime
    prod_x, prod_modulus = prod_crt_pair(prod_x, prod_modulus, residue, prime)
    verify_x, verify_modulus = verify_crt_pair(verify_x, verify_modulus, residue, prime)
    assert (verify_x, verify_modulus) == (prod_x, prod_modulus)

with tempfile.TemporaryDirectory() as td:
    t = Path(td)
    binary = t / "solver.bin"
    binary.write_bytes(b"synthetic-b300-verifier\x00")
    bsha = sha(binary)
    psha = "a" * 64

    # Build the fixture with production math, then require the independent
    # verifier to reconstruct and accept it.
    n = 1
    bound, _ = prod_simple_path_upper_bound(n)
    prefix = prod_primes_for_bound(bound)
    assert len(prefix) == 1
    prime = prefix[0]
    residue = 1
    assert residue <= bound < prime
    x, modulus = prod_crt_pair(0, 1, residue, prime)
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
    assert d["verifier_math"] == VERIFIER_MATH_VERSION
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

    # Exercise the canonical shell wrapper separately. The real Python verifier
    # above already proves CRT/certificate correctness; this shim only lets the
    # wrapper reach its post-verification certificate binding without needing an
    # n=27 exact fixture.
    wrapper_work = t / "wrapper-work"
    wrapper_work.mkdir()
    wrapper_binary = t / "wrapper-solver.bin"
    wrapper_binary.write_bytes(b"synthetic-wrapper-solver\x00")
    wrapper_binary.chmod(0o755)
    wrapper_profile = t / "wrapper-profile.env"
    wrapper_profile.write_text("BUCKET_THREADS=256\n")
    wrapper_checkpoint = t / "wrapper-checkpoint.json"
    wrapper_checkpoint.write_text("{}\n")
    wrapper_exact = wrapper_work / "exact.txt"
    wrapper_exact.write_text("synthetic-n27-exact\n")
    wrapper_bsha = sha(wrapper_binary)
    wrapper_psha = sha(wrapper_profile)

    shim_dir = t / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python3"
    shim.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == */scripts/solve/verify_b300_exact_result.py ]]; then
  shift
  [[ "${{1:-}}" == 27 ]] || exit 91
  shift
  checkpoint=''; exact=''; binary=''; psha=''; cert=''
  while (($#)); do
    case "$1" in
      --checkpoint) checkpoint="$2"; shift 2 ;;
      --exact) exact="$2"; shift 2 ;;
      --binary) binary="$2"; shift 2 ;;
      --profile-sha256) psha="$2"; shift 2 ;;
      --certificate) cert="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  [[ -n "$checkpoint" && -n "$exact" && -n "$binary" && -n "$psha" && -n "$cert" ]] || exit 92
  bsha="$(sha256sum "$binary" | awk '{{print $1}}')"
  csha="$(sha256sum "$checkpoint" | awk '{{print $1}}')"
  esha="$(sha256sum "$exact" | awk '{{print $1}}')"
  cat >"$cert" <<EOF
{{"schema":1,"verified":true,"n":27,"solver_binary_sha256":"$bsha","solver_profile_sha256":"$psha","checkpoint_sha256":"$csha","exact_txt_sha256":"$esha"}}
EOF
  echo 'B300_EXACT_VERIFY_OK synthetic_wrapper=1'
  exit 0
fi
exec {shlex.quote(sys.executable)} "$@"
""")
    shim.chmod(0o755)

    def selected_contract(schema: int, path: Path, *, shadow_controls: bool = False) -> None:
        lines = [
            f"B300_GRAND_SELECTED_SCHEMA={schema}",
            "B300_GRAND_SELECTED_VALIDATED=1",
            "B300_GRAND_SELECTED_N=27",
            f"B300_GRAND_SELECTED_PROFILE_FILE={shlex.quote(str(wrapper_profile))}",
            f"B300_GRAND_SELECTED_PROFILE_SHA256={wrapper_psha}",
            f"B300_GRAND_SELECTED_BINARY={shlex.quote(str(wrapper_binary))}",
            f"B300_GRAND_SELECTED_BINARY_SHA256={wrapper_bsha}",
            f"B300_GRAND_SELECTED_WORK_DIR={shlex.quote(str(wrapper_work))}",
            f"B300_GRAND_SELECTED_CHECKPOINT={shlex.quote(str(wrapper_checkpoint))}",
        ]
        if shadow_controls:
            lines += [
                "SELECTED_ENV=/tmp/oneesan-shadow-selected.env",
                "CERTIFICATE=/tmp/oneesan-shadow-certificate.json",
                "ONEESAN_ROOT=/tmp/oneesan-shadow-root",
            ]
        path.write_text("\n".join(lines) + "\n")

    selected1 = t / "wrapper-schema1.env"
    selected3 = t / "wrapper-schema3.env"
    selected4 = t / "wrapper-schema4.env"
    selected_contract(1, selected1)
    selected_contract(3, selected3, shadow_controls=True)
    selected_contract(4, selected4)

    cert1 = t / "wrapper-schema1.verify.json"
    cert3 = t / "wrapper-schema3.verify.json"
    run_wrapper(selected1, cert1, shim_dir, ok=True, needle="B300 GRAND VERIFY COMPLETE schema=1")
    run_wrapper(selected3, cert3, shim_dir, ok=True, needle="B300 GRAND VERIFY COMPLETE schema=3")
    assert cert1.is_file() and cert3.is_file()
    assert not Path("/tmp/oneesan-shadow-certificate.json").exists()
    run_wrapper(selected4, t / "wrapper-schema4.verify.json", shim_dir, ok=False, needle="unsupported grand selection schema=4")

print(
    "b300-exact-result-verifier-preflight OK "
    "independent_math=1 no_solver_math_import=1 cross_impl_bound=1 cross_impl_crt=1 "
    "synthetic_complete_crt=1 certificate=1 exact_tamper_rejected=1 "
    "checkpoint_tamper_rejected=1 binary_tamper_rejected=1 "
    "wrapper_schema1=1 wrapper_schema3=1 wrapper_schema4_rejected=1 "
    "wrapper_control_paths_pinned=1 gpu_work=0"
)
