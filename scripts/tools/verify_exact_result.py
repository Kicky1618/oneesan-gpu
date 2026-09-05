#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))

from build_provenance import FORMAT_V1, FORMAT_V2, load_provenance  # noqa: E402
from exact_safety import (  # noqa: E402
    RESULT_CHECKSUM_FIELD,
    admission_certificate_identity,
    binary_sha256,
    load_checkpoint,
    load_exact_result_manifest,
    validate_exact_provenance_admissible,
)


def _columns_compatible(x: int, y: int, height: int) -> bool:
    if height <= 1:
        return True
    mask = (1 << (height - 1)) - 1
    vertical_x = (x ^ (x >> 1)) & mask
    vertical_y = (y ^ (y >> 1)) & mask
    top_diff = (x ^ y) & mask
    return (vertical_x & vertical_y & top_diff) == 0


def _checkerboard_strip_count(height: int, width: int) -> int:
    states = 1 << height
    dp = [1] * states
    for _ in range(1, width):
        ndp = [0] * states
        for x, value in enumerate(dp):
            if not value:
                continue
            for y in range(states):
                if _columns_compatible(x, y, height):
                    ndp[y] += value
        dp = ndp
    return sum(dp)


def independent_path_upper_bound(n: int, max_strip_height: int = 9) -> tuple[int, list[int]]:
    if n < 1:
        return 1, []
    hmax = min(max_strip_height, n)
    counts = {h: _checkerboard_strip_count(h, n) for h in range(1, hmax + 1)}
    best: list[int | None] = [None] * (n + 1)
    parts: list[list[int] | None] = [None] * (n + 1)
    best[0], parts[0] = 1, []
    for rows in range(1, n + 1):
        for h, count in counts.items():
            if h > rows or best[rows - h] is None:
                continue
            candidate = best[rows - h] * count
            if best[rows] is None or candidate < best[rows]:
                best[rows] = candidate
                parts[rows] = parts[rows - h] + [h]
    assert best[n] is not None and parts[n] is not None
    return best[n], parts[n]


def _is_prime_u64(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        s += 1
        d //= 2
    # Deterministic Miller-Rabin bases for all unsigned 64-bit integers.
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _crt_pair_checked(x: int, m: int, residue: int, modulus: int) -> tuple[int, int]:
    if math.gcd(m, modulus) != 1:
        raise ValueError(f"CRT modulus {modulus} is not coprime to accumulated modulus")
    t = ((residue - x) % modulus) * pow(m, -1, modulus) % modulus
    return x + m * t, m * modulus


def decimal_field(data: dict, key: str) -> int:
    value = data.get(key)
    if not isinstance(value, str) or not value or not value.isdigit():
        raise ValueError(f"{key} must be a canonical nonnegative decimal string")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError(f"{key} is not canonical decimal: {value!r}")
    return parsed



ROW6_CRT20_HEADER = "src/cuda/b300/row6_automaton_crt20_generated.hpp"
ROW6_CRT20_AUXILIARY = {
    "row6-crt20-generator": "scripts/tools/gen_row6_crt20.py",
    "row6-crt20-verifier": "scripts/tools/verify_row6_crt20.py",
    "row6-path-bound-source": "scripts/solve/path_bound.py",
    "row6-rational-certificate": "formal/certificates/row6_rational_dump.txt.xz",
}


def _verify_generation_evidence(pdata: dict) -> None:
    dependency_paths = {rec["path"] for rec in pdata["dependencies"]}
    if ROW6_CRT20_HEADER not in dependency_paths:
        return
    if pdata["format"] == FORMAT_V1:
        # Historical V1 sidecars predate auxiliary-generation evidence. Their
        # direct source/header hashes remain verifiable, but no stronger claim
        # about how the generated row-6 header was derived is made.
        return
    if pdata["format"] != FORMAT_V2:
        raise ValueError(f"unsupported generated-source provenance format: {pdata['format']!r}")
    actual = {rec["role"]: rec["path"] for rec in pdata["auxiliary_dependencies"]}
    missing = {
        role: expected_path
        for role, expected_path in ROW6_CRT20_AUXILIARY.items()
        if actual.get(role) != expected_path
    }
    if missing:
        raise ValueError(f"row6 CRT20 provenance is missing canonical auxiliary generation evidence: {missing}")
    from verify_row6_crt20 import verify_all
    verify_all()

def verify(path: Path, binary: Path | None = None, *, verify_sources: bool = False) -> dict:
    data = load_exact_result_manifest(path)
    expected_fields = {
        "format", "n", "exact_decimal", "path_bound_decimal", "bound_bits",
        "bound_method", "strip_partition", "modulus_product_decimal", "modulus_bits",
        "primes_used", "congruences", "solver", "solver_wall_s_sum",
        "checkpoint_file", "checkpoint_sha256", "exact_file", "exact_sha256",
        "build_provenance_file", "build_provenance_sha256",
        "admission_certificate_file", "admission_certificate_sha256",
        "admission_certificate_schema", RESULT_CHECKSUM_FIELD,
    }
    if set(data) != expected_fields:
        raise ValueError(
            f"manifest field mismatch: missing={sorted(expected_fields-set(data))} "
            f"unknown={sorted(set(data)-expected_fields)}"
        )
    n = data["n"]
    if type(n) is not int or n < 1:
        raise ValueError("manifest n must be a positive JSON integer")
    if data["bound_method"] != "checkerboard_strip_v1":
        raise ValueError(f"unsupported bound method: {data['bound_method']!r}")

    exact = decimal_field(data, "exact_decimal")
    path_bound = decimal_field(data, "path_bound_decimal")
    modulus_product = decimal_field(data, "modulus_product_decimal")
    recomputed_bound, recomputed_partition = independent_path_upper_bound(n)
    if path_bound != recomputed_bound:
        raise ValueError(f"path bound mismatch: {path_bound} != recomputed {recomputed_bound}")
    if data["strip_partition"] != recomputed_partition:
        raise ValueError(
            f"strip partition mismatch: {data['strip_partition']} != {recomputed_partition}"
        )
    if data["bound_bits"] != path_bound.bit_length():
        raise ValueError("bound_bits mismatch")
    if data["modulus_bits"] != modulus_product.bit_length():
        raise ValueError("modulus_bits mismatch")

    congruences = data["congruences"]
    if not isinstance(congruences, list):
        raise ValueError("congruences must be a JSON array")
    if type(data["primes_used"]) is not int or data["primes_used"] != len(congruences):
        raise ValueError("primes_used mismatch")
    if not congruences:
        raise ValueError("at least one CRT congruence is required")

    x, m = 0, 1
    used_residues: list[tuple[int, int]] = []
    used_moduli: list[int] = []
    wall_sum = 0.0
    for idx, rec in enumerate(congruences, 1):
        if not isinstance(rec, dict):
            raise ValueError(f"congruence {idx}: record must be an object")
        allowed = {"modulus", "residue", "wall_s", "log"}
        if set(rec) - allowed or not {"modulus", "residue", "wall_s"} <= set(rec):
            raise ValueError(f"congruence {idx}: field mismatch")
        p, residue, wall = rec["modulus"], rec["residue"], rec["wall_s"]
        if type(p) is not int or p < 2 or p >= 1 << 64:
            raise ValueError(f"congruence {idx}: invalid modulus {p!r}")
        if not _is_prime_u64(p):
            raise ValueError(f"congruence {idx}: modulus {p} is not prime")
        if p in used_moduli:
            raise ValueError(f"congruence {idx}: duplicate modulus {p}")
        if type(residue) is not int or not 0 <= residue < p:
            raise ValueError(f"congruence {idx}: non-canonical residue {residue!r}")
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or wall < 0:
            raise ValueError(f"congruence {idx}: invalid wall_s")
        if "log" in rec and not isinstance(rec["log"], str):
            raise ValueError(f"congruence {idx}: log must be a string")
        x, m = _crt_pair_checked(x, m, residue, p)
        used_residues.append((p, residue))
        used_moduli.append(p)
        wall_sum += float(wall)

    if m <= path_bound:
        raise ValueError(f"CRT modulus product does not exceed path bound: {m} <= {path_bound}")
    if len(used_moduli) > 1 and m // used_moduli[-1] > path_bound:
        raise ValueError("CRT congruence list contains a redundant trailing modulus")

    if m != modulus_product:
        raise ValueError(f"modulus product mismatch: {modulus_product} != recomputed {m}")
    if x != exact:
        raise ValueError(f"CRT exact mismatch: manifest {exact} != recomputed {x}")
    if not 0 <= exact <= path_bound:
        raise ValueError(f"exact value violates rigorous path bound: {exact} > {path_bound}")
    for p, residue in used_residues:
        if exact % p != residue:
            raise ValueError(f"exact value fails congruence for modulus {p}")

    reported_wall = data["solver_wall_s_sum"]
    if isinstance(reported_wall, bool) or not isinstance(reported_wall, (int, float)) or not math.isfinite(float(reported_wall)) or reported_wall < 0:
        raise ValueError("invalid solver_wall_s_sum")
    if not math.isclose(float(reported_wall), wall_sum, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f"solver wall-time sum mismatch: {reported_wall} != {wall_sum}")

    solver = data["solver"]
    if not isinstance(solver, dict) or set(solver) != {"binary_name", "binary_sha256", "git_commit"}:
        raise ValueError("solver identity field mismatch")
    if not all(isinstance(v, str) for v in solver.values()):
        raise ValueError("solver identity values must be strings")
    solver_sha = solver["binary_sha256"]
    if len(solver_sha) != 64 or any(c not in "0123456789abcdef" for c in solver_sha):
        raise ValueError("invalid solver SHA-256")
    if binary is not None:
        actual_binary_sha = binary_sha256(binary)
        if actual_binary_sha != solver_sha:
            raise ValueError(
                f"solver binary SHA-256 mismatch: {actual_binary_sha} != {solver_sha}"
            )

    admission_name = data["admission_certificate_file"]
    admission_sha = data["admission_certificate_sha256"]
    admission_schema = data["admission_certificate_schema"]
    admission_identity = None
    if admission_name is None or admission_sha is None or admission_schema is None:
        if not (admission_name is None and admission_sha is None and admission_schema is None):
            raise ValueError(
                "admission certificate file/hash/schema must either all be null or all be present"
            )
    else:
        if not isinstance(admission_name, str) or Path(admission_name).name != admission_name:
            raise ValueError("admission_certificate_file must be a basename")
        if not isinstance(admission_sha, str) or len(admission_sha) != 64 or any(
            c not in "0123456789abcdef" for c in admission_sha
        ):
            raise ValueError("invalid admission certificate SHA-256")
        if not isinstance(admission_schema, str) or not admission_schema:
            raise ValueError("invalid admission certificate schema")
        admission_path = path.parent / admission_name
        if not admission_path.is_file():
            raise ValueError(
                f"admission certificate referenced by manifest is missing: {admission_path}"
            )
        actual_admission_sha = binary_sha256(admission_path)
        if actual_admission_sha != admission_sha:
            raise ValueError(
                f"admission certificate SHA-256 mismatch: {actual_admission_sha} != {admission_sha}"
            )
        admission_identity = admission_certificate_identity(admission_path)
        if admission_identity["schema"] != admission_schema:
            raise ValueError("admission certificate schema mismatch")
        if admission_schema == "oneesan-row8-gridfp-structural-v2":
            from row8_gridfp_structural_cert import verify as verify_row8_admission
            try:
                verify_row8_admission(admission_path)
            except SystemExit as exc:
                raise ValueError(f"row8 admission certificate verification failed: {exc}") from exc
            cert_data = json.loads(admission_path.read_text(encoding="utf-8"))
            if cert_data.get("n") != n or cert_data.get("width") != n + 1:
                raise ValueError(
                    f"row8 admission certificate target mismatch: "
                    f"n={cert_data.get('n')} width={cert_data.get('width')} expected n={n}"
                )
        else:
            raise ValueError(f"unsupported admission certificate schema: {admission_schema!r}")

    checkpoint_name = data["checkpoint_file"]
    checkpoint_sha = data["checkpoint_sha256"]
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_file must be a basename")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha):
        raise ValueError("invalid checkpoint SHA-256")
    checkpoint = path.parent / checkpoint_name
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint referenced by manifest is missing: {checkpoint}")
    actual_checkpoint_sha = binary_sha256(checkpoint)
    if actual_checkpoint_sha != checkpoint_sha:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {actual_checkpoint_sha} != {checkpoint_sha}"
        )

    # The checkpoint is evidence, not merely a blob whose hash is recorded.
    # Parse it with the same strict V3 validator and require every CRT record
    # used by the manifest to match its durable checkpoint record exactly.
    checkpoint_residues = load_checkpoint(
        checkpoint, n=n, identity=solver, admission_certificate=admission_identity
    )
    if set(checkpoint_residues) != set(used_moduli):
        raise ValueError(
            f"checkpoint modulus set mismatch: {sorted(checkpoint_residues)} != {sorted(used_moduli)}"
        )
    for idx, rec in enumerate(congruences):
        p = rec["modulus"]
        saved = checkpoint_residues[p]
        if saved.get("residue") != rec["residue"]:
            raise ValueError(f"checkpoint/manifest residue mismatch for modulus {p}")
        saved_wall = float(saved.get("wall_s", 0.0))
        if not math.isclose(saved_wall, float(rec["wall_s"]), rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"checkpoint/manifest wall_s mismatch for modulus {p}")
        if saved.get("log") != rec.get("log"):
            raise ValueError(f"checkpoint/manifest log mismatch for modulus {p}")

    exact_name = data["exact_file"]
    exact_sha = data["exact_sha256"]
    if not isinstance(exact_name, str) or Path(exact_name).name != exact_name:
        raise ValueError("exact_file must be a basename")
    if not isinstance(exact_sha, str) or len(exact_sha) != 64 or any(c not in "0123456789abcdef" for c in exact_sha):
        raise ValueError("invalid exact.txt SHA-256")
    exact_path = path.parent / exact_name
    if not exact_path.is_file():
        raise ValueError(f"exact result file referenced by manifest is missing: {exact_path}")
    actual_exact_sha = binary_sha256(exact_path)
    if actual_exact_sha != exact_sha:
        raise ValueError(f"exact.txt SHA-256 mismatch: {actual_exact_sha} != {exact_sha}")
    expected_exact_text = (
        f"n={n}\n"
        f"exact={exact}\n"
        f"bound_bits={path_bound.bit_length()}\n"
        f"modulus_bits={modulus_product.bit_length()}\n"
        f"primes_used={len(congruences)}\n"
        f"solver_wall_s_sum={float(reported_wall):.9f}\n"
        f"manifest_file={path.name}\n"
    )
    if exact_path.read_text(encoding="utf-8") != expected_exact_text:
        raise ValueError("exact.txt contents do not match verified manifest fields")

    provenance_name = data["build_provenance_file"]
    provenance_sha = data["build_provenance_sha256"]
    if provenance_name is None or provenance_sha is None:
        if provenance_name is not None or provenance_sha is not None:
            raise ValueError("build provenance file/hash must either both be null or both be present")
    else:
        if not isinstance(provenance_name, str) or Path(provenance_name).name != provenance_name:
            raise ValueError("build_provenance_file must be a basename")
        if not isinstance(provenance_sha, str) or len(provenance_sha) != 64 or any(c not in "0123456789abcdef" for c in provenance_sha):
            raise ValueError("invalid build provenance SHA-256")
        provenance = path.parent / provenance_name
        if not provenance.is_file():
            raise ValueError(f"build provenance referenced by manifest is missing: {provenance}")
        actual_provenance_sha = binary_sha256(provenance)
        if actual_provenance_sha != provenance_sha:
            raise ValueError(
                f"build provenance SHA-256 mismatch: {actual_provenance_sha} != {provenance_sha}"
            )
        pdata = load_provenance(
            provenance, binary=binary, root=ROOT if verify_sources else None,
            verify_sources=verify_sources, expected_compile_args=[f"-DTARGET_W={n + 1}"],
        )
        validate_exact_provenance_admissible(pdata, admission_identity)
        if verify_sources:
            _verify_generation_evidence(pdata)
        if pdata["binary_sha256"] != solver_sha:
            raise ValueError("build provenance solver SHA-256 does not match exact manifest")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify an ONEESAN exact-result manifest from first principles")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--binary", type=Path, help="also verify the solver executable SHA-256")
    ap.add_argument("--verify-sources", action="store_true",
                    help="also compare build-provenance source/header hashes with this checkout")
    args = ap.parse_args()
    try:
        data = verify(args.manifest, args.binary, verify_sources=args.verify_sources)
    except (ValueError, OSError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"format={data['format']}")
    print(f"n={data['n']}")
    print(f"exact={data['exact_decimal']}")
    print(f"primes_used={data['primes_used']}")
    print(f"manifest_sha256={data[RESULT_CHECKSUM_FIELD]}")
    print("valid=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
