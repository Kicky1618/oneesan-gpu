#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PRIMES = [
    4294967291, 4294967279, 4294967231, 4294967197,
    4294967189, 4294967161, 4294967143, 4294967111,
    4294967087, 4294967029, 4294966997, 4294966981,
    4294966943, 4294966927, 4294966909, 4294966877,
    4294966829, 4294966813, 4294966769, 4294966667,
    4294966661, 4294966657, 4294966651, 4294966639,
    4294966619, 4294966591, 4294966583, 4294966553,
    4294966477, 4294966447, 4294966441, 4294966427,
    4294966373, 4294966367, 4294966337, 4294966297,
    4294966243, 4294966237, 4294966231, 4294966217,
    4294966187, 4294966177, 4294966163, 4294966153,
    4294966129, 4294966121, 4294966099, 4294966087,
]

VERIFIER_MATH_VERSION = "independent-v1"


def _is_prime_32(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small:
        return True
    if any(n % p == 0 for p in small):
        return False

    d = n - 1
    s = 0
    while d % 2 == 0:
        s += 1
        d //= 2
    for a in (2, 3, 5, 7, 11):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _validate_prime_table() -> None:
    if len(PRIMES) != len(set(PRIMES)):
        raise SystemExit("independent verifier prime table contains duplicates")
    if any(a <= b for a, b in zip(PRIMES, PRIMES[1:])):
        raise SystemExit("independent verifier prime table is not strictly descending")
    for p in PRIMES:
        if p >= 1 << 32 or not _is_prime_32(p):
            raise SystemExit(f"independent verifier prime table contains non-prime p={p}")


def _strip_compatible(x: int, y: int, height: int) -> bool:
    if height <= 1:
        return True
    row_mask = (1 << (height - 1)) - 1
    x_changes = (x ^ (x >> 1)) & row_mask
    y_changes = (y ^ (y >> 1)) & row_mask
    top_diff = (x ^ y) & row_mask
    return (x_changes & y_changes & top_diff) == 0


def checkerboard_strip_count(height: int, width: int) -> int:
    if height < 1 or width < 1:
        raise ValueError("strip dimensions must be positive")
    states = 1 << height
    transitions = [
        [y for y in range(states) if _strip_compatible(x, y, height)]
        for x in range(states)
    ]
    dp = [1] * states
    for _ in range(width - 1):
        nxt = [0] * states
        for x, ys in enumerate(transitions):
            count = dp[x]
            if count == 0:
                continue
            for y in ys:
                nxt[y] += count
        dp = nxt
    return sum(dp)


def simple_path_upper_bound(n: int, max_strip_height: int = 9) -> tuple[int, list[int]]:
    if n < 1:
        return 1, []
    if max_strip_height < 1:
        raise ValueError("max_strip_height must be positive")

    hmax = min(n, max_strip_height)
    strip_counts = [0] + [
        checkerboard_strip_count(height, n)
        for height in range(1, hmax + 1)
    ]

    best: list[int | None] = [None] * (n + 1)
    partition: list[list[int] | None] = [None] * (n + 1)
    best[0] = 1
    partition[0] = []
    for rows in range(1, n + 1):
        for height in range(1, min(hmax, rows) + 1):
            previous = best[rows - height]
            previous_parts = partition[rows - height]
            if previous is None or previous_parts is None:
                continue
            candidate = previous * strip_counts[height]
            if best[rows] is None or candidate < best[rows]:
                best[rows] = candidate
                partition[rows] = [*previous_parts, height]

    if best[n] is None or partition[n] is None:
        raise RuntimeError("failed to construct independent strip bound")
    return best[n], partition[n]


def primes_for_bound(bound: int) -> list[int]:
    if bound < 0:
        raise ValueError("bound must be non-negative")
    _validate_prime_table()

    product = 1
    prefix: list[int] = []
    for p in PRIMES:
        prefix.append(p)
        product *= p
        if product > bound:
            return prefix
    raise SystemExit(
        f"independent CRT prime capacity insufficient: product has {product.bit_length()} bits, "
        f"bound has {bound.bit_length()} bits"
    )


def _mod_inverse(a: int, modulus: int) -> int:
    old_r, r = modulus, a % modulus
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_t, t = t, old_t - q * t
    if old_r != 1:
        raise ValueError(f"no inverse for {a} modulo {modulus}")
    return old_t % modulus


def crt_pair(x: int, m: int, residue: int, prime: int) -> tuple[int, int]:
    if m <= 0 or prime <= 1:
        raise ValueError("CRT moduli must be positive")
    if not 0 <= residue < prime:
        raise ValueError(f"residue {residue} outside [0,{prime})")
    delta = (residue - x) % prime
    step = delta * _mod_inverse(m, prime) % prime
    return x + m * step, m * prime


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_exact(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        if "=" not in raw:
            raise SystemExit(f"malformed exact.txt line: {raw!r}")
        k, v = raw.split("=", 1)
        if k in out:
            raise SystemExit(f"duplicate exact.txt key: {k}")
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Independently verify a completed B300 exact CRT artifact")
    ap.add_argument("n", nargs="?", type=int, default=27)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--profile-sha256", default=None)
    ap.add_argument("--certificate", default=None)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    exact_path = Path(args.exact).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not exact_path.is_file():
        raise SystemExit(f"exact artifact not found: {exact_path}")

    data = json.loads(checkpoint.read_text())
    if int(data.get("n", -1)) != args.n:
        raise SystemExit(f"checkpoint n mismatch: {data.get('n')} != {args.n}")
    fp = data.get("solver_fingerprint")
    if not isinstance(fp, dict) or fp.get("schema") not in (2, 3):
        raise SystemExit(f"unsupported checkpoint fingerprint: {fp!r}")
    bsha = fp.get("binary_sha256")
    if not isinstance(bsha, str) or len(bsha) != 64:
        raise SystemExit("checkpoint binary SHA is malformed")
    psha = fp.get("profile_sha256")
    if fp["schema"] == 3 and (not isinstance(psha, str) or len(psha) != 64):
        raise SystemExit("schema-3 checkpoint profile SHA is malformed")
    if fp["schema"] == 2 and psha is not None:
        raise SystemExit("schema-2 checkpoint unexpectedly carries profile SHA")

    if args.binary:
        binary = Path(args.binary).resolve()
        if not binary.is_file():
            raise SystemExit(f"binary not found: {binary}")
        got = sha256(binary)
        if got != bsha:
            raise SystemExit(f"binary SHA mismatch: checkpoint={bsha} actual={got}")
    else:
        binary = None
    if args.profile_sha256 is not None:
        if fp["schema"] != 3:
            raise SystemExit("--profile-sha256 requires schema-3 checkpoint")
        if args.profile_sha256 != psha:
            raise SystemExit(f"profile SHA mismatch: checkpoint={psha} expected={args.profile_sha256}")

    path_bound, strips = simple_path_upper_bound(args.n)
    prefix = primes_for_bound(path_bound)
    residues_raw = data.get("residues", {})
    if not isinstance(residues_raw, dict):
        raise SystemExit("checkpoint residues must be an object")

    x, modulus, total_wall = 0, 1, 0.0
    used = 0
    used_records: list[dict[str, object]] = []
    for p in prefix:
        rec = residues_raw.get(str(p))
        if rec is None:
            break
        if not isinstance(rec, dict) or "residue" not in rec:
            raise SystemExit(f"malformed residue record for p={p}")
        r = int(rec["residue"])
        wall = float(rec.get("wall_s", 0.0))
        if not 0 <= r < p:
            raise SystemExit(f"invalid residue {r} mod {p}")
        if not math.isfinite(wall) or wall < 0:
            raise SystemExit(f"invalid wall_s={wall} for p={p}")
        x, modulus = crt_pair(x, modulus, r, p)
        total_wall += wall
        used += 1
        used_records.append({"modulus": p, "residue": r, "wall_s": wall})
        if modulus > path_bound:
            break

    if modulus <= path_bound:
        raise SystemExit(
            f"CRT capacity insufficient: modulus_bits={modulus.bit_length()} "
            f"bound_bits={path_bound.bit_length()} contiguous_primes={used}"
        )
    if x > path_bound:
        raise SystemExit(f"CRT representative exceeds rigorous bound: {x} > {path_bound}")

    fields = parse_exact(exact_path)
    expected_int = {
        "n": args.n,
        "exact": x,
        "bound_bits": path_bound.bit_length(),
        "modulus_bits": modulus.bit_length(),
        "primes_used": used,
        "checkpoint_schema": int(fp["schema"]),
    }
    for key, expected in expected_int.items():
        if key not in fields or int(fields[key]) != expected:
            raise SystemExit(f"exact.txt mismatch {key}: got={fields.get(key)!r} expected={expected}")
    wall_text = f"{total_wall:.9f}"
    if fields.get("solver_wall_s_sum") != wall_text:
        raise SystemExit(
            f"exact.txt mismatch solver_wall_s_sum: got={fields.get('solver_wall_s_sum')!r} expected={wall_text}"
        )
    checkpoint_sha = sha256(checkpoint)
    if fields.get("checkpoint_sha256") != checkpoint_sha:
        raise SystemExit("exact.txt checkpoint SHA mismatch")
    if fields.get("solver_binary_sha256") != bsha:
        raise SystemExit("exact.txt binary SHA mismatch")
    if fp["schema"] == 3:
        if fields.get("solver_profile_sha256") != psha:
            raise SystemExit("exact.txt profile SHA mismatch")
    elif "solver_profile_sha256" in fields:
        raise SystemExit("schema-2 exact.txt unexpectedly carries profile SHA")

    cert_path = (
        Path(args.certificate).resolve()
        if args.certificate
        else exact_path.with_name("exact.verify.json")
    )
    certificate = {
        "schema": 1,
        "verified": True,
        "verifier_math": VERIFIER_MATH_VERSION,
        "n": args.n,
        "exact": x,
        "rigorous_bound": path_bound,
        "bound_bits": path_bound.bit_length(),
        "strip_partition": strips,
        "crt_modulus": modulus,
        "modulus_bits": modulus.bit_length(),
        "primes_used": used,
        "solver_wall_s_sum": total_wall,
        "checkpoint_schema": fp["schema"],
        "checkpoint_sha256": checkpoint_sha,
        "exact_txt_sha256": sha256(exact_path),
        "solver_binary_sha256": bsha,
        "solver_profile_sha256": psha,
        "binary_path": str(binary) if binary else None,
        "checkpoint_path": str(checkpoint),
        "exact_path": str(exact_path),
        "residues_used": used_records,
    }
    tmp = cert_path.with_suffix(cert_path.suffix + ".tmp")
    tmp.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    tmp.replace(cert_path)

    print(
        f"B300_EXACT_VERIFY_OK n={args.n} exact={x} primes_used={used} "
        f"modulus_bits={modulus.bit_length()} bound_bits={path_bound.bit_length()} "
        f"verifier_math={VERIFIER_MATH_VERSION}"
    )
    print(f"checkpoint_sha256={checkpoint_sha}")
    print(f"exact_txt_sha256={certificate['exact_txt_sha256']}")
    print(f"solver_binary_sha256={bsha}")
    if psha is not None:
        print(f"solver_profile_sha256={psha}")
    print(f"certificate={cert_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
