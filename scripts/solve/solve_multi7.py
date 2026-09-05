#!/usr/bin/env python3
import re
import subprocess
import sys

PRIMES = [
    2305843009213693921,
    2305843009213692799,
    2305843009213691767,
    2305843009213690657,
    2305843009213689601,
    2305843009213688569,
    2305843009213687519,
]

def crt(residues: list[int]) -> tuple[int, int]:
    x, M = 0, 1
    for r, p in zip(residues, PRIMES):
        t = ((r - x) % p) * pow(M, -1, p) % p
        x += M * t
        M *= p
    return x, M

def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    exe = sys.argv[2] if len(sys.argv) > 2 else "./build/oneesan_cuda_multi7_rawbuf"
    proc = subprocess.run([exe, str(n)], text=True, capture_output=True, check=True)
    m = re.search(r"residues=([0-9,]+).*?moduli=([0-9,]+).*?peak_states=(\d+).*?gpu_ms=([0-9.]+)", proc.stdout)
    if not m:
        raise RuntimeError(proc.stdout)
    residues = [int(x) for x in m.group(1).split(',')]
    moduli = [int(x) for x in m.group(2).split(',')]
    if moduli != PRIMES:
        raise RuntimeError(f"solver modulus echo mismatch: got {moduli}, expected {PRIMES}")
    if len(residues) != len(PRIMES):
        raise RuntimeError(f"expected {len(PRIMES)} residues, got {len(residues)}")
    for residue, modulus in zip(residues, PRIMES):
        if not 0 <= residue < modulus:
            raise RuntimeError(f"non-canonical residue {residue} for modulus {modulus}")
    x, M = crt(residues)
    edge_count = 2 * n * (n + 1)
    path_bound = 1 << edge_count
    if not 0 <= x < path_bound:
        raise RuntimeError(f"CRT reconstruction violates subset bound: {x} >= 2^{edge_count}")
    bound_bits = edge_count + 1
    if M.bit_length() <= bound_bits:
        raise RuntimeError(f"CRT modulus only {M.bit_length()} bits, need > {bound_bits}")
    print(f"n={n}")
    print(f"paths={x}")
    print(f"decimal_digits={len(str(x))}")
    print(f"bound_bits={bound_bits}")
    print(f"modulus_bits={M.bit_length()}")
    print(f"peak_states={m.group(3)}")
    print(f"gpu_ms={float(m.group(4)):.3f}")

if __name__ == "__main__":
    main()
