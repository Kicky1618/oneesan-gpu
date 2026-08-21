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
    exe = sys.argv[2] if len(sys.argv) > 2 else "./oneesan_cuda_multi7_rawbuf"
    proc = subprocess.run([exe, str(n)], text=True, capture_output=True, check=True)
    m = re.search(r"residues=([0-9,]+).*peak_states=(\d+).*gpu_ms=([0-9.]+)", proc.stdout)
    if not m:
        raise RuntimeError(proc.stdout)
    residues = [int(x) for x in m.group(1).split(',')]
    if len(residues) != len(PRIMES):
        raise RuntimeError(f"expected {len(PRIMES)} residues, got {len(residues)}")
    x, M = crt(residues)
    bound_bits = 2 * n * (n + 1) + 1
    if M.bit_length() <= bound_bits:
        raise RuntimeError(f"CRT modulus only {M.bit_length()} bits, need > {bound_bits}")
    print(f"n={n}")
    print(f"paths={x}")
    print(f"decimal_digits={len(str(x))}")
    print(f"bound_bits={bound_bits}")
    print(f"modulus_bits={M.bit_length()}")
    print(f"peak_states={m.group(2)}")
    print(f"gpu_ms={float(m.group(3)):.3f}")

if __name__ == "__main__":
    main()
