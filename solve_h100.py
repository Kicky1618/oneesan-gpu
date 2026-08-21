#!/usr/bin/env python3
import math
import re
import subprocess
import sys

PRIMES = [
    2305843009213693951,
    2305843009213693921,
    2305843009213693907,
    2305843009213693723,
    2305843009213693693,
    2305843009213693669,
    2305843009213693613,
    2305843009213693561,
    2305843009213693549,
    2305843009213693487,
    2305843009213693421,
    2305843009213693373,
]

def crt(residues):
    x, m = 0, 1
    for r, p in zip(residues, PRIMES):
        t = ((r - x) % p) * pow(m, -1, p) % p
        x += m * t
        m *= p
    return x, m

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    bound_bits = 2 * n * (n + 1) + 1
    modulus_bits = math.prod(PRIMES).bit_length()
    if modulus_bits <= bound_bits:
        raise SystemExit(
            f"CRT capacity insufficient for n={n}: modulus_bits={modulus_bits}, "
            f"need > {bound_bits}. Extend PRIMES/NRES first."
        )
    proc = subprocess.run(["./oneesan_cuda_h100", str(n)], text=True, capture_output=True, check=True)
    print(proc.stderr, end="", file=sys.stderr)
    m = re.search(r"residues=([0-9,]+).*peak_states=(\d+).*gpu_ms=([0-9.]+)", proc.stdout)
    if not m:
        raise SystemExit(f"could not parse solver output: {proc.stdout!r}")
    residues = [int(x) for x in m.group(1).split(',')]
    if len(residues) != len(PRIMES):
        raise SystemExit(f"expected {len(PRIMES)} residues, got {len(residues)}")
    x, mod = crt(residues)
    print(f"n={n}")
    print(f"paths={x}")
    print(f"decimal_digits={len(str(x))}")
    print(f"bound_bits={bound_bits}")
    print(f"modulus_bits={mod.bit_length()}")
    print(f"peak_states={m.group(2)}")
    print(f"gpu_ms={m.group(3)}")

if __name__ == '__main__':
    main()
