#!/usr/bin/env python3
import math
import re
import subprocess
import sys
import time

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
    2305843009213693277,
]
BATCH = 4


def crt(residues):
    x, m = 0, 1
    for r, p in zip(residues, PRIMES):
        t = ((r - x) % p) * pow(m, -1, p) % p
        x += m * t
        m *= p
    return x, m


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    bound_bits = 2 * n * (n + 1) + 1
    modulus_bits = math.prod(PRIMES).bit_length()
    if modulus_bits <= bound_bits:
        raise SystemExit(
            f"CRT capacity insufficient for n={n}: modulus_bits={modulus_bits}, need > {bound_bits}"
        )

    residues = []
    total_gpu_ms = 0.0
    total_wall = 0.0
    peak_states = 0
    peak_alloc = 0
    peak_slots = 0

    for base in range(0, len(PRIMES), BATCH):
        active = min(BATCH, len(PRIMES) - base)
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["./oneesan_cuda_hopper_compact", str(n), str(base), str(active)],
            text=True, capture_output=True, check=True,
        )
        wall = time.perf_counter() - t0
        print(proc.stderr, end='', file=sys.stderr)
        m = re.search(
            r"residues=([0-9,]+).*peak_states=(\d+).*hash_slots=(\d+).*peak_alloc_bytes=(\d+).*gpu_ms=([0-9.]+)",
            proc.stdout,
        )
        if not m:
            raise SystemExit(f"could not parse solver output: {proc.stdout!r}")
        rs = [int(x) for x in m.group(1).split(',')]
        if len(rs) != active:
            raise SystemExit(f"batch {base}: expected {active} residues, got {len(rs)}")
        residues.extend(rs)
        total_gpu_ms += float(m.group(5))
        total_wall += wall
        peak_states = max(peak_states, int(m.group(2)))
        peak_slots = max(peak_slots, int(m.group(3)))
        peak_alloc = max(peak_alloc, int(m.group(4)))
        print(
            f"batch {base//BATCH+1}: primes {base}..{base+active-1} "
            f"gpu={float(m.group(5))/1000:.3f}s mem={int(m.group(4))/2**30:.3f}GiB",
            file=sys.stderr,
        )

    x, mod = crt(residues)
    print(f"n={n}")
    print(f"paths={x}")
    print(f"decimal_digits={len(str(x))}")
    print(f"bound_bits={bound_bits}")
    print(f"modulus_bits={mod.bit_length()}")
    print(f"peak_states={peak_states}")
    print(f"hash_slots={peak_slots}")
    print(f"peak_alloc_bytes={peak_alloc}")
    print(f"peak_alloc_gib={peak_alloc/2**30:.3f}")
    print(f"sum_gpu_ms={total_gpu_ms:.3f}")
    print(f"sum_wall_s={total_wall:.3f}")


if __name__ == '__main__':
    main()
