#!/usr/bin/env python3
import math
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


def crt_pair(x: int, mod_product: int, residue: int, p: int) -> tuple[int, int]:
    t = ((residue - x) % p) * pow(mod_product, -1, p) % p
    return x + mod_product * t, mod_product * p


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    edges = 2 * n * (n + 1)
    # Every simple path is a distinct subset of grid edges, hence answer < 2^edges.
    required_bits = edges + 1

    x, M = 0, 1
    used = 0
    total_gpu_ms = 0.0

    for p in PRIMES:
        proc = subprocess.run(
            ["./oneesan_cuda", str(n), str(p)],
            text=True,
            capture_output=True,
            check=True,
        )
        m = re.search(r"residue=(\d+).*gpu_ms=([0-9.]+)", proc.stdout)
        if not m:
            raise RuntimeError(f"could not parse solver output: {proc.stdout!r}")
        residue = int(m.group(1))
        total_gpu_ms += float(m.group(2))
        x, M = crt_pair(x, M, residue, p)
        used += 1
        print(f"prime {used}: p={p} residue={residue} modulus_bits={M.bit_length()}", file=sys.stderr)
        if M.bit_length() > required_bits:
            break
    else:
        raise RuntimeError(
            f"not enough CRT primes: need >{required_bits} bits, got {M.bit_length()} bits"
        )

    print(f"n={n}")
    print(f"paths={x}")
    print(f"decimal_digits={len(str(x))}")
    print(f"crt_primes={used}")
    print(f"bound_bits={required_bits}")
    print(f"modulus_bits={M.bit_length()}")
    print(f"sum_gpu_ms={total_gpu_ms:.3f}")


if __name__ == "__main__":
    main()
