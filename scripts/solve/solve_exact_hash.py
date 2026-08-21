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

def crt_pair(x: int, M: int, r: int, p: int) -> tuple[int, int]:
    t = ((r - x) % p) * pow(M, -1, p) % p
    return x + M * t, M * p

def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    exe = sys.argv[2] if len(sys.argv) > 2 else "./build/oneesan_cuda_hash"
    required_bits = 2 * n * (n + 1) + 1
    x, M = 0, 1
    total_ms = 0.0
    peak = 0
    for i, p in enumerate(PRIMES, 1):
        proc = subprocess.run([exe, str(n), str(p)], text=True, capture_output=True, check=True)
        m = re.search(r"residue=(\d+).*peak_states=(\d+).*gpu_ms=([0-9.]+)", proc.stdout)
        if not m:
            raise RuntimeError(proc.stdout)
        r = int(m.group(1))
        peak = max(peak, int(m.group(2)))
        total_ms += float(m.group(3))
        x, M = crt_pair(x, M, r, p)
        print(f"prime {i}: residue={r} modulus_bits={M.bit_length()}", file=sys.stderr)
        if M.bit_length() > required_bits:
            print(f"n={n}")
            print(f"paths={x}")
            print(f"decimal_digits={len(str(x))}")
            print(f"crt_primes={i}")
            print(f"bound_bits={required_bits}")
            print(f"modulus_bits={M.bit_length()}")
            print(f"peak_states={peak}")
            print(f"sum_gpu_ms={total_ms:.3f}")
            return
    raise RuntimeError(f"not enough primes: need > {required_bits} bits, got {M.bit_length()}")

if __name__ == "__main__":
    main()
