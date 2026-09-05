#!/usr/bin/env python3
"""Generate exact row-6 coefficients for 22 packed pairs of <2^15 CRT primes."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
from verify_row6_crt20 import EXPECTED_COUNTS, load_certificate  # noqa: E402

DEFAULT_OUTPUT = ROOT / "src" / "cuda" / "b300" / "row6_automaton_rns15x2_generated.hpp"
NPAIRS = 22


def primes_below(n: int) -> list[int]:
    sieve = bytearray(b"\x01") * n
    sieve[:2] = b"\x00\x00"
    p = 2
    while p * p < n:
        if sieve[p]:
            sieve[p * p : n : p] = b"\x00" * (((n - 1 - p * p) // p) + 1)
        p += 1
    return [i for i in range(2, n) if sieve[i]]


def choose_primes(transitions: dict[str, list[tuple[int, int, int, int]]]) -> list[int]:
    den = {d for rows in transitions.values() for _, _, _, d in rows}
    good = [p for p in primes_below(1 << 15) if p >= 3 and all(d % p for d in den)]
    need = 2 * NPAIRS
    if len(good) < need:
        raise RuntimeError("not enough denominator-safe 15-bit primes")
    return list(reversed(good[-need:]))


def coeff(num: int, den: int, p: int) -> int:
    if den % p == 0:
        raise RuntimeError(f"denominator vanishes modulo {p}")
    return (num % p) * pow(den, -1, p) % p


def generate(output: Path) -> None:
    _, transitions = load_certificate()
    primes = choose_primes(transitions)
    pairs = list(zip(primes[::2], primes[1::2]))
    bits = sum(math.log2(p) for p in primes)
    if bits <= 633.0:
        raise RuntimeError(f"RNS15x2 capacity too small: {bits:.6f} bits")

    with output.open("w") as f:
        f.write("#pragma once\n#include <cstdint>\nnamespace oneesan::row6rns15 {\n")
        f.write(f"static constexpr int NPAIRS={NPAIRS};\n")
        f.write(f"static constexpr double CRT_BITS={bits:.15f};\n")
        f.write("static constexpr uint16_t P0[NPAIRS]={" + ",".join(str(a) for a, _ in pairs) + "};\n")
        f.write("static constexpr uint16_t P1[NPAIRS]={" + ",".join(str(b) for _, b in pairs) + "};\n")
        f.write(
            "static inline int pair_index(uint32_t a,uint32_t b){for(int i=0;i<NPAIRS;++i)"
            "if(P0[i]==a&&P1[i]==b)return i;return -1;}\n"
        )
        for sym in "NRL":
            exact = transitions[sym]
            if len(exact) != EXPECTED_COUNTS[sym]:
                raise RuntimeError(f"unexpected {sym} transition count")
            f.write(f"static constexpr uint32_t CO_{sym}[NPAIRS][{len(exact)}]={{\n")
            for pi, (p0, p1) in enumerate(pairs):
                values = []
                for _, _, num, den in exact:
                    a = coeff(num, den, p0)
                    b = coeff(num, den, p1)
                    values.append(f"{a | (b << 16)}u")
                f.write("{" + ",".join(values) + "}" + ("," if pi + 1 < NPAIRS else "") + "\n")
            f.write("};\n")
        f.write("}\n")

    print(f"generated {output}")
    print("pairs=" + " ".join(f"{a}/{b}" for a, b in pairs))
    print(f"crt_bits={bits:.12f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    generate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
