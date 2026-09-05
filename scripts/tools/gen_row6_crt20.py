#!/usr/bin/env python3
"""Regenerate the production row-6 CRT20 coefficient header from the exact certificate."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "solve"))
sys.path.insert(0, str(ROOT / "scripts" / "tools"))
from path_bound import primes_for_bound, simple_path_upper_bound  # noqa: E402
from verify_row6_crt20 import EXPECTED_COUNTS, load_certificate  # noqa: E402

DEFAULT_OUTPUT = ROOT / "src" / "cuda" / "b300" / "row6_automaton_crt20_generated.hpp"


def generate(output: Path) -> None:
    _, transitions = load_certificate()
    bound, _ = simple_path_upper_bound(27)
    primes = primes_for_bound(bound)
    if len(primes) != 20:
        raise RuntimeError(f"n=27 bound now requires {len(primes)} CRT primes, expected 20")

    with output.open("w") as f:
        f.write("#pragma once\n#include <cstdint>\nnamespace oneesan::row6crt {\n")
        f.write("static constexpr int NPRIMES=20;\n")
        f.write(
            "static constexpr uint32_t PRIMES[NPRIMES]={"
            + ",".join(f"{p}u" for p in primes)
            + "};\n"
        )
        f.write(
            "static inline int prime_index(uint32_t p){for(int i=0;i<NPRIMES;++i)"
            "if(PRIMES[i]==p)return i;return -1;}\n"
        )
        for sym in "NRL":
            exact = transitions[sym]
            if len(exact) != EXPECTED_COUNTS[sym]:
                raise RuntimeError(f"unexpected {sym} transition count")
            f.write(f"static constexpr uint32_t CO_{sym}[NPRIMES][{len(exact)}]={{\n")
            denominators = {den for _, _, _, den in exact}
            for pi, p in enumerate(primes):
                inverse_denominator = {}
                for den in denominators:
                    if den % p == 0:
                        raise RuntimeError(f"denominator vanishes mod {p}")
                    inverse_denominator[den] = pow(den, -1, p)
                values = [
                    f"{(num % p) * inverse_denominator[den] % p}u"
                    for _, _, num, den in exact
                ]
                f.write("{" + ",".join(values) + "}" + ("," if pi + 1 < len(primes) else "") + "\n")
            f.write("};\n")
        f.write("}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
