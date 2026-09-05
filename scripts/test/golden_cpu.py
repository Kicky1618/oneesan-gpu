#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

GOLDEN = {
    1: 2,
    2: 12,
    3: 184,
    4: 8512,
    5: 1262816,
    6: 575780564,
    7: 789360053252,
    8: 3266598486981642,
    9: 41044208702632496804,
}


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <solver> [max_n]")
    solver = Path(sys.argv[1])
    max_n = int(sys.argv[2]) if len(sys.argv) > 2 else max(GOLDEN)
    for n, expected in GOLDEN.items():
        if n > max_n:
            continue
        proc = subprocess.run([str(solver), str(n)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise SystemExit(f"{solver} n={n} failed ({proc.returncode}): {proc.stderr}")
        m = re.search(r"paths=(\d+)", proc.stdout)
        if not m:
            raise SystemExit(f"{solver} n={n}: could not parse paths from: {proc.stdout!r}")
        got = int(m.group(1))
        if got != expected:
            raise SystemExit(f"{solver} n={n}: got {got}, expected {expected}")
        print(f"ok n={n} paths={got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
