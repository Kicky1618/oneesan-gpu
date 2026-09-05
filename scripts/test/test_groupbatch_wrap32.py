#!/usr/bin/env python3
from __future__ import annotations

import random
import re
from pathlib import Path

M = 1 << 32
ROOT = Path(__file__).resolve().parents[2]
HDR = ROOT / 'src/cuda/b300/row6_automaton_crt20_generated.hpp'


def load_primes() -> list[int]:
    text = HDR.read_text(errors='ignore')
    m = re.search(r'PRIMES\[NPRIMES\]\s*=\s*\{([^}]*)\}', text)
    if not m:
        raise RuntimeError('CRT20 PRIMES table not found')
    return [int(x) for x in re.findall(r'(\d+)u', m.group(1))]


def normalize32(x: int, p: int) -> int:
    assert 0 <= x < M
    # Production WRAP32 uses one subtract; the mode guard must imply this is enough.
    return x - p if x >= p else x


def end_around_add(x: int, add: int, p: int) -> int:
    """Sequential model of native atomicAdd + pseudo-Mersenne end-around carry."""
    gap = M - p
    while add:
        total = x + add
        x = total & 0xFFFFFFFF
        if total < M:
            return x
        add = gap
    return x


def check_sequential(primes: list[int]) -> None:
    rng = random.Random(0x6F6E656573616E)
    for p in primes:
        assert 2 * p > M, p
        for _ in range(20_000):
            x0 = rng.randrange(p)
            vals = [rng.randrange(p) for _ in range(rng.randrange(1, 24))]
            x = x0
            for v in vals:
                x = end_around_add(x, v, p)
            got = normalize32(x, p)
            want = (x0 + sum(vals)) % p
            assert got == want, (p, x0, vals, got, want)


def check_interleavings(primes: list[int]) -> None:
    """Randomly interleave primary additions with their deferred carry compensation."""
    rng = random.Random(0x575241503332)
    for p in primes:
        gap = M - p
        for _ in range(1_000):
            x0 = rng.randrange(p)
            vals = [rng.randrange(p) for _ in range(64)]
            x = x0
            pending = vals[:]
            steps = 0
            while pending:
                i = rng.randrange(len(pending))
                add = pending.pop(i)
                total = x + add
                x = total & 0xFFFFFFFF
                if total >= M:
                    pending.append(gap)
                steps += 1
                assert steps < 100_000
            got = normalize32(x, p)
            want = (x0 + sum(vals)) % p
            assert got == want, (p, got, want)


def check_small_exhaustive() -> None:
    # Same algebra at 8 bits. p > 2^7 guarantees one-subtract normalization.
    m = 1 << 8
    for p in (251, 239, 223, 193):
        gap = m - p
        for x0 in range(p):
            for v in range(p):
                x = x0
                add = v
                while add:
                    total = x + add
                    x = total & 0xFF
                    if total < m:
                        break
                    add = gap
                got = x - p if x >= p else x
                assert got == (x0 + v) % p


def main() -> None:
    primes = load_primes()
    assert len(primes) == 20
    check_small_exhaustive()
    check_sequential(primes)
    check_interleavings(primes)
    print(f'groupbatch WRAP32 end-around: ok ({len(primes)} CRT primes)')


if __name__ == '__main__':
    main()
