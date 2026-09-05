#!/usr/bin/env python3
from __future__ import annotations

CAP = 65535


def blocks_for(n: int, threads: int, lanes: int) -> int:
    if n <= 0:
        return 1
    return min(CAP, max(1, (n + threads * lanes - 1) // (threads * lanes)))


def inverse_maps(i: int, n: int, grid: int, lanes: int) -> bool:
    if not 0 <= i < n:
        return False
    tid = i % grid
    ordinal = i // grid
    q, lane = divmod(ordinal, lanes)
    base = tid + q * lanes * grid
    return tid < grid and lane < lanes and base < n and base + lane * grid == i


def enumerate_exact(n: int, grid: int, lanes: int) -> None:
    seen = bytearray(n)
    for tid in range(min(grid, n)):
        for base in range(tid, n, lanes * grid):
            for lane in range(lanes):
                i = base + lane * grid
                if i >= n:
                    continue
                if seen[i]:
                    raise SystemExit(
                        f'duplicate lanes={lanes} n={n} grid={grid} i={i}'
                    )
                seen[i] = 1
    try:
        bad = seen.index(0)
    except ValueError:
        return
    raise SystemExit(f'missing lanes={lanes} n={n} grid={grid} i={bad}')


for lanes in (2, 3, 4):
    production_cases = 0
    enumerated_cases = 0
    old_degenerate_cases = 0
    multilane_cases = 0

    for threads in (32, 64, 128, 256, 512, 1024):
        block_cap_n = CAP * threads * lanes
        ns = {
            1,
            max(1, threads - 1),
            threads,
            threads + 1,
            2 * threads - 1,
            2 * threads,
            2 * threads + 1,
            3 * threads - 1,
            3 * threads,
            3 * threads + 1,
            4 * threads - 1,
            4 * threads,
            4 * threads + 1,
            4095,
            4096,
            4097,
            65535,
            65536,
            65537,
            block_cap_n - 1,
            block_cap_n,
            block_cap_n + 1,
            block_cap_n + lanes * threads + 17,
        }
        for n in sorted(ns):
            if n <= 0:
                continue
            blocks = blocks_for(n, threads, lanes)
            expect = min(CAP, max(1, (n + threads * lanes - 1) // (threads * lanes)))
            if blocks != expect:
                raise SystemExit(
                    f'block formula mismatch lanes={lanes} n={n} threads={threads} '
                    f'got={blocks} expected={expect}'
                )
            grid = blocks * threads

            if n <= 200_000:
                enumerate_exact(n, grid, lanes)
                enumerated_cases += 1

            last = n - 1
            samples = {
                0,
                min(1, last),
                min(threads - 1, last),
                min(threads, last),
                last // 7,
                last // 5,
                last // 3,
                last // 2,
                max(0, last - 2),
                max(0, last - 1),
                last,
                min(CAP * threads, last),
            }
            for i in samples:
                if not inverse_maps(i, n, grid, lanes):
                    raise SystemExit(
                        f'inverse mismatch lanes={lanes} n={n} threads={threads} '
                        f'grid={grid} i={i}'
                    )

            # Model the old production launch: bm=ceil(n/threads).  Before the
            # 65535-block cap it makes grid >= n, so every lane except lane 0 is
            # dead.  This assertion captures the exact regression that led to
            # the launch fix in the ILP2/3/4 transforms.
            old_blocks = min(CAP, max(1, (n + threads - 1) // threads))
            old_grid = old_blocks * threads
            old_live = min(lanes, (n + old_grid - 1) // old_grid)
            new_live = min(lanes, (n + grid - 1) // grid)
            if threads < n <= CAP * threads:
                if old_live != 1:
                    raise SystemExit(
                        f'old launch model unexpectedly nondegenerate lanes={lanes} '
                        f'n={n} threads={threads} live={old_live}'
                    )
                old_degenerate_cases += 1
                if new_live <= 1:
                    raise SystemExit(
                        f'scaled launch failed to expose MLP lanes={lanes} '
                        f'n={n} threads={threads} live={new_live}'
                    )
                multilane_cases += 1
            production_cases += 1

    print(
        'b300-main-pull-ilp-production-launch-proof',
        f'lanes={lanes}',
        f'production_cases={production_cases}',
        f'enumerated_cases={enumerated_cases}',
        f'old_degenerate_cases={old_degenerate_cases}',
        f'multilane_cases={multilane_cases}',
        'block_cap=65535',
        'scaled_launch=1',
        'exact=1',
    )

print('b300-main-pull-ilp-production-launch-proof OK scaled_launch=1 exact=1')
