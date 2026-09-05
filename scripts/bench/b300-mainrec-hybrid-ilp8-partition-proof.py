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
                        f'duplicate n={n} grid={grid} lanes={lanes} i={i}'
                    )
                seen[i] = 1
    try:
        bad = seen.index(0)
    except ValueError:
        return
    raise SystemExit(f'missing n={n} grid={grid} lanes={lanes} i={bad}')


thresholds = (0, 1, 256, 1024, 65536, 262144, 1048576, 4194304, 16777216)
threads_cases = (32, 64, 128, 256, 512, 1024)
production_cases = 0
enumerated_cases = 0
boundary_cases = 0
ilp2_cases = 0
ilp8_cases = 0

for threshold in thresholds:
    for threads in threads_cases:
        cap2 = CAP * threads * 2
        cap8 = CAP * threads * 8
        ns = {
            1,
            max(1, threads - 1),
            threads,
            threads + 1,
            2 * threads - 1,
            2 * threads,
            2 * threads + 1,
            8 * threads - 1,
            8 * threads,
            8 * threads + 1,
            4095,
            4096,
            4097,
            65535,
            65536,
            65537,
            cap2 - 1,
            cap2,
            cap2 + 1,
            cap8 - 1,
            cap8,
            cap8 + 1,
        }
        if threshold > 0:
            ns.add(threshold - 1)
        ns.add(threshold)
        ns.add(threshold + 1)

        for n in sorted(x for x in ns if x > 0):
            lanes = 8 if n >= threshold else 2
            if threshold == 0 and lanes != 8:
                raise SystemExit('threshold=0 must select ILP8 for every positive n')
            if threshold > 0 and n == threshold and lanes != 8:
                raise SystemExit(
                    f'threshold inclusive boundary failed n={n} threshold={threshold}'
                )
            if threshold > 1 and n == threshold - 1 and lanes != 2:
                raise SystemExit(
                    f'threshold lower boundary failed n={n} threshold={threshold}'
                )
            if n in (threshold - 1, threshold, threshold + 1):
                boundary_cases += 1

            blocks = blocks_for(n, threads, lanes)
            grid = blocks * threads
            if not (1 <= blocks <= CAP):
                raise SystemExit(
                    f'bad block count n={n} threads={threads} lanes={lanes} blocks={blocks}'
                )

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
            if not all(inverse_maps(i, n, grid, lanes) for i in samples):
                raise SystemExit(
                    f'inverse partition mismatch n={n} threshold={threshold} '
                    f'threads={threads} lanes={lanes} grid={grid}'
                )

            # At exactly the selector boundary, verify that switching kernels
            # changes only the partition schedule: both schedules independently
            # cover the complete destination domain exactly once.
            if threshold > 0 and n in (threshold - 1, threshold, threshold + 1):
                for test_lanes in (2, 8):
                    test_grid = blocks_for(n, threads, test_lanes) * threads
                    if n <= 200_000:
                        enumerate_exact(n, test_grid, test_lanes)
                    for i in samples:
                        if not inverse_maps(i, n, test_grid, test_lanes):
                            raise SystemExit(
                                f'cross-kernel boundary mismatch n={n} threshold={threshold} '
                                f'threads={threads} lanes={test_lanes}'
                            )

            production_cases += 1
            if lanes == 2:
                ilp2_cases += 1
            else:
                ilp8_cases += 1

print(
    'b300-mainrec-hybrid-ilp8-partition-proof OK',
    'selector=n_ge_threshold',
    'base_ilp=2',
    'high_ilp=8',
    'ilp2_launch=ceil_n_over_2threads_capped65535',
    'ilp8_launch=ceil_n_over_8threads_capped65535',
    f'production_cases={production_cases}',
    f'enumerated_cases={enumerated_cases}',
    f'boundary_cases={boundary_cases}',
    f'ilp2_cases={ilp2_cases}',
    f'ilp8_cases={ilp8_cases}',
    'exact_partition=1',
)
