#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math

COUNT_BYTES = 4
MAXW = 28


def full_dp(maxw: int = MAXW) -> list[list[int]]:
    dp = [[0] * (maxw + 2) for _ in range(maxw + 1)]
    dp[0][0] = 1
    for w in range(1, maxw + 1):
        for h in range(maxw + 1):
            x = dp[w - 1][h]
            if h > 0:
                x += dp[w - 1][h - 1]
            if h < maxw + 1:
                x += dp[w - 1][h + 1]
            dp[w][h] = x
    return dp


def make_spec_size(width: int, fixed: int, occ: int, maxw: int = MAXW) -> int:
    prev = [0] * (maxw + 2)
    prev[0] = 1
    for w in range(1, width + 1):
        pos = w - 1
        f = (fixed >> pos) & 1
        o = (occ >> pos) & 1
        cur = [0] * (maxw + 2)
        for h in range(maxw + 1):
            x = 0
            if not f or not o:
                x += prev[h]
            if not f or o:
                if h > 0:
                    x += prev[h - 1]
                if h < maxw + 1:
                    x += prev[h + 1]
            cur[h] = x
        prev = cur
    return prev[1]


def window_candidates(width: int, hi: int, lo: int) -> list[int]:
    return [q for q in range(width - 1, -1, -1) if q < lo - 1 or q > hi]


def window_masks(width: int, hi: int, lo: int, fixed_pos: list[int], group: int) -> tuple[int, int, int, int]:
    mf = mo = bf = bo = 0
    for i, q in enumerate(fixed_pos):
        one = (group >> i) & 1
        mf |= 1 << q
        if one:
            mo |= 1 << q
        bq = q if q < lo - 1 else q - 1
        bf |= 1 << bq
        if one:
            bo |= 1 << bq
    return mf, mo, bf, bo


def forced_window_plan(width: int) -> list[dict[str, int | float]]:
    low = width // 2
    ranges = [(width - 1, low + 1), (low, 1)]
    out: list[dict[str, int | float]] = []
    for hi, lo in ranges:
        fp = window_candidates(width, hi, lo)
        max_bytes = 0
        max_main = 0
        max_block = 0
        argmax = 0
        for group in range(1 << len(fp)):
            mf, mo, bf, bo = window_masks(width, hi, lo, fp, group)
            main = make_spec_size(width, mf, mo)
            block = make_spec_size(width - 1, bf, bo)
            need = (2 * main + 2 * block) * COUNT_BYTES
            if need > max_bytes:
                max_bytes = need
                max_main = main
                max_block = block
                argmax = group
        out.append({
            "p_hi": hi,
            "p_lo": lo,
            "fixed_bits": len(fp),
            "groups": 1 << len(fp),
            "max_scratch_bytes": max_bytes,
            "max_scratch_mib": max_bytes / (1 << 20),
            "max_main_states": max_main,
            "max_blocked_states": max_block,
            "argmax_group": argmax,
        })
    return out


def nonnegative_walk_count(length: int, start: int) -> int:
    cur = {start: 1}
    for _ in range(length):
        nxt: dict[int, int] = {}
        for h, c in cur.items():
            nxt[h] = nxt.get(h, 0) + c
            if h > 0:
                nxt[h - 1] = nxt.get(h - 1, 0) + c
            nxt[h + 1] = nxt.get(h + 1, 0) + c
        cur = nxt
    return sum(cur.values())


def factor_table_bytes(width: int) -> tuple[int, dict[str, int]]:
    low = width // 2
    high = width - 1 - low
    stride = MAXW + 2

    # LOW codes are paths that consume low symbols and return to height zero,
    # summed over all possible starting heights. This equals the number of
    # nonnegative prefixes of length low from height zero by reversal symmetry.
    low_all = nonnegative_walk_count(low, 0)
    # HIGH codes are nonnegative prefixes starting at height one.
    high_all = nonnegative_walk_count(high, 1)

    pieces = {
        "low_all_codes": low_all * 4,
        "low_mask_codes": low_all * 4,
        "low_mask_off": (1 << low) * stride * 4,
        "low_packed_rank": (3 ** low) * 4,
        "high_all_codes": high_all * 4,
        "high_mask_codes": high_all * 4,
        "high_mask_off": (1 << high) * stride * 4,
        "high_packed_rank": (1 << (2 * high)) * 4,
        "high_main_base": high_all * 8,
        "high_block_base": high_all * 8,
        "trit7": (1 << 14) * 2,
    }
    return sum(pieces.values()), pieces


def plan(width: int, gpus: int, reserve_mib: int, safety_mib: int) -> dict:
    if not 2 <= width <= MAXW:
        raise SystemExit(f"width must be in 2..{MAXW}")
    if not 1 <= gpus <= 8:
        raise SystemExit("gpus must be in 1..8")
    dp = full_dp()
    main = dp[width][1]
    blocked = dp[width - 1][1]
    main_chunk = math.ceil(main / gpus)
    blocked_chunk = math.ceil(blocked / gpus)
    auth_bytes = (main_chunk + blocked_chunk) * COUNT_BYTES
    factors, factor_pieces = factor_table_bytes(width)
    windows = forced_window_plan(width)
    max_scratch = max(x["max_scratch_bytes"] for x in windows)
    min_total = auth_bytes + factors + max_scratch + (reserve_mib + safety_mib) * (1 << 20)
    return {
        "width": width,
        "n": width - 1,
        "gpus": gpus,
        "main_states": main,
        "blocked_states": blocked,
        "main_chunk": main_chunk,
        "blocked_chunk": blocked_chunk,
        "auth_per_gpu_bytes": auth_bytes,
        "auth_per_gpu_mib": auth_bytes / (1 << 20),
        "factor_table_bytes": factors,
        "factor_table_mib": factors / (1 << 20),
        "factor_pieces_bytes": factor_pieces,
        "forced_windows": windows,
        "max_forced_scratch_bytes": max_scratch,
        "max_forced_scratch_mib": max_scratch / (1 << 20),
        "reserve_mib": reserve_mib,
        "safety_mib": safety_mib,
        "minimum_total_bytes": min_total,
        "minimum_total_mib": min_total / (1 << 20),
        "minimum_total_gib": min_total / (1 << 30),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Host-only HBM plan for row8 Grid-FP/structural certification")
    ap.add_argument("width", type=int)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--reserve-mib", type=int, default=8192)
    ap.add_argument("--safety-mib", type=int, default=1024)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    d = plan(args.width, args.gpus, args.reserve_mib, args.safety_mib)
    if args.json:
        print(json.dumps(d, indent=2, sort_keys=True))
    else:
        print(f"n={d['n']} width={d['width']} gpus={d['gpus']}")
        print(f"main_states={d['main_states']} blocked_states={d['blocked_states']}")
        print(f"auth_per_gpu_mib={d['auth_per_gpu_mib']:.3f}")
        print(f"factor_table_mib={d['factor_table_mib']:.3f}")
        for x in d["forced_windows"]:
            print(f"window={x['p_hi']}..{x['p_lo']} groups={x['groups']} max_scratch_mib={x['max_scratch_mib']:.3f}")
        print(f"reserve_mib={d['reserve_mib']} safety_mib={d['safety_mib']}")
        print(f"minimum_total_mib={d['minimum_total_mib']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
