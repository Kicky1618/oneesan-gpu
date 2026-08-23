#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

COUNT_BYTES = 4
MAXW = 28
MIB = 1 << 20
TIB = 1 << 40


@dataclass(frozen=True)
class Group:
    main_states: int
    blocked_states: int

    @property
    def roundtrip_bytes(self) -> int:
        # One HIGH pass loads main+blocked and writes them back once.
        return 2 * (self.main_states + self.blocked_states) * COUNT_BYTES


def make_spec_size(width: int, fixed: int, occ: int) -> int:
    dp = [[0] * (MAXW + 2) for _ in range(width + 1)]
    dp[0][0] = 1
    for w in range(1, width + 1):
        pos = w - 1
        is_fixed = (fixed >> pos) & 1
        is_occ = (occ >> pos) & 1
        prev = dp[w - 1]
        cur = dp[w]
        for h in range(MAXW + 1):
            x = 0
            if not is_fixed or not is_occ:
                x += prev[h]
            if not is_fixed or is_occ:
                if h > 0:
                    x += prev[h - 1]
                if h < MAXW + 1:
                    x += prev[h + 1]
            cur[h] = x
    return dp[width][1]


def window_candidates(width: int, hi: int, lo: int) -> list[int]:
    return [q for q in range(width - 1, -1, -1) if q < lo - 1 or q > hi]


def window_masks(
    width: int, hi: int, lo: int, fixed_pos: list[int], group: int
) -> tuple[int, int, int, int]:
    del width
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


def build_high_groups(n: int, low_lut_k: int) -> tuple[int, int, list[Group]]:
    width = n + 1
    high_lut_k = width - low_lut_k - 1
    if n < 2 or width > MAXW:
        raise ValueError(f"n must be in [2, {MAXW - 1}]")
    if low_lut_k <= 0 or high_lut_k <= 0:
        raise ValueError("LOW_LUT_K and HIGH_LUT_K must be positive")

    hi = width - 1
    lo = low_lut_k + 1
    fixed_pos = window_candidates(width, hi, lo)
    groups: list[Group] = []
    for g in range(1 << len(fixed_pos)):
        mf, mo, bf, bo = window_masks(width, hi, lo, fixed_pos, g)
        groups.append(
            Group(
                make_spec_size(width, mf, mo),
                make_spec_size(width - 1, bf, bo),
            )
        )
    return width, high_lut_k, groups


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    x = q * (len(sorted_values) - 1)
    a = math.floor(x)
    b = math.ceil(x)
    if a == b:
        return float(sorted_values[a])
    f = x - a
    return sorted_values[a] * (1.0 - f) + sorted_values[b] * f


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile forced-two-window HIGH group PCIe transfer sizes."
    )
    parser.add_argument("n", type=int, nargs="?", default=27)
    parser.add_argument("--low-lut-k", type=int, default=None)
    parser.add_argument(
        "--threshold-mib",
        type=float,
        action="append",
        default=None,
        help="CPU-offload threshold; may be specified multiple times",
    )
    args = parser.parse_args()

    width = args.n + 1
    low_lut_k = args.low_lut_k if args.low_lut_k is not None else width // 2
    width, high_lut_k, groups = build_high_groups(args.n, low_lut_k)

    main_sum = sum(g.main_states for g in groups)
    blocked_sum = sum(g.blocked_states for g in groups)
    transfers = sorted(g.roundtrip_bytes for g in groups if g.roundtrip_bytes)
    total_per_row = sum(transfers)
    total_per_residue = total_per_row * width

    # Independent partition check: fixed LOW occupancy groups must exactly cover
    # the authoritative main and blocked state spaces once per HIGH pass.
    full_main = make_spec_size(width, 0, 0)
    full_blocked = make_spec_size(width - 1, 0, 0)
    if (main_sum, blocked_sum) != (full_main, full_blocked):
        raise RuntimeError(
            "HIGH groups do not partition authoritative state space: "
            f"main={main_sum}/{full_main} blocked={blocked_sum}/{full_blocked}"
        )

    print(
        f"n={args.n} width={width} low_lut_k={low_lut_k} "
        f"high_lut_k={high_lut_k} groups={len(groups)}"
    )
    print(f"main_states={main_sum} blocked_states={blocked_sum}")
    print(
        f"high_roundtrip_tib_per_row={total_per_row / TIB:.9f} "
        f"high_roundtrip_tib_per_residue={total_per_residue / TIB:.9f}"
    )

    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"group_p{int(q * 100):02d}_mib={percentile(transfers, q) / MIB:.6f}")
    print(f"group_min_mib={transfers[0] / MIB:.6f}")
    print(f"group_max_mib={transfers[-1] / MIB:.6f}")

    thresholds = args.threshold_mib or [4, 16, 64, 256, 1024, 4096]
    for threshold_mib in thresholds:
        threshold_bytes = threshold_mib * MIB
        selected = [x for x in transfers if x <= threshold_bytes]
        selected_bytes = sum(selected)
        group_frac = len(selected) / len(transfers)
        transfer_frac = selected_bytes / total_per_row
        saved_tib = total_per_residue * transfer_frac / TIB
        print(
            f"threshold_mib={threshold_mib:g} "
            f"cpu_groups={len(selected)} "
            f"cpu_group_fraction={group_frac:.9f} "
            f"pcie_fraction_removed={transfer_frac:.9f} "
            f"pcie_tib_removed_per_residue={saved_tib:.9f}"
        )

    desc = sorted(transfers, reverse=True)
    for frac in (0.01, 0.05, 0.10, 0.20):
        count = max(1, int(len(desc) * frac))
        share = sum(desc[:count]) / total_per_row
        print(
            f"largest_group_fraction={frac:.2f} count={count} "
            f"transfer_fraction={share:.9f}"
        )


if __name__ == "__main__":
    main()
