#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass

GIB = 1 << 30
MIB = 1 << 20


@dataclass(frozen=True)
class GroupCost:
    group: int
    mask: int
    roundtrip_bytes: int
    nn_cells: int
    nrnl_cells: int
    block_cells: int
    cross_cells: int

    def weighted_cells(
        self, nn_weight: float, nrnl_weight: float,
        block_weight: float, cross_weight: float,
    ) -> float:
        return (
            self.nn_cells * nn_weight
            + self.nrnl_cells * nrnl_weight
            + self.block_cells * block_weight
            + self.cross_cells * cross_weight
        )


@dataclass(frozen=True)
class Candidate:
    group: GroupCost
    forced: bool
    cpu_s: float
    dma_s: float
    cells: float

    @property
    def margin_s(self) -> float:
        return self.dma_s - self.cpu_s

    @property
    def efficiency(self) -> float:
        if self.cpu_s == 0:
            return math.inf
        return self.dma_s / self.cpu_s


def load_costs(path: str) -> list[GroupCost]:
    out: list[GroupCost] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "group", "mask", "roundtrip_bytes", "nn_cells", "nrnl_cells",
            "block_closure_cells", "cross_closure_cells",
        }
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing cost columns: {', '.join(sorted(missing))}")
        for row in rd:
            out.append(
                GroupCost(
                    group=int(row["group"]),
                    mask=int(row["mask"]),
                    roundtrip_bytes=int(row["roundtrip_bytes"]),
                    nn_cells=int(row["nn_cells"]),
                    nrnl_cells=int(row["nrnl_cells"]),
                    block_cells=int(row["block_closure_cells"]),
                    cross_cells=int(row["cross_closure_cells"]),
                )
            )
    if not out:
        raise SystemExit("cost TSV contains no groups")
    seen = {g.group for g in out}
    if len(seen) != len(out):
        raise SystemExit("duplicate group IDs in cost TSV")
    return out


def select_sequential(
    candidates: list[Candidate], min_margin_s: float,
    max_groups: int | None,
) -> list[Candidate]:
    forced = [x for x in candidates if x.forced]
    optional = [
        x for x in candidates
        if not x.forced and x.margin_s >= min_margin_s
    ]
    optional.sort(
        key=lambda x: (x.margin_s, x.group.roundtrip_bytes), reverse=True
    )
    if max_groups is None:
        return forced + optional
    room = max(0, max_groups - len(forced))
    return forced + optional[:room]


def select_overlap(
    candidates: list[Candidate], min_margin_s: float | None,
    max_groups: int | None,
) -> tuple[list[Candidate], float, float, float]:
    """Greedy fractional-knapsack relaxation for the overlapped HIGH critical path.

    The modeled HIGH phase is

        max(sum(cpu_i for selected),
            sum(dma_i for unselected)).

    Forced groups are always on CPU. Optional groups are ordered by dma_i/cpu_i,
    then every prefix is evaluated. This is exact for the fractional relaxation
    and a deterministic O(n log n) approximation to the discrete partition.
    """
    forced = [x for x in candidates if x.forced]
    optional = [x for x in candidates if not x.forced]
    if min_margin_s is not None:
        optional = [x for x in optional if x.margin_s >= min_margin_s]
    optional.sort(
        key=lambda x: (x.efficiency, x.dma_s, -x.cpu_s), reverse=True
    )

    if max_groups is not None:
        optional = optional[: max(0, max_groups - len(forced))]

    total_dma_s = sum(x.dma_s for x in candidates)
    cpu_s = sum(x.cpu_s for x in forced)
    saved_dma_s = sum(x.dma_s for x in forced)
    best_k = 0
    best_cpu_s = cpu_s
    best_remaining_dma_s = max(0.0, total_dma_s - saved_dma_s)
    best_critical_s = max(best_cpu_s, best_remaining_dma_s)

    for k, x in enumerate(optional, 1):
        cpu_s += x.cpu_s
        saved_dma_s += x.dma_s
        remaining_dma_s = max(0.0, total_dma_s - saved_dma_s)
        critical_s = max(cpu_s, remaining_dma_s)
        if critical_s < best_critical_s:
            best_k = k
            best_cpu_s = cpu_s
            best_remaining_dma_s = remaining_dma_s
            best_critical_s = critical_s

    return (
        forced + optional[:best_k],
        best_cpu_s,
        best_remaining_dma_s,
        best_critical_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Select CPU HIGH occupancy groups using an explicit DMA-vs-CPU cost model. "
            "The output is directly consumable as CPU_HIGH_GROUPS_FILE."
        )
    )
    ap.add_argument("cost_tsv")
    ap.add_argument("--pcie-gib-s", type=float, required=True,
                    help="measured aggregate H2D+D2H throughput in GiB/s")
    ap.add_argument("--cpu-gcell-s", type=float, required=True,
                    help="measured aggregate direct-executor throughput in weighted Gcells/s")
    ap.add_argument("--nn-weight", type=float, default=1.0)
    ap.add_argument("--nrnl-weight", type=float, default=1.0)
    ap.add_argument("--block-weight", type=float, default=1.0)
    ap.add_argument("--cross-weight", type=float, default=1.0)
    ap.add_argument("--group-overhead-us", type=float, default=0.0,
                    help="per-group CPU scheduling/fixed overhead")
    ap.add_argument("--min-margin-us", type=float, default=None,
                    help=(
                        "optional per-group sequential margin filter. Default is 0 us "
                        "for sequential mode and no filter for --overlap"
                    ))
    ap.add_argument("--gpu-target-mib", type=float, default=None,
                    help="force groups larger than the available GPU scratch target onto CPU")
    ap.add_argument("--max-groups", type=int, default=None,
                    help="optional cap; forced groups are always retained")
    ap.add_argument("--overlap", action="store_true",
                    help=(
                        "optimize max(CPU-HIGH time, remaining DMA time) instead of "
                        "independent sequential margins"
                    ))
    ap.add_argument("--report-top", type=int, default=12)
    args = ap.parse_args()

    for name in (
        "pcie_gib_s", "cpu_gcell_s", "nn_weight", "nrnl_weight",
        "block_weight", "cross_weight",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.group_overhead_us < 0:
        raise SystemExit("--group-overhead-us must be non-negative")
    if args.min_margin_us is not None and args.min_margin_us < 0:
        raise SystemExit("--min-margin-us must be non-negative")
    if args.gpu_target_mib is not None and args.gpu_target_mib <= 0:
        raise SystemExit("--gpu-target-mib must be positive")
    if args.max_groups is not None and args.max_groups <= 0:
        raise SystemExit("--max-groups must be positive")

    costs = load_costs(args.cost_tsv)
    cpu_cell_rate = args.cpu_gcell_s * 1e9
    pcie_rate = args.pcie_gib_s * GIB
    fixed_cpu_s = args.group_overhead_us * 1e-6
    min_margin_s = (
        None if args.min_margin_us is None else args.min_margin_us * 1e-6
    )
    gpu_target_bytes = None if args.gpu_target_mib is None else args.gpu_target_mib * MIB

    candidates: list[Candidate] = []
    for g in costs:
        cells = g.weighted_cells(
            args.nn_weight, args.nrnl_weight, args.block_weight, args.cross_weight
        )
        cpu_s = cells / cpu_cell_rate + fixed_cpu_s
        dma_s = g.roundtrip_bytes / pcie_rate
        forced = gpu_target_bytes is not None and g.roundtrip_bytes > gpu_target_bytes
        candidates.append(Candidate(g, forced, cpu_s, dma_s, cells))

    forced_rows = [x for x in candidates if x.forced]
    total_dma_s = sum(x.dma_s for x in candidates)

    if args.overlap:
        selected, overlap_cpu_s, overlap_remaining_dma_s, overlap_critical_s = select_overlap(
            candidates, min_margin_s, args.max_groups
        )
    else:
        selected = select_sequential(
            candidates, 0.0 if min_margin_s is None else min_margin_s,
            args.max_groups,
        )
        overlap_cpu_s = overlap_remaining_dma_s = overlap_critical_s = math.nan

    selected_ids = sorted({x.group.group for x in selected})
    selected_set = set(selected_ids)

    for group in selected_ids:
        print(group)

    total_bytes = sum(g.roundtrip_bytes for g in costs)
    selected_bytes = sum(g.roundtrip_bytes for g in costs if g.group in selected_set)
    selected_cells = sum(
        g.weighted_cells(args.nn_weight, args.nrnl_weight, args.block_weight, args.cross_weight)
        for g in costs if g.group in selected_set
    )
    est_cpu_s = selected_cells / cpu_cell_rate + len(selected_ids) * fixed_cpu_s
    est_dma_saved_s = selected_bytes / pcie_rate
    est_margin_s = est_dma_saved_s - est_cpu_s
    est_remaining_dma_s = max(0.0, total_dma_s - est_dma_saved_s)

    print(
        "planner "
        f"mode={'overlap' if args.overlap else 'sequential'} "
        f"groups={len(costs)} selected={len(selected_ids)} forced={len(forced_rows)} "
        f"removed_gib={selected_bytes / GIB:.6f} "
        f"removed_fraction={selected_bytes / total_bytes:.9f} "
        f"weighted_gcells={selected_cells / 1e9:.6f} "
        f"est_cpu_s={est_cpu_s:.6f} est_dma_saved_s={est_dma_saved_s:.6f} "
        f"est_remaining_dma_s={est_remaining_dma_s:.6f} "
        f"est_sequential_margin_s={est_margin_s:.6f}",
        file=sys.stderr,
    )
    if args.overlap:
        print(
            "overlap_model "
            f"baseline_dma_s={total_dma_s:.6f} "
            f"cpu_s={overlap_cpu_s:.6f} "
            f"remaining_dma_s={overlap_remaining_dma_s:.6f} "
            f"critical_s={overlap_critical_s:.6f} "
            f"modeled_speedup={total_dma_s / overlap_critical_s if overlap_critical_s else math.inf:.9f}x",
            file=sys.stderr,
        )

    if forced_rows and any(x.margin_s < 0 for x in forced_rows):
        bad = sum(1 for x in forced_rows if x.margin_s < 0)
        print(
            f"warning: {bad} GPU-fit-forced groups have negative modeled CPU margin",
            file=sys.stderr,
        )

    if args.overlap:
        best = sorted(
            candidates, key=lambda x: (x.efficiency, x.dma_s), reverse=True
        )[: max(0, args.report_top)]
    else:
        best = sorted(
            candidates, key=lambda x: x.margin_s, reverse=True
        )[: max(0, args.report_top)]

    for x in best:
        print(
            "top "
            f"group={x.group.group} mask={x.group.mask} forced={int(x.forced)} "
            f"roundtrip_mib={x.group.roundtrip_bytes / MIB:.6f} "
            f"weighted_mcells={x.cells / 1e6:.6f} "
            f"cpu_us={x.cpu_s * 1e6:.3f} dma_us={x.dma_s * 1e6:.3f} "
            f"margin_us={x.margin_s * 1e6:.3f} "
            f"efficiency={x.efficiency:.9f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
