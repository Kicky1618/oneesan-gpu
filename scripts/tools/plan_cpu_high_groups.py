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
    pcie_copy_calls: int
    gpu_state_steps: int
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
    gpu_kernel_s: float
    cells: float

    @property
    def gpu_saved_s(self) -> float:
        return self.dma_s + self.gpu_kernel_s

    @property
    def margin_s(self) -> float:
        return self.gpu_saved_s - self.cpu_s

    @property
    def efficiency(self) -> float:
        if self.cpu_s == 0:
            return math.inf
        return self.gpu_saved_s / self.cpu_s


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
        fields = rd.fieldnames or []
        has_gpu_steps = "gpu_state_steps" in fields
        has_copy_calls = "pcie_copy_calls" in fields
        for row in rd:
            out.append(
                GroupCost(
                    group=int(row["group"]),
                    mask=int(row["mask"]),
                    roundtrip_bytes=int(row["roundtrip_bytes"]),
                    pcie_copy_calls=int(row["pcie_copy_calls"]) if has_copy_calls else 0,
                    gpu_state_steps=int(row["gpu_state_steps"]) if has_gpu_steps else 0,
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
    """Approximate min max(CPU selected, GPU unselected) by ratio-prefix search."""
    forced = [x for x in candidates if x.forced]
    optional = [x for x in candidates if not x.forced]
    if min_margin_s is not None:
        optional = [x for x in optional if x.margin_s >= min_margin_s]
    optional.sort(
        key=lambda x: (x.efficiency, x.gpu_saved_s, -x.cpu_s), reverse=True
    )

    if max_groups is not None:
        optional = optional[: max(0, max_groups - len(forced))]

    total_gpu_s = sum(x.gpu_saved_s for x in candidates)
    cpu_s = sum(x.cpu_s for x in forced)
    saved_gpu_s = sum(x.gpu_saved_s for x in forced)
    best_k = 0
    best_cpu_s = cpu_s
    best_remaining_gpu_s = max(0.0, total_gpu_s - saved_gpu_s)
    best_critical_s = max(best_cpu_s, best_remaining_gpu_s)

    for k, x in enumerate(optional, 1):
        cpu_s += x.cpu_s
        saved_gpu_s += x.gpu_saved_s
        remaining_gpu_s = max(0.0, total_gpu_s - saved_gpu_s)
        critical_s = max(cpu_s, remaining_gpu_s)
        if critical_s < best_critical_s:
            best_k = k
            best_cpu_s = cpu_s
            best_remaining_gpu_s = remaining_gpu_s
            best_critical_s = critical_s

    return (
        forced + optional[:best_k],
        best_cpu_s,
        best_remaining_gpu_s,
        best_critical_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Select CPU HIGH occupancy groups using measured PCIe, GPU-HIGH, and "
            "CPU-direct cost models. Output is CPU_HIGH_GROUPS_FILE compatible."
        )
    )
    ap.add_argument("cost_tsv")
    ap.add_argument("--pcie-gib-s", type=float, required=True,
                    help="measured aggregate H2D+D2H payload throughput in GiB/s")
    ap.add_argument("--pcie-copy-overhead-us", type=float, default=0.0,
                    help="fixed cost per H2D/D2H factor-slice copy call")
    ap.add_argument("--cpu-gcell-s", type=float, required=True,
                    help="measured aggregate direct-executor throughput in weighted Gcells/s")
    ap.add_argument("--gpu-gstate-s", type=float, default=None,
                    help="measured GPU HIGH throughput in billion state-steps/s")
    ap.add_argument("--nn-weight", type=float, default=1.0)
    ap.add_argument("--nrnl-weight", type=float, default=1.0)
    ap.add_argument("--block-weight", type=float, default=1.0)
    ap.add_argument("--cross-weight", type=float, default=1.0)
    ap.add_argument("--group-overhead-us", type=float, default=0.0,
                    help="per-group CPU scheduling/fixed overhead")
    ap.add_argument("--gpu-group-overhead-us", type=float, default=0.0,
                    help="per-group GPU HIGH launch/fixed overhead")
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
                    help="minimize max(CPU-HIGH, remaining GPU-HIGH) instead of sequential sum")
    ap.add_argument("--report-top", type=int, default=12)
    args = ap.parse_args()

    for name in (
        "pcie_gib_s", "cpu_gcell_s", "nn_weight", "nrnl_weight",
        "block_weight", "cross_weight",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.gpu_gstate_s is not None and args.gpu_gstate_s <= 0:
        raise SystemExit("--gpu-gstate-s must be positive")
    if args.pcie_copy_overhead_us < 0:
        raise SystemExit("--pcie-copy-overhead-us must be non-negative")
    if args.group_overhead_us < 0 or args.gpu_group_overhead_us < 0:
        raise SystemExit("group overheads must be non-negative")
    if args.min_margin_us is not None and args.min_margin_us < 0:
        raise SystemExit("--min-margin-us must be non-negative")
    if args.gpu_target_mib is not None and args.gpu_target_mib <= 0:
        raise SystemExit("--gpu-target-mib must be positive")
    if args.max_groups is not None and args.max_groups <= 0:
        raise SystemExit("--max-groups must be positive")

    costs = load_costs(args.cost_tsv)
    cpu_cell_rate = args.cpu_gcell_s * 1e9
    pcie_rate = args.pcie_gib_s * GIB
    gpu_state_rate = None if args.gpu_gstate_s is None else args.gpu_gstate_s * 1e9
    fixed_pcie_copy_s = args.pcie_copy_overhead_us * 1e-6
    fixed_cpu_s = args.group_overhead_us * 1e-6
    fixed_gpu_s = args.gpu_group_overhead_us * 1e-6
    min_margin_s = None if args.min_margin_us is None else args.min_margin_us * 1e-6
    gpu_target_bytes = None if args.gpu_target_mib is None else args.gpu_target_mib * MIB

    if gpu_state_rate is not None and not any(g.gpu_state_steps for g in costs):
        raise SystemExit("--gpu-gstate-s requires gpu_state_steps in the cost plan")
    if fixed_pcie_copy_s > 0 and not any(g.pcie_copy_calls for g in costs):
        raise SystemExit("--pcie-copy-overhead-us requires pcie_copy_calls in the cost plan")

    candidates: list[Candidate] = []
    for g in costs:
        if g.roundtrip_bytes <= 0:
            continue
        cells = g.weighted_cells(
            args.nn_weight, args.nrnl_weight, args.block_weight, args.cross_weight
        )
        cpu_s = cells / cpu_cell_rate + fixed_cpu_s
        dma_s = g.roundtrip_bytes / pcie_rate + g.pcie_copy_calls * fixed_pcie_copy_s
        gpu_kernel_s = fixed_gpu_s
        if gpu_state_rate is not None:
            gpu_kernel_s += g.gpu_state_steps / gpu_state_rate
        forced = gpu_target_bytes is not None and g.roundtrip_bytes > gpu_target_bytes
        candidates.append(Candidate(g, forced, cpu_s, dma_s, gpu_kernel_s, cells))

    if not candidates:
        return

    forced_rows = [x for x in candidates if x.forced]
    total_gpu_s = sum(x.gpu_saved_s for x in candidates)

    if args.overlap:
        selected, overlap_cpu_s, overlap_remaining_gpu_s, overlap_critical_s = select_overlap(
            candidates, min_margin_s, args.max_groups
        )
    else:
        selected = select_sequential(
            candidates, 0.0 if min_margin_s is None else min_margin_s,
            args.max_groups,
        )
        overlap_cpu_s = overlap_remaining_gpu_s = overlap_critical_s = math.nan

    selected_ids = sorted({x.group.group for x in selected})

    for group in selected_ids:
        print(group)

    total_bytes = sum(x.group.roundtrip_bytes for x in candidates)
    selected_bytes = sum(x.group.roundtrip_bytes for x in selected)
    selected_cells = sum(x.cells for x in selected)
    selected_gpu_steps = sum(x.group.gpu_state_steps for x in selected)
    selected_copy_calls = sum(x.group.pcie_copy_calls for x in selected)
    est_cpu_s = sum(x.cpu_s for x in selected)
    est_dma_saved_s = sum(x.dma_s for x in selected)
    est_kernel_saved_s = sum(x.gpu_kernel_s for x in selected)
    est_gpu_saved_s = est_dma_saved_s + est_kernel_saved_s
    est_margin_s = est_gpu_saved_s - est_cpu_s
    est_remaining_gpu_s = max(0.0, total_gpu_s - est_gpu_saved_s)

    print(
        "planner "
        f"mode={'overlap' if args.overlap else 'sequential'} "
        f"groups={len(candidates)} selected={len(selected_ids)} forced={len(forced_rows)} "
        f"removed_gib={selected_bytes / GIB:.6f} "
        f"removed_fraction={selected_bytes / total_bytes:.9f} "
        f"pcie_copy_calls={selected_copy_calls} "
        f"weighted_gcells={selected_cells / 1e9:.6f} "
        f"gpu_gstate_steps={selected_gpu_steps / 1e9:.6f} "
        f"pcie_copy_overhead_us={args.pcie_copy_overhead_us:.6f} "
        f"cpu_group_overhead_us={args.group_overhead_us:.6f} "
        f"gpu_group_overhead_us={args.gpu_group_overhead_us:.6f} "
        f"est_cpu_s={est_cpu_s:.6f} est_dma_saved_s={est_dma_saved_s:.6f} "
        f"est_gpu_kernel_saved_s={est_kernel_saved_s:.6f} "
        f"est_gpu_saved_s={est_gpu_saved_s:.6f} "
        f"est_remaining_gpu_s={est_remaining_gpu_s:.6f} "
        f"est_sequential_margin_s={est_margin_s:.6f}",
        file=sys.stderr,
    )
    if args.overlap:
        print(
            "overlap_model "
            f"baseline_gpu_s={total_gpu_s:.6f} "
            f"cpu_s={overlap_cpu_s:.6f} "
            f"remaining_gpu_s={overlap_remaining_gpu_s:.6f} "
            f"critical_s={overlap_critical_s:.6f} "
            f"modeled_speedup={total_gpu_s / overlap_critical_s if overlap_critical_s else math.inf:.9f}x",
            file=sys.stderr,
        )

    if forced_rows and any(x.margin_s < 0 for x in forced_rows):
        bad = sum(1 for x in forced_rows if x.margin_s < 0)
        print(
            f"warning: {bad} GPU-fit-forced groups have negative modeled CPU margin",
            file=sys.stderr,
        )

    if args.overlap:
        best = sorted(candidates, key=lambda x: (x.efficiency, x.gpu_saved_s), reverse=True)
    else:
        best = sorted(candidates, key=lambda x: x.margin_s, reverse=True)
    best = best[: max(0, args.report_top)]

    for x in best:
        print(
            "top "
            f"group={x.group.group} mask={x.group.mask} forced={int(x.forced)} "
            f"roundtrip_mib={x.group.roundtrip_bytes / MIB:.6f} "
            f"pcie_copy_calls={x.group.pcie_copy_calls} "
            f"weighted_mcells={x.cells / 1e6:.6f} "
            f"gpu_mstate_steps={x.group.gpu_state_steps / 1e6:.6f} "
            f"cpu_us={x.cpu_s * 1e6:.3f} dma_us={x.dma_s * 1e6:.3f} "
            f"gpu_kernel_us={x.gpu_kernel_s * 1e6:.3f} "
            f"margin_us={x.margin_s * 1e6:.3f} "
            f"efficiency={x.efficiency:.9f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
