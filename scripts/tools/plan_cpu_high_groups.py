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
    ap.add_argument("--min-margin-us", type=float, default=0.0,
                    help="require at least this much estimated saving per group")
    ap.add_argument("--gpu-target-mib", type=float, default=None,
                    help="force groups larger than the available GPU scratch target onto CPU")
    ap.add_argument("--max-groups", type=int, default=None,
                    help="optional cap; forced groups are always retained")
    ap.add_argument("--report-top", type=int, default=12)
    args = ap.parse_args()

    for name in (
        "pcie_gib_s", "cpu_gcell_s", "nn_weight", "nrnl_weight",
        "block_weight", "cross_weight",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.group_overhead_us < 0 or args.min_margin_us < 0:
        raise SystemExit("overhead and margin must be non-negative")
    if args.gpu_target_mib is not None and args.gpu_target_mib <= 0:
        raise SystemExit("--gpu-target-mib must be positive")
    if args.max_groups is not None and args.max_groups <= 0:
        raise SystemExit("--max-groups must be positive")

    costs = load_costs(args.cost_tsv)
    cpu_cell_rate = args.cpu_gcell_s * 1e9
    pcie_rate = args.pcie_gib_s * GIB
    fixed_cpu_s = args.group_overhead_us * 1e-6
    min_margin_s = args.min_margin_us * 1e-6
    gpu_target_bytes = None if args.gpu_target_mib is None else args.gpu_target_mib * MIB

    ranked: list[tuple[float, bool, float, float, float, GroupCost]] = []
    for g in costs:
        cells = g.weighted_cells(
            args.nn_weight, args.nrnl_weight, args.block_weight, args.cross_weight
        )
        cpu_s = cells / cpu_cell_rate + fixed_cpu_s
        dma_s = g.roundtrip_bytes / pcie_rate
        margin_s = dma_s - cpu_s
        forced = gpu_target_bytes is not None and g.roundtrip_bytes > gpu_target_bytes
        ranked.append((margin_s, forced, cpu_s, dma_s, cells, g))

    forced_rows = [x for x in ranked if x[1]]
    profitable = [x for x in ranked if not x[1] and x[0] >= min_margin_s]
    profitable.sort(key=lambda x: (x[0], x[5].roundtrip_bytes), reverse=True)

    selected = list(forced_rows)
    if args.max_groups is None:
        selected.extend(profitable)
    else:
        room = max(0, args.max_groups - len(forced_rows))
        selected.extend(profitable[:room])

    selected_ids = sorted({x[5].group for x in selected})
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

    print(
        "planner "
        f"groups={len(costs)} selected={len(selected_ids)} forced={len(forced_rows)} "
        f"removed_gib={selected_bytes / GIB:.6f} "
        f"removed_fraction={selected_bytes / total_bytes:.9f} "
        f"weighted_gcells={selected_cells / 1e9:.6f} "
        f"est_cpu_s={est_cpu_s:.6f} est_dma_saved_s={est_dma_saved_s:.6f} "
        f"est_sequential_margin_s={est_margin_s:.6f}",
        file=sys.stderr,
    )

    if forced_rows and any(x[0] < 0 for x in forced_rows):
        bad = sum(1 for x in forced_rows if x[0] < 0)
        print(
            f"warning: {bad} GPU-fit-forced groups have negative modeled CPU margin",
            file=sys.stderr,
        )

    best = sorted(ranked, key=lambda x: x[0], reverse=True)[: max(0, args.report_top)]
    for margin_s, forced, cpu_s, dma_s, cells, g in best:
        print(
            "top "
            f"group={g.group} mask={g.mask} forced={int(forced)} "
            f"roundtrip_mib={g.roundtrip_bytes / MIB:.6f} "
            f"weighted_mcells={cells / 1e6:.6f} "
            f"cpu_us={cpu_s * 1e6:.3f} dma_us={dma_s * 1e6:.3f} "
            f"margin_us={margin_s * 1e6:.3f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
