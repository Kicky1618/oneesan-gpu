#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

GIB = 1 << 30
MIB = 1 << 20
TIB = 1 << 40


@dataclass
class Sample:
    mode: str
    overlap: int
    threshold_mib: float
    wall_s: float
    dma_s: float
    cpu_high_wall_s: float
    cpu_low_wall_s: float
    removed_tib: float
    remaining_tib: float
    cpu_high_groups: int


@dataclass(frozen=True)
class CostGroup:
    group: int
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


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else math.nan


def load(path: str) -> list[Sample]:
    out: list[Sample] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "mode", "threshold_mib", "wall_s", "h2d_s", "d2h_s",
            "cpu_high_wall_s", "cpu_low_wall_s", "pcie_removed_tib",
            "pcie_remaining_tib", "cpu_high_groups",
        }
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing TSV columns: {', '.join(sorted(missing))}")

        overlap_column = "overlap" if "overlap" in (rd.fieldnames or []) else None
        for row in rd:
            out.append(
                Sample(
                    mode=row["mode"],
                    overlap=int(row[overlap_column]) if overlap_column else 0,
                    threshold_mib=float(row["threshold_mib"]),
                    wall_s=float(row["wall_s"]),
                    dma_s=float(row["h2d_s"]) + float(row["d2h_s"]),
                    cpu_high_wall_s=float(row["cpu_high_wall_s"]),
                    cpu_low_wall_s=float(row["cpu_low_wall_s"]),
                    removed_tib=float(row["pcie_removed_tib"]),
                    remaining_tib=float(row["pcie_remaining_tib"]),
                    cpu_high_groups=int(row["cpu_high_groups"]),
                )
            )
    if not out:
        raise SystemExit("TSV contains no samples")
    return out


def load_costs(path: str) -> list[CostGroup]:
    out: list[CostGroup] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "group", "roundtrip_bytes", "nn_cells", "nrnl_cells",
            "block_closure_cells", "cross_closure_cells",
        }
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing cost columns: {', '.join(sorted(missing))}")
        for row in rd:
            out.append(
                CostGroup(
                    group=int(row["group"]),
                    roundtrip_bytes=int(row["roundtrip_bytes"]),
                    nn_cells=int(row["nn_cells"]),
                    nrnl_cells=int(row["nrnl_cells"]),
                    block_cells=int(row["block_closure_cells"]),
                    cross_cells=int(row["cross_closure_cells"]),
                )
            )
    if not out:
        raise SystemExit("cost TSV contains no groups")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze CPU HIGH threshold sweeps, measure DMA/offload break-even, "
            "and optionally calibrate the exact per-group direct cost model."
        )
    )
    ap.add_argument("tsv")
    ap.add_argument("--cost-plan", default=None,
                    help="TSV from ramstream32_cpu_high_cost_plan")
    ap.add_argument("--nn-weight", type=float, default=1.0)
    ap.add_argument("--nrnl-weight", type=float, default=1.0)
    ap.add_argument("--block-weight", type=float, default=1.0)
    ap.add_argument("--cross-weight", type=float, default=1.0)
    args = ap.parse_args()

    for name in ("nn_weight", "nrnl_weight", "block_weight", "cross_weight"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    samples = load(args.tsv)
    costs = load_costs(args.cost_plan) if args.cost_plan else None
    grouped: dict[tuple[str, int, float], list[Sample]] = defaultdict(list)
    for s in samples:
        grouped[(s.mode, s.overlap, s.threshold_mib)].append(s)

    configs = sorted({(s.mode, s.overlap) for s in samples})
    for mode, overlap in configs:
        rows = {
            threshold: group
            for (m, o, threshold), group in grouped.items()
            if m == mode and o == overlap
        }
        if not rows:
            continue
        baseline_threshold = 0.0 if 0.0 in rows else min(rows)
        baseline = rows[baseline_threshold]
        baseline_dma = mean([s.dma_s for s in baseline])
        baseline_wall = mean([s.wall_s for s in baseline])

        print(
            f"config mode={mode} overlap={overlap} "
            f"baseline_threshold_mib={baseline_threshold:g} "
            f"baseline_wall_s={baseline_wall:.9f} baseline_dma_s={baseline_dma:.9f}"
        )

        best_threshold = baseline_threshold
        best_wall = baseline_wall
        pcie_rates: list[float] = []
        cpu_rates: list[float] = []

        for threshold in sorted(rows):
            group = rows[threshold]
            wall = mean([s.wall_s for s in group])
            dma = mean([s.dma_s for s in group])
            cpu_high = mean([s.cpu_high_wall_s for s in group])
            cpu_low = mean([s.cpu_low_wall_s for s in group])
            removed = mean([s.removed_tib for s in group])
            remaining = mean([s.remaining_tib for s in group])
            cpu_groups = mean([float(s.cpu_high_groups) for s in group])

            dma_saved = baseline_dma - dma
            wall_saved = baseline_wall - wall
            dma_saved_per_tib = dma_saved / removed if removed > 0 else 0.0
            cpu_cost_per_tib = cpu_high / removed if removed > 0 else 0.0
            sequential_margin = dma_saved - cpu_high
            efficiency = dma_saved / cpu_high if cpu_high > 0 else math.inf

            extra = ""
            if removed > 0 and dma_saved > 0:
                measured_pcie_gib_s = removed * TIB / GIB / dma_saved
                pcie_rates.append(measured_pcie_gib_s)
                extra += f" measured_pcie_gib_s={measured_pcie_gib_s:.9f}"

            if costs is not None and mode == "direct" and threshold > 0 and cpu_high > 0:
                limit_bytes = int(threshold * MIB)
                selected_costs = [g for g in costs if g.roundtrip_bytes <= limit_bytes]
                selected_bytes_per_row = sum(g.roundtrip_bytes for g in selected_costs)
                selected_cells_per_row = sum(
                    g.weighted_cells(
                        args.nn_weight, args.nrnl_weight,
                        args.block_weight, args.cross_weight,
                    )
                    for g in selected_costs
                )
                if selected_bytes_per_row > 0 and removed > 0:
                    rows_est = removed * TIB / selected_bytes_per_row
                    total_cells = selected_cells_per_row * rows_est
                    measured_cpu_gcell_s = total_cells / cpu_high / 1e9
                    if math.isfinite(measured_cpu_gcell_s) and measured_cpu_gcell_s > 0:
                        cpu_rates.append(measured_cpu_gcell_s)
                        extra += (
                            f" model_rows={rows_est:.6f}"
                            f" measured_cpu_gcell_s={measured_cpu_gcell_s:.9f}"
                        )

            print(
                f"threshold_mib={threshold:g} runs={len(group)} "
                f"mean_wall_s={wall:.9f} wall_saved_s={wall_saved:.9f} "
                f"mean_dma_s={dma:.9f} dma_saved_s={dma_saved:.9f} "
                f"mean_cpu_high_wall_s={cpu_high:.9f} "
                f"mean_cpu_low_wall_s={cpu_low:.9f} "
                f"removed_tib={removed:.9f} remaining_tib={remaining:.9f} "
                f"cpu_high_groups={cpu_groups:.3f} "
                f"dma_saved_s_per_tib={dma_saved_per_tib:.9f} "
                f"cpu_cost_s_per_tib={cpu_cost_per_tib:.9f} "
                f"sequential_margin_s={sequential_margin:.9f} "
                f"offload_efficiency={efficiency:.9f}{extra}"
            )
            if wall < best_wall:
                best_wall = wall
                best_threshold = threshold

        print(
            f"best mode={mode} overlap={overlap} "
            f"threshold_mib={best_threshold:g} mean_wall_s={best_wall:.9f} "
            f"speedup_vs_baseline={baseline_wall / best_wall:.9f}x"
        )

        if pcie_rates:
            pcie_med = statistics.median(pcie_rates)
            print(f"calibration mode={mode} overlap={overlap} pcie_gib_s={pcie_med:.9f}")
        else:
            pcie_med = math.nan
        if cpu_rates:
            cpu_med = statistics.median(cpu_rates)
            print(f"calibration mode={mode} overlap={overlap} cpu_gcell_s={cpu_med:.9f}")
        else:
            cpu_med = math.nan

        if costs is not None and mode == "direct" and math.isfinite(pcie_med) and math.isfinite(cpu_med):
            overlap_arg = " --overlap" if overlap else ""
            print(
                "planner_command "
                f"python3 scripts/tools/plan_cpu_high_groups.py {args.cost_plan} "
                f"--pcie-gib-s {pcie_med:.9f} --cpu-gcell-s {cpu_med:.9f} "
                f"--nn-weight {args.nn_weight:g} --nrnl-weight {args.nrnl_weight:g} "
                f"--block-weight {args.block_weight:g} --cross-weight {args.cross_weight:g}"
                f"{overlap_arg}"
            )


if __name__ == "__main__":
    main()
