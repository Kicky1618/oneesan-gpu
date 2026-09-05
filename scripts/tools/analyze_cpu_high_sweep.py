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
    gpu_kernel_s: float
    cpu_high_wall_s: float
    cpu_low_wall_s: float
    removed_tib: float
    remaining_tib: float
    cpu_high_groups: int


@dataclass(frozen=True)
class CostGroup:
    group: int
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


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else math.nan


def nnls_two_predictors(rows: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """Fit y ~= a*x + b*g with a,b >= 0 and no intercept."""
    if len(rows) < 2:
        return None
    sxx = sum(x * x for x, _, _ in rows)
    sgg = sum(g * g for _, g, _ in rows)
    sxg = sum(x * g for x, g, _ in rows)
    sxy = sum(x * y for x, _, y in rows)
    sgy = sum(g * y for _, g, y in rows)

    candidates: list[tuple[float, float]] = []
    det = sxx * sgg - sxg * sxg
    scale = max(1.0, sxx * sgg)
    if det > 1e-12 * scale:
        a = (sxy * sgg - sgy * sxg) / det
        b = (sgy * sxx - sxy * sxg) / det
        if a >= 0.0 and b >= 0.0:
            candidates.append((a, b))
    if sxx > 0.0:
        candidates.append((max(0.0, sxy / sxx), 0.0))
    if sgg > 0.0:
        candidates.append((0.0, max(0.0, sgy / sgg)))
    if not candidates:
        return None

    def sse(ab: tuple[float, float]) -> float:
        a, b = ab
        return sum((y - a * x - b * g) ** 2 for x, g, y in rows)

    return min(candidates, key=sse)


def load(path: str) -> list[Sample]:
    out: list[Sample] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "mode", "threshold_mib", "wall_s", "h2d_s", "d2h_s",
            "gpu_kernel_s", "cpu_high_wall_s", "cpu_low_wall_s",
            "pcie_removed_tib", "pcie_remaining_tib", "cpu_high_groups",
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
                    gpu_kernel_s=float(row["gpu_kernel_s"]),
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
        fields = rd.fieldnames or []
        has_gpu_steps = "gpu_state_steps" in fields
        has_copy_calls = "pcie_copy_calls" in fields
        for row in rd:
            out.append(
                CostGroup(
                    group=int(row["group"]),
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze CPU HIGH threshold sweeps and calibrate PCIe, GPU-HIGH, and "
            "CPU-direct per-group cost models."
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
    nonempty_costs = [g for g in costs or [] if g.roundtrip_bytes > 0]
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
        baseline_gpu_kernel = mean([s.gpu_kernel_s for s in baseline])
        baseline_wall = mean([s.wall_s for s in baseline])

        print(
            f"config mode={mode} overlap={overlap} "
            f"baseline_threshold_mib={baseline_threshold:g} "
            f"baseline_wall_s={baseline_wall:.9f} "
            f"baseline_dma_s={baseline_dma:.9f} "
            f"baseline_gpu_kernel_s={baseline_gpu_kernel:.9f}"
        )

        best_threshold = baseline_threshold
        best_wall = baseline_wall
        pcie_rates: list[float] = []
        cpu_rates: list[float] = []
        gpu_rates: list[float] = []
        model_rows_samples: list[float] = []
        cpu_affine_rows: list[tuple[float, float, float]] = []

        for threshold in sorted(rows):
            group = rows[threshold]
            wall = mean([s.wall_s for s in group])
            dma = mean([s.dma_s for s in group])
            gpu_kernel = mean([s.gpu_kernel_s for s in group])
            cpu_high = mean([s.cpu_high_wall_s for s in group])
            cpu_low = mean([s.cpu_low_wall_s for s in group])
            removed = mean([s.removed_tib for s in group])
            remaining = mean([s.remaining_tib for s in group])
            cpu_groups = mean([float(s.cpu_high_groups) for s in group])

            dma_saved = baseline_dma - dma
            gpu_kernel_saved = baseline_gpu_kernel - gpu_kernel
            gpu_total_saved = dma_saved + gpu_kernel_saved
            wall_saved = baseline_wall - wall
            dma_saved_per_tib = dma_saved / removed if removed > 0 else 0.0
            cpu_cost_per_tib = cpu_high / removed if removed > 0 else 0.0
            sequential_margin = gpu_total_saved - cpu_high
            efficiency = gpu_total_saved / cpu_high if cpu_high > 0 else math.inf

            extra = ""
            if removed > 0 and dma_saved > 0:
                measured_pcie_gib_s = removed * TIB / GIB / dma_saved
                pcie_rates.append(measured_pcie_gib_s)
                extra += f" measured_pcie_gib_s={measured_pcie_gib_s:.9f}"

            if costs is not None and mode == "direct" and threshold > 0 and cpu_high > 0:
                limit_bytes = int(threshold * MIB)
                selected_costs = [
                    g for g in nonempty_costs if g.roundtrip_bytes <= limit_bytes
                ]
                selected_bytes_per_row = sum(g.roundtrip_bytes for g in selected_costs)
                selected_cells_per_row = sum(
                    g.weighted_cells(
                        args.nn_weight, args.nrnl_weight,
                        args.block_weight, args.cross_weight,
                    )
                    for g in selected_costs
                )
                selected_gpu_steps_per_row = sum(g.gpu_state_steps for g in selected_costs)
                if selected_bytes_per_row > 0 and removed > 0:
                    rows_est = removed * TIB / selected_bytes_per_row
                    model_rows_samples.append(rows_est)
                    total_cells = selected_cells_per_row * rows_est
                    total_group_execs = len(selected_costs) * rows_est
                    measured_cpu_gcell_s = total_cells / cpu_high / 1e9
                    if math.isfinite(measured_cpu_gcell_s) and measured_cpu_gcell_s > 0:
                        cpu_rates.append(measured_cpu_gcell_s)
                        extra += (
                            f" model_rows={rows_est:.6f}"
                            f" measured_cpu_gcell_s={measured_cpu_gcell_s:.9f}"
                        )
                    cpu_affine_rows.append((total_cells, total_group_execs, cpu_high))
                    if selected_gpu_steps_per_row > 0 and gpu_kernel_saved > 0:
                        total_gpu_steps = selected_gpu_steps_per_row * rows_est
                        measured_gpu_gstate_s = total_gpu_steps / gpu_kernel_saved / 1e9
                        if math.isfinite(measured_gpu_gstate_s) and measured_gpu_gstate_s > 0:
                            gpu_rates.append(measured_gpu_gstate_s)
                            extra += f" measured_gpu_gstate_s={measured_gpu_gstate_s:.9f}"

            print(
                f"threshold_mib={threshold:g} runs={len(group)} "
                f"mean_wall_s={wall:.9f} wall_saved_s={wall_saved:.9f} "
                f"mean_dma_s={dma:.9f} dma_saved_s={dma_saved:.9f} "
                f"mean_gpu_kernel_s={gpu_kernel:.9f} "
                f"gpu_kernel_saved_s={gpu_kernel_saved:.9f} "
                f"gpu_total_saved_s={gpu_total_saved:.9f} "
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

        pcie_med = statistics.median(pcie_rates) if pcie_rates else math.nan
        cpu_med = statistics.median(cpu_rates) if cpu_rates else math.nan
        gpu_med = statistics.median(gpu_rates) if gpu_rates else math.nan
        if math.isfinite(pcie_med):
            print(f"calibration mode={mode} overlap={overlap} pcie_gib_s={pcie_med:.9f}")
        if math.isfinite(cpu_med):
            print(f"calibration mode={mode} overlap={overlap} cpu_gcell_s={cpu_med:.9f}")
        if math.isfinite(gpu_med):
            print(f"calibration mode={mode} overlap={overlap} gpu_gstate_s={gpu_med:.9f}")

        rows_est = statistics.median(model_rows_samples) if model_rows_samples else math.nan

        pcie_affine_rows: list[tuple[float, float, float]] = []
        gpu_affine_rows: list[tuple[float, float, float]] = []
        if costs is not None and math.isfinite(rows_est):
            for threshold, group in rows.items():
                limit_bytes = int(threshold * MIB)
                selected_costs = [
                    g for g in nonempty_costs
                    if threshold > 0 and g.roundtrip_bytes <= limit_bytes
                ]
                selected_ids = {g.group for g in selected_costs}
                remaining_costs = [g for g in nonempty_costs if g.group not in selected_ids]
                remaining_bytes = sum(g.roundtrip_bytes for g in remaining_costs) * rows_est
                remaining_calls = sum(g.pcie_copy_calls for g in remaining_costs) * rows_est
                remaining_steps = sum(g.gpu_state_steps for g in remaining_costs) * rows_est
                remaining_execs = len(remaining_costs) * rows_est
                pcie_y = mean([s.dma_s for s in group])
                gpu_y = mean([s.gpu_kernel_s for s in group])
                pcie_affine_rows.append((remaining_bytes, remaining_calls, pcie_y))
                gpu_affine_rows.append((remaining_steps, remaining_execs, gpu_y))

        pcie_affine = nnls_two_predictors(pcie_affine_rows)
        affine_pcie_rate = math.nan
        affine_pcie_copy_us = math.nan
        if pcie_affine is not None and pcie_affine[0] > 0.0:
            affine_pcie_rate = 1.0 / pcie_affine[0] / GIB
            affine_pcie_copy_us = pcie_affine[1] * 1e6
            print(
                f"affine_calibration mode={mode} overlap={overlap} "
                f"pcie_gib_s={affine_pcie_rate:.9f} "
                f"pcie_copy_overhead_us={affine_pcie_copy_us:.9f}"
            )

        cpu_affine = nnls_two_predictors(cpu_affine_rows)
        affine_cpu_rate = math.nan
        affine_cpu_group_us = math.nan
        if cpu_affine is not None and cpu_affine[0] > 0.0:
            affine_cpu_rate = 1.0 / cpu_affine[0] / 1e9
            affine_cpu_group_us = cpu_affine[1] * 1e6
            print(
                f"affine_calibration mode={mode} overlap={overlap} "
                f"cpu_gcell_s={affine_cpu_rate:.9f} "
                f"cpu_group_overhead_us={affine_cpu_group_us:.9f}"
            )

        gpu_affine = nnls_two_predictors(gpu_affine_rows)
        affine_gpu_rate = math.nan
        affine_gpu_group_us = math.nan
        if gpu_affine is not None and gpu_affine[0] > 0.0:
            affine_gpu_rate = 1.0 / gpu_affine[0] / 1e9
            affine_gpu_group_us = gpu_affine[1] * 1e6
            print(
                f"affine_calibration mode={mode} overlap={overlap} "
                f"gpu_gstate_s={affine_gpu_rate:.9f} "
                f"gpu_group_overhead_us={affine_gpu_group_us:.9f}"
            )

        if costs is not None and mode == "direct":
            use_pcie_rate = affine_pcie_rate if math.isfinite(affine_pcie_rate) else pcie_med
            use_cpu_rate = affine_cpu_rate if math.isfinite(affine_cpu_rate) else cpu_med
            use_gpu_rate = affine_gpu_rate if math.isfinite(affine_gpu_rate) else gpu_med
            if math.isfinite(use_pcie_rate) and math.isfinite(use_cpu_rate):
                overlap_arg = " --overlap" if overlap else ""
                gpu_arg = f" --gpu-gstate-s {use_gpu_rate:.9f}" if math.isfinite(use_gpu_rate) else ""
                pcie_overhead_arg = (
                    f" --pcie-copy-overhead-us {affine_pcie_copy_us:.9f}"
                    if math.isfinite(affine_pcie_copy_us) and affine_pcie_copy_us > 0 else ""
                )
                cpu_overhead_arg = (
                    f" --group-overhead-us {affine_cpu_group_us:.9f}"
                    if math.isfinite(affine_cpu_group_us) and affine_cpu_group_us > 0 else ""
                )
                gpu_overhead_arg = (
                    f" --gpu-group-overhead-us {affine_gpu_group_us:.9f}"
                    if math.isfinite(affine_gpu_group_us) and affine_gpu_group_us > 0 else ""
                )
                print(
                    "planner_command "
                    f"python3 scripts/tools/plan_cpu_high_groups.py {args.cost_plan} "
                    f"--pcie-gib-s {use_pcie_rate:.9f} --cpu-gcell-s {use_cpu_rate:.9f} "
                    f"--nn-weight {args.nn_weight:g} --nrnl-weight {args.nrnl_weight:g} "
                    f"--block-weight {args.block_weight:g} --cross-weight {args.cross_weight:g}"
                    f"{gpu_arg}{pcie_overhead_arg}{cpu_overhead_arg}{gpu_overhead_arg}{overlap_arg}"
                )


if __name__ == "__main__":
    main()
