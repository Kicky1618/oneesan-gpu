#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

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


def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else math.nan


def load(path: str) -> list[Sample]:
    out: list[Sample] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "mode",
            "threshold_mib",
            "wall_s",
            "h2d_s",
            "d2h_s",
            "cpu_high_wall_s",
            "cpu_low_wall_s",
            "pcie_removed_tib",
            "pcie_remaining_tib",
            "cpu_high_groups",
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze CPU HIGH threshold sweeps and estimate whether CPU offload "
            "cost beats the measured H2D+D2H time it removes."
        )
    )
    ap.add_argument("tsv")
    args = ap.parse_args()

    samples = load(args.tsv)
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
                f"offload_efficiency={efficiency:.9f}"
            )
            if wall < best_wall:
                best_wall = wall
                best_threshold = threshold

        print(
            f"best mode={mode} overlap={overlap} "
            f"threshold_mib={best_threshold:g} mean_wall_s={best_wall:.9f} "
            f"speedup_vs_baseline={baseline_wall / best_wall:.9f}x"
        )


if __name__ == "__main__":
    main()
