#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Cost:
    group: int
    nn: int
    nrnl: int
    block: int
    cross: int


@dataclass
class Run:
    sample: int
    group: int
    role: str
    cpu_s: float
    executions: float


def load_costs(path: str) -> dict[int, Cost]:
    out: dict[int, Cost] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "group", "nn_cells", "nrnl_cells",
            "block_closure_cells", "cross_closure_cells",
        }
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing cost columns: {', '.join(sorted(missing))}")
        for row in rd:
            g = int(row["group"])
            out[g] = Cost(
                g,
                int(row["nn_cells"]),
                int(row["nrnl_cells"]),
                int(row["block_closure_cells"]),
                int(row["cross_closure_cells"]),
            )
    return out


def load_runs(path: str) -> list[Run]:
    out: list[Run] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {"role", "sample", "group", "cpu_high_wall_s", "cpu_high_groups"}
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing run columns: {', '.join(sorted(missing))}")
        for row in rd:
            out.append(
                Run(
                    sample=int(row["sample"]),
                    group=int(row["group"]),
                    role=row["role"],
                    cpu_s=float(row["cpu_high_wall_s"]),
                    executions=float(row["cpu_high_groups"]),
                )
            )
    if not out:
        raise SystemExit("run TSV contains no rows")
    return out


def nnls_coordinate_descent(
    x: list[list[float]], y: list[float],
    max_iter: int = 100000, tol: float = 1e-12,
) -> list[float]:
    n = len(x)
    p = len(x[0])
    scales = []
    for j in range(p):
        rms = math.sqrt(sum(row[j] * row[j] for row in x) / n)
        scales.append(rms if rms > 0 else 1.0)
    z = [[row[j] / scales[j] for j in range(p)] for row in x]

    b = [0.0] * p
    pred = [0.0] * n
    norms = [sum(row[j] * row[j] for row in z) for j in range(p)]
    for _ in range(max_iter):
        max_change = 0.0
        max_value = 0.0
        for j in range(p):
            if norms[j] == 0:
                continue
            old = b[j]
            num = 0.0
            for i in range(n):
                residual_without_j = y[i] - pred[i] + z[i][j] * old
                num += z[i][j] * residual_without_j
            new = max(0.0, num / norms[j])
            if new != old:
                delta = new - old
                for i in range(n):
                    pred[i] += z[i][j] * delta
                b[j] = new
                max_change = max(max_change, abs(delta))
            max_value = max(max_value, abs(b[j]))
        if max_change <= tol * (1.0 + max_value):
            break
    return [b[j] / scales[j] for j in range(p)]


def predict(beta: list[float], row: list[float]) -> float:
    return sum(a * b for a, b in zip(beta, row))


def geometric_mean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fit non-negative NN/NRNL/BLOCK/CROSS CPU HIGH direct costs from "
            "stream-calibration benchmark results."
        )
    )
    ap.add_argument("cost_tsv")
    ap.add_argument("run_tsv")
    ap.add_argument("--write-args", default=None,
                    help="optional file receiving planner weight arguments")
    args = ap.parse_args()

    costs = load_costs(args.cost_tsv)
    runs = load_runs(args.run_tsv)

    grouped: dict[int, list[Run]] = defaultdict(list)
    sample_group: dict[int, int] = {}
    validation_runs: list[Run] = []
    for r in runs:
        if r.role == "validation":
            validation_runs.append(r)
            continue
        if r.role != "sample":
            continue
        if r.group not in costs:
            raise SystemExit(f"group {r.group} missing from cost plan")
        if r.sample in sample_group and sample_group[r.sample] != r.group:
            raise SystemExit(f"sample {r.sample} maps to multiple groups")
        sample_group[r.sample] = r.group
        grouped[r.sample].append(r)

    if len(grouped) < 5:
        raise SystemExit("need at least five distinct calibration samples")

    rows: list[list[float]] = []
    ys: list[float] = []
    labels: list[tuple[int, int]] = []
    for sample in sorted(grouped):
        rs = grouped[sample]
        group = sample_group[sample]
        c = costs[group]
        cpu_s = statistics.fmean(r.cpu_s for r in rs)
        executions = statistics.fmean(r.executions for r in rs)
        # Each calibration policy contains one group, so cpu_high_groups is the
        # number of complete per-row group executions in the measured run.
        row = [
            c.nn * executions,
            c.nrnl * executions,
            c.block * executions,
            c.cross * executions,
            executions,
        ]
        rows.append([float(x) for x in row])
        ys.append(cpu_s)
        labels.append((sample, group))

    beta = nnls_coordinate_descent(rows, ys)
    names = ["nn", "nrnl", "block", "cross", "group"]
    preds = [predict(beta, row) for row in rows]
    residuals = [p - y for p, y in zip(preds, ys)]
    rmse = math.sqrt(statistics.fmean(r * r for r in residuals))
    mean_y = statistics.fmean(ys)
    rel_rmse = rmse / mean_y if mean_y else math.inf
    max_abs = max(abs(r) for r in residuals)

    for name, coef in zip(names[:4], beta[:4]):
        print(
            f"coefficient stream={name} seconds_per_cell={coef:.12e} "
            f"gcell_s={(1.0 / coef / 1e9) if coef > 0 else math.inf:.9f}"
        )
    print(
        f"coefficient stream=group seconds_per_execution={beta[4]:.12e} "
        f"overhead_us={beta[4] * 1e6:.9f}"
    )
    print(
        f"fit samples={len(rows)} rmse_s={rmse:.9f} rel_rmse={rel_rmse:.9f} "
        f"max_abs_s={max_abs:.9f}"
    )

    positive = [x for x in beta[:4] if x > 0]
    if not positive:
        raise SystemExit("all fitted stream coefficients are zero")
    base = geometric_mean(positive)
    weights = [x / base for x in beta[:4]]
    cpu_gcell_s = 1.0 / base / 1e9
    planner = (
        f"--cpu-gcell-s {cpu_gcell_s:.9f} "
        f"--nn-weight {weights[0]:.9f} "
        f"--nrnl-weight {weights[1]:.9f} "
        f"--block-weight {weights[2]:.9f} "
        f"--cross-weight {weights[3]:.9f} "
        f"--group-overhead-us {beta[4] * 1e6:.9f}"
    )
    print("planner_args " + planner)

    for (sample, group), y, p in zip(labels, ys, preds):
        print(
            f"sample sample={sample} group={group} measured_s={y:.9f} "
            f"predicted_s={p:.9f} residual_s={p-y:.9f}"
        )

    if validation_runs:
        selected_groups = sorted(set(sample_group.values()))
        total_exec = statistics.fmean(r.executions for r in validation_runs)
        per_group_exec = total_exec / len(selected_groups)
        agg = [0.0] * 5
        for group in selected_groups:
            c = costs[group]
            agg[0] += c.nn * per_group_exec
            agg[1] += c.nrnl * per_group_exec
            agg[2] += c.block * per_group_exec
            agg[3] += c.cross * per_group_exec
        agg[4] = total_exec
        measured = statistics.fmean(r.cpu_s for r in validation_runs)
        predicted = predict(beta, agg)
        print(
            f"validation groups={len(selected_groups)} measured_s={measured:.9f} "
            f"predicted_s={predicted:.9f} residual_s={predicted-measured:.9f} "
            f"relative_error={abs(predicted-measured)/measured if measured else math.inf:.9f}"
        )

    if args.write_args:
        with open(args.write_args, "w", encoding="utf-8") as f:
            f.write(planner + "\n")


if __name__ == "__main__":
    main()
