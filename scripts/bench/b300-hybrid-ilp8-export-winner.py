#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shlex
import statistics
from pathlib import Path


def median_num(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if row[key] not in ("", "nan")]
    return statistics.median(values) if values else math.nan


def spill_free(rows: list[dict[str, str]]) -> bool:
    values: list[tuple[int, int]] = []
    for row in rows:
        try:
            values.append(
                (int(row["spill_store_max_bytes"]), int(row["spill_load_max_bytes"]))
            )
        except (KeyError, ValueError):
            return False
    return bool(values) and all(store == 0 and load == 0 for store, load in values)


def binary_path(
    build_dir: Path, mode: str, threshold: str, random_cg: bool, warp_scan: bool
) -> Path:
    if mode == "ilp4":
        tag = "n27_ilp4warp_dualmask_closuretab"
    elif mode == "hybrid":
        tag = f"n27_mainhybrid8_t{threshold}_warp_dualmask_closuretab"
    else:
        raise ValueError(f"unsupported mode={mode}")
    if random_cg:
        tag += "_cg"
    if warp_scan:
        tag += "_warpscan"
    return build_dir / f"oneesan_cuda_gridfp_b300_hbm32_{tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("output")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--random-cg", type=int, choices=(0, 1), required=True)
    parser.add_argument("--warp-scan", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()

    with open(args.result, newline="") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows:
        raise SystemExit("no hybrid ILP8 rows")

    residues = {row["residue"] for row in rows}
    if len(residues) != 1:
        raise SystemExit(f"FATAL hybrid winner residue mismatch: {residues!r}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["mode"], row["threshold"]), []).append(row)

    candidates: list[dict[str, object]] = []
    for (mode, threshold), group in grouped.items():
        candidates.append(
            {
                "mode": mode,
                "threshold": threshold,
                "wall": median_num(group, "wall_s"),
                "mc": median_num(group, "mc_avg_pct"),
                "spill_free": spill_free(group),
                "regs": max(
                    (int(row["regs_max"]) for row in group if row["regs_max"].isdigit()),
                    default=-1,
                ),
            }
        )

    base = next((candidate for candidate in candidates if candidate["mode"] == "ilp4"), None)
    if base is None:
        raise SystemExit("missing ILP4 baseline")

    safe_hybrid = [
        candidate
        for candidate in candidates
        if candidate["mode"] == "hybrid" and candidate["spill_free"]
    ]
    pool = [base, *safe_hybrid]

    def sort_key(candidate: dict[str, object]) -> tuple[float, float]:
        wall = float(candidate["wall"])
        mc = float(candidate["mc"])
        return wall, -mc if not math.isnan(mc) else math.inf

    winner = min(pool, key=sort_key)
    winner_path = binary_path(
        Path(args.build_dir),
        str(winner["mode"]),
        str(winner["threshold"]),
        bool(args.random_cg),
        bool(args.warp_scan),
    )
    base_path = binary_path(
        Path(args.build_dir), "ilp4", "NA", bool(args.random_cg), bool(args.warp_scan)
    )
    for label, path in (("winner", winner_path), ("baseline", base_path)):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"{label} binary is not executable: {path}")

    speedup = float(base["wall"]) / float(winner["wall"])
    transformed = int(winner["mode"] == "hybrid")
    with open(args.output, "w") as file:
        values = {
            "B300_HYBRID_WINNER_MODE": winner["mode"],
            "B300_HYBRID_WINNER_THRESHOLD": winner["threshold"],
            "B300_HYBRID_WINNER_BIN": winner_path,
            "B300_HYBRID_WINNER_THREADS": args.threads,
            "B300_HYBRID_WINNER_WALL_S": f"{float(winner['wall']):.9f}",
            "B300_HYBRID_WINNER_MC_AVG_PCT": f"{float(winner['mc']):.3f}",
            "B300_HYBRID_WINNER_REGISTERS": winner["regs"],
            "B300_HYBRID_WINNER_SPILL_FREE": int(bool(winner["spill_free"])),
            "B300_HYBRID_WINNER_SPEEDUP_VS_ILP4": f"{speedup:.9f}",
            "B300_HYBRID_WINNER_TRANSFORMED": transformed,
            "B300_HYBRID_BASE_BIN": base_path,
            "B300_HYBRID_BASE_THREADS": args.threads,
            "B300_HYBRID_RESIDUE": next(iter(residues)),
        }
        for key, value in values.items():
            file.write(f"{key}={shlex.quote(str(value))}\n")

    print(f"b300_hybrid_winner_mode={winner['mode']}")
    print(f"b300_hybrid_winner_threshold={winner['threshold']}")
    print(f"b300_hybrid_winner_wall_s={float(winner['wall']):.9f}")
    print(f"b300_hybrid_winner_speedup_vs_ilp4={speedup:.9f}x")
    print(f"b300_hybrid_winner_spill_free={int(bool(winner['spill_free']))}")
    print(f"b300_hybrid_winner_env={args.output}")


if __name__ == "__main__":
    main()
