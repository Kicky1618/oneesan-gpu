#!/usr/bin/env python3
"""Build and alternate baseline/optimized runs of the same factor solver."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
EXPECTED = {9: 41044208702632496804 % 4294967291, 18: 503411004, 20: 2308006916, 23: 2762394459}
EXPECTED_SECOND = {9: 3332982389, 20: 3704549185, 21: 2124618149}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=range(9, 28), default=20)
    parser.add_argument("--arch", default="native")
    parser.add_argument("--gpus", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--scratch-mib", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--modulus", type=int, default=4294967291)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--optimization", choices=("division", "reverse", "sparse", "shard", "config", "scatter", "block-cache", "factor-memory", "frontier", "graphs", "graph-io", "transfer", "direct-index", "frontier-compiled", "transfer-batch"), default="division")
    args = parser.parse_args()
    if args.repeats < 1 or args.scratch_mib < 1:
        parser.error("repeats and scratch-mib must be positive")
    if not args.compile_only:
        subprocess.run(["nvidia-smi", "-L"], check=True)
    output = Path(tempfile.mkdtemp(prefix="oneesan-factor-div-"))
    print(f"Logs and binaries: {output}", flush=True)
    low = (args.n + 1) // 2
    binaries = {}
    commands = {}
    for mode in (0, 1):
        binary = output / f"solver-{mode}"
        switches = {
            "division": [f"-DFACTOR_RECIPROCAL_DIV={mode}"],
            "reverse": [f"-DGRIDFP_REUSE_MAIN_RANK={mode}", "-DGRIDFP_SPARSE_REVERSE=0"],
            "sparse": [f"-DGRIDFP_SPARSE_REVERSE={mode}"],
            "shard": [f"-DGRIDFP_SHARD_SEARCH={mode}"],
            "config": [f"-DGRIDFP_BATCH_FACTOR_CONFIG={mode}", f"-DGRIDFP_LEGACY_GROUP_UPLOAD={1-mode}"],
            "scatter": [f"-DGRIDFP_REUSE_SCATTER_MATE={mode}"],
            "transfer": [],
            "transfer-batch": [],
            "direct-index": [f"-DGRIDFP_DIRECT_GLOBAL_INDEX={mode}"],
            "frontier-compiled": [f"-DGRIDFP_DIRECT_GLOBAL_INDEX={mode}", f"-DGRIDFP_LOW_RANK_DELTA={mode}"],
            "block-cache": [],
            "graphs": [],
            "graph-io": [],
            "factor-memory": [f"-DGRIDFP_GLOBAL_FACTOR_CONFIG={mode}"],
            "frontier": [f"-DGRIDFP_BATCH_FACTOR_CONFIG={mode}",
                         f"-DGRIDFP_LEGACY_GROUP_UPLOAD={1-mode}",
                         "-DGRIDFP_REUSE_SCATTER_MATE=0"],
        }[args.optimization]
        command = ["nvcc", "-O3", "-std=c++17", "-lineinfo", f"-arch={args.arch}",
                   f"-DTARGET_W={args.n + 1}", f"-DLOW_LUT_K={low}",
                   f"-DHIGH_LUT_K={args.n-low}", *switches,
                   "-Xptxas=-v", str(SOURCE), "-o", str(binary)]
        with (output / f"build-{mode}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        binaries[mode] = binary
        commands[mode] = command
    env = os.environ.copy()
    env["GRIDFP_PLAN_ONLY"] = "0"
    env["GRIDFP_ROW6_INIT_ONLY"] = "0"
    results = {0: [], 1: []}
    report = {"args": vars(args), "commands": commands,
              "environment": {k: v for k, v in env.items()
                              if k.startswith(("GRIDFP_", "CUDA_VISIBLE_DEVICES"))},
              "results": results}

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    if args.compile_only:
        return
    expected = (EXPECTED if args.modulus==4294967291 else EXPECTED_SECOND if args.modulus==4294966997 else {}).get(args.n)
    # First pair warms both binaries; alternating order reduces drift bias.
    for iteration in range(args.repeats + 1):
        for mode in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            command = [str(binaries[mode]), str(args.n), str(args.scratch_mib),
                       "14", str(args.gpus), str(args.modulus)]
            run_env = env.copy()
            run_env.update(GRIDFP_TRANSFER_MAP_MIB="0", GRIDFP_VERIFY_TRANSFER_MAPS="0", GRIDFP_PIPELINE_GROUPS="0",
                           GRIDFP_TRANSFER_BATCH="1", GRIDFP_PROFILE_BATCH="0")
            if args.optimization == "transfer-batch":
                run_env.update(GRIDFP_TRANSFER_MAP_MIB="512", GRIDFP_PIPELINE_GROUPS="1",
                               GRIDFP_TRANSITION_GRAPHS="2", GRIDFP_TRANSFER_BATCH="32" if mode else "1")
            if args.optimization in ("transfer", "frontier-compiled"):
                run_env.update(GRIDFP_TRANSFER_MAP_MIB="512" if mode else "0",
                               GRIDFP_PIPELINE_GROUPS=str(mode), GRIDFP_TRANSITION_GRAPHS="2")
            if args.optimization in ("config", "scatter", "block-cache", "frontier"):
                run_env["GRIDFP_SINGLE_STREAM_MAX_STATES"] = "0"
                run_env["GRIDFP_CACHE_BLOCK_MATES"] = str(mode) if args.optimization in ("block-cache", "frontier") else "0"
            if args.optimization == "graphs":
                run_env.update(GRIDFP_TRANSITION_GRAPHS=str(2*mode),
                               GRIDFP_SINGLE_STREAM_MAX_STATES="0",
                               GRIDFP_CACHE_BLOCK_MATES="1",
                               GRIDFP_PARALLEL_GROUP_IO="0")
            if args.optimization == "graph-io":
                run_env.update(GRIDFP_TRANSITION_GRAPHS="2",
                               GRIDFP_SINGLE_STREAM_MAX_STATES="0",
                               GRIDFP_CACHE_BLOCK_MATES="1",
                               GRIDFP_PARALLEL_GROUP_IO=str(mode))
            report.setdefault("runtime_environment_by_mode", {})[mode] = {
                k: v for k, v in run_env.items() if k.startswith(("GRIDFP_", "CUDA_VISIBLE_DEVICES"))}
            log_path = output / f"run-{iteration}-{mode}.log"
            with log_path.open("w") as log:
                subprocess.run(command, env=run_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            lines = [line for line in log_path.read_text().splitlines()
                     if line.startswith("backend=")]
            if len(lines) != 1:
                raise RuntimeError(f"Expected one complete result in {log_path}")
            fields = dict(re.findall(r"(\w+)=([^\s]+)", lines[0]))
            residue = int(fields["residue"])
            if expected is None:
                expected = residue
            if residue != expected:
                raise RuntimeError(f"Residue mismatch: {residue} != {expected} ({log_path})")
            print(f"iteration={iteration} optimization={args.optimization} enabled={mode} residue={residue} "
                  f"wall_s={fields['wall_s']}", flush=True)
            if iteration:
                results[mode].append(fields)
            save()
    summary = {}
    for field in ("wall_s", "active_sum_s", "gather_sum_s", "transition_sum_s", "scatter_sum_s", "group_graph_sum_s"):
        medians = [statistics.median(float(r[field]) for r in results[m]) for m in (0, 1)]
        summary[field] = {"baseline": medians[0], "optimized": medians[1],
                          "speedup": medians[0] / medians[1] if medians[1] else None}
    report["median_summary"] = summary
    save()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
