#!/usr/bin/env python3
"""Build v0.56 once and sweep runtime LOW CTA targets on identical residues."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_PATH = HERE / "b300_maskshard_ab.py"
spec = importlib.util.spec_from_file_location("b300_maskshard_ab", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load base A/B driver: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SOURCE = "src/cuda/b300/oneesan_b300_maskshard_v056_lowmaskbatch_runtime_tune.cu"
DEFAULT_TARGETS = [16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=27)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--low", type=int, default=14)
    ap.add_argument("--high", type=int, default=13)
    ap.add_argument("--arch", default="native")
    ap.add_argument("--target", type=int, action="append", dest="targets")
    ap.add_argument("--max-replicas", type=int, default=1024)
    ap.add_argument("--modulus", type=int, action="append", dest="moduli")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "maskshard-runtime")
    ap.add_argument("--result-dir", type=Path)
    ap.add_argument("--vram-reserve-mib", type=int)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-gpu-preflight", action="store_true")
    args = ap.parse_args()

    targets = args.targets or DEFAULT_TARGETS
    moduli = args.moduli or [4294967291]
    if args.n < 2 or args.low < 1 or args.high < 1 or args.low + args.high != args.n:
        ap.error("LOW+HIGH must equal n and both must be positive")
    if args.gpus < 1 or args.gpus > 8:
        ap.error("gpus must be in [1,8]")
    if args.threads < 32 or args.threads > 1024 or args.threads % 32:
        ap.error("threads must be a multiple of 32 in [32,1024]")
    if args.max_replicas < 1 or args.max_replicas > 65535:
        ap.error("max-replicas must be in [1,65535]")
    if any(t < 1 for t in targets):
        ap.error("all targets must be positive")
    if any(m < 2 or m > 0xFFFFFFFF for m in moduli):
        ap.error("modulus must fit uint32 and be >=2")

    if not args.dry_run and not args.skip_gpu_preflight:
        visible, source = base.visible_gpu_count_hint()
        if visible is not None and visible < args.gpus:
            raise SystemExit(
                f"GPU preflight failed: requested {args.gpus}, only {visible} "
                f"visible according to {source}"
            )

    args.build_dir.mkdir(parents=True, exist_ok=True)
    binary = args.build_dir / f"b300_maskshard_v056_n{args.n}"
    if args.dry_run:
        for target in targets:
            cmd = [str(binary), str(args.n), str(args.gpus), str(args.threads)]
            cmd += [str(m) for m in moduli]
            print(
                f"ONEESAN_LOW_TARGET_TASKS_PER_CTA={target} "
                f"ONEESAN_LOW_MAX_REPLICAS={args.max_replicas} "
                + shlex.join(cmd)
            )
        return 0

    if args.skip_build:
        if not binary.is_file():
            raise SystemExit(f"missing binary for --skip-build: {binary}")
        base.validate_build_provenance("v0.56", binary, SOURCE, args)
    else:
        binary = base.build_variant("v0.56", SOURCE, args)
        base.validate_build_provenance("v0.56", binary, SOURCE, args)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = args.result_dir or (
        ROOT / "build" / "bench" / f"maskshard-runtime-target-{stamp}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    common_env = os.environ.copy()
    if args.vram_reserve_mib is not None:
        common_env["GRIDFP_VRAM_RESERVE_MIB"] = str(args.vram_reserve_mib)

    expected = {}
    summary = []
    for target in targets:
        env = common_env.copy()
        env["ONEESAN_LOW_TARGET_TASKS_PER_CTA"] = str(target)
        env["ONEESAN_LOW_MAX_REPLICAS"] = str(args.max_replicas)
        cmd = [str(binary), str(args.n), str(args.gpus), str(args.threads)]
        cmd += [str(m) for m in moduli]
        print(f"\n=== target={target} ===", flush=True)
        rc, stdout, stderr = base.tee_process(cmd, env)
        tag = f"target-{target}"
        (result_dir / f"{tag}.stdout.log").write_text(stdout)
        (result_dir / f"{tag}.stderr.log").write_text(stderr)
        if rc:
            print(f"ERROR: target={target} exited rc={rc}", file=sys.stderr)
            return rc
        rows = base.result_rows(stdout)
        try:
            base.validate_result_rows(
                tag, rows, n=args.n, gpus=args.gpus, moduli=moduli
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 90
        for row in rows:
            mod = row["modulus"]
            residue = row["residue"]
            if mod not in expected:
                expected[mod] = residue
            elif expected[mod] != residue:
                print(
                    f"RESIDUE MISMATCH modulus={mod}: reference={expected[mod]} "
                    f"target={target} got={residue}",
                    file=sys.stderr,
                )
                return 91
        summary.append({"target_tasks_per_cta": target, "rows": rows})
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== runtime target sweep ===")
    print("target\tmodulus\tresidue\twall_s\tsetup_s")
    for item in summary:
        target = item["target_tasks_per_cta"]
        for row in item["rows"]:
            print(
                f"{target}\t{row['modulus']}\t{row['residue']}\t"
                f"{row['wall_s']}\t{row['setup_s']}"
            )
    print(f"all target residues agree; result_dir={result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
