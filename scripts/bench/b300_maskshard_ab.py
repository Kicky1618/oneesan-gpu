#!/usr/bin/env python3
"""Build/run B300 mask-shard research variants on identical inputs.

The driver intentionally runs variants sequentially so each process releases its
HBM before the next candidate starts.  It verifies that every successful run
returns the same residue for each modulus and writes raw stdout/stderr plus a
machine-readable summary under the result directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build" / "b300-hbm32-batch.sh"

VARIANTS = {
    "v0.4": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu",
    "v0.7": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_batch_guarded.cu",
    "v0.8": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_highclosurerows_batch_guarded.cu",
    "v0.9": "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosurerows_batch_guarded.cu",
}

PHASE_KEYS = (
    "setup_s",
    "wall_s",
    "high_io_sum_s",
    "high_orbit_sum_s",
    "high_closure_sum_s",
    "low_orbit_sum_s",
    "low_closure_sum_s",
    "max_scratch_gib",
)


def parse_kv_line(line: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            out[key] = value
    return out


def result_rows(stdout: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for line in stdout.splitlines():
        row = parse_kv_line(line)
        if "residue" in row and "modulus" in row and "wall_s" in row:
            rows.append(row)
    return rows


def tee_process(cmd: List[str], env: Dict[str, str]) -> Tuple[int, str, str]:
    print("+ " + shlex.join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []

    def pump(stream, dst: List[str], target) -> None:
        assert stream is not None
        for line in stream:
            dst.append(line)
            print(line, end="", file=target, flush=True)

    ts = [
        threading.Thread(target=pump, args=(proc.stdout, stdout_lines, sys.stdout)),
        threading.Thread(target=pump, args=(proc.stderr, stderr_lines, sys.stderr)),
    ]
    for t in ts:
        t.start()
    rc = proc.wait()
    for t in ts:
        t.join()
    return rc, "".join(stdout_lines), "".join(stderr_lines)


def build_variant(name: str, source: str, args: argparse.Namespace) -> Path:
    out = args.build_dir / f"b300_maskshard_{name.replace('.', '')}_n{args.n}"
    env = os.environ.copy()
    env.update(
        {
            "N": str(args.n),
            "SRC": source,
            "OUT": str(out),
            "ARCH": args.arch,
            "LOW_LUT_K": str(args.low),
            "HIGH_LUT_K": str(args.high),
        }
    )
    rc = subprocess.call([str(BUILD_SCRIPT)], cwd=ROOT, env=env)
    if rc:
        raise RuntimeError(f"build failed for {name}, rc={rc}")
    if not out.is_file():
        raise RuntimeError(f"build reported success but binary is missing: {out}")
    return out


def numeric(row: Dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return None


def print_comparison(summary: List[Dict[str, object]]) -> None:
    if not summary:
        return
    print("\n=== B300 mask-shard comparison ===")
    header = ["variant", "modulus", "residue"] + list(PHASE_KEYS)
    print("\t".join(header))
    for item in summary:
        variant = str(item["variant"])
        for row in item["rows"]:  # type: ignore[index]
            values = [variant, row.get("modulus", ""), row.get("residue", "")]
            values += [row.get(k, "") for k in PHASE_KEYS]
            print("\t".join(values))

    first_by_mod: Dict[str, float] = {}
    print("\nwall-time ratios vs first variant:")
    for item in summary:
        variant = str(item["variant"])
        for row in item["rows"]:  # type: ignore[index]
            mod = row.get("modulus", "")
            wall = numeric(row, "wall_s")
            if wall is None:
                continue
            if mod not in first_by_mod:
                first_by_mod[mod] = wall
            base = first_by_mod[mod]
            ratio = wall / base if base else float("nan")
            print(f"  modulus={mod} {variant}: {wall:.6f}s  ratio={ratio:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=27)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--low", type=int, default=14)
    ap.add_argument("--high", type=int, default=13)
    ap.add_argument("--arch", default="native")
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["v0.4", "v0.7", "v0.8", "v0.9"],
        choices=sorted(VARIANTS),
    )
    ap.add_argument(
        "--modulus",
        type=int,
        action="append",
        dest="moduli",
        help="repeat to test multiple CRT primes; default 4294967291",
    )
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--build-dir", type=Path, default=ROOT / "build" / "maskshard-ab")
    ap.add_argument("--result-dir", type=Path)
    ap.add_argument("--vram-reserve-mib", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.n + 1 != args.low + args.high + 1:
        ap.error("LOW+HIGH must equal n")
    if args.threads < 32 or args.threads > 1024 or args.threads % 32:
        ap.error("threads must be a multiple of 32 in [32,1024]")
    moduli = args.moduli or [4294967291]
    for mod in moduli:
        if not 2 <= mod <= 0xFFFFFFFF:
            ap.error(f"modulus out of uint32 range: {mod}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = args.result_dir or (ROOT / "build" / "bench" / f"maskshard-ab-{stamp}")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    binaries: Dict[str, Path] = {}
    for name in args.variants:
        expected = args.build_dir / f"b300_maskshard_{name.replace('.', '')}_n{args.n}"
        if args.skip_build:
            if not expected.is_file():
                raise SystemExit(f"missing binary for --skip-build: {expected}")
            binaries[name] = expected
        elif args.dry_run:
            binaries[name] = expected
        else:
            binaries[name] = build_variant(name, VARIANTS[name], args)

    manifest = {
        "n": args.n,
        "gpus": args.gpus,
        "threads": args.threads,
        "low": args.low,
        "high": args.high,
        "arch": args.arch,
        "moduli": moduli,
        "variants": args.variants,
        "sources": {v: VARIANTS[v] for v in args.variants},
        "binaries": {v: str(binaries[v]) for v in args.variants},
        "git_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip(),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.dry_run:
        for name in args.variants:
            cmd = [str(binaries[name]), str(args.n), str(args.gpus), str(args.threads)]
            cmd += [str(m) for m in moduli]
            print(f"{name}: {shlex.join(cmd)}")
        print(f"result_dir={result_dir}")
        return 0

    expected_residue: Dict[str, str] = {}
    summary: List[Dict[str, object]] = []
    env = os.environ.copy()
    if args.vram_reserve_mib is not None:
        env["GRIDFP_VRAM_RESERVE_MIB"] = str(args.vram_reserve_mib)

    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        cmd = [str(binaries[name]), str(args.n), str(args.gpus), str(args.threads)]
        cmd += [str(m) for m in moduli]
        rc, stdout, stderr = tee_process(cmd, env)
        (result_dir / f"{name}.stdout.log").write_text(stdout)
        (result_dir / f"{name}.stderr.log").write_text(stderr)
        if rc:
            print(f"ERROR: {name} exited rc={rc}; logs kept in {result_dir}", file=sys.stderr)
            return rc

        rows = result_rows(stdout)
        if len(rows) != len(moduli):
            print(
                f"ERROR: {name} produced {len(rows)} result rows for {len(moduli)} moduli",
                file=sys.stderr,
            )
            return 90
        for row in rows:
            mod = row["modulus"]
            residue = row["residue"]
            if mod not in expected_residue:
                expected_residue[mod] = residue
            elif residue != expected_residue[mod]:
                print(
                    f"RESIDUE MISMATCH modulus={mod}: reference={expected_residue[mod]} "
                    f"{name}={residue}",
                    file=sys.stderr,
                )
                return 91
        summary.append({"variant": name, "rows": rows})
        (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print_comparison(summary)
    print(f"\nall residues agree; result_dir={result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
