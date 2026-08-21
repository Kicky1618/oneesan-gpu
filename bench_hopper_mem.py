#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

PRIMES = [
    2305843009213693951,
    2305843009213693921,
    2305843009213693907,
    2305843009213693723,
    2305843009213693693,
    2305843009213693669,
    2305843009213693613,
    2305843009213693561,
    2305843009213693549,
    2305843009213693487,
    2305843009213693421,
    2305843009213693373,
    2305843009213693277,
]

LINE_RE = re.compile(
    r"backend=(\S+)\s+n=(\d+)\s+residues=([0-9,]+)\s+"
    r"peak_states=(\d+)\s+hash_slots=(\d+)\s+"
    r"peak_alloc_bytes=(\d+)\s+gpu_ms=([0-9.]+)"
)


def crt(residues: list[int]) -> tuple[int, int]:
    x, m = 0, 1
    for r, p in zip(residues, PRIMES):
        t = ((r - x) % p) * pow(m, -1, p) % p
        x += m * t
        m *= p
    return x, m


def gpu_info() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(cmd, text=True).splitlines()[0]
        name, mem_mib, driver = [x.strip() for x in line.split(",", 2)]
        return {"gpu": name, "gpu_memory_mib": mem_mib, "driver": driver}
    except Exception as exc:
        return {"gpu": "unknown", "gpu_memory_mib": "", "driver": "", "gpu_info_error": str(exc)}


def run_one(binary: str, n: int, verbose: bool) -> dict[str, object]:
    bound_bits = 2 * n * (n + 1) + 1
    modulus_bits = math.prod(PRIMES).bit_length()
    if modulus_bits <= bound_bits:
        raise RuntimeError(
            f"CRT capacity insufficient for n={n}: modulus_bits={modulus_bits}, need > {bound_bits}"
        )

    t0 = time.perf_counter()
    proc = subprocess.run([binary, str(n)], text=True, capture_output=True)
    wall_s = time.perf_counter() - t0
    if verbose and proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"solver failed for n={n}, exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    match = LINE_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"could not parse solver output for n={n}: {proc.stdout!r}")

    backend, got_n, residues_s, peak_states, hash_slots, peak_alloc_bytes, gpu_ms = match.groups()
    if int(got_n) != n:
        raise RuntimeError(f"solver returned n={got_n}, expected n={n}")
    residues = [int(x) for x in residues_s.split(",")]
    if len(residues) != len(PRIMES):
        raise RuntimeError(f"expected {len(PRIMES)} residues, got {len(residues)}")

    paths, modulus = crt(residues)
    alloc = int(peak_alloc_bytes)
    return {
        "n": n,
        "backend": backend,
        "paths": str(paths),
        "decimal_digits": len(str(paths)),
        "bound_bits": bound_bits,
        "modulus_bits": modulus.bit_length(),
        "peak_states": int(peak_states),
        "hash_slots": int(hash_slots),
        "peak_alloc_bytes": alloc,
        "peak_alloc_gib": alloc / (1024**3),
        "gpu_ms": float(gpu_ms),
        "wall_s": wall_s,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark exact oneesan frontier DP on Hopper GPUs")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=18)
    ap.add_argument("--binary", default="./oneesan_cuda_hopper_mem")
    ap.add_argument("--csv", default="hopper_n1_18.csv")
    ap.add_argument("--json", default="hopper_n1_18.json")
    ap.add_argument("--verbose", action="store_true", help="show per-edge solver progress")
    args = ap.parse_args()

    if args.start < 1 or args.end < args.start:
        raise SystemExit("invalid range")

    info = gpu_info()
    print(f"GPU: {info.get('gpu')}  memory={info.get('gpu_memory_mib')} MiB  driver={info.get('driver')}")
    print(f"range: n={args.start}..{args.end}")

    rows: list[dict[str, object]] = []
    csv_path = Path(args.csv)
    json_path = Path(args.json)

    for n in range(args.start, args.end + 1):
        try:
            row = run_one(args.binary, n, args.verbose)
        except Exception as exc:
            print(f"n={n}: FAILED: {exc}", file=sys.stderr)
            break
        rows.append(row)
        write_csv(csv_path, rows)
        json_path.write_text(json.dumps({"system": info, "results": rows}, indent=2) + "\n")
        print(
            f"n={n:2d}  states={row['peak_states']:,}  "
            f"mem={row['peak_alloc_gib']:.2f} GiB  "
            f"gpu={row['gpu_ms']/1000:.3f} s  wall={row['wall_s']:.3f} s  "
            f"digits={row['decimal_digits']}"
        )

    print(f"saved: {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
