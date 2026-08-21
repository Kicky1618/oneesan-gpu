#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRIMES = [
    4294967291, 4294967279, 4294967231, 4294967197,
    4294967189, 4294967161, 4294967143, 4294967111,
    4294967087, 4294967029, 4294966997, 4294966981,
    4294966943, 4294966927, 4294966909, 4294966877,
    4294966829, 4294966813, 4294966769, 4294966667,
    4294966661, 4294966657, 4294966651, 4294966639,
    4294966619, 4294966591, 4294966583, 4294966553,
    4294966477, 4294966447, 4294966441, 4294966427,
    4294966373, 4294966367, 4294966337, 4294966297,
    4294966243, 4294966237, 4294966231, 4294966217,
    4294966187, 4294966177, 4294966163, 4294966153,
    4294966129, 4294966121, 4294966099, 4294966087,
]

RESULT_RE = re.compile(r"residue=(\d+).*?modulus=(\d+).*?wall_s=([0-9.eE+-]+)")


def crt_pair(x: int, m: int, r: int, p: int) -> tuple[int, int]:
    t = ((r - x) % p) * pow(m, -1, p) % p
    return x + m * t, m * p


def load_checkpoint(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {int(k): v for k, v in data.get("residues", {}).items()}


def save_checkpoint(path: Path, n: int, residues: dict[int, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "n": n,
        "residues": {str(k): v for k, v in residues.items()},
    }, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact n x (n+1) oneesan count on the B300 HBM32 solver using CRT")
    ap.add_argument("n", nargs="?", type=int, default=27)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--target-mib", type=int, default=16384)
    ap.add_argument("--max-window", type=int, default=14)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--max-runs", type=int, default=0, help="0 = run until the exact bound is reached; useful for smoke tests")
    args = ap.parse_args()

    n = args.n
    # Every simple s-t path is a T-join.  In a connected graph, all T-joins
    # form one affine coset of the cycle space, hence there are exactly
    # 2^(E-V+1) T-joins.  For an (n+1)x(n+1) grid, E-V+1 = n^2.
    # Therefore the path count is <= 2^(n^2), a much tighter rigorous bound
    # than the old 2^E edge-subset bound.
    required_bits = n * n
    if len(PRIMES) != len(set(PRIMES)):
        raise SystemExit("internal error: duplicate CRT primes")
    total_modulus = 1
    for p in PRIMES:
        total_modulus *= p
    if total_modulus.bit_length() <= required_bits:
        raise SystemExit(
            f"CRT prime capacity insufficient: {total_modulus.bit_length()} bits, need >{required_bits}"
        )
    binary = Path(args.binary) if args.binary else REPO_ROOT / "build" / f"oneesan_cuda_gridfp_b300_hbm32_n{n}"
    if not binary.exists():
        raise SystemExit(f"binary not found: {binary}")
    binary = binary.resolve()

    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "work" / f"b300_exact_n{n}"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint.json"
    residues = load_checkpoint(checkpoint)

    # Reject a checkpoint for another n instead of silently combining residues.
    if checkpoint.exists():
        meta = json.loads(checkpoint.read_text())
        if int(meta.get("n", -1)) != n:
            raise SystemExit(f"checkpoint {checkpoint} belongs to n={meta.get('n')}")

    x, M = 0, 1
    used = 0
    total_wall = 0.0
    runs_this_invocation = 0

    for idx, p in enumerate(PRIMES, 1):
        if p in residues:
            rec = residues[p]
            r = int(rec["residue"])
            wall = float(rec.get("wall_s", 0.0))
            print(f"[{idx:02d}/{len(PRIMES)}] cached p={p} residue={r}", file=sys.stderr)
        else:
            if args.max_runs and runs_this_invocation >= args.max_runs:
                break
            log = work / f"p{p}.log"
            cmd = [str(binary), str(n), str(p), str(args.target_mib), str(args.max_window), str(args.gpus)]
            print(f"[{idx:02d}/{len(PRIMES)}] run p={p}: {' '.join(cmd)}", file=sys.stderr, flush=True)
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
            log.write_text(proc.stderr + proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                print(proc.stdout, file=sys.stderr)
                raise SystemExit(f"solver failed for p={p}, rc={proc.returncode}; log={log}")
            match = RESULT_RE.search(proc.stdout)
            if not match:
                raise SystemExit(f"could not parse solver output for p={p}; log={log}")
            r, got_p, wall = int(match.group(1)), int(match.group(2)), float(match.group(3))
            if got_p != p:
                raise SystemExit(f"solver returned modulus {got_p}, expected {p}")
            residues[p] = {"residue": r, "wall_s": wall, "log": str(log)}
            save_checkpoint(checkpoint, n, residues)
            runs_this_invocation += 1
            print(f"[{idx:02d}/{len(PRIMES)}] done p={p} residue={r} wall_s={wall:.6f}", file=sys.stderr, flush=True)

        x, M = crt_pair(x, M, r, p)
        used += 1
        total_wall += wall
        print(f"  CRT bits={M.bit_length()} / need>{required_bits}", file=sys.stderr, flush=True)
        if M.bit_length() > required_bits:
            out = work / "exact.txt"
            out.write_text(
                f"n={n}\n"
                f"exact={x}\n"
                f"bound_bits={required_bits}\n"
                f"modulus_bits={M.bit_length()}\n"
                f"primes_used={used}\n"
                f"solver_wall_s_sum={total_wall:.9f}\n"
            )
            print(f"n={n}")
            print(f"exact={x}")
            print(f"bound_bits={required_bits}")
            print(f"modulus_bits={M.bit_length()}")
            print(f"primes_used={used}")
            print(f"solver_wall_s_sum={total_wall:.9f}")
            print(f"result_file={out}")
            return 0

    print(f"partial: CRT bits={M.bit_length()} need>{required_bits}; cached_residues={len(residues)}", file=sys.stderr)
    return 0 if args.max_runs else 2


if __name__ == "__main__":
    raise SystemExit(main())
