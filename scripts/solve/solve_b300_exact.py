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

# Exact 9x27 strip counts for the stronger planar-component condition used by
# the n=27 bound.  They are independently recomputed by
# src/cpp/probes/pq_component_strip_bound.cpp.
#
# Split the outer boundary into the fixed s-t arcs
#   P0 = top + right,  Q0 = left + bottom.
# For a valid simple path P, write boundary(F) = P XOR P0.  Every connected
# component of 1-faces must touch P0; otherwise one boundary cycle of that
# component survives in P.  Taking the complement gives
# boundary(~F) = P XOR Q0, so every 0-component must touch Q0.  On an internal
# strip boundary either color is allowed to escape into the adjacent strip.
# Ignoring consistency between the three strips only enlarges the set and is
# therefore still a rigorous upper bound.
_PQ27_TOP9 = 1439363966680482394681847048772970007433626003156790462009370
_PQ27_MIDDLE9 = 22942552281959548690313451479726513472161304029234933083393982
_PQ27_BOTTOM9 = _PQ27_TOP9
_PQ27_BOUND = _PQ27_TOP9 * _PQ27_MIDDLE9 * _PQ27_BOTTOM9


def _strip_compatible(x: int, y: int, height: int) -> bool:
    """Whether two adjacent face-bit columns avoid a 2x2 checkerboard."""
    for r in range(height - 1):
        a = (x >> r) & 1
        b = (x >> (r + 1)) & 1
        c = (y >> r) & 1
        d = (y >> (r + 1)) & 1
        if a != b and c != d and a != c:
            return False
    return True


def checkerboard_strip_count(height: int, width: int) -> int:
    """Count height x width binary matrices with no checkerboard 2x2 block."""
    states = 1 << height
    nxt = [
        [y for y in range(states) if _strip_compatible(x, y, height)]
        for x in range(states)
    ]
    dp = [1] * states
    for _ in range(1, width):
        ndp = [0] * states
        for x, ys in enumerate(nxt):
            v = dp[x]
            if not v:
                continue
            for y in ys:
                ndp[y] += v
        dp = ndp
    return sum(dp)


def simple_path_upper_bound(n: int, max_strip_height: int = 9) -> tuple[int, list[int]]:
    """Rigorous upper bound for corner-to-corner simple paths.

    Fix one outer-boundary s-t path P0. Every s-t path is a T-join and can be
    written uniquely as P0 XOR boundary(F), where F is a subset of the n^2
    bounded faces. At an interior vertex P0 has degree zero. If the four
    surrounding face bits form either checkerboard pattern, boundary(F) uses
    all four incident edges, giving degree four, which a simple path cannot
    have. Thus path count is at most the number of n x n face-bit matrices
    without checkerboard 2x2 blocks.

    To keep the generic bound cheap to compute, partition the rows into
    independent strips and ignore constraints across strip boundaries. For
    n=27 and strip height >=9, also use the stronger precomputed P0/Q0
    component bound documented above and take the smaller rigorous bound.
    """
    if n < 1:
        return 1, []
    hmax = min(max_strip_height, n)
    strip = {h: checkerboard_strip_count(h, n) for h in range(1, hmax + 1)}
    best: list[int | None] = [None] * (n + 1)
    parts: list[list[int] | None] = [None] * (n + 1)
    best[0], parts[0] = 1, []
    for rows in range(1, n + 1):
        for h, cnt in strip.items():
            if h > rows or best[rows - h] is None:
                continue
            cand = best[rows - h] * cnt
            if best[rows] is None or cand < best[rows]:
                best[rows] = cand
                parts[rows] = parts[rows - h] + [h]
    assert best[n] is not None and parts[n] is not None

    generic_bound = best[n]
    generic_parts = parts[n]
    if n == 27 and hmax >= 9 and _PQ27_BOUND < generic_bound:
        return _PQ27_BOUND, [9, 9, 9]
    return generic_bound, generic_parts


def primes_for_bound(bound: int) -> list[int]:
    m = 1
    out: list[int] = []
    for p in PRIMES:
        out.append(p)
        m *= p
        if m > bound:
            return out
    raise SystemExit(
        f"CRT prime capacity insufficient: product has {m.bit_length()} bits, "
        f"bound has {bound.bit_length()} bits"
    )


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
    path_bound, strip_partition = simple_path_upper_bound(n)
    prefix = primes_for_bound(path_bound)
    required_bits = path_bound.bit_length()
    print(
        f"rigorous path bound: <={path_bound} "
        f"({required_bits} bits), strips={strip_partition}, CRT primes={len(prefix)}",
        file=sys.stderr,
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

    for idx, p in enumerate(prefix, 1):
        if p in residues:
            rec = residues[p]
            r = int(rec["residue"])
            wall = float(rec.get("wall_s", 0.0))
            print(f"[{idx:02d}/{len(prefix)}] cached p={p} residue={r}", file=sys.stderr)
        else:
            if args.max_runs and runs_this_invocation >= args.max_runs:
                break
            log = work / f"p{p}.log"
            cmd = [str(binary), str(n), str(p), str(args.target_mib), str(args.max_window), str(args.gpus)]
            print(f"[{idx:02d}/{len(prefix)}] run p={p}: {' '.join(cmd)}", file=sys.stderr, flush=True)
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
            print(f"[{idx:02d}/{len(prefix)}] done p={p} residue={r} wall_s={wall:.6f}", file=sys.stderr, flush=True)

        x, M = crt_pair(x, M, r, p)
        used += 1
        total_wall += wall
        print(f"  CRT bits={M.bit_length()} / bound_bits={required_bits}", file=sys.stderr, flush=True)
        if M > path_bound:
            if x > path_bound:
                raise SystemExit(
                    f"CRT reconstruction exceeds rigorous path bound: exact={x} > bound={path_bound}"
                )
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

    print(f"partial: CRT bits={M.bit_length()} bound_bits={required_bits}; cached_residues={len(residues)}", file=sys.stderr)
    return 0 if args.max_runs else 2


if __name__ == "__main__":
    raise SystemExit(main())
