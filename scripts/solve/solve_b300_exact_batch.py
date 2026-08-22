#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from solve_b300_exact import PRIMES, crt_pair, simple_path_upper_bound, primes_for_bound

RESULT_RE = re.compile(r"residue=(\d+).*?modulus=(\d+).*?wall_s=([0-9.eE+-]+)")


def load_checkpoint(path: Path, n: int) -> dict[int, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if int(data.get("n", -1)) != n:
        raise SystemExit(f"checkpoint {path} belongs to n={data.get('n')}")
    return {int(k): v for k, v in data.get("residues", {}).items()}


def save_checkpoint(path: Path, n: int, residues: dict[int, dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "n": n,
        "residues": {str(k): v for k, v in residues.items()},
    }, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def reconstruct(prefix: list[int], residues: dict[int, dict]) -> tuple[int, int, float]:
    x, m, wall = 0, 1, 0.0
    for p in prefix:
        if p not in residues:
            break
        rec = residues[p]
        x, m = crt_pair(x, m, int(rec["residue"]), p)
        wall += float(rec.get("wall_s", 0.0))
    return x, m, wall


def main() -> int:
    ap = argparse.ArgumentParser(description="Checkpointed multi-modulus B300 HBM32 CRT runner")
    ap.add_argument("n", nargs="?", type=int, default=27)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--target-mib", type=int, default=16384)
    ap.add_argument("--max-window", type=int, default=14)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--max-runs", type=int, default=0, help="limit newly computed residues for smoke tests")
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
    binary = Path(args.binary) if args.binary else REPO_ROOT / "build" / f"oneesan_cuda_gridfp_b300_hbm32_batch_n{n}"
    if not binary.exists():
        raise SystemExit(f"batch binary not found: {binary}")
    binary = binary.resolve()

    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "work" / f"b300_exact_n{n}"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint.json"
    residues = load_checkpoint(checkpoint, n)

    x, m, total_wall = reconstruct(prefix, residues)
    if m > path_bound:
        return finish(work, n, path_bound, prefix, residues)

    missing = [p for p in prefix if p not in residues]
    if args.max_runs:
        missing = missing[:args.max_runs]
    if not missing:
        print(f"partial: CRT bits={m.bit_length()} bound_bits={required_bits}", file=sys.stderr)
        return 0 if args.max_runs else 2

    cmd = [
        str(binary), str(n), str(args.target_mib), str(args.max_window), str(args.gpus),
        *map(str, missing),
    ]
    print(f"batch run: {len(missing)} residue(s), first={missing[0]}, last={missing[-1]}", file=sys.stderr, flush=True)
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        env=os.environ.copy(),
        bufsize=1,
    )
    assert proc.stdout is not None
    seen = set()
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        match = RESULT_RE.search(line)
        if not match:
            continue
        r, p, wall = int(match.group(1)), int(match.group(2)), float(match.group(3))
        if p not in missing:
            proc.terminate()
            raise SystemExit(f"batch binary returned unexpected modulus {p}")
        residues[p] = {"residue": r, "wall_s": wall}
        seen.add(p)
        save_checkpoint(checkpoint, n, residues)
        xx, mm, _ = reconstruct(prefix, residues)
        print(f"checkpoint p={p}: contiguous CRT bits={mm.bit_length()} / bound_bits={required_bits}", file=sys.stderr, flush=True)

    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"batch solver failed, rc={rc}; completed this invocation={len(seen)}/{len(missing)}")
    if seen != set(missing):
        missing_output = sorted(set(missing) - seen, reverse=True)
        raise SystemExit(f"batch solver exited without residues for: {missing_output}")

    x, m, total_wall = reconstruct(prefix, residues)
    if m > path_bound:
        return finish(work, n, path_bound, prefix, residues)
    print(f"partial: CRT bits={m.bit_length()} bound_bits={required_bits}; cached={len(residues)}", file=sys.stderr)
    return 0 if args.max_runs else 2


def finish(work: Path, n: int, path_bound: int, prefix: list[int], residues: dict[int, dict]) -> int:
    required_bits = path_bound.bit_length()
    x, m, total_wall = reconstruct(prefix, residues)
    if m <= path_bound:
        raise RuntimeError("finish called before CRT capacity reached")
    used = 0
    mm = 1
    for p in prefix:
        if p not in residues:
            break
        used += 1
        mm *= p
        if mm > path_bound:
            break
    out = work / "exact.txt"
    out.write_text(
        f"n={n}\n"
        f"exact={x}\n"
        f"bound_bits={required_bits}\n"
        f"modulus_bits={m.bit_length()}\n"
        f"primes_used={used}\n"
        f"solver_wall_s_sum={total_wall:.9f}\n"
    )
    print(f"n={n}")
    print(f"exact={x}")
    print(f"bound_bits={required_bits}")
    print(f"modulus_bits={m.bit_length()}")
    print(f"primes_used={used}")
    print(f"solver_wall_s_sum={total_wall:.9f}")
    print(f"result_file={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
