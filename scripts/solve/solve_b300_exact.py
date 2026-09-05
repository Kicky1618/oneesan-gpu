#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

from exact_safety import (
    acquire_workdir_lock,
    load_checkpoint,
    save_checkpoint,
    solver_identity,
    validate_exact_reconstruction,
    validate_residue,
    write_exact_result,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

from path_bound import (
    PRIMES,
    _strip_compatible,
    checkerboard_strip_count,
    primes_for_bound,
    simple_path_upper_bound,
)

from solver_output import parse_result_line

def crt_pair(x: int, m: int, r: int, p: int) -> tuple[int, int]:
    t = ((r - x) % p) * pow(m, -1, p) % p
    return x + m * t, m * p


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
    if not 2 <= args.n <= 27:
        ap.error("n must be in 2..27 for the B300 production solvers")
    if args.target_mib < 1:
        ap.error("--target-mib must be at least 1")
    if args.max_window < 1:
        ap.error("--max-window must be at least 1")
    if not 0 <= args.gpus <= 8:
        ap.error("--gpus must be in 0..8 (0 = all visible GPUs)")
    if args.max_runs < 0:
        ap.error("--max-runs must be nonnegative")

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
        raise SystemExit(f"binary not found: {binary}")
    binary = binary.resolve()

    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "work" / f"b300_exact_n{n}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        _lock_fd = acquire_workdir_lock(work)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint = work / "checkpoint.json"
    try:
        identity = solver_identity(
            binary, REPO_ROOT, expected_compile_args=[f"-DTARGET_W={n + 1}"],
        )
        residues = load_checkpoint(checkpoint, n=n, identity=identity)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    unexpected_moduli = sorted(set(residues) - set(prefix))
    if unexpected_moduli:
        raise SystemExit(f"checkpoint contains unexpected moduli: {unexpected_moduli}")
    x, M = 0, 1
    used = 0
    total_wall = 0.0
    runs_this_invocation = 0

    for idx, p in enumerate(prefix, 1):
        if p in residues:
            rec = residues[p]
            r = int(rec["residue"])
            wall = float(rec.get("wall_s", 0.0))
            try:
                validate_residue(p, r, source=f"cached modulus {p}")
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            print(f"[{idx:02d}/{len(prefix)}] cached p={p} residue={r}", file=sys.stderr)
        else:
            if args.max_runs and runs_this_invocation >= args.max_runs:
                break
            log = work / f"p{p}.log"
            cmd = [str(binary), str(n), str(args.target_mib), str(args.max_window), str(args.gpus), str(p)]
            print(f"[{idx:02d}/{len(prefix)}] run p={p}: {' '.join(cmd)}", file=sys.stderr, flush=True)
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
            log.write_text(proc.stderr + proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                print(proc.stdout, file=sys.stderr)
                raise SystemExit(f"solver failed for p={p}, rc={proc.returncode}; log={log}")
            parsed = []
            try:
                for line in proc.stdout.splitlines():
                    result = parse_result_line(line)
                    if result is not None:
                        parsed.append(result)
            except (ValueError, OverflowError) as exc:
                raise SystemExit(f"invalid solver result for p={p}: {exc}; log={log}") from exc
            if len(parsed) != 1:
                raise SystemExit(
                    f"expected exactly one solver result for p={p}, got {len(parsed)}; log={log}"
                )
            result = parsed[0]
            got_n, r, got_p, wall = result.n, result.residue, result.modulus, result.wall_s
            if got_n != n:
                raise SystemExit(f"solver returned n={got_n}, expected n={n}")
            if got_p != p:
                raise SystemExit(f"solver returned modulus {got_p}, expected {p}")
            try:
                validate_residue(p, r, source=f"solver modulus {p}")
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            residues[p] = {"residue": r, "wall_s": wall, "log": str(log)}
            save_checkpoint(checkpoint, n=n, identity=identity, residues=residues)
            runs_this_invocation += 1
            print(f"[{idx:02d}/{len(prefix)}] done p={p} residue={r} wall_s={wall:.6f}", file=sys.stderr, flush=True)

        x, M = crt_pair(x, M, r, p)
        used += 1
        total_wall += wall
        print(f"  CRT bits={M.bit_length()} / bound_bits={required_bits}", file=sys.stderr, flush=True)
        if M > path_bound:
            used_residues = [(q, int(residues[q]["residue"])) for q in prefix[:used]]
            try:
                validate_exact_reconstruction(x, M, path_bound, used_residues)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            out, manifest = write_exact_result(
                work, n=n, exact=x, path_bound=path_bound, modulus_product=M,
                used_moduli=prefix[:used], residues=residues, identity=identity,
                checkpoint_path=checkpoint, total_wall_s=total_wall,
                strip_partition=strip_partition, binary_path=binary,
            )
            print(f"n={n}")
            print(f"exact={x}")
            print(f"bound_bits={required_bits}")
            print(f"modulus_bits={M.bit_length()}")
            print(f"primes_used={used}")
            print(f"solver_wall_s_sum={total_wall:.9f}")
            print(f"result_file={out}")
            print(f"manifest_file={manifest}")
            return 0

    print(f"partial: CRT bits={M.bit_length()} bound_bits={required_bits}; cached_residues={len(residues)}", file=sys.stderr)
    return 0 if args.max_runs else 2


if __name__ == "__main__":
    raise SystemExit(main())
