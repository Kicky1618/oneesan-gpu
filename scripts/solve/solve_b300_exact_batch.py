#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from solve_b300_exact import PRIMES, crt_pair, simple_path_upper_bound, primes_for_bound

RESULT_RE = re.compile(r"residue=(\d+).*?modulus=(\d+).*?wall_s=([0-9.eE+-]+)")
CHECKPOINT_SCHEMA = 2


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def solver_fingerprint(binary: Path) -> dict:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "binary_sha256": file_sha256(binary),
    }


def load_checkpoint(path: Path, n: int, fingerprint: dict) -> dict[int, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if int(data.get("n", -1)) != n:
        raise SystemExit(f"checkpoint {path} belongs to n={data.get('n')}")
    stored = data.get("solver_fingerprint")
    if stored != fingerprint:
        raise SystemExit(
            f"checkpoint {path} has no compatible solver fingerprint; "
            "move/remove it or use a separate --work-dir"
        )
    residues = {int(k): v for k, v in data.get("residues", {}).items()}
    for p, rec in residues.items():
        r = int(rec["residue"])
        if p <= 1 or not (0 <= r < p):
            raise SystemExit(f"checkpoint {path} has invalid residue {r} mod {p}")
    return residues


def save_checkpoint(
    path: Path, n: int, fingerprint: dict, residues: dict[int, dict]
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "n": n,
        "solver_fingerprint": fingerprint,
        "residues": {str(k): v for k, v in residues.items()},
    }, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def reconstruct(prefix: list[int], residues: dict[int, dict]) -> tuple[int, int, float]:
    x, m, wall = 0, 1, 0.0
    for p in prefix:
        if p not in residues:
            break
        rec = residues[p]
        r = int(rec["residue"])
        if not (0 <= r < p):
            raise RuntimeError(f"invalid cached residue {r} mod {p}")
        x, m = crt_pair(x, m, r, p)
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
    fingerprint = solver_fingerprint(binary)
    print(
        f"solver fingerprint: sha256={fingerprint['binary_sha256']}",
        file=sys.stderr,
    )

    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "work" / f"b300_exact_n{n}"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / "checkpoint.json"
    residues = load_checkpoint(checkpoint, n, fingerprint)

    x, m, total_wall = reconstruct(prefix, residues)
    if m > path_bound:
        return finish(work, n, path_bound, prefix, residues, fingerprint)

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
        if p in seen:
            proc.terminate()
            raise SystemExit(f"batch binary returned duplicate modulus {p}")
        if not (0 <= r < p):
            proc.terminate()
            raise SystemExit(f"batch binary returned invalid residue {r} mod {p}")
        if not math.isfinite(wall) or wall < 0:
            proc.terminate()
            raise SystemExit(f"batch binary returned invalid wall_s={wall} for modulus {p}")
        residues[p] = {"residue": r, "wall_s": wall}
        seen.add(p)
        save_checkpoint(checkpoint, n, fingerprint, residues)
        _, mm, _ = reconstruct(prefix, residues)
        print(f"checkpoint p={p}: contiguous CRT bits={mm.bit_length()} / bound_bits={required_bits}", file=sys.stderr, flush=True)

    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"batch solver failed, rc={rc}; completed this invocation={len(seen)}/{len(missing)}")
    if seen != set(missing):
        missing_output = sorted(set(missing) - seen, reverse=True)
        raise SystemExit(f"batch solver exited without residues for: {missing_output}")

    x, m, total_wall = reconstruct(prefix, residues)
    if m > path_bound:
        return finish(work, n, path_bound, prefix, residues, fingerprint)
    print(f"partial: CRT bits={m.bit_length()} bound_bits={required_bits}; cached={len(residues)}", file=sys.stderr)
    return 0 if args.max_runs else 2


def finish(
    work: Path,
    n: int,
    path_bound: int,
    prefix: list[int],
    residues: dict[int, dict],
    fingerprint: dict,
) -> int:
    required_bits = path_bound.bit_length()
    x, m, total_wall = reconstruct(prefix, residues)
    if m <= path_bound:
        raise RuntimeError("finish called before CRT capacity reached")
    # Since the proven path bound satisfies exact <= path_bound < CRT modulus,
    # the canonical CRT representative must itself lie below the bound.  A
    # larger value proves that at least one residue/checkpoint/solver result is
    # inconsistent and must never be emitted as an exact answer.
    if x > path_bound:
        raise RuntimeError(
            f"CRT reconstruction exceeds rigorous path bound: exact_candidate={x} "
            f"> bound={path_bound}"
        )
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
        f"solver_binary_sha256={fingerprint['binary_sha256']}\n"
    )
    print(f"n={n}")
    print(f"exact={x}")
    print(f"bound_bits={required_bits}")
    print(f"modulus_bits={m.bit_length()}")
    print(f"primes_used={used}")
    print(f"solver_wall_s_sum={total_wall:.9f}")
    print(f"solver_binary_sha256={fingerprint['binary_sha256']}")
    print(f"result_file={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
