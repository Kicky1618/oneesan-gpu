#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from exact_safety import (
    acquire_workdir_lock,
    admission_certificate_identity,
    load_checkpoint,
    save_checkpoint,
    solver_identity,
    validate_exact_reconstruction,
    validate_residue,
    write_exact_result,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

from path_bound import PRIMES, primes_for_bound, simple_path_upper_bound
from solve_b300_exact import crt_pair

from solver_output import parse_result_line


def reconstruct(prefix: list[int], residues: dict[int, dict]) -> tuple[int, int, float]:
    x, m, wall = 0, 1, 0.0
    for p in prefix:
        if p not in residues:
            break
        rec = residues[p]
        r = int(rec["residue"])
        validate_residue(p, r, source=f"checkpoint modulus {p}")
        x, m = crt_pair(x, m, r, p)
        wall += float(rec.get("wall_s", 0.0))
    return x, m, wall


def verify_admission_certificate(path: Path | None, n: int) -> dict[str, str] | None:
    admission = admission_certificate_identity(path)
    if admission is None:
        return None
    schema = admission["schema"]
    if schema != "oneesan-row8-gridfp-structural-v2":
        raise ValueError(f"unsupported exact admission certificate schema: {schema!r}")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/tools/row8_gridfp_structural_cert.py"),
         "--verify", str(path)],
        cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise ValueError(f"row8 admission certificate verification failed: {detail}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse row8 admission certificate {path}: {exc}") from exc
    if data.get("n") != n or data.get("width") != n + 1 or data.get("integer_vector_equal") is not True:
        raise ValueError(
            f"row8 admission certificate target mismatch: cert n={data.get('n')} "
            f"width={data.get('width')} requested n={n}"
        )
    return admission


def main() -> int:
    ap = argparse.ArgumentParser(description="Checkpointed multi-modulus B300 HBM32 CRT runner")
    ap.add_argument("n", nargs="?", type=int, default=27)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--target-mib", type=int, default=16384)
    ap.add_argument("--max-window", type=int, default=14)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--max-runs", type=int, default=0, help="limit newly computed residues for smoke tests")
    ap.add_argument("--admission-certificate", type=Path, default=None,
                    help="bind a verified exact-admission certificate to checkpoint and result manifest")
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
        raise SystemExit(f"batch binary not found: {binary}")
    binary = binary.resolve()
    fingerprint = solver_fingerprint(binary)
    print(
        f"solver fingerprint: sha256={fingerprint['binary_sha256']}",
        file=sys.stderr,
    )

    work = Path(args.work_dir) if args.work_dir else REPO_ROOT / "work" / f"b300_exact_n{n}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        _lock_fd = acquire_workdir_lock(work)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint = work / "checkpoint.json"
    admission_path = args.admission_certificate.resolve() if args.admission_certificate else None
    try:
        admission = verify_admission_certificate(admission_path, n)
        identity = solver_identity(
            binary, REPO_ROOT, expected_compile_args=[f"-DTARGET_W={n + 1}"],
            admission_certificate=admission,
        )
        residues = load_checkpoint(
            checkpoint, n=n, identity=identity, admission_certificate=admission
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    unexpected_moduli = sorted(set(residues) - set(prefix))
    if unexpected_moduli:
        raise SystemExit(f"checkpoint contains unexpected moduli: {unexpected_moduli}")
    x, m, total_wall = reconstruct(prefix, residues)
    if m > path_bound:
        return finish(
            work, n, path_bound, prefix, residues, strip_partition, identity, checkpoint, binary,
            admission_path, admission,
        )

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
    child_env = os.environ.copy()
    if admission and admission.get("schema") == "oneesan-row8-gridfp-structural-v2":
        child_env["GRIDFP_BOUNDED_PREFIX_K"] = "8"
        child_env["GRIDFP_DIRECT_ROW8_TENSOR"] = "1"
        child_env["GRIDFP_ROW8_STRUCTURAL"] = "1"
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        env=child_env,
        bufsize=1,
    )
    assert proc.stdout is not None
    seen = set()
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        try:
            result = parse_result_line(line)
        except (ValueError, OverflowError) as exc:
            proc.terminate()
            raise SystemExit(f"invalid batch solver result line: {line.strip()}: {exc}") from exc
        if result is None:
            continue
        got_n, r, p, wall = result.n, result.residue, result.modulus, result.wall_s
        if got_n != n:
            proc.terminate()
            raise SystemExit(f"batch binary returned n={got_n}, expected n={n}")
        if p not in missing:
            proc.terminate()
            raise SystemExit(f"batch binary returned unexpected modulus {p}")
        if p in seen:
            proc.terminate()
            raise SystemExit(f"batch binary returned duplicate modulus {p}")
        try:
            validate_residue(p, r, source=f"batch solver modulus {p}")
        except ValueError as exc:
            proc.terminate()
            raise SystemExit(str(exc)) from exc
        residues[p] = {"residue": r, "wall_s": wall}
        seen.add(p)
        save_checkpoint(
            checkpoint, n=n, identity=identity, residues=residues, admission_certificate=admission
        )
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
        return finish(
            work, n, path_bound, prefix, residues, strip_partition, identity, checkpoint, binary,
            admission_path, admission,
        )
    print(f"partial: CRT bits={m.bit_length()} bound_bits={required_bits}; cached={len(residues)}", file=sys.stderr)
    return 0 if args.max_runs else 2


def finish(
    work: Path, n: int, path_bound: int, prefix: list[int], residues: dict[int, dict],
    strip_partition: list[int] | None = None, identity: dict[str, str] | None = None,
    checkpoint: Path | None = None, binary: Path | None = None,
    admission_certificate_path: Path | None = None,
    admission_certificate: dict[str, str] | None = None,
) -> int:
    required_bits = path_bound.bit_length()
    x, m, total_wall = reconstruct(prefix, residues)
    if m <= path_bound:
        raise RuntimeError("finish called before CRT capacity reached")
    used_residues = []
    for p in prefix:
        if p not in residues:
            break
        used_residues.append((p, int(residues[p]["residue"])))
    validate_exact_reconstruction(x, m, path_bound, used_residues)
    used = 0
    mm = 1
    for p in prefix:
        if p not in residues:
            break
        used += 1
        mm *= p
        if mm > path_bound:
            break
    if strip_partition is None or identity is None or checkpoint is None:
        raise RuntimeError("finish requires result-manifest provenance after CRT validation")
    used_moduli = prefix[:used]
    out, manifest = write_exact_result(
        work, n=n, exact=x, path_bound=path_bound, modulus_product=m,
        used_moduli=used_moduli, residues=residues, identity=identity,
        checkpoint_path=checkpoint, total_wall_s=total_wall,
        strip_partition=strip_partition, binary_path=binary,
        admission_certificate_path=admission_certificate_path,
        admission_certificate=admission_certificate,
    )
    print(f"n={n}")
    print(f"exact={x}")
    print(f"bound_bits={required_bits}")
    print(f"modulus_bits={m.bit_length()}")
    print(f"primes_used={used}")
    print(f"solver_wall_s_sum={total_wall:.9f}")
    print(f"result_file={out}")
    print(f"manifest_file={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
