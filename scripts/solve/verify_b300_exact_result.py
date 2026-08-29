#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from solve_b300_exact import crt_pair, primes_for_bound, simple_path_upper_bound


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_exact(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        if "=" not in raw:
            raise SystemExit(f"malformed exact.txt line: {raw!r}")
        k, v = raw.split("=", 1)
        if k in out:
            raise SystemExit(f"duplicate exact.txt key: {k}")
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Independently verify a completed B300 exact CRT artifact")
    ap.add_argument("n", nargs="?", type=int, default=27)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--profile-sha256", default=None)
    ap.add_argument("--certificate", default=None)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    exact_path = Path(args.exact).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not exact_path.is_file():
        raise SystemExit(f"exact artifact not found: {exact_path}")

    data = json.loads(checkpoint.read_text())
    if int(data.get("n", -1)) != args.n:
        raise SystemExit(f"checkpoint n mismatch: {data.get('n')} != {args.n}")
    fp = data.get("solver_fingerprint")
    if not isinstance(fp, dict) or fp.get("schema") not in (2, 3):
        raise SystemExit(f"unsupported checkpoint fingerprint: {fp!r}")
    bsha = fp.get("binary_sha256")
    if not isinstance(bsha, str) or len(bsha) != 64:
        raise SystemExit("checkpoint binary SHA is malformed")
    psha = fp.get("profile_sha256")
    if fp["schema"] == 3 and (not isinstance(psha, str) or len(psha) != 64):
        raise SystemExit("schema-3 checkpoint profile SHA is malformed")
    if fp["schema"] == 2 and psha is not None:
        raise SystemExit("schema-2 checkpoint unexpectedly carries profile SHA")

    if args.binary:
        binary = Path(args.binary).resolve()
        if not binary.is_file():
            raise SystemExit(f"binary not found: {binary}")
        got = sha256(binary)
        if got != bsha:
            raise SystemExit(f"binary SHA mismatch: checkpoint={bsha} actual={got}")
    else:
        binary = None
    if args.profile_sha256 is not None:
        if fp["schema"] != 3:
            raise SystemExit("--profile-sha256 requires schema-3 checkpoint")
        if args.profile_sha256 != psha:
            raise SystemExit(f"profile SHA mismatch: checkpoint={psha} expected={args.profile_sha256}")

    path_bound, strips = simple_path_upper_bound(args.n)
    prefix = primes_for_bound(path_bound)
    residues_raw = data.get("residues", {})
    if not isinstance(residues_raw, dict):
        raise SystemExit("checkpoint residues must be an object")

    x, modulus, total_wall = 0, 1, 0.0
    used = 0
    used_records: list[dict[str, object]] = []
    for p in prefix:
        rec = residues_raw.get(str(p))
        if rec is None:
            break
        if not isinstance(rec, dict) or "residue" not in rec:
            raise SystemExit(f"malformed residue record for p={p}")
        r = int(rec["residue"])
        wall = float(rec.get("wall_s", 0.0))
        if not 0 <= r < p:
            raise SystemExit(f"invalid residue {r} mod {p}")
        if not math.isfinite(wall) or wall < 0:
            raise SystemExit(f"invalid wall_s={wall} for p={p}")
        x, modulus = crt_pair(x, modulus, r, p)
        total_wall += wall
        used += 1
        used_records.append({"modulus": p, "residue": r, "wall_s": wall})
        if modulus > path_bound:
            break

    if modulus <= path_bound:
        raise SystemExit(
            f"CRT capacity insufficient: modulus_bits={modulus.bit_length()} "
            f"bound_bits={path_bound.bit_length()} contiguous_primes={used}"
        )
    if x > path_bound:
        raise SystemExit(f"CRT representative exceeds rigorous bound: {x} > {path_bound}")

    fields = parse_exact(exact_path)
    expected_int = {
        "n": args.n,
        "exact": x,
        "bound_bits": path_bound.bit_length(),
        "modulus_bits": modulus.bit_length(),
        "primes_used": used,
        "checkpoint_schema": int(fp["schema"]),
    }
    for key, expected in expected_int.items():
        if key not in fields or int(fields[key]) != expected:
            raise SystemExit(f"exact.txt mismatch {key}: got={fields.get(key)!r} expected={expected}")
    wall_text = f"{total_wall:.9f}"
    if fields.get("solver_wall_s_sum") != wall_text:
        raise SystemExit(
            f"exact.txt mismatch solver_wall_s_sum: got={fields.get('solver_wall_s_sum')!r} expected={wall_text}"
        )
    checkpoint_sha = sha256(checkpoint)
    if fields.get("checkpoint_sha256") != checkpoint_sha:
        raise SystemExit("exact.txt checkpoint SHA mismatch")
    if fields.get("solver_binary_sha256") != bsha:
        raise SystemExit("exact.txt binary SHA mismatch")
    if fp["schema"] == 3:
        if fields.get("solver_profile_sha256") != psha:
            raise SystemExit("exact.txt profile SHA mismatch")
    elif "solver_profile_sha256" in fields:
        raise SystemExit("schema-2 exact.txt unexpectedly carries profile SHA")

    cert_path = (
        Path(args.certificate).resolve()
        if args.certificate
        else exact_path.with_name("exact.verify.json")
    )
    certificate = {
        "schema": 1,
        "verified": True,
        "n": args.n,
        "exact": x,
        "rigorous_bound": path_bound,
        "bound_bits": path_bound.bit_length(),
        "strip_partition": strips,
        "crt_modulus": modulus,
        "modulus_bits": modulus.bit_length(),
        "primes_used": used,
        "solver_wall_s_sum": total_wall,
        "checkpoint_schema": fp["schema"],
        "checkpoint_sha256": checkpoint_sha,
        "exact_txt_sha256": sha256(exact_path),
        "solver_binary_sha256": bsha,
        "solver_profile_sha256": psha,
        "binary_path": str(binary) if binary else None,
        "checkpoint_path": str(checkpoint),
        "exact_path": str(exact_path),
        "residues_used": used_records,
    }
    tmp = cert_path.with_suffix(cert_path.suffix + ".tmp")
    tmp.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    tmp.replace(cert_path)

    print(f"B300_EXACT_VERIFY_OK n={args.n} exact={x} primes_used={used} modulus_bits={modulus.bit_length()} bound_bits={path_bound.bit_length()}")
    print(f"checkpoint_sha256={checkpoint_sha}")
    print(f"exact_txt_sha256={certificate['exact_txt_sha256']}")
    print(f"solver_binary_sha256={bsha}")
    if psha is not None:
        print(f"solver_profile_sha256={psha}")
    print(f"certificate={cert_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
