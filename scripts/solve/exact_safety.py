#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, TextIO

from build_provenance import load_provenance

CHECKPOINT_FORMAT_VERSION = 4
CHECKSUM_FIELD = "checkpoint_sha256"

# The row7 tensor initializer currently depends on a 41 MiB exact compact table
# whose 600+ MiB extraction inputs live only under work/formal-probes.  The
# table hash is reproducible, but its generation semantics are not yet part of
# the clean-clone certificate chain.  Keep it available for benchmarking, but
# do not admit it into a result labeled exact until that evidence is formalized.
_EXACT_PROVENANCE_FORBIDDEN_AUX_ROLES = {"row7-exact-compact"}
_ROW8_STRUCTURAL_AUX_ROLE = "row8-structural-int-v1"
_ROW8_STRUCTURAL_ADMISSION_SCHEMA = "oneesan-row8-gridfp-structural-v2"


def validate_exact_provenance_admissible(
    provenance: dict[str, Any],
    admission_certificate: dict[str, str] | None = None,
) -> None:
    roles = {
        rec.get("role")
        for rec in provenance.get("auxiliary_dependencies", [])
        if isinstance(rec, dict)
    }
    forbidden = sorted(roles & _EXACT_PROVENANCE_FORBIDDEN_AUX_ROLES)
    if forbidden:
        raise ValueError(
            "build provenance is not admitted for exact CRT results: "
            f"uncertified auxiliary generation evidence {forbidden}; "
            "use this binary only for benchmarking until the row7 exact table "
            "has a clean-clone reproducible certificate chain"
        )
    if _ROW8_STRUCTURAL_AUX_ROLE in roles:
        admission = _validate_admission_identity(
            admission_certificate, source="row8 exact admission certificate"
        )
        if admission is None or admission.get("schema") != _ROW8_STRUCTURAL_ADMISSION_SCHEMA:
            raise ValueError(
                "build provenance is not admitted for exact CRT results: row8 structural "
                "binary requires a verified width-specific Grid-FP/structural admission certificate"
            )


def binary_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def solver_identity(
    binary: Path, repo_root: Path, *, expected_compile_args: list[str] | None = None,
    admission_certificate: dict[str, str] | None = None,
) -> dict[str, str]:
    binary = binary.resolve()
    sidecar = Path(str(binary) + ".provenance.json")
    # git_commit is part of the solver/build identity, not the runtime checkout.
    # If a verified build sidecar exists, use the commit captured at build time.
    # Without build provenance we cannot honestly infer which commit produced an
    # arbitrary executable, so record "unknown" rather than the current HEAD.
    build_commit = "unknown"
    if sidecar.is_file():
        provenance = load_provenance(
            sidecar, binary=binary, expected_compile_args=expected_compile_args,
        )
        validate_exact_provenance_admissible(provenance, admission_certificate)
        build_commit = provenance["git_commit"]
    return {
        "binary_name": binary.name,
        "binary_sha256": binary_sha256(binary),
        "git_commit": build_commit,
    }


def validate_residue(modulus: int, residue: int, *, source: str = "residue") -> None:
    if modulus < 2:
        raise ValueError(f"{source}: invalid modulus {modulus}")
    if not 0 <= residue < modulus:
        raise ValueError(
            f"{source}: residue {residue} is outside canonical range [0, {modulus})"
        )


def validate_wall_s(value: int | float, *, source: str = "wall_s") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: wall_s must be numeric")
    wall = float(value)
    if not math.isfinite(wall) or wall < 0:
        raise ValueError(f"{source}: wall_s must be finite and nonnegative")
    return wall


def validate_residue_records(residues: dict[int, dict]) -> None:
    for modulus, rec in residues.items():
        if type(modulus) is not int:
            raise ValueError(f"checkpoint modulus key must be int, got {type(modulus).__name__}")
        if not isinstance(rec, dict):
            raise ValueError(f"checkpoint modulus {modulus}: record must be an object")
        if "residue" not in rec:
            raise ValueError(f"checkpoint modulus {modulus}: missing residue")
        residue = rec["residue"]
        if type(residue) is not int:
            raise ValueError(f"checkpoint modulus {modulus}: residue must be a JSON integer")
        validate_residue(modulus, residue, source=f"checkpoint modulus {modulus}")
        if "wall_s" in rec:
            validate_wall_s(rec["wall_s"], source=f"checkpoint modulus {modulus}")
        if "log" in rec and not isinstance(rec["log"], str):
            raise ValueError(f"checkpoint modulus {modulus}: log must be a string")


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key in exact checkpoint: {key}")
        out[key] = value
    return out


def _canonical_checkpoint_bytes(data: dict[str, Any]) -> bytes:
    payload = dict(data)
    payload.pop(CHECKSUM_FIELD, None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def checkpoint_checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_checkpoint_bytes(data)).hexdigest()


def attach_checkpoint_checksum(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out[CHECKSUM_FIELD] = checkpoint_checksum(out)
    return out


def _load_json_strict(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read exact checkpoint {path}: {exc}") from exc
    try:
        def reject_constant(token: str) -> None:
            raise ValueError(f"non-standard JSON constant: {token}")
        data = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed exact checkpoint {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"checkpoint {path}: root must be a JSON object")
    return data


_ADMISSION_IDENTITY_KEYS = {"sha256", "schema"}

def admission_certificate_identity(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    path = path.resolve()
    data = _load_json_strict(path)
    schema = data.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"admission certificate {path}: missing schema")
    return {
        "sha256": binary_sha256(path),
        "schema": schema,
    }

def _validate_admission_identity(value: Any, *, source: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _ADMISSION_IDENTITY_KEYS:
        raise ValueError(f"{source}: admission certificate identity fields mismatch")
    if not all(isinstance(value[k], str) and value[k] for k in _ADMISSION_IDENTITY_KEYS):
        raise ValueError(f"{source}: admission certificate identity values must be nonempty strings")
    sha = value["sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"{source}: invalid admission certificate SHA-256")
    return dict(value)


def _validate_checkpoint_checksum(path: Path, data: dict[str, Any]) -> None:
    saved = data.get(CHECKSUM_FIELD)
    if not isinstance(saved, str) or len(saved) != 64:
        raise ValueError(
            f"checkpoint {path} has no compatible integrity checksum; "
            "move/delete the legacy checkpoint and recompute residues"
        )
    actual = checkpoint_checksum(data)
    if not hmac.compare_digest(saved, actual):
        raise ValueError(f"checkpoint {path}: SHA-256 integrity checksum mismatch")


def load_checkpoint(
    path: Path,
    *,
    n: int,
    identity: dict[str, str],
    admission_certificate: dict[str, str] | None = None,
) -> dict[int, dict]:
    if not path.exists():
        return {}
    data = _load_json_strict(path)
    if "format_version" not in data:
        raise ValueError(
            f"checkpoint {path} has no compatible solver fingerprint/integrity format; "
            "move/delete the legacy checkpoint and recompute residues"
        )
    if type(data.get("format_version")) is not int or type(data.get("n")) is not int:
        raise ValueError(f"checkpoint {path}: format_version and n must be JSON integers")
    if data["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint {path} has no compatible solver fingerprint/integrity format; "
            "move/delete the legacy checkpoint and recompute residues"
        )
    _validate_checkpoint_checksum(path, data)
    if data["n"] != n:
        raise ValueError(f"checkpoint {path} belongs to n={data.get('n')}, expected n={n}")

    expected_keys = {"format_version", "n", "solver", "admission_certificate", "residues", CHECKSUM_FIELD}
    unknown_keys = sorted(set(data) - expected_keys)
    missing_keys = sorted(expected_keys - set(data))
    if unknown_keys:
        raise ValueError(f"checkpoint {path}: unknown fields: {unknown_keys}")
    if missing_keys:
        raise ValueError(f"checkpoint {path}: missing fields: {missing_keys}")

    saved_solver = data.get("solver")
    if not isinstance(saved_solver, dict):
        raise ValueError(f"checkpoint {path}: missing solver identity")
    solver_keys = {"binary_name", "binary_sha256", "git_commit"}
    if set(saved_solver) != solver_keys:
        raise ValueError(
            f"checkpoint {path}: solver identity fields mismatch: {sorted(saved_solver)}"
        )
    if not all(isinstance(saved_solver[k], str) for k in solver_keys):
        raise ValueError(f"checkpoint {path}: solver identity values must be strings")
    saved_sha = saved_solver.get("binary_sha256")
    if len(saved_sha) != 64 or any(c not in "0123456789abcdef" for c in saved_sha):
        raise ValueError(f"checkpoint {path}: invalid solver binary SHA-256")
    if saved_sha != identity["binary_sha256"]:
        raise ValueError(
            f"checkpoint {path}: solver binary changed "
            f"({saved_sha!r} != {identity['binary_sha256']!r})"
        )
    expected_admission = _validate_admission_identity(
        admission_certificate, source="requested admission certificate"
    )
    saved_admission = _validate_admission_identity(
        data.get("admission_certificate"), source=f"checkpoint {path}"
    )
    if saved_admission != expected_admission:
        raise ValueError(
            f"checkpoint {path}: admission certificate changed "
            f"({saved_admission!r} != {expected_admission!r})"
        )

    raw = data.get("residues", {})
    if not isinstance(raw, dict):
        raise ValueError(f"checkpoint {path}: residues must be an object")
    residues: dict[int, dict] = {}
    for key, rec in raw.items():
        try:
            modulus = int(key)
        except ValueError as exc:
            raise ValueError(f"checkpoint {path}: invalid modulus key {key!r}") from exc
        if str(modulus) != key:
            raise ValueError(f"checkpoint {path}: non-canonical modulus key {key!r}")
        if not isinstance(rec, dict):
            raise ValueError(f"checkpoint modulus {modulus}: record must be an object")
        allowed_rec = {"residue", "wall_s", "log"}
        unknown_rec = sorted(set(rec) - allowed_rec)
        if unknown_rec:
            raise ValueError(f"checkpoint modulus {modulus}: unknown fields: {unknown_rec}")
        residues[modulus] = rec
    validate_residue_records(residues)
    return residues


def _durable_atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        stream = os.fdopen(fd, "w", encoding="utf-8", closefd=True)
        fd = -1  # ownership transferred to stream; avoid a double-close on exceptions
        with stream as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def save_checkpoint(
    path: Path,
    *,
    n: int,
    identity: dict[str, str],
    residues: dict[int, dict],
    admission_certificate: dict[str, str] | None = None,
) -> None:
    validate_residue_records(residues)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "n": n,
        "solver": identity,
        "admission_certificate": _validate_admission_identity(
            admission_certificate, source="checkpoint admission certificate"
        ),
        "residues": {str(k): v for k, v in residues.items()},
    }
    data = attach_checkpoint_checksum(payload)
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    _durable_atomic_write(path, text)


def validate_exact_reconstruction(
    x: int,
    modulus_product: int,
    path_bound: int,
    used_residues: list[tuple[int, int]],
) -> None:
    if modulus_product <= path_bound:
        raise ValueError(
            f"CRT capacity insufficient: M={modulus_product} <= path_bound={path_bound}"
        )
    if not 0 <= x <= path_bound:
        raise ValueError(
            f"CRT reconstruction violates rigorous path bound: exact={x}, bound={path_bound}"
        )
    for modulus, residue in used_residues:
        validate_residue(modulus, residue, source=f"final modulus {modulus}")
        if x % modulus != residue:
            raise ValueError(
                f"CRT reconstruction failed final congruence check for modulus {modulus}: "
                f"{x % modulus} != {residue}"
            )


def acquire_workdir_lock(work: Path) -> TextIO:
    lock_path = work / ".solve.lock"
    fd = lock_path.open("a+")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fd.close()
        raise RuntimeError(f"work directory is already in use: {work}") from exc
    return fd

RESULT_MANIFEST_FORMAT = "ONEESAN_EXACT_RESULT_V4"
RESULT_CHECKSUM_FIELD = "manifest_sha256"


def _canonical_result_bytes(data: dict[str, Any]) -> bytes:
    payload = dict(data)
    payload.pop(RESULT_CHECKSUM_FIELD, None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def result_manifest_checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_result_bytes(data)).hexdigest()


def attach_result_manifest_checksum(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out[RESULT_CHECKSUM_FIELD] = result_manifest_checksum(out)
    return out


def write_exact_result(
    work: Path,
    *,
    n: int,
    exact: int,
    path_bound: int,
    modulus_product: int,
    used_moduli: list[int],
    residues: dict[int, dict],
    identity: dict[str, str],
    checkpoint_path: Path,
    total_wall_s: float,
    strip_partition: list[int],
    binary_path: Path | None = None,
    admission_certificate_path: Path | None = None,
    admission_certificate: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    validate_residue_records(residues)
    if len(set(used_moduli)) != len(used_moduli):
        raise ValueError("exact result contains duplicate CRT moduli")
    missing = [p for p in used_moduli if p not in residues]
    if missing:
        raise ValueError(f"exact result is missing residues for moduli: {missing}")
    product = 1
    for p in used_moduli:
        product *= p
    if product != modulus_product:
        raise ValueError(f"CRT modulus product mismatch before finalization: {product} != {modulus_product}")
    used_residues = [(p, residues[p]["residue"]) for p in used_moduli]
    validate_exact_reconstruction(exact, modulus_product, path_bound, used_residues)
    if not checkpoint_path.exists():
        raise ValueError(f"cannot finalize exact result without checkpoint: {checkpoint_path}")
    if not math.isfinite(total_wall_s) or total_wall_s < 0:
        raise ValueError("total solver wall time must be finite and nonnegative")

    congruences = []
    for p in used_moduli:
        rec = residues[p]
        item: dict[str, Any] = {
            "modulus": p,
            "residue": rec["residue"],
            "wall_s": float(rec.get("wall_s", 0.0)),
        }
        if "log" in rec:
            item["log"] = rec["log"]
        congruences.append(item)

    congruence_wall_sum = sum(item["wall_s"] for item in congruences)
    if not math.isclose(float(total_wall_s), congruence_wall_sum, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            f"solver wall-time sum mismatch before finalization: {total_wall_s} != {congruence_wall_sum}"
        )

    manifest_path = work / "exact_manifest.json"
    exact_path = work / "exact.txt"
    exact_text = (
        f"n={n}\n"
        f"exact={exact}\n"
        f"bound_bits={path_bound.bit_length()}\n"
        f"modulus_bits={modulus_product.bit_length()}\n"
        f"primes_used={len(used_moduli)}\n"
        f"solver_wall_s_sum={total_wall_s:.9f}\n"
        f"manifest_file={manifest_path.name}\n"
    )
    exact_sha256 = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()

    expected_admission = _validate_admission_identity(
        admission_certificate, source="exact result admission certificate"
    )

    provenance_file: str | None = None
    provenance_sha256: str | None = None
    if binary_path is not None:
        binary_path = binary_path.resolve()
        sidecar = Path(str(binary_path) + ".provenance.json")
        if sidecar.is_file():
            # Validate the build sidecar against the exact binary before it becomes
            # part of the durable result bundle. Source/header hashes are already
            # embedded in the sidecar and can be checked independently later.
            provenance = load_provenance(
                sidecar, binary=binary_path,
                expected_compile_args=[f"-DTARGET_W={n + 1}"],
            )
            validate_exact_provenance_admissible(provenance, expected_admission)
            provenance_copy = work / "solver_build_provenance.json"
            _durable_atomic_write(provenance_copy, sidecar.read_text(encoding="utf-8"))
            provenance_file = provenance_copy.name
            provenance_sha256 = binary_sha256(provenance_copy)

    admission_file: str | None = None
    admission_sha256: str | None = None
    admission_schema: str | None = None
    if admission_certificate_path is not None:
        actual_admission = admission_certificate_identity(admission_certificate_path)
        if actual_admission != expected_admission:
            raise ValueError(
                f"admission certificate identity mismatch before finalization: "
                f"{actual_admission!r} != {expected_admission!r}"
            )
        admission_copy = work / "admission_certificate.json"
        _durable_atomic_write(
            admission_copy, admission_certificate_path.read_text(encoding="utf-8")
        )
        admission_file = admission_copy.name
        admission_sha256 = binary_sha256(admission_copy)
        admission_schema = actual_admission["schema"]
    elif expected_admission is not None:
        raise ValueError("admission certificate identity supplied without certificate path")

    payload: dict[str, Any] = {
        "format": RESULT_MANIFEST_FORMAT,
        "n": n,
        "exact_decimal": str(exact),
        "path_bound_decimal": str(path_bound),
        "bound_bits": path_bound.bit_length(),
        "bound_method": "checkerboard_strip_v1",
        "strip_partition": strip_partition,
        "modulus_product_decimal": str(modulus_product),
        "modulus_bits": modulus_product.bit_length(),
        "primes_used": len(used_moduli),
        "congruences": congruences,
        "solver": identity,
        "solver_wall_s_sum": float(total_wall_s),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": binary_sha256(checkpoint_path),
        "exact_file": exact_path.name,
        "exact_sha256": exact_sha256,
        "build_provenance_file": provenance_file,
        "build_provenance_sha256": provenance_sha256,
        "admission_certificate_file": admission_file,
        "admission_certificate_sha256": admission_sha256,
        "admission_certificate_schema": admission_schema,
    }
    manifest = attach_result_manifest_checksum(payload)

    # Publish the manifest before exact.txt.  exact.txt is the completion marker:
    # if a crash happens between these writes the verifier rejects the missing
    # exact file, and the next finalization can safely recreate it.
    _durable_atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
    )
    _durable_atomic_write(exact_path, exact_text)
    return exact_path, manifest_path


def load_exact_result_manifest(path: Path) -> dict[str, Any]:
    data = _load_json_strict(path)
    saved = data.get(RESULT_CHECKSUM_FIELD)
    if not isinstance(saved, str) or len(saved) != 64:
        raise ValueError(f"exact result manifest {path}: missing SHA-256 checksum")
    actual = result_manifest_checksum(data)
    if not hmac.compare_digest(saved, actual):
        raise ValueError(f"exact result manifest {path}: SHA-256 integrity checksum mismatch")
    if data.get("format") != RESULT_MANIFEST_FORMAT:
        raise ValueError(f"exact result manifest {path}: unsupported format {data.get('format')!r}")
    return data
