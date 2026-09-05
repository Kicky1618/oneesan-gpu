#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

FORMAT_V1 = "ONEESAN_BUILD_PROVENANCE_V1"
FORMAT_V2 = "ONEESAN_BUILD_PROVENANCE_V2"
FORMAT = FORMAT_V2
CHECKSUM_FIELD = "provenance_sha256"
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def source_closure(source: Path, root: Path) -> list[Path]:
    root = root.resolve()
    pending = [source.resolve()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise ValueError(f"source dependency is missing: {path}")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source dependency escaped repository: {path}") from exc
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"quoted-include source is not UTF-8 text: {path}") from exc
        for line in text.splitlines():
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            dep = (path.parent / m.group(1)).resolve()
            try:
                dep.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"include from {path} escaped repository: {m.group(1)}") from exc
            pending.append(dep)
    return sorted(seen)


def _canonical_bytes(data: dict[str, Any]) -> bytes:
    payload = dict(data)
    payload.pop(CHECKSUM_FIELD, None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def checksum(data: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(data)).hexdigest()


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate build-provenance JSON key: {k}")
        out[k] = v
    return out


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        stream = os.fdopen(fd, "w", encoding="utf-8", closefd=True)
        fd = -1
        with stream as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def compiler_version(compiler: Path) -> str:
    proc = subprocess.run(
        [str(compiler), "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"compiler --version failed with rc={proc.returncode}: {compiler}")
    out = proc.stdout.strip()
    if not out:
        raise ValueError(f"compiler --version returned empty output: {compiler}")
    return out


def _repo_record(path: Path, root: Path, *, role: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"provenance dependency is missing: {path}")
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"provenance dependency is outside repository: {path}") from exc
    rec: dict[str, Any] = {
        "path": rel,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }
    if role is not None:
        if not role or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", role):
            raise ValueError(f"invalid auxiliary dependency role: {role!r}")
        rec["role"] = role
    return rec


def create_provenance(
    *, root: Path, binary: Path, source: Path, compiler: Path,
    compile_args: list[str], output: Path | None = None,
    auxiliary_dependencies: list[tuple[str, Path]] | None = None,
) -> Path:
    root = root.resolve()
    binary = binary.resolve()
    source = source.resolve()
    compiler = compiler.resolve()
    if not binary.is_file():
        raise ValueError(f"binary is missing: {binary}")
    try:
        source_rel = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"source is outside repository: {source}") from exc
    deps = source_closure(source, root)
    dep_records = [_repo_record(dep, root) for dep in deps]
    aux_records = [
        _repo_record(path, root, role=role)
        for role, path in (auxiliary_dependencies or [])
    ]
    aux_records.sort(key=lambda rec: (rec["role"], rec["path"]))
    if len({rec["role"] for rec in aux_records}) != len(aux_records):
        raise ValueError("auxiliary dependency roles must be unique")
    if len({rec["path"] for rec in aux_records}) != len(aux_records):
        raise ValueError("auxiliary dependency paths must be unique")
    payload: dict[str, Any] = {
        "format": FORMAT_V2,
        "binary_file": binary.name,
        "binary_sha256": sha256_file(binary),
        "binary_size": binary.stat().st_size,
        "source": source_rel,
        "dependencies": dep_records,
        "auxiliary_dependencies": aux_records,
        "compiler_path": str(compiler),
        "compiler_version": compiler_version(compiler),
        "compile_args": list(compile_args),
        "git_commit": git_commit(root),
    }
    payload[CHECKSUM_FIELD] = checksum(payload)
    if output is None:
        output = Path(str(binary) + ".provenance.json")
    _atomic_write(output, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return output


def _validate_sha256(value: Any, *, what: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid {what} SHA-256")
    return value


def _validate_dependency_records(deps: Any, *, auxiliary: bool) -> list[str]:
    label = "auxiliary dependencies" if auxiliary else "dependencies"
    if not isinstance(deps, list):
        raise ValueError(f"build provenance {label} must be an array")
    paths: list[str] = []
    roles: list[str] = []
    expected_fields = {"path", "sha256", "size", "role"} if auxiliary else {"path", "sha256", "size"}
    for idx, rec in enumerate(deps, 1):
        if not isinstance(rec, dict) or set(rec) != expected_fields:
            raise ValueError(f"build provenance {label} {idx}: field mismatch")
        rp, rh, rs = rec["path"], rec["sha256"], rec["size"]
        if not isinstance(rp, str) or Path(rp).is_absolute() or ".." in Path(rp).parts:
            raise ValueError(f"build provenance {label} {idx}: unsafe path")
        _validate_sha256(rh, what=f"{label} {idx}")
        if type(rs) is not int or rs < 0:
            raise ValueError(f"build provenance {label} {idx}: invalid size")
        paths.append(rp)
        if auxiliary:
            role = rec["role"]
            if not isinstance(role, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", role):
                raise ValueError(f"build provenance {label} {idx}: invalid role")
            roles.append(role)
    if len(paths) != len(set(paths)):
        raise ValueError(f"build provenance {label} paths must be unique")
    if auxiliary:
        if len(roles) != len(set(roles)):
            raise ValueError("build provenance auxiliary dependency roles must be unique")
        if deps != sorted(deps, key=lambda rec: (rec["role"], rec["path"])):
            raise ValueError("build provenance auxiliary dependencies must be sorted by role/path")
    elif paths != sorted(paths):
        raise ValueError("build provenance dependencies must be unique and sorted")
    return paths


def load_provenance(
    path: Path, *, binary: Path | None = None, root: Path | None = None,
    verify_sources: bool = False, expected_compile_args: list[str] | None = None,
) -> dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-standard JSON constant: {token}")),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed build provenance {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("build provenance root must be an object")
    fmt = data.get("format")
    common = {
        "format", "binary_file", "binary_sha256", "binary_size", "source",
        "dependencies", "compiler_path", "compiler_version", "compile_args",
        "git_commit", CHECKSUM_FIELD,
    }
    if fmt == FORMAT_V1:
        expected = common
    elif fmt == FORMAT_V2:
        expected = common | {"auxiliary_dependencies"}
    else:
        raise ValueError(f"unsupported build provenance format: {fmt!r}")
    if set(data) != expected:
        raise ValueError(
            f"build provenance field mismatch: missing={sorted(expected-set(data))} "
            f"unknown={sorted(set(data)-expected)}"
        )
    saved = _validate_sha256(data[CHECKSUM_FIELD], what="build provenance checksum")
    if not hmac.compare_digest(saved, checksum(data)):
        raise ValueError("build provenance SHA-256 checksum mismatch")
    for key in ("binary_file", "binary_sha256", "source", "compiler_path", "compiler_version", "git_commit"):
        if not isinstance(data[key], str):
            raise ValueError(f"build provenance {key} must be a string")
    if Path(data["binary_file"]).name != data["binary_file"]:
        raise ValueError("build provenance binary_file must be a basename")
    bsha = _validate_sha256(data["binary_sha256"], what="build provenance binary")
    if type(data["binary_size"]) is not int or data["binary_size"] < 0:
        raise ValueError("invalid build provenance binary_size")
    if not isinstance(data["compile_args"], list) or not all(isinstance(x, str) for x in data["compile_args"]):
        raise ValueError("build provenance compile_args must be a string array")
    if expected_compile_args is not None:
        missing_args = [arg for arg in expected_compile_args if arg not in data["compile_args"]]
        if missing_args:
            raise ValueError(f"build provenance is missing required compile args: {missing_args}")
    deps = data["dependencies"]
    paths = _validate_dependency_records(deps, auxiliary=False)
    aux = data.get("auxiliary_dependencies", [])
    _validate_dependency_records(aux, auxiliary=True)
    if binary is not None:
        binary = binary.resolve()
        if binary.name != data["binary_file"]:
            raise ValueError(f"provenance binary name mismatch: {binary.name} != {data['binary_file']}")
        if binary.stat().st_size != data["binary_size"] or sha256_file(binary) != bsha:
            raise ValueError("build provenance binary hash/size mismatch")
    if verify_sources:
        if root is None:
            raise ValueError("root is required when verify_sources=True")
        root = root.resolve()
        source = (root / data["source"]).resolve()
        expected_deps = source_closure(source, root)
        expected_paths = [p.relative_to(root).as_posix() for p in expected_deps]
        if expected_paths != paths:
            raise ValueError("build provenance dependency closure mismatch")
        by_path = {rec["path"]: rec for rec in deps}
        for dep in expected_deps:
            rel = dep.relative_to(root).as_posix()
            rec = by_path[rel]
            if dep.stat().st_size != rec["size"] or sha256_file(dep) != rec["sha256"]:
                raise ValueError(f"build provenance source hash/size mismatch: {rel}")
        for rec in aux:
            dep = (root / rec["path"]).resolve()
            try:
                dep.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"build provenance auxiliary dependency escaped repository: {rec['path']}") from exc
            if not dep.is_file():
                raise ValueError(f"build provenance auxiliary dependency is missing: {rec['path']}")
            if dep.stat().st_size != rec["size"] or sha256_file(dep) != rec["sha256"]:
                raise ValueError(f"build provenance auxiliary source hash/size mismatch: {rec['path']}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Create or verify deterministic Oneesan build provenance")
    sub = ap.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--binary", type=Path, required=True)
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--compiler", type=Path, required=True)
    create.add_argument("--compile-arg", action="append", default=[])
    create.add_argument(
        "--auxiliary-dependency", action="append", nargs=2, metavar=("ROLE", "PATH"), default=[],
        help="record a non-include generation input such as an exact certificate",
    )
    create.add_argument("--output", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("provenance", type=Path)
    verify.add_argument("--binary", type=Path)
    verify.add_argument("--root", type=Path)
    verify.add_argument("--verify-sources", action="store_true")
    verify.add_argument("--expect-compile-arg", action="append", default=[])
    args = ap.parse_args()
    try:
        if args.command == "create":
            out = create_provenance(
                root=args.root, binary=args.binary, source=args.source, compiler=args.compiler,
                compile_args=args.compile_arg, output=args.output,
                auxiliary_dependencies=[(role, Path(dep)) for role, dep in args.auxiliary_dependency],
            )
            print(out)
        else:
            data = load_provenance(
                args.provenance, binary=args.binary, root=args.root,
                verify_sources=args.verify_sources, expected_compile_args=args.expect_compile_arg,
            )
            print(f"format={data['format']}")
            print(f"binary_sha256={data['binary_sha256']}")
            print(f"dependencies={len(data['dependencies'])}")
            print(f"auxiliary_dependencies={len(data.get('auxiliary_dependencies', []))}")
            print("valid=1")
    except (ValueError, OSError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
