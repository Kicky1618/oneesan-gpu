#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"')


def closure(source: Path) -> set[Path]:
    pending = [source.resolve()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"production source dependency is missing: {path}")
        try:
            path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError(f"local include escaped repository: {path}") from exc
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            m = INCLUDE_RE.match(line)
            if not m:
                continue
            dep = (path.parent / m.group(1)).resolve()
            try:
                dep.relative_to(ROOT)
            except ValueError as exc:
                raise ValueError(f"include from {path} escaped repository: {m.group(1)}") from exc
            pending.append(dep)
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description="Check recursive quoted-include closure for production sources")
    ap.add_argument("source", nargs="+", type=Path)
    args = ap.parse_args()
    all_files: set[Path] = set()
    try:
        for raw in args.source:
            src = raw if raw.is_absolute() else ROOT / raw
            all_files |= closure(src)
    except (OSError, ValueError) as exc:
        print(f"source closure INVALID: {exc}", file=sys.stderr)
        return 1
    for path in sorted(all_files):
        print(path.relative_to(ROOT))
    print(f"source closure: PASS files={len(all_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
