#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass

MIB = 1 << 20


@dataclass(frozen=True)
class Row:
    group: int
    roundtrip_bytes: int
    authoritative_bytes: int
    page4k_upper: int
    page2m_upper: int


def load(path: str) -> list[Row]:
    out: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "group", "roundtrip_bytes", "authoritative_bytes",
            "page4k_boundary_upper_bytes", "page2m_boundary_upper_bytes",
        }
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing NUMA columns: {', '.join(sorted(missing))}")
        for x in rd:
            out.append(
                Row(
                    group=int(x["group"]),
                    roundtrip_bytes=int(x["roundtrip_bytes"]),
                    authoritative_bytes=int(x["authoritative_bytes"]),
                    page4k_upper=int(x["page4k_boundary_upper_bytes"]),
                    page2m_upper=int(x["page2m_boundary_upper_bytes"]),
                )
            )
    if not out:
        raise SystemExit("cost plan contains no groups")
    return out


def load_groups(path: str) -> set[int]:
    out: set[int] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                out.add(int(line))
            except ValueError as e:
                raise SystemExit(f"invalid group at {path}:{lineno}") from e
    return out


def frac(a: int, b: int) -> float:
    return a / b if b else 0.0


def report(name: str, rows: list[Row]) -> None:
    auth = sum(x.authoritative_bytes for x in rows)
    rt = sum(x.roundtrip_bytes for x in rows)
    p4 = sum(x.page4k_upper for x in rows)
    p2 = sum(x.page2m_upper for x in rows)
    full4 = sum(1 for x in rows if x.authoritative_bytes and x.page4k_upper >= x.authoritative_bytes)
    full2 = sum(1 for x in rows if x.authoritative_bytes and x.page2m_upper >= x.authoritative_bytes)
    print(
        f"selection={name} groups={len(rows)} "
        f"authoritative_gib={auth / (1 << 30):.9f} "
        f"roundtrip_gib_per_row={rt / (1 << 30):.9f} "
        f"page4k_boundary_upper_fraction={frac(p4, auth):.9f} "
        f"page2m_boundary_upper_fraction={frac(p2, auth):.9f} "
        f"groups_100pct_4k_upper={full4} groups_100pct_2m_upper={full2}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize the upper bound on authoritative bytes that may lie in "
            "pages shared across neighboring LOW-occupancy HIGH groups."
        )
    )
    ap.add_argument("cost_plan")
    ap.add_argument("--groups-file", default=None)
    ap.add_argument("--threshold-mib", type=float, action="append", default=None)
    args = ap.parse_args()

    rows = load(args.cost_plan)
    report("all", rows)

    if args.groups_file:
        groups = load_groups(args.groups_file)
        known = {x.group for x in rows}
        bad = sorted(groups - known)
        if bad:
            raise SystemExit(f"unknown group IDs: {bad[:8]}")
        report("file", [x for x in rows if x.group in groups])

    for t in args.threshold_mib or []:
        if t < 0:
            raise SystemExit("--threshold-mib must be non-negative")
        limit = t * MIB
        report(f"threshold-{t:g}MiB", [x for x in rows if x.roundtrip_bytes <= limit])


if __name__ == "__main__":
    main()
