#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field


@dataclass
class Sample:
    tag: str
    array: str
    bytes: int
    samples: int
    success: int
    requested_spacing_mib: float
    actual_spacing_mib: float
    nodes: dict[int, int] = field(default_factory=dict)
    errors: dict[int, int] = field(default_factory=dict)
    syscall_errno: int = 0

    def fractions(self) -> dict[int, float]:
        if self.success <= 0:
            return {}
        return {n: c / self.success for n, c in self.nodes.items()}


def parse_line(line: str) -> Sample | None:
    if not line.startswith("numa_sample "):
        return None
    fields: dict[str, str] = {}
    nodes: dict[int, int] = {}
    errors: dict[int, int] = {}
    for token in line.strip().split()[1:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if re.fullmatch(r"N\d+", k):
            nodes[int(k[1:])] = int(v)
        elif re.fullmatch(r"E-?\d+", k):
            errors[int(k[1:])] = int(v)
        else:
            fields[k] = v
    required = {
        "tag", "array", "bytes", "samples", "success",
        "requested_spacing_mib", "actual_spacing_mib",
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"NUMA sample line missing {sorted(missing)}: {line.rstrip()}")
    return Sample(
        tag=fields["tag"],
        array=fields["array"],
        bytes=int(fields["bytes"]),
        samples=int(fields["samples"]),
        success=int(fields["success"]),
        requested_spacing_mib=float(fields["requested_spacing_mib"]),
        actual_spacing_mib=float(fields["actual_spacing_mib"]),
        nodes=nodes,
        errors=errors,
        syscall_errno=int(fields.get("syscall_errno", "0")),
    )


def load(path: str) -> list[Sample]:
    out: list[Sample] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            try:
                s = parse_line(line)
            except Exception as e:
                raise SystemExit(f"{path}:{lineno}: {e}") from e
            if s is not None:
                out.append(s)
    if not out:
        raise SystemExit("no numa_sample lines found")
    return out


def fmt_nodes(s: Sample) -> str:
    if not s.nodes:
        return "none"
    fr = s.fractions()
    return ",".join(
        f"N{n}:{s.nodes[n]}:{fr[n]:.6f}" for n in sorted(s.nodes)
    )


def l1(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize RAMstream NUMA page samples and row1->final drift."
    )
    ap.add_argument("log")
    ap.add_argument(
        "--max-unplaced-fraction", type=float, default=None,
        help="exit nonzero if any sample has more unsuccessful queries than this fraction",
    )
    args = ap.parse_args()
    if args.max_unplaced_fraction is not None and not 0 <= args.max_unplaced_fraction <= 1:
        raise SystemExit("--max-unplaced-fraction must be in [0,1]")

    samples = load(args.log)
    by_key = {(s.tag, s.array): s for s in samples}
    bad = False
    for s in samples:
        placed_fraction = s.success / s.samples if s.samples else 0.0
        dominant_node = -1
        dominant_fraction = 0.0
        if s.nodes and s.success:
            dominant_node, dominant_count = max(s.nodes.items(), key=lambda x: (x[1], -x[0]))
            dominant_fraction = dominant_count / s.success
        print(
            f"sample tag={s.tag} array={s.array} bytes={s.bytes} "
            f"samples={s.samples} success={s.success} placed_fraction={placed_fraction:.9f} "
            f"dominant_node={dominant_node} dominant_fraction={dominant_fraction:.9f} "
            f"requested_spacing_mib={s.requested_spacing_mib:g} "
            f"actual_spacing_mib={s.actual_spacing_mib:g} "
            f"syscall_errno={s.syscall_errno} nodes={fmt_nodes(s)}"
        )
        if args.max_unplaced_fraction is not None:
            unplaced = 1.0 - placed_fraction
            if unplaced > args.max_unplaced_fraction:
                bad = True

    arrays = sorted({s.array for s in samples})
    for array in arrays:
        a = by_key.get(("row1", array))
        b = by_key.get(("final", array))
        if a is None or b is None:
            continue
        drift = l1(a.fractions(), b.fractions())
        print(
            f"drift array={array} row1_success={a.success} final_success={b.success} "
            f"node_fraction_l1={drift:.9f}"
        )

    for tag in sorted({s.tag for s in samples}):
        rows = [s for s in samples if s.tag == tag]
        success = sum(s.success for s in rows)
        total = sum(s.samples for s in rows)
        node_counts: dict[int, int] = {}
        for s in rows:
            for node, count in s.nodes.items():
                node_counts[node] = node_counts.get(node, 0) + count
        dominant = max(node_counts.items(), key=lambda x: (x[1], -x[0])) if node_counts else (-1, 0)
        print(
            f"aggregate tag={tag} samples={total} success={success} "
            f"placed_fraction={success / total if total else math.nan:.9f} "
            f"dominant_node={dominant[0]} "
            f"dominant_fraction={dominant[1] / success if success else 0.0:.9f}"
        )

    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
