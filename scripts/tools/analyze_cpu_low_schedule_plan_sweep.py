#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    scheme: str
    workers: int
    domain_size: int
    domains: int
    imbalance: float
    cross4k: int
    cross2m: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find Pareto-efficient LOW schedule topology plans."
    )
    p.add_argument("tsv", type=Path)
    p.add_argument(
        "--max-imbalance",
        type=float,
        default=None,
        help="Only consider points with imbalance <= this value.",
    )
    p.add_argument(
        "--scheme",
        action="append",
        choices=("lpt", "contiguous", "domain", "hybrid"),
        help="Restrict to one or more schemes. 'hybrid' is an alias for 'domain'.",
    )
    return p.parse_args()


def finite_float(value: str, name: str, lineno: int) -> float:
    try:
        x = float(value)
    except ValueError as exc:
        raise ValueError(f"line {lineno}: invalid {name}: {value!r}") from exc
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f"line {lineno}: invalid {name}: {value!r}")
    return x


def nonnegative_int(value: str, name: str, lineno: int) -> int:
    try:
        x = int(value)
    except ValueError as exc:
        raise ValueError(f"line {lineno}: invalid {name}: {value!r}") from exc
    if x < 0:
        raise ValueError(f"line {lineno}: invalid {name}: {value!r}")
    return x


def domain_prefix(fieldnames: list[str]) -> str:
    if "domain_imbalance" in fieldnames:
        return "domain"
    if "hybrid_imbalance" in fieldnames:
        return "hybrid"
    raise ValueError("missing domain_imbalance (or legacy hybrid_imbalance) column")


def read_points(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        names = list(reader.fieldnames or [])
        dpfx = domain_prefix(names)
        required = {
            "workers",
            "domain_size",
            "domains",
            "lpt_imbalance",
            "lpt_cross_domain_4k",
            "lpt_cross_domain_2m",
            "contiguous_imbalance",
            "contiguous_cross_domain_4k",
            "contiguous_cross_domain_2m",
            f"{dpfx}_imbalance",
            f"{dpfx}_cross_domain_4k",
            f"{dpfx}_cross_domain_2m",
        }
        missing = required - set(names)
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")

        for lineno, row in enumerate(reader, start=2):
            workers = nonnegative_int(row["workers"], "workers", lineno)
            dsize = nonnegative_int(row["domain_size"], "domain_size", lineno)
            domains = nonnegative_int(row["domains"], "domains", lineno)
            if workers <= 0 or dsize <= 0 or domains <= 0:
                raise ValueError(f"line {lineno}: worker/domain counts must be positive")

            for scheme, pfx in (
                ("lpt", "lpt"),
                ("contiguous", "contiguous"),
                ("domain", dpfx),
            ):
                points.append(
                    Point(
                        scheme=scheme,
                        workers=workers,
                        domain_size=dsize,
                        domains=domains,
                        imbalance=finite_float(row[f"{pfx}_imbalance"], f"{pfx}_imbalance", lineno),
                        cross4k=nonnegative_int(row[f"{pfx}_cross_domain_4k"], f"{pfx}_cross_domain_4k", lineno),
                        cross2m=nonnegative_int(row[f"{pfx}_cross_domain_2m"], f"{pfx}_cross_domain_2m", lineno),
                    )
                )
    if not points:
        raise ValueError("no plan rows")
    return points


def dominates(a: Point, b: Point) -> bool:
    le = (
        a.imbalance <= b.imbalance
        and a.cross4k <= b.cross4k
        and a.cross2m <= b.cross2m
    )
    lt = (
        a.imbalance < b.imbalance
        or a.cross4k < b.cross4k
        or a.cross2m < b.cross2m
    )
    return le and lt


def fmt(p: Point, prefix: str) -> str:
    return (
        f"{prefix} scheme={p.scheme} workers={p.workers} "
        f"domain_size={p.domain_size} domains={p.domains} "
        f"imbalance={p.imbalance:.9f} "
        f"cross_domain_4k={p.cross4k} cross_domain_2m={p.cross2m}"
    )


def main() -> int:
    args = parse_args()
    try:
        points = read_points(args.tsv)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    requested = args.scheme or ["lpt", "contiguous", "domain"]
    allowed = {"domain" if x == "hybrid" else x for x in requested}
    points = [p for p in points if p.scheme in allowed]
    if args.max_imbalance is not None:
        if not math.isfinite(args.max_imbalance) or args.max_imbalance < 0.0:
            print("error: --max-imbalance must be a finite non-negative value", file=sys.stderr)
            return 2
        points = [p for p in points if p.imbalance <= args.max_imbalance]
    if not points:
        print("no points satisfy filters")
        return 1

    frontier = [p for p in points if not any(dominates(q, p) for q in points if q != p)]
    frontier.sort(key=lambda p: (p.cross2m, p.cross4k, p.imbalance, p.workers, p.scheme))

    print(
        f"plans={len(points)} pareto={len(frontier)} "
        f"schemes={','.join(sorted(allowed))}"
    )
    for p in frontier:
        print(fmt(p, "pareto"))

    best_balance = min(points, key=lambda p: (p.imbalance, p.cross2m, p.cross4k))
    best_4k = min(points, key=lambda p: (p.cross4k, p.imbalance, p.cross2m))
    best_2m = min(points, key=lambda p: (p.cross2m, p.imbalance, p.cross4k))
    print(fmt(best_balance, "best_balance"))
    print(fmt(best_4k, "best_4k"))
    print(fmt(best_2m, "best_2m"))

    for scheme in ("lpt", "contiguous", "domain"):
        subset = [p for p in points if p.scheme == scheme]
        if not subset:
            continue
        best = min(subset, key=lambda p: (p.imbalance, p.cross2m, p.cross4k))
        print(fmt(best, "scheme_best_balance"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
