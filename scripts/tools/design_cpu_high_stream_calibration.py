#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

MIB = 1 << 20


@dataclass(frozen=True)
class Group:
    group: int
    roundtrip_bytes: int
    nn: int
    nrnl: int
    block: int
    cross: int

    @property
    def total(self) -> int:
        return self.nn + self.nrnl + self.block + self.cross

    def raw_features(self) -> list[float]:
        return [float(self.nn), float(self.nrnl), float(self.block), float(self.cross), 1.0]


def load_groups(path: str) -> list[Group]:
    out: list[Group] = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        required = {
            "group", "roundtrip_bytes", "nn_cells", "nrnl_cells",
            "block_closure_cells", "cross_closure_cells",
        }
        missing = required.difference(rd.fieldnames or [])
        if missing:
            raise SystemExit(f"missing cost columns: {', '.join(sorted(missing))}")
        for row in rd:
            g = Group(
                group=int(row["group"]),
                roundtrip_bytes=int(row["roundtrip_bytes"]),
                nn=int(row["nn_cells"]),
                nrnl=int(row["nrnl_cells"]),
                block=int(row["block_closure_cells"]),
                cross=int(row["cross_closure_cells"]),
            )
            if g.roundtrip_bytes > 0 and g.total > 0:
                out.append(g)
    if not out:
        raise SystemExit("cost plan has no non-empty groups")
    return out


def logdet(matrix: list[list[float]]) -> float:
    a = [row[:] for row in matrix]
    n = len(a)
    ans = 0.0
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[p][i]) < 1e-300:
            return -math.inf
        if p != i:
            a[i], a[p] = a[p], a[i]
        pivot = a[i][i]
        if pivot <= 0.0:
            return -math.inf
        ans += math.log(pivot)
        for r in range(i + 1, n):
            q = a[r][i] / pivot
            if q == 0.0:
                continue
            for c in range(i + 1, n):
                a[r][c] -= q * a[i][c]
    return ans


def gram_with(rows: list[list[float]], dim: int, ridge: float) -> list[list[float]]:
    g = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        g[i][i] = ridge
    for x in rows:
        for i in range(dim):
            xi = x[i]
            for j in range(i, dim):
                z = xi * x[j]
                g[i][j] += z
                if i != j:
                    g[j][i] += z
    return g


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Choose topology-diverse HIGH occupancy groups for estimating separate "
            "NN/NRNL/BLOCK/CROSS CPU direct costs."
        )
    )
    ap.add_argument("cost_tsv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--min-roundtrip-mib", type=float, default=64.0)
    ap.add_argument("--max-roundtrip-mib", type=float, default=1024.0)
    ap.add_argument("--max-candidates", type=int, default=4096)
    ap.add_argument("--ridge", type=float, default=1e-6)
    args = ap.parse_args()

    if args.samples < 5:
        raise SystemExit("--samples must be at least 5 for four stream costs plus overhead")
    if args.min_roundtrip_mib < 0 or args.max_roundtrip_mib <= args.min_roundtrip_mib:
        raise SystemExit("invalid roundtrip MiB range")
    if args.max_candidates < args.samples:
        raise SystemExit("--max-candidates must be >= --samples")
    if args.ridge <= 0:
        raise SystemExit("--ridge must be positive")

    groups = load_groups(args.cost_tsv)
    lo = args.min_roundtrip_mib * MIB
    hi = args.max_roundtrip_mib * MIB
    candidates = [g for g in groups if lo <= g.roundtrip_bytes <= hi]
    if len(candidates) < args.samples:
        raise SystemExit(
            f"only {len(candidates)} groups in requested size range; need {args.samples}"
        )

    # Preserve feature extremes before applying the candidate cap. This avoids
    # throwing away a rare CROSS-heavy group merely because its total work is modest.
    keep: dict[int, Group] = {}
    for k in range(4):
        ranked = sorted(
            candidates,
            key=lambda g: (
                [g.nn, g.nrnl, g.block, g.cross][k] / g.total,
                g.total,
            ),
            reverse=True,
        )
        for g in ranked[: min(args.samples * 4, len(ranked))]:
            keep[g.group] = g
    for g in sorted(candidates, key=lambda g: g.total, reverse=True):
        if len(keep) >= args.max_candidates:
            break
        keep[g.group] = g
    candidates = list(keep.values())

    raw = [g.raw_features() for g in candidates]
    dim = 5
    scales: list[float] = []
    for j in range(dim):
        rms = math.sqrt(sum(x[j] * x[j] for x in raw) / len(raw))
        scales.append(rms if rms > 0 else 1.0)
    normalized = {
        g.group: [x / scales[j] for j, x in enumerate(g.raw_features())]
        for g in candidates
    }

    selected: list[Group] = []
    selected_rows: list[list[float]] = []
    remaining = {g.group: g for g in candidates}
    logdets: list[float] = []

    for _ in range(args.samples):
        best: Group | None = None
        best_ld = -math.inf
        for g in remaining.values():
            rows = selected_rows + [normalized[g.group]]
            ld = logdet(gram_with(rows, dim, args.ridge))
            if ld > best_ld:
                best_ld = ld
                best = g
        assert best is not None
        selected.append(best)
        selected_rows.append(normalized[best.group])
        del remaining[best.group]
        logdets.append(best_ld)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "sample\tgroups_file\tgroup\troundtrip_mib\tnn_cells\tnrnl_cells"
            "\tblock_cells\tcross_cells\tnn_fraction\tnrnl_fraction"
            "\tblock_fraction\tcross_fraction\tdesign_logdet\n"
        )
        for i, (g, ld) in enumerate(zip(selected, logdets), 1):
            name = f"sample-{i:02d}-g{g.group}.groups"
            (out_dir / name).write_text(f"{g.group}\n", encoding="utf-8")
            parts = [g.nn, g.nrnl, g.block, g.cross]
            fracs = [x / g.total for x in parts]
            f.write(
                f"{i}\t{name}\t{g.group}\t{g.roundtrip_bytes / MIB:.9f}"
                f"\t{g.nn}\t{g.nrnl}\t{g.block}\t{g.cross}"
                f"\t{fracs[0]:.9f}\t{fracs[1]:.9f}"
                f"\t{fracs[2]:.9f}\t{fracs[3]:.9f}\t{ld:.9f}\n"
            )

    validation = out_dir / "validation-all.groups"
    validation.write_text(
        "".join(f"{g.group}\n" for g in selected), encoding="utf-8"
    )

    final_gram = gram_with(selected_rows, dim, args.ridge)
    print(
        f"designed samples={len(selected)} candidates={len(candidates)} "
        f"final_logdet={logdet(final_gram):.9f} manifest={manifest} "
        f"validation={validation}"
    )
    print("selected_groups=" + ",".join(str(g.group) for g in selected))


if __name__ == "__main__":
    main()
