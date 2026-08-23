#!/usr/bin/env python3
"""Partition an exported HIGH-mask transition graph into 8 weighted shards.

Input comes from factor_highmask_graph_export.cpp. The script uses recursive
normalized spectral bisection for a balanced seed, then capacity-constrained
single-node moves and optional pair swaps. It is a research heuristic, not an
optimal graph-partition solver.

Requires numpy and scipy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from typing import List, Tuple

try:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except ImportError as exc:  # pragma: no cover - research environment dependency
    raise SystemExit("highmask_spectral_partition.py requires numpy and scipy") from exc


def read_nodes(path: Path) -> np.ndarray:
    rows: List[Tuple[int, int]] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append((int(row["mask"]), int(row["state_weight"])))
    rows.sort()
    if not rows or [x[0] for x in rows] != list(range(len(rows))):
        raise ValueError("node masks must be contiguous from zero")
    return np.array([x[1] for x in rows], dtype=np.int64)


def read_edges(path: Path, n: int):
    edge_rows: List[int] = []
    edge_cols: List[int] = []
    edge_vals: List[float] = []
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    edge_weight = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            a = int(row["mask_a"])
            b = int(row["mask_b"])
            w = int(row["transition_weight"])
            if not (0 <= a < n and 0 <= b < n and a < b and w > 0):
                raise ValueError(f"invalid edge row: {row}")
            adjacency[a].append((b, w))
            adjacency[b].append((a, w))
            edge_weight[(a, b)] = w
            edge_rows.extend((a, b))
            edge_cols.extend((b, a))
            edge_vals.extend((float(w), float(w)))
    matrix = sp.csr_matrix((edge_vals, (edge_rows, edge_cols)), shape=(n, n))
    return matrix, adjacency, edge_weight


def weighted_split(matrix, weights: np.ndarray, nodes: List[int]) -> Tuple[List[int], List[int]]:
    ids = np.array(nodes, dtype=np.int32)
    sub = matrix[ids][:, ids]
    degree = np.asarray(sub.sum(axis=1)).ravel()
    invsqrt = np.zeros_like(degree)
    nz = degree > 0
    invsqrt[nz] = 1.0 / np.sqrt(degree[nz])
    lap = sp.eye(len(ids), format="csr") - sp.diags(invsqrt) @ sub @ sp.diags(invsqrt)
    values, vectors = spla.eigsh(lap, k=2, which="SM", tol=1e-5, maxiter=10000)
    order_eig = np.argsort(values)
    fiedler = vectors[:, order_eig[1]]
    order = np.argsort(fiedler)
    ordered_ids = ids[order]
    ordered_weights = weights[ordered_ids]
    cumulative = np.cumsum(ordered_weights)
    half = float(ordered_weights.sum()) / 2.0
    j = int(np.searchsorted(cumulative, half))
    candidates = {max(1, min(len(ids) - 1, j)), max(1, min(len(ids) - 1, j + 1))}
    cut = min(candidates, key=lambda x: abs(float(cumulative[x - 1]) - half))
    return ordered_ids[:cut].tolist(), ordered_ids[cut:].tolist()


def recursive_spectral(matrix, weights: np.ndarray, parts: int = 8) -> List[int]:
    if parts != 8:
        raise ValueError("this research script currently targets exactly 8 shards")
    groups = [list(range(len(weights)))]
    for _ in range(3):
        nxt = []
        for group in groups:
            a, b = weighted_split(matrix, weights, group)
            nxt.extend((a, b))
        groups = nxt
    owner = [0] * len(weights)
    for d, group in enumerate(groups):
        for u in group:
            owner[u] = d
    return owner


def shard_loads(owner: List[int], weights: np.ndarray) -> List[int]:
    loads = [0] * 8
    for u, d in enumerate(owner):
        loads[d] += int(weights[u])
    return loads


def cut_value(owner: List[int], edge_weight) -> int:
    return sum(w for (a, b), w in edge_weight.items() if owner[a] != owner[b])


def local_moves(
    owner: List[int],
    loads: List[int],
    weights: np.ndarray,
    adjacency,
    max_load: float,
    seed: int,
    passes: int,
) -> Tuple[List[int], List[int]]:
    owner = owner.copy()
    loads = loads.copy()
    rng = random.Random(seed)
    nodes = list(range(len(owner)))
    for _ in range(passes):
        rng.shuffle(nodes)
        moved = 0
        for u in nodes:
            a = owner[u]
            by = [0] * 8
            for v, w in adjacency[u]:
                by[owner[v]] += w
            best = a
            best_gain = 0
            wu = int(weights[u])
            for b in range(8):
                if b == a or loads[b] + wu > max_load:
                    continue
                gain = by[b] - by[a]
                if gain > best_gain:
                    best_gain = gain
                    best = b
            if best != a:
                owner[u] = best
                loads[a] -= wu
                loads[best] += wu
                moved += 1
        if moved == 0:
            break
    return owner, loads


def pair_swaps(
    owner: List[int],
    loads: List[int],
    weights: np.ndarray,
    adjacency,
    max_load: float,
    rounds: int,
    candidates_per_pair: int,
) -> Tuple[List[int], List[int]]:
    owner = owner.copy()
    loads = loads.copy()
    edge_lookup = []
    for u in range(len(owner)):
        edge_lookup.append(dict(adjacency[u]))

    for _ in range(rounds):
        candidates = {}
        by_part = [[u for u, d in enumerate(owner) if d == a] for a in range(8)]
        for a in range(8):
            for b in range(8):
                if a == b:
                    continue
                scored = []
                for u in by_part[a]:
                    affinity = [0] * 8
                    for v, w in adjacency[u]:
                        affinity[owner[v]] += w
                    scored.append((affinity[b] - affinity[a], u))
                scored.sort(reverse=True)
                candidates[(a, b)] = scored[:candidates_per_pair]

        best = None
        for a in range(8):
            for b in range(a + 1, 8):
                for gu, u in candidates[(a, b)]:
                    wu = int(weights[u])
                    for gv, v in candidates[(b, a)]:
                        wv = int(weights[v])
                        if loads[a] - wu + wv > max_load:
                            continue
                        if loads[b] - wv + wu > max_load:
                            continue
                        gain = gu + gv - 2 * edge_lookup[u].get(v, 0)
                        if gain > 0 and (best is None or gain > best[0]):
                            best = (gain, u, v, a, b)
        if best is None:
            break
        _, u, v, a, b = best
        owner[u], owner[v] = b, a
        wu = int(weights[u])
        wv = int(weights[v])
        loads[a] += wv - wu
        loads[b] += wu - wv
    return owner, loads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", type=Path, help="prefix used by factor_highmask_graph_export")
    ap.add_argument("--max-load-ratio", type=float, default=1.025)
    ap.add_argument("--move-passes", type=int, default=30)
    ap.add_argument("--swap-rounds", type=int, default=100)
    ap.add_argument("--swap-candidates", type=int, default=60)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1618)
    ap.add_argument("--dp-rows", type=int, default=28)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.max_load_ratio < 1.0:
        ap.error("max-load-ratio must be >= 1")

    nodes_path = Path(str(args.prefix) + ".nodes.tsv")
    edges_path = Path(str(args.prefix) + ".edges.tsv")
    weights = read_nodes(nodes_path)
    matrix, adjacency, edge_weight = read_edges(edges_path, len(weights))
    average = float(weights.sum()) / 8.0
    max_load = average * args.max_load_ratio

    owner = recursive_spectral(matrix, weights)
    loads = shard_loads(owner, weights)
    if max(loads) > max_load:
        raise RuntimeError("spectral seed exceeds requested load cap")

    for cycle in range(args.cycles):
        owner, loads = local_moves(
            owner, loads, weights, adjacency, max_load,
            args.seed + cycle, args.move_passes,
        )
        if args.swap_rounds:
            owner, loads = pair_swaps(
                owner, loads, weights, adjacency, max_load,
                args.swap_rounds, args.swap_candidates,
            )

    cut = cut_value(owner, edge_weight)
    total_cuttable = sum(edge_weight.values())
    same_mask_n27 = 73_007_659_168 if len(weights) == 8192 else 0
    total_updates = total_cuttable + same_mask_n27
    peer_tib = cut * 4.0 * args.dp_rows / float(1 << 40)
    remote_fraction = cut / total_updates if total_updates else 0.0

    result = {
        "vertices": len(weights),
        "edges": len(edge_weight),
        "total_state_weight": int(weights.sum()),
        "max_load_ratio_requested": args.max_load_ratio,
        "shard_loads": loads,
        "max_load_ratio_actual": max(loads) / average,
        "min_load_ratio_actual": min(loads) / average,
        "cut_update_weight": cut,
        "remote_update_fraction": remote_fraction,
        "direct_peer_tib_per_residue": peer_tib,
        "owner": owner,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "owner"}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"partition_json={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
