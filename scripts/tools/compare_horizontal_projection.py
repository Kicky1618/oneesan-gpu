#!/usr/bin/env python3
import argparse


def load(path):
    vars_by_level = {}
    nodes = {}
    root = None
    grid_n = None
    with open(path, encoding="utf-8") as f:
        if f.readline().strip() != "ONEESAN_ZDD_V1":
            raise ValueError(f"{path}: bad magic")
        for line in f:
            p = line.split()
            if not p or p[0].startswith("#"):
                continue
            if p[0] == "grid_n":
                grid_n = int(p[1])
            elif p[0] == "root":
                root = int(p[1])
            elif p[0] == "var":
                vars_by_level[int(p[1])] = int(p[2])
            elif p[0] == "node":
                nodes[int(p[1])] = tuple(map(int, p[2:]))
            elif p[0] == "end":
                break
    if root is None or grid_n is None:
        raise ValueError(f"{path}: missing metadata")
    return grid_n, root, vars_by_level, nodes


def enumerate_family(path):
    n, root, vars_by_level, nodes = load(path)
    memo = {0: set(), 1: {()}}

    def visit(node_id):
        if node_id in memo:
            return memo[node_id]
        level, low, high = nodes[node_id]
        out = set(visit(low))
        edge = vars_by_level[level]
        out.update(tuple(sorted(item + (edge,))) for item in visit(high))
        memo[node_id] = out
        return out

    return n, visit(root)


def canonical_edges(n):
    W = n + 1
    edges = []
    eid = {}
    def vid(r, c): return r * W + c
    for r in range(W):
        for c in range(W):
            if c + 1 < W:
                u, v = vid(r, c), vid(r, c + 1)
                eid[tuple(sorted((u, v)))] = len(edges)
                edges.append((u, v))
            if r + 1 < W:
                u, v = vid(r, c), vid(r + 1, c)
                eid[tuple(sorted((u, v)))] = len(edges)
                edges.append((u, v))
    return edges, eid


def decode_horizontal(n, selected):
    W = n + 1
    V = W * W
    s, t = 0, V - 1
    edges, eid = canonical_edges(n)
    chosen = set(selected)
    hdeg = [0] * V

    for e in selected:
        u, v = edges[e]
        ru, cu = divmod(u, W)
        rv, cv = divmod(v, W)
        if ru != rv:
            raise ValueError(f"projection contains non-horizontal edge id {e}")
        hdeg[u] += 1
        hdeg[v] += 1

    for c in range(W):
        above = 0
        for r in range(W):
            v = r * W + c
            target_parity = 1 if v in (s, t) else 0
            vertical_degree_parity = target_parity ^ (hdeg[v] & 1)
            below = above ^ vertical_degree_parity
            if r + 1 < W:
                if below:
                    e = eid[tuple(sorted((v, (r + 1) * W + c)))]
                    chosen.add(e)
            elif below:
                raise ValueError("horizontal projection violates bottom boundary parity")
            above = below

    # Strong validation for the reconstructed object.
    deg = [0] * V
    adj = [[] for _ in range(V)]
    for e in chosen:
        u, v = edges[e]
        deg[u] += 1
        deg[v] += 1
        adj[u].append(v)
        adj[v].append(u)
    for v, d in enumerate(deg):
        want_terminal = v in (s, t)
        if want_terminal:
            if d != 1:
                raise ValueError(f"terminal {v} has degree {d}")
        elif d not in (0, 2):
            raise ValueError(f"vertex {v} has degree {d}")
    seen = set()
    stack = [s]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u])
    if t not in seen:
        raise ValueError("reconstructed graph does not connect terminals")
    if any(deg[v] and v not in seen for v in range(V)):
        raise ValueError("reconstructed graph has a disconnected cycle/component")

    return tuple(sorted(chosen))


def main():
    ap = argparse.ArgumentParser(description="Decode a horizontal-projection ZDD and compare with a full-edge ZDD")
    ap.add_argument("horizontal")
    ap.add_argument("full")
    args = ap.parse_args()

    n1, hp = enumerate_family(args.horizontal)
    n2, ff = enumerate_family(args.full)
    if n1 != n2:
        raise SystemExit(f"grid_n mismatch: {n1} vs {n2}")
    decoded = {decode_horizontal(n1, x) for x in hp}
    print(f"horizontal={len(hp)} decoded={len(decoded)} full={len(ff)}")
    print(f"only_decoded={len(decoded-ff)} only_full={len(ff-decoded)}")
    print(f"equal={int(decoded == ff)}")
    return 0 if decoded == ff else 1


if __name__ == "__main__":
    raise SystemExit(main())
