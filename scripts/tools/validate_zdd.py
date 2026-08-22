#!/usr/bin/env python3
import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ONEESAN_ZDD_V1 and compute exact family cardinality")
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    meta = {}
    vars_by_level = {}
    nodes = {}
    with args.path.open("r", encoding="utf-8") as f:
        magic = f.readline().strip()
        if magic != "ONEESAN_ZDD_V1":
            raise SystemExit(f"bad magic: {magic!r}")
        for line_no, line in enumerate(f, 2):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if p[0] == "var":
                if len(p) != 5:
                    raise SystemExit(f"line {line_no}: malformed var")
                level, edge_i, u, v = map(int, p[1:])
                if level in vars_by_level:
                    raise SystemExit(f"line {line_no}: duplicate level {level}")
                vars_by_level[level] = (edge_i, u, v)
            elif p[0] == "node":
                if len(p) != 5:
                    raise SystemExit(f"line {line_no}: malformed node")
                nid, level, low, high = map(int, p[1:])
                if nid <= 1 or nid in nodes:
                    raise SystemExit(f"line {line_no}: invalid/duplicate node {nid}")
                if high == 0:
                    raise SystemExit(f"line {line_no}: non-reduced ZDD node {nid} has high=0")
                nodes[nid] = (level, low, high)
            elif p[0] == "end":
                break
            elif len(p) == 2:
                meta[p[0]] = int(p[1])
            else:
                raise SystemExit(f"line {line_no}: unknown record")

    variables = meta.get("variables")
    root = meta.get("root")
    if variables is None or root is None:
        raise SystemExit("missing variables/root metadata")
    if len(vars_by_level) != variables:
        raise SystemExit(f"variable map has {len(vars_by_level)} entries, expected {variables}")
    if set(vars_by_level) != set(range(1, variables + 1)):
        raise SystemExit("variable levels are not exactly 1..variables")
    if root > 1 and root not in nodes:
        raise SystemExit(f"root node {root} not present")

    # Validate ordering and reachability while computing exact cardinalities.
    memo = {0: 0, 1: 1}
    visiting = set()

    def count(nid: int) -> int:
        if nid in memo:
            return memo[nid]
        if nid not in nodes:
            raise SystemExit(f"dangling node reference {nid}")
        if nid in visiting:
            raise SystemExit(f"cycle involving node {nid}")
        visiting.add(nid)
        level, low, high = nodes[nid]
        if not (1 <= level <= variables):
            raise SystemExit(f"node {nid}: bad level {level}")
        for child in (low, high):
            if child > 1:
                if child not in nodes:
                    raise SystemExit(f"node {nid}: dangling child {child}")
                child_level = nodes[child][0]
                if child_level >= level:
                    raise SystemExit(
                        f"node {nid}: child {child} level {child_level} is not below parent level {level}"
                    )
        value = count(low) + count(high)
        visiting.remove(nid)
        memo[nid] = value
        return value

    exact = count(root)

    reachable = set()
    stack = [root]
    while stack:
        nid = stack.pop()
        if nid <= 1 or nid in reachable:
            continue
        reachable.add(nid)
        _, lo, hi = nodes[nid]
        stack.extend((lo, hi))
    if len(reachable) != len(nodes):
        raise SystemExit(f"file contains unreachable nodes: reachable={len(reachable)} stored={len(nodes)}")

    print(f"format=ONEESAN_ZDD_V1")
    print(f"grid_n={meta.get('grid_n')}")
    print(f"variables={variables}")
    print(f"nodes={len(nodes)}")
    print(f"root={root}")
    print(f"exact_cardinality={exact}")
    print("valid=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
