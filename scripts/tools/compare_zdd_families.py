#!/usr/bin/env python3
import argparse


def load(path):
    vars_by_level = {}
    nodes = {}
    root = None
    with open(path, encoding="utf-8") as f:
        if f.readline().strip() != "ONEESAN_ZDD_V1":
            raise ValueError(f"{path}: bad magic")
        for line in f:
            p = line.split()
            if not p or p[0].startswith("#"):
                continue
            if p[0] == "root":
                root = int(p[1])
            elif p[0] == "var":
                vars_by_level[int(p[1])] = int(p[2])
            elif p[0] == "node":
                nodes[int(p[1])] = tuple(map(int, p[2:]))
            elif p[0] == "end":
                break
    if root is None:
        raise ValueError(f"{path}: no root")
    return root, vars_by_level, nodes


def enumerate_family(path):
    root, vars_by_level, nodes = load(path)
    memo = {0: set(), 1: {()}}

    def visit(node_id):
        if node_id in memo:
            return memo[node_id]
        level, low, high = nodes[node_id]
        result = set(visit(low))
        edge = vars_by_level[level]
        result.update(tuple(sorted(item + (edge,))) for item in visit(high))
        memo[node_id] = result
        return result

    return visit(root)


def main():
    ap = argparse.ArgumentParser(description="Enumerate two small ONEESAN ZDDs and compare edge-set families")
    ap.add_argument("left")
    ap.add_argument("right")
    args = ap.parse_args()
    a = enumerate_family(args.left)
    b = enumerate_family(args.right)
    print(f"left={len(a)} right={len(b)}")
    print(f"only_left={len(a-b)} only_right={len(b-a)}")
    print(f"equal={int(a == b)}")
    return 0 if a == b else 1


if __name__ == "__main__":
    raise SystemExit(main())
