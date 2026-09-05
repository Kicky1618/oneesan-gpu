#!/usr/bin/env python3
"""Independently check the single-loop pairing matrix and its determinant.

The all-order proof is in docs/research/frontier-batch-and-rank.md.
This finite test generates pairings directly and counts graph components.
"""
import argparse
from functools import cache
import json
from math import comb

import numpy as np


@cache
def pairings(n):
    if n == 0:
        return ((),)
    result = []
    for k in range(n):
        right = 2*k + 1
        for inside in pairings(k):
            for outside in pairings(n-k-1):
                mate = [right] + [j+1 for j in inside] + [0]
                mate += [j+right+1 for j in outside]
                result.append(tuple(mate))
    return tuple(result)


def connected(a, b):
    seen = {0}
    pending = [0]
    while pending:
        i = pending.pop()
        for j in (a[i], b[i]):
            if j not in seen:
                seen.add(j)
                pending.append(j)
    return len(seen) == len(a)


def determinant_mod(matrix, prime):
    a = matrix.astype(np.uint64)
    n = len(a)
    determinant = 1
    rank = 0
    for col in range(n):
        nonzero = np.flatnonzero(a[rank:, col])
        if not len(nonzero):
            continue
        pivot = rank + int(nonzero[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
            determinant = -determinant
        value = int(a[rank, col])
        determinant = determinant * value % prime
        # Both factors are < prime < 2^32, so uint64 multiplication is exact.
        factors = a[rank+1:, col] * np.uint64(pow(value, -1, prime)) % np.uint64(prime)
        products = factors[:, None] * a[rank, col:][None, :] % np.uint64(prime)
        a[rank+1:, col:] = (a[rank+1:, col:] + np.uint64(prime) - products) % np.uint64(prime)
        rank += 1
        if rank == n:
            break
    return rank, determinant % prime if rank == n else 0


def predicted_determinant(n, prime):
    def choose(k):
        return comb(2*n, k) if 0 <= k <= 2*n else 0
    order = 0
    determinant = 1
    for j in range(1, n+1):
        exponent = choose(n-j) - 2*choose(n-j-1) + choose(n-j-2)
        if j % 2:
            order += exponent
            leading = (-1)**((j-1)//2) * ((j+1)//2)
        else:
            leading = (-1)**(j//2)
        determinant = determinant * pow(leading % prime, exponent, prime) % prime
    assert order == comb(2*n, n)//(n+1)
    return determinant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pairs", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 1 <= args.max_pairs <= 8:
        parser.error("max-pairs must be in [1,8]")
    results = []
    for n in range(1, args.max_pairs+1):
        states = pairings(n)
        assert len(states) == len(set(states)) == comb(2*n, n)//(n+1)
        for a in states:
            assert all(a[a[i]] == i and a[i] != i for i in range(2*n))
        matrix = np.array([[connected(a, b) for b in states] for a in states], dtype=np.uint8)
        for prime in (1000000007, 4294967291, 4294966997):
            rank, determinant = determinant_mod(matrix, prime)
            expected = predicted_determinant(n, prime)
            assert rank == len(states) and determinant == expected
            row = dict(pairs=n, states=len(states), prime=prime, rank=rank, determinant=determinant)
            results.append(row)
            print(json.dumps(row), flush=True)
    # The symbolic leading coefficient is nonzero for every n; these checks also
    # exercise negative exponents, which start appearing at n=8.
    for n in range(1, 65):
        assert predicted_determinant(n, 4294967291)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
