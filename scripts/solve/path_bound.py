#!/usr/bin/env python3
"""Rigorous checkerboard-strip path bound and the production CRT prime table."""
from __future__ import annotations

PRIMES = [
    4294967291, 4294967279, 4294967231, 4294967197,
    4294967189, 4294967161, 4294967143, 4294967111,
    4294967087, 4294967029, 4294966997, 4294966981,
    4294966943, 4294966927, 4294966909, 4294966877,
    4294966829, 4294966813, 4294966769, 4294966667,
    4294966661, 4294966657, 4294966651, 4294966639,
    4294966619, 4294966591, 4294966583, 4294966553,
    4294966477, 4294966447, 4294966441, 4294966427,
    4294966373, 4294966367, 4294966337, 4294966297,
    4294966243, 4294966237, 4294966231, 4294966217,
    4294966187, 4294966177, 4294966163, 4294966153,
    4294966129, 4294966121, 4294966099, 4294966087,
]


def _strip_compatible(x: int, y: int, height: int) -> bool:
    """Whether two adjacent face-bit columns avoid a 2x2 checkerboard."""
    for r in range(height - 1):
        a = (x >> r) & 1
        b = (x >> (r + 1)) & 1
        c = (y >> r) & 1
        d = (y >> (r + 1)) & 1
        if a != b and c != d and a != c:
            return False
    return True


def _strip_orbits(height: int) -> tuple[list[int], list[int], list[int]]:
    """Column orbits under bit complement and vertical reflection."""
    states = 1 << height
    mask = states - 1
    representatives: list[int] = []
    orbit_of = [-1] * states
    sizes: list[int] = []
    for x in range(states):
        if orbit_of[x] >= 0:
            continue
        reflected = int(f"{x:0{height}b}"[::-1], 2)
        orbit = {x, x ^ mask, reflected, reflected ^ mask}
        index = len(representatives)
        representatives.append(x)
        sizes.append(len(orbit))
        for y in orbit:
            orbit_of[y] = index
    return representatives, orbit_of, sizes


def checkerboard_strip_count(height: int, width: int) -> int:
    """Exact no-checkerboard count using the symmetry quotient transfer.

    Complement and reflection preserve compatibility. Hence all columns in
    an orbit have the same continuation count, starting from the all-ones
    vector. An edge weight counts compatible members of the destination
    orbit; the final sum weights each orbit by its size. See the proof in
    docs/research/strip-orbit-quotient.md (this is not a division by orbit size).
    """
    representatives, orbit_of, sizes = _strip_orbits(height)
    nxt: list[list[tuple[int, int]]] = []
    for x in representatives:
        weights: dict[int, int] = {}
        for y, orbit in enumerate(orbit_of):
            if _strip_compatible(x, y, height):
                weights[orbit] = weights.get(orbit, 0) + 1
        nxt.append(list(weights.items()))
    dp = [1] * len(representatives)
    for _ in range(1, width):
        dp = [sum(weight * dp[y] for y, weight in row) for row in nxt]
    return sum(size * count for size, count in zip(sizes, dp))


def simple_path_upper_bound(n: int, max_strip_height: int = 9) -> tuple[int, list[int]]:
    """Rigorous upper bound for corner-to-corner simple paths.

    Fix one outer-boundary s-t path P0. Every s-t path is a T-join and can be
    written uniquely as P0 XOR boundary(F), where F is a subset of the n^2
    bounded faces. At an interior vertex P0 has degree zero. If the four
    surrounding face bits form either checkerboard pattern, boundary(F) uses
    all four incident edges, giving degree four, which a simple path cannot
    have. Thus path count is at most the number of n x n face-bit matrices
    without checkerboard 2x2 blocks.

    Rows are partitioned into independent strips and cross-strip constraints
    are dropped, which can only enlarge the set. Dynamic programming chooses
    the product-minimizing strip partition up to max_strip_height.
    """
    if n < 1:
        return 1, []
    hmax = min(max_strip_height, n)
    strip = {h: checkerboard_strip_count(h, n) for h in range(1, hmax + 1)}
    best: list[int | None] = [None] * (n + 1)
    parts: list[list[int] | None] = [None] * (n + 1)
    best[0], parts[0] = 1, []
    for rows in range(1, n + 1):
        for h, count in strip.items():
            if h > rows or best[rows - h] is None:
                continue
            candidate = best[rows - h] * count
            if best[rows] is None or candidate < best[rows]:
                best[rows] = candidate
                parts[rows] = parts[rows - h] + [h]
    assert best[n] is not None and parts[n] is not None
    return best[n], parts[n]


def primes_for_bound(bound: int) -> list[int]:
    product = 1
    out: list[int] = []
    for p in PRIMES:
        out.append(p)
        product *= p
        if product > bound:
            return out
    raise ValueError(
        f"CRT prime capacity insufficient: product has {product.bit_length()} bits, "
        f"bound has {bound.bit_length()} bits"
    )
