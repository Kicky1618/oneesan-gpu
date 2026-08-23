# B300 row-depth structural sparsity study

This is a follow-up to v0.12. v0.12 proves that BLOCKED is zero at every row
boundary. The next question is whether large parts of MAIN are also known zero
from geometry, without inspecting count values.

## Frontier depth

Interpret a canonical frontier state as the usual Motzkin-like walk read from
high position to low position, starting at height 1:

- `N`: height unchanged;
- `R`: height -1;
- `L`: height +1.

Let `depth(m)` be the maximum height visited by that walk.

Exhaustive boolean-reachability experiments through W=14 show the exact row
boundary identity

```text
support after row r = { MAIN states m : depth(m) <= r }
```

until the maximum possible depth is reached. The same experiment checks that,
during the HIGH portion of row `r`, every reachable MAIN and BLOCKED state has
frontier depth at most `r`.

`factor_row_depth_support.cpp` verifies this without using arithmetic count
values: it propagates state-set reachability through the same
`include_horizontal()` / `blocked_exclude()` semantics. It checks W=6,8,10,12,14
and pins the W=14 row-boundary sequence

```text
r=1  8,192
r=2  80,782
r=3  159,094
r=4  190,400
r=5  196,406
r=6  196,924
r=7  196,938  (all states)
```

These are exactly the numbers of W=14 frontier states with maximum height at
most 1,2,...,7.

The geometric interpretation is that a frontier nesting of depth `h` cannot be
realized before enough grid rows exist to embed `h` nested noncrossing
connections.

## n=27 capped state counts

The height-capped state count is computed by a tiny DP that forbids height above
the chosen cap. For W=28 MAIN:

```text
cap 1       134,217,728   0.0348%
cap 2    18,457,556,052   4.7852%
cap 3   112,925,875,764  29.2767%
cap 4   240,539,369,472  62.3612%
cap 5   329,056,985,516  85.3099%
cap 6   369,274,024,420  95.7364%
cap 7   382,187,801,740  99.0844%
cap 8   385,169,379,172  99.8574%
cap 9   385,659,538,996  99.9845%
cap 10  385,715,191,452  99.9989%
cap 11  385,719,320,672  99.99995%
cap 12  385,719,502,616  99.9999990%
cap 13  385,719,506,592  ~100%
cap 14  385,719,506,620  100%
```

So structural sparsity is substantial only in roughly the first five or six
rows. It should not be treated as a whole-run sparse-DP strategy.

## Communication upper-bound model

Starting from v0.12:

- row-boundary BLOCKED gather is already removed;
- before row `r>1`, MAIN above depth `r-1` is structurally zero;
- after the HIGH portion of row `r`, MAIN/BLOCKED above depth `r` is
  structurally unreachable.

If an implementation could transfer exactly the height-capped regions while
zero-filling the rest locally, the n=27 logical HIGH-I/O word count would change
from

```text
v0.12 dense words     25,380,726,522,116
row-depth cap words   22,074,394,853,240
ratio                  0.869730614
reduction              13.0269386%
```

or in logical payload:

```text
92.334545196 -> 80.306180655 TiB/residue
```

Under the same 7/8 peer-fraction model used elsewhere, the capped payload is
about

```text
70.267908074 TiB/residue
```

This is a safe structural upper-bound opportunity, not a runtime result. The
current factorized storage is not ordered directly by global maximum frontier
height, so exploiting the cap requires either cheap per-factor depth metadata or
row-specific compact task lists. Building a huge per-state table would defeat
the purpose.

## Implementation direction

A promising low-metadata route is to precompute small per-factor quantities for
the existing LOW/HIGH code tables, sufficient to decide whether a composed
`(HIGH code, center, LOW code)` state can exceed a row depth cap. The HIGH I/O
kernel can then replace remote loads/stores outside the cap by local zeros or no
store.

Before implementing this candidate, profile v0.12 on real B300x8. If
`high_io_sum_s` remains dominant after the free BLOCKED-gather removal, the
additional ~13% logical-I/O opportunity is worth pursuing. If HIGH I/O is no
longer dominant, row-depth filtering adds branch/metadata cost for limited
whole-run benefit.
