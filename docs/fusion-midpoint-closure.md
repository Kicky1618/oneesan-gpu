# Fusion midpoint closure

Research branch: `research/odd-tl-midpoint`.

This note records the point-reflection closure after the Grid-FP state has already
been evolved in the finite beta=0 fusion basis.

For a dense occupied sector of odd width `m`, let `T_m` be the finite odd-TL
change of coordinates, `D_m` the canonical weighted-involution Gram form, and `R_m`
the physical horizontal reflection in the diagram basis.  The midpoint operator in
fusion coordinates is

```text
K_m = D_m T_m R_m T_m^{-1}.
```

No odd-TL transform is required at the midpoint when the DP itself uses the fusion
backend; the state is already in `T_m` coordinates.

## Sparse support and A004148

`src/cpp/probes/fusion_midpoint_closure_probe.cpp` applies `K_m` one basis vector at
a time without forming a dense inverse.  It reuses the exact finite odd-TL transform,
its explicitly inverted triangular recursion, diagram reflection, and the canonical
Gram recursion.

The observed nonzero counts are

```text
m   dim W_m^1   nnz K_m
1       1             1
3       2             2
5       5             8
7      14            37
9      42           185
11    132           978
13    429          5373
15   1430         30372
```

These are exactly OEIS A004148 evaluated at the same odd `m`.  A004148 counts
peakless Motzkin paths, equivalently noncrossing partial matchings in which adjacent
vertices are not paired (RNA secondary structures).  This strongly suggests that the
support of the fusion reflection closure is naturally indexed by those partial
matchings.  A full explicit bijection between a nonzero pair of fusion paths and the
matching remains to be written down.

The A004148 recurrence is

```text
a(n+1) = a(n) + sum_{k=1}^{n-1} a(k) a(n-1-k),
a(0)=a(1)=1.
```

The first term corresponds to an unmatched first vertex; the convolution corresponds
to pairing it with a nonadjacent vertex and independently choosing the nested and
remaining noncrossing structures.  Diagrammatically, the absence of adjacent pairs is
consistent with beta=0: an immediately created-and-closed local cup produces one
contractible loop and vanishes.

## Width-28 closure work

For dilute width 28, the directed sparse closure work predicted from A004148 is

```text
sum_{m odd} binom(28,m) a(m)
= 23,977,709,765,604 contributions / CRT residue.
```

Point reflection has no fixed odd occupancy mask at even width, so mask pairs can be
processed once and doubled.  The actual bilinear work is therefore

```text
11,988,854,882,802 contributions / CRT residue.
```

The dominant occupied widths are `m=19,21,23`; `m=25,27` together contribute only
about 3.26% of the reflected-mask-pair work.

The dense-table storage is much smaller than the dilute work because one `K_m` is
shared by every occupancy mask of size `m`.  Cumulative edge counts are approximately

```text
m<=23 :    44,550,375 edges
m<=25 :   271,011,268 edges
m<=27 : 1,663,262,280 edges
```

Thus a practical first GPU implementation can keep CSR tables through `m=23` or
`m=25` and use a generated/F-move fallback only for the very rare largest occupied
sectors.

For `m<=23`, `dim W_m^1 <= 208012`, so an 18-bit destination rank is sufficient.
The closure coefficients also form a small alphabet at the sizes checked.  Exact
rational enumeration gives

```text
m=11: 11 distinct nonzero coefficients, denominators {1,2}
m=13: 14 distinct nonzero coefficients, denominators {1,2}
m=15: 22 distinct nonzero coefficients, denominators {1,2,3,4}
```

This makes a packed `destination + coefficient-code` edge representation plausible.

## Shared-memory execution

For each reflected occupancy-mask pair `(S, reverse(S))`, load the two contiguous
fusion blocks, then contract the same `K_m` CSR.  The existing occupancy-major fusion
ID makes these blocks contiguous by construction.

At the important widths, the two input vectors require roughly

```text
m=19 : 131 KiB
m=21 : 459 KiB
m=23 : 1.59 MiB
```

so `m=19` fits in one large Blackwell CTA and `m=21/23` naturally map to distributed
shared memory clusters.  The much rarer `m>=25` sectors can use a global-memory
fallback.

## Compact-MPO check

The coincidence with peakless Motzkin paths suggested that `K_m` might have a tiny
MPO bond dimension.  Exact operator-Schmidt ranks in the raw two-colour fusion-word
alphabet do not support that stronger conjecture.

Examples of cut ranks are

```text
m=9,  r=4: middle cut rank 62 out of a 100 x 100 prefix/suffix pair space
m=11, r=5: near-middle cut rank 96 out of 100 on one side
m=13, r=6: middle cut rank 643 out of 1225
```

The bond dimension is therefore growing substantially; a direct sparse closure is a
more credible near-term implementation than a small fixed-bond MPO.

The A004148 support structure is still useful: it should allow direct combinatorial
CSR generation without constructing `T R T^{-1}` by linear algebra, and may expose a
more specialized factorization than a generic MPO.
