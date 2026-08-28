# Fusion-basis midpoint contraction

Research branch: `research/odd-tl-midpoint`.

This note records how point-symmetry MITM should be combined with the fusion-basis Grid-FP backend.

## Coordinate convention

For one exact odd occupancy mask `S`, let `v_S` be the dense one-defect link-state vector in the ordinary diagram basis and let

```text
x_S = T_m v_S
```

be the finite beta=0 odd-TL fusion coordinates used by the fusion backend, where `m=popcount(S)`.

The Gram factorization is

```text
G_m = T_m^T D_m T_m,
```

where `D_m` is the recursively weighted involution described by the odd-TL factorization probes.

For the reflected occupancy mask `rho(S)`, the midpoint term is

```text
v_S^T G_m R_m v_rho(S).
```

Because the DP already stores `x=T v`, the left member needs no basis transform.  Only one member of each unordered reflected mask pair is round-tripped:

```text
x_rho
  -- T^-1 -->  v_rho
  -- R    -->  R v_rho
  -- T    -->  y_rho
```

and then

```text
term(S) = x_S^T D_m y_rho.
```

Thus a reflected mask pair costs two sparse transforms on only half of the states.  Since forward and inverse transforms have the same sparse-map count, the total transform work over all pairs is approximately one full-vector transform, not two.

`src/cpp/probes/fusion_midpoint_roundtrip_probe.cpp` checks this identity on arbitrary fusion-coordinate vectors, independently of Grid-FP reachability.

## Reflection permutation in the diagram basis

`R_m` is only a permutation.  For each dense link state, reflect the noncrossing matching order and re-rank the resulting ballot word.  A table

```text
reflect_rank_m[rank]
```

is enough.  Across all odd `m<=27`, the dense fusion/link-state dimensions sum to only `3707851`, so one uint32 reflection table for every width is about 14.14 MiB.

The permutation is an involution and can be executed in place by swapping only pairs with

```text
rank < reflect_rank_m[rank].
```

No second right-hand vector is required merely for reflection.

## Cost at W=28

The previously measured unnormalised sparse odd-TL transform has about

```text
2,659,582,660,624
```

cross-sector contributions over the complete width-28 dilute state space.  An inverse transform has the same map count.  Applying inverse+forward to one member of each reflected occupancy pair therefore has approximately this same total contribution count.

The final `D_m` contraction visits half of the `385719506620` states:

```text
192,859,753,310
```

weighted products.

Hence the fusion-basis midpoint closure can remain of order

```text
~2.66e12 sparse transform contributions
+0.193e12 final products,
```

before implementation-specific diagonal rescaling or modular-reduction costs.  This is much smaller than one complete width-28 Grid-FP row in the current full-state work model.

If the integer-normalised odd-TL transform is used instead, the extra diagonal rescalings raise the transform arithmetic-site count to roughly `3.35e12`, but most of those operations use tiny integer coefficients.  The rational and integer-normalised variants should therefore both be benchmarked on B300 rather than chosen from operation count alone.

## Shared-memory/DSM execution

The authoritative fusion ID is occupancy-major, so each exact occupancy sector is already one contiguous Catalan block.  No midpoint regroup/permutation of the global frontier is necessary.

For an unordered reflected mask pair `(S,rho(S))`:

1. load `x_S` and `x_rho`;
2. leave `x_S` unchanged;
3. apply `T_m^-1` to `x_rho` in place;
4. apply the in-place `reflect_rank_m` involution;
5. apply `T_m` in place;
6. contract against `x_S` using `D_m`'s partner/weight table;
7. reduce one scalar per block/cluster.

Two uint32 vectors require approximately:

```text
m=19 : 131 KiB
m=21 : 459 KiB
m=23 : 1.59 MiB
m=25 : 5.67 MiB
m=27 : 20.4 MiB
```

so `m<=19` is a natural one-CTA path, `m=21/23` a cluster-DSM path, and the very rare `m=25/27` sectors can use a global-memory fallback initially.

## Half-twist identity

There is also a useful independent description of reflection.  In the ordinary beta=0 TL module, let `e_i` be the TL generator and choose the normalised braid generator

```text
b_i = I + i e_i,
```

which is the Kauffman-bracket braid representation at `A^-2=i` (`beta=-A^2-A^-2=0`).  Small exact tests give

```text
Delta_n = (b_1)(b_2 b_1)...(b_{n-1}...b_1)
        = gamma_n R_n
```

on the one-defect standard module, through `n=9`, with `gamma_n` a global fourth-root phase (`gamma_{2r+1}` is consistent with `i^(r^2)`).

This gives a second exact route to reflection because the fusion-basis `e_i` action is local.  It is not the preferred midpoint implementation: the half twist contains `m(m-1)/2` local braid factors, whereas the `T^-1 -> R -> T` roundtrip is substantially cheaper.  It is nevertheless useful for proofs and independent regression tests.

## Sparse closure matrix and A004148

Define the direct fusion closure matrix

```text
K_m = D_m (T_m R_m T_m^-1).
```

Its total nonzero count for odd dense widths is

```text
m=3  : 2
m=5  : 8
m=7  : 37
m=9  : 185
m=11 : 978
m=13 : 5373
m=15 : 30372
```

The sequence is exactly `A004148(m)` for every tested odd `m`.  A004148 counts peakless Motzkin paths and equivalently RNA secondary structures / noncrossing partial matchings with no adjacent matched pair.

This is currently an experimentally verified identity, not yet a proof.  The agreement through `m=15` strongly suggests that the support of the fusion half-twist/closure has a direct RNA-structure parametrisation.

Using A004148 through `m=27`, the predicted W=28 direct-CSR contraction work is

```text
sum_{m odd} binom(28,m) A004148(m)
= 23,977,709,765,604
```

or about `62.16` nonzeros per dilute state on average.  This is useful as an independent closure implementation, but it is still much more arithmetic than the transform roundtrip above.

The storage picture is more favourable:

```text
sum_{m odd, m<=23} A004148(m) = 44,550,375 edges
```

which is only about 170 MiB if one edge can be packed into 32 bits.  The large tables are `m=25` and especially `m=27`; those sectors together contain only about 0.65% of W=28 states.  Therefore an `A004148`-CSR implementation remains attractive as a regression path or hybrid fallback.

## Research questions

1. Prove `nnz(K_m)=A004148(m)` and construct the explicit bijection from nonzero fusion matrix entries to peakless Motzkin paths / RNA secondary structures.
2. Prove the half-twist/reflection identity and its global phase at beta=0.
3. Benchmark rational versus integer-normalised `T^-1/R/T` on B300.
4. Fuse inverse-transform, reflection permutation, forward-transform and `D` contraction into one occupancy-block kernel so the transformed right vector never returns to HBM.
5. Combine this closure with the `M+D` atomic-free fusion Grid-FP recurrence; then the first 14 rows and the midpoint closure use the same occupancy-major authoritative layout.
