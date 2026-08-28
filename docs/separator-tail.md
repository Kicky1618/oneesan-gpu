# Separator tail contraction: local OSR and renewal metric

This note records the current tail-contraction picture for the exact separator-channel branch.

## 1. Generic qutrit local operator is not free-fermionic

At q=i, dense TL(beta=0) is the familiar free-fermion Temperley-Lieb representation.  However, the dilute/vacancy sector needs the full 3-state Motzkin representation V=V(0)+V(1).

`src/cpp/probes/motzkin_local_osr_probe.cpp` evaluates the two-site generators over F_998244353 using a square root q of -1 and computes exact operator-Schmidt ranks by realignment.

Expected ranks are:

```text
operator                 ordinary rank   operator-Schmidt rank
I                        9               1
r                        3               3
l                        3               3
full Motzkin e            1               9
PTL epsilon               1               4
I + full e                9               9
I + epsilon               9               5
I + r + l + full e        7               9
I + r + l + epsilon       7               9
```

The important negative result is `OSR(full e)=9`, the maximum possible for a 3x3 local Hilbert space.  The NN -> LR creation branch of Grid-FP needs the vacancy-containing/full Motzkin sector, so a generic crossing face cannot be replaced by the free-fermion dense-TL generator alone.

Consequently the naive generic MPO bound for `t` remaining rows is

```text
one layer:    <= 9^t
double layer: <= 81^t
```

which is far too pessimistic to explain the measured A111960 ranks.

## 2. Renewal channels have a much smaller reachable tail

Use the candidate channel alphabet

```text
0, +, -, |
```

with `|` allowed only when the current +/- charge is zero.  A completed renewal channel additionally requires zero charge after the last symbol.

The number of completed channels at length r is A111961, the row sum of A111960.  If the last gap is allowed to carry a pending charge, the number of raw length-t tail extensions from one completed channel begins

```text
t = 0,1,2,3,4,5,6
F = 1,4,14,48,162,542,1802
```

and satisfies

```text
F_(t+1) = 3 F_t + A111961(t).
```

Thus for the proposed row-11 stop (`t=3`) the reachable raw separator tail has at most 48 labels per incoming renewal channel, versus the generic qutrit MPO bound 9^3=729.

`src/cpp/probes/separator_channel_metric_probe.cpp` checks both the A111960 channel cardinalities and this raw-tail count.

## 3. Candidate channel metric is a weighted involution

Doty--Giaquinto's tensor-space realization uses nondegenerate local bilinear forms whose nonzero qutrit pairings are only

```text
0 <-> 0
+ <-> -
- <-> +
```

(up to q-dependent scalar weights).

This motivates the candidate renewal-channel dual

```text
0 -> 0
+ -> -
- -> +
| -> |
```

applied symbolwise.  It is an involution and preserves the zero-charge condition in every renewal gap.  Therefore the candidate Gram matrix on renewal words has exactly one nonzero per row/column, rather than a dense `F_t x F_t` tail kernel.

This is not yet a proof that the Grid-FP separator basis pulls the link-state Gram form back to exactly this word metric.  The missing theorem is the explicit triangular map from separator tangles to the charge-word / colored-Motzkin basis.

## 4. Why this would matter

For W=28 after 11 completed rows the exact Schmidt factorization estimate is roughly

```text
sum_h rank_h        = 188366
factor entries      = 99979313981
uint32 storage      = about 372.45 GiB / residue
sum_h rank_h^2      = 6530108868
```

A dense channel-channel closure would immediately reintroduce billions of channel pairs.  A weighted-involution channel metric instead permits columnwise pairing and keeps closure work proportional to factor storage (times the small remaining-tail overhead), not to `rank^2`.

The research target is therefore stronger than proving the A111960 rank formula:

1. construct an explicit separator-tangle -> renewal-channel transform;
2. prove that the pullback of the local qutrit bilinear form is monomial/weighted-involutive on channels;
3. express one crossing grid face as a sparse update on these labels;
4. contract the final 3 rows without expanding to the full W=28 frontier vector.

## 5. Hardware side note: exact dense GEMM on B300

HGX B300 is unusually weak in FP64 compared with B200, so an FP64 exact-integer GEMM strategy is not attractive on B300.  If dense channel contractions become useful, an alternative is an RNS with pairwise-coprime prime powers <=1024:

```text
m_p = largest p^a <= 1024, for every prime p <= 1024.
```

There are 172 such moduli and their product has about 1478.9 bits.  A 28x28 self-avoiding path count has the rigorous non-backtracking upper bound

```text
#paths < 2 * 3^783,
```

which is about 1242 bits, so this RNS is sufficient in principle.

Centered residues have magnitude <=512 and are exactly representable in FP16. Products are <=2^18; FP32 accumulation remains integer-exact for short K chunks (for example <=32 products before modular reduction).  This suggests an exact FP16-Tensor-Core RNS GEMM path if the separator algorithm ultimately exposes large dense matrix multiplications.  It is only a hardware experiment candidate, not part of the correctness argument.
