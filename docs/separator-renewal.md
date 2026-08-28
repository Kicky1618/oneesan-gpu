# Exact separator-channel / renewal-array experiment

This note records a second line of attack, independent of the odd-TL midpoint transform.

## Balanced Schmidt cut

For even frontier width `W=2k`, split the canonical Motzkin state at the balanced spatial cut

```text
HIGH k | LOW k
```

and condition on the intermediate Motzkin height `h`.  The count vector at a completed-row boundary is then a direct sum of matrices

```text
V = direct_sum_h V_h,
V_h in F_p^(H_h x L_h).
```

The experiment uses `p=4294967291` and computes exact matrix ranks by Gaussian elimination for `W<=14`.

The important reason to use the balanced cut instead of the production `13|center|14` split is geometric: only one horizontal cell update per completed grid row crosses the balanced cut.  Updates strictly inside either half cannot increase Schmidt rank.

Probe:

```text
src/cpp/probes/frontier_schmidt_rank_probe.cpp
```

## Experimental rank triangle

Before finite side dimensions saturate, the measured ranks are

```text
r=0: 1
r=1: 1 1
r=2: 3 2 1
r=3: 7 7 3 1
r=4: 19 20 12 4 1
r=5: 51 61 40 18 5 1
r=6: 141 182 135 68 25 6 1
...
```

This is exactly OEIS A111960, the renewal/convolution triangle of the central trinomial coefficients.  Its Riordan representation is

```text
(G(x), x G(x)),
G(x)=1/sqrt(1-2x-3x^2).
```

Hence the universal unsaturated channel count is

```text
C(r,h) = [x^(r-h)] G(x)^(h+1)
       = [x^(r-h)] (1-2x-3x^2)^(-(h+1)/2).
```

A convenient exact recurrence, writing `n=r-h`, is

```text
c_0 = 1
(n+1)c_(n+1) = (2n+h+1)c_n + 3(n+h)c_(n-1).
```

The working finite-width conjecture, verified by the exact small-width experiments so far, is

```text
rank V_h^(r) = min(C(r,h), H_h, L_h).
```

`frontier_schmidt_rank_probe.cpp` now treats this formula as a regression assertion.

A lightweight formula-only probe is

```text
src/cpp/probes/separator_channel_formula_probe.cpp
```

## W=28 projection

For a balanced `14|14` split the segment dimensions used by the experiment are

```text
h : H_h     L_h
0 : 196938  113634
1 : 345957  196938
2 : 417522  232323
3 : 409500  220584
4 : 343278  177177
5 : 250887  122694
6 : 161070   73710
7 :  90909   38376
8 :  44928   17199
9 :  19278    6552
10:   7084    2079
11:   2183     532
12:    546     104
13:    105      14
14:     14       1
```

The full canonical main vector has `385719506620` entries.  If each block is stored as an exact rank factorization `X_h Y_h^T`, the projected storage is:

```text
completed rows   factor/full     factor entries       uint32 storage
8                0.007798        3,007,976,835        11.21 GiB
9                0.025115        9,687,281,274        36.09 GiB
10               0.080749       31,146,613,441       116.03 GiB
11               0.259202       99,979,313,981       372.45 GiB
12               0.830710      320,420,966,238      1193.66 GiB
13               1.530234      590,241,270,246      2198.82 GiB
14               1.539312      593,742,784,829      2211.86 GiB
```

Thus the exact Schmidt structure is very strong for roughly the first 10--11 rows, then finite side dimensions force saturation.

The row sums of the universal channel triangle are OEIS A111961:

```text
1,2,6,18,56,176,558,1778,5686,18230,58558,188366,...
```

with asymptotic growth `(1+sqrt(5))^r/sqrt(5)`.

## Important: do not expand after row 11

The tempting strategy

```text
compressed first 11 rows -> materialize full frontier -> ordinary rows 12..14
```

is not viable.  Materializing `V_h=X_hY_h^T` costs approximately

```text
sum_h H_h L_h rank_h
```

field products.  At row 11 this is about `1.287e16` products per residue.

Therefore this line only becomes useful if the final contraction is performed directly in factor/channel form.

## Candidate canonical channel basis

A111960 is a renewal array.  Equivalently

```text
C(r,h) = sum_{n_0+...+n_h=r-h} prod_j T(n_j),
```

where `T(n)` is the central trinomial coefficient.  This suggests a separator-tangle normal form consisting of `h+1` neutral blocks separated by `h` renewal events.

The row sum A111961 also has a Motzkin-path interpretation: it counts Motzkin paths with a marking/color choice on steps returning to height zero.  This is promising because it may turn the numerical Schmidt basis into a combinatorial, sparse-update basis and remove Gaussian recompression entirely.

## Next mathematical task

Construct an explicit map

```text
separator tangle <-> A111960 renewal/colored-Motzkin channel
```

and prove:

1. the channels span the row-boundary coefficient matrix;
2. distinct channels are linearly independent until a side dimension saturates;
3. one additional grid row acts by a sparse local rule on channel labels;
4. the point-symmetry / beta=0 Gram closure can be evaluated directly on the channel factors without reconstructing the full `W=28` frontier vector.

The fourth item is essential.  A successful direct closure would make this more than a memory compression trick; it could remove most of the first-half full-frontier work before the odd-TL midpoint join.
