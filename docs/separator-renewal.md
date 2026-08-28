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

## Renewal-gap model

A111960 is a renewal array.  Equivalently

```text
C(r,h) = sum_{n_0+...+n_h=r-h} prod_j T(n_j),
```

where `T(n)` is the central trinomial coefficient.  Since `T(n)` counts ternary words in `{-1,0,+1}^n` with total sum zero, a channel of type `(r,h)` can be counted explicitly as

```text
w_0 | w_1 | ... | w_h
```

where every `w_i` is a zero-sum ternary word and the sum of their lengths is `r-h`.  Diagrammatically, the proposed interpretation is that `h` propagating separator strands divide the strip into `h+1` independent neutral gaps.

This is already an explicit combinatorial labelling of the correct cardinality, but appending a row is not obviously local in this representation because a new ternary letter can temporarily unbalance the last neutral word.

## Colored-Motzkin channel model

The continued fraction recorded for A111960 has the Jacobi form

```text
1 / (1 - (1+y)x - 2x^2/(1-x-x^2/(1-x-x^2/(...))))
```

which gives a more useful local path model.

A channel is represented by a Motzkin path of length `r` with the following colors:

- an up step has one color;
- a horizontal step above height zero has one color;
- a down step landing above height zero has one color;
- a down step from height one to zero has two colors;
- a horizontal step at height zero has two colors, one ordinary and one distinguished/marked.

Then `C(r,h)` is the number of such colored paths having exactly `h` distinguished ground-horizontal steps.  For example, at length two:

```text
HH contributes (1+y)^2
UD contributes 2
=> 3 + 2y + y^2
```

so the row is `3,2,1` exactly.

This model is checked independently by

```text
src/cpp/probes/separator_colored_motzkin_probe.cpp
```

against the coefficient formula for A111960.

This is a substantially better candidate for an actual separator basis than the renewal-gap words: one additional grid row corresponds to appending one local Motzkin step and at most choosing one of two colors.  The remaining problem is to construct the linear map from an actual separator tangle to this colored path basis and prove triangularity/non-singularity.

## Production matrix and alternating Catalans

Writing the Riordan array as `(g,f)=(G,xG)`, its Riordan `A` and `Z` series are

```text
A(y) = y / f^{-1}(y) = y + sqrt(1+4y^2)
Z(y) = (A(y)-1)/y.
```

Therefore

```text
A(y) = 1 + y + 2y^2 - 2y^4 + 4y^6 - 10y^8 + ...
Z(y) = 1 + 2y - 2y^3 + 4y^5 - 10y^7 + ...
```

and the nontrivial coefficients are

```text
2*(-1)^(m-1)*Catalan(m-1).
```

Consequently the dimension triangle obeys a fixed Hessenberg-Toeplitz production rule.  For `h>=1`,

```text
C(r+1,h)
 = C(r,h-1) + C(r,h)
 + sum_{m>=1} 2*(-1)^(m-1)*Cat(m-1)*C(r,h+2m-1),
```

with the analogous first-column `Z` rule for `h=0`.

The alternating Catalan tail is highly suggestive of the beta=0 Temperley-Lieb basis changes already found in the odd-TL midpoint work: lowering a large separator height amounts to closing nested noncrossing strands, and Catalan coefficients are the natural noncrossing inversion coefficients.

## Pascal times pure-connectivity factorisation

OEIS also records

```text
A111960 = Pascal * A111959,
```

where A111959 is the renewal array for the aerated central binomial coefficients,

```text
(1/sqrt(1-4x^2), x/sqrt(1-4x^2)).
```

This separates the ternary channel into a binomial/vacancy part and a pure pair-connectivity part.  A111959 is especially structured: its infinitesimal generator has only one nonzero subdiagonal,

```text
B[k+2,k] = 2*(k+1),
```

so formally

```text
A111959 = exp(B).
```

Pascal itself is `exp(A)` with `A[k+1,k]=k+1`.  Hence the channel-dimension array factors into two extremely local raising generators.  This may be the algebraic route to an explicit basis: one generator inserts a neutral/vacancy event, while the other inserts a pair-connectivity event.

## Relation to Motzkin / dilute-TL diagrams

The appearance of Motzkin paths is structurally plausible rather than merely numerical.  The standard basis diagrams of the Motzkin algebra and dilute Temperley-Lieb algebra coincide as diagram sets, although their multiplications differ.  Therefore no argument should identify the two algebras outright, but a Motzkin path basis for the separator module is compatible with the underlying dilute-TL diagram combinatorics.

## Next mathematical task

Construct an explicit linear map

```text
separator tangle <-> colored Motzkin channel
```

and prove:

1. the colored paths span the row-boundary Schmidt space;
2. distinct paths are linearly independent until a side dimension saturates;
3. one additional grid row acts by a sparse local rule on colored-path labels;
4. the alternating-Catalan production matrix is the matrix of the same local rule after eliminating the hidden Motzkin height;
5. the point-symmetry / beta=0 Gram closure can be evaluated directly in channel coordinates without reconstructing the full `W=28` frontier vector.

A concrete route for item 1 is to compute lexicographic pivot bases of the exact small-width Schmidt matrices, recursively label the new pivots by colored-Motzkin extensions, and test whether the resulting basis-change matrices become triangular with unit or small-integer diagonal.  If that pattern stabilizes, it should expose the required diagrammatic map.
