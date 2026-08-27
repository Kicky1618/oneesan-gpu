# Odd-TL midpoint contraction research

This branch studies a point-symmetry meet-in-the-middle replacement for the second half of the Grid-FP sweep.

The intended n=27 configuration has vertex/frontier width `W=28`, with the production factor split

```text
HIGH13 | center | LOW14
```

and modulus `4294967291` used by the probes below.

## 1. Verified beta=0 odd Temperley-Lieb factorization

`src/cpp/probes/odd_tl_gram_factorization_probe.cpp` represents the standard module `W_n^d` by ballot words and orders the final two steps as

```text
UU | UD | DU | DD
```

so that

```text
W_n^d -> W_{n-2}^{d-2} + 2 W_{n-2}^d + W_{n-2}^{d+2}.
```

For odd `d`, the finite beta=0 basis transform is

```text
A <- A + partial_d(C) + Q_d(D)
B <- B + partial_{d+2}(D)
C <- C + J_{d+2}(D)
D <- D
```

followed recursively in the four child sectors.

The resulting Gram form recurses as

```text
G'(n,d)
 = G'(n-2,d-2)
 + (H tensor G'(n-2,d))
 + (-(d+3)/(d+1)) G'(n-2,d+2),

H = [0 1; 1 0].
```

The explicit-Gram probe matched this factorization for every odd sector

```text
n = 1,3,5,...,13
1 <= d <= n, d odd
```

with four random vector pairs per sector modulo `4294967291`.

## 2. Integer-only normalization for CUDA

The rational coefficients of `Q_d` share the denominator

```text
s = (d+1)/2.
```

`src/cpp/probes/odd_tl_integer_normalization_probe.cpp` uses the coordinate

```text
Ahat = s * A'
```

so the local transform becomes

```text
Ahat <- s*A + s*partial_d(C) + Qtilde_d(D)
B'   <- B + partial_{d+2}(D)
C'   <- C + J_{d+2}(D)
D'   <- D
```

where every transform coefficient is a signed small integer. The canonical Gram recursion changes only by giving the A child the scalar `1/s^2`; the D child retains `-(s+1)/s`.

The integer-normalized transform has been independently checked against the explicit Gram form. This is the CUDA form: modular inverses move into the final leaf weights instead of appearing in trillions of transform contributions.

For W=28, the one-transform-per-occupancy-sector workload is

```text
2,659,582,660,624 cross-sector sparse contributions
  692,722,353,416 A-coordinate small-integer rescalings
-----------------------------------------------
3,352,305,014,040 transform arithmetic sites
```

plus `192,859,753,310` weighted leaf products after pairing reflected occupancy sectors. The earlier 2.852e12 estimate omitted the A-coordinate rescalings required by the integer normalization.

## 3. Static gather-stage compiler

`src/cpp/probes/odd_tl_transform_compiler_probe.cpp` compiles the recursive integer transform into breadth-by-depth stages.

At each recursion depth:

```text
phase 1: update A and B, reading old C,D
barrier
phase 2: update C, reading D
barrier
next recursion depth
```

Child ranges are disjoint, so all nodes at the same depth can execute concurrently. No global or shared-memory atomic is required.

Each sparse edge needs only

```text
22 bits : absolute source index (D_27 = 2,674,440 < 2^22)
 5 bits : signed coefficient in [-14,14]
```

so one contribution fits in a uint32. The transform compiler checks a CPU executor of this staged representation against the recursive transform.

The sum of cross-edge tables for one copy of every odd width through 27 is about 50.98 million edges, or about 194.5 MiB at four bytes per edge. This is small relative to B300 HBM and avoids generating topology maps on device.

## 4. Canonical Gram becomes a weighted involution

`src/cpp/probes/odd_tl_partner_weight_probe.cpp` recursively flattens the normalized canonical Gram to

```text
partner[i]
weight[i]
```

with

```text
partner[partner[i]] == i
weight[partner[i]] == weight[i].
```

The contraction is therefore

```text
sum_i weight[i] * x[i] * y[partner[i]].
```

There are only `3,707,851` dense one-defect states in total over occupied widths 1,3,...,27, so a uint32 partner table is about 14.14 MiB. The partner table is modulus-independent; only the weight table must be regenerated for each CRT prime.

## 5. Reflection and factor-to-TL maps

The existing point-symmetry probe represents a topology by a pairing signature. Converting a signature to the dense TL ballot word is simple:

- signature labels run high-to-low;
- TL terminals run low-to-high;
- a terminal is `U` iff it is the defect or its partner occurs later in low-to-high order.

An independent exhaustive Motzkin-state check through W=8 found `compat(signature_a, signature_b)` identical to the beta=0 TL Gram for every topology pair.

`src/cpp/probes/midpoint_factor_map_probe.cpp` compiles dense TL rank to factorized segment coordinates. For a state with `kH` occupied HIGH terminals, center occupancy `c`, and `kL` occupied LOW terminals, one coordinate fits in 26 bits:

```text
10 bits : LOW mask-local topology rank
10 bits : HIGH mask-local topology rank
 2 bits : center symbol N/R/L
 4 bits : HIGH ending height
```

The maximum HIGH/LOW segment-local rank is below 1024.

There are 210 valid `(kH,c,kL)` tuples and

```text
16,878,801
```

factor-map entries in total, so a uint32 table is about 64.4 MiB. Dense reflection needs another `3,707,851` uint32 ranks, about 14.14 MiB.

The exact physical occupancy positions do not affect mask-local topology rank; only the occupied counts and boundary heights matter. Actual mask-dependent all-ranks are recovered from the existing `*_mask_codes` and `*_packed_rank` tables.

## 6. LOW14/HIGH13 reflection pairing

`src/cpp/probes/midpoint_group_pairing_probe.cpp` records the exact coarse-group relation.

Let `h` be the reflected/right HIGH13 occupancy. The corresponding left LOW14 group is

```text
L = reverse13(h) | (b << 13),  b in {0,1}.
```

For a left HIGH13 occupancy `H` and left center occupancy `c`, the reflected state has

```text
right HIGH13 = h
right center = b
right LOW14  = reverse13(H) | (c << 13).
```

For fixed left `L` and `H`, exactly one value of `c` makes the full occupancy odd. Hence one CTA corresponds naturally to one exact odd occupancy sector.

There are `2^27` odd masks and no reflection-fixed odd mask at even width 28, so the join uses exactly `2^26 = 67,108,864` unordered occupancy pairs and multiplies the accumulated bilinear form by two.

## 7. Existing canonical B300 backend can host the first midpoint kernel

The production experimental file

```text
src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu
```

already gathers LOW14- or HIGH13-fixed groups into the same factorized scratch layout required by the midpoint kernel. Its authoritative HBM remains canonical Motzkin-rank order, but that is no longer a blocker for the first implementation.

A low-risk first path is:

```text
for each HIGH13 group h:
    gather reflected/right HIGH13 group once

    for b in {0,1}:
        L = reverse13(h) | (b<<13)
        gather left LOW14 group L

        process exact HIGH masks H in batches:
            choose the unique center occupancy c giving odd total occupancy
            skip unless S < reverse28(S)
            pack left exact sector into TL rank
            pack reflected right sector into TL rank
            integer odd-TL transform both vectors
            weighted-involution dot
```

The right HIGH13 group is reused for the two LOW14 groups differing only at position 13. Across all groups, every authoritative state is needed only once as one side of an unordered reflected occupancy pair.

This route changes no DP transition or authoritative state numbering. It can establish the real midpoint speedup before committing to a storage-layout migration.

## 8. Factorized-authoritative storage remains the second optimization

The existing RAM backend already has occupancy-major authoritative storage in

```text
src/cuda/gridfp/ramstream32_factorized_storage.hpp
```

and `src/cpp/probes/factorized_storage_geometry_probe.cpp` gives the W=28 geometry:

```text
main_states=385719506620
LOW14 avg contiguous run=21.588 uint32
LOW14 one-warp/row lane efficiency=50.672%
HIGH13 max nonempty factor blocks/group=22
HIGH13 owner-aware local-HBM weighted fraction=30.849%
```

`src/cuda/b300/probes/factorized_storage_rect_probe.cu` validates the compact descriptor proposed for a future B300 storage-major backend:

```cpp
(global base, local base, rows, global stride, width)
```

A LOW-fixed FBlock is one strided rectangle; it must not be expanded into billions of per-row `PeerInterval` records. A HIGH-fixed FBlock is one contiguous rectangle.

Storage-major HBM can remove per-state canonical rank/unrank from group I/O, but it is now deliberately postponed until the canonical-HBM midpoint kernel is measured.

## 9. Midpoint CTA classes

For an exact occupancy width `m`, the two uint32 vectors require

```text
m=17 :   37.98 KiB
m=19 :  131.22 KiB
m=21 :  459.27 KiB
m=23 :    1.59 MiB
m=25 :    5.67 MiB
m=27 :   20.40 MiB
```

The intended hierarchy is therefore:

```text
m <= 19 : one CTA, ordinary shared memory
m = 21  : small thread-block cluster / DSM
m = 23  : up to an 8-block cluster / DSM
m >= 25 : global-memory fallback (very small state fraction)
```

A fixed LOW group has fixed `kL`. HIGH masks can be batched by popcount; adjacent HIGH-popcount classes produce the same odd `m`, reducing the number of distinct shared-memory launch sizes.

## 10. Current implementation order

1. Keep production sources unchanged.
2. Use `point_symmetry_mitm_odd_tl.cpp` to connect actual half-DP vectors to the odd-TL join and compare against both explicit meander join and full DP.
3. Use the integer-normalized stage compiler and weighted involution as the CUDA table source.
4. Implement the midpoint kernel on the existing canonical-authoritative B300 factorized backend.
5. Stop after 14 rows for W=28 and compare residues against the full 28-row solver.
6. Measure midpoint group-gather time separately from TL transform time.
7. Only if canonical group I/O is material, switch authoritative HBM to the storage-major rectangle layout.
8. Then add owner-aware HIGH scheduling and DSM specializations.

Do not run broad GitHub Actions matrices for this branch; use targeted local/B300 regression runs.