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

For odd `d`, the finite beta=0 basis transform used by the probe is

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

The probe independently builds the beta=0 Gram form by overlaying pairs of link states. A component is accepted only when it contains one defect from each side; a closed component therefore contributes zero.

Local compilation used:

```bash
g++ -std=c++20 -O2 -Wall -Wextra \
  src/cpp/probes/odd_tl_gram_factorization_probe.cpp \
  -o odd_tl_gram_factorization_probe
```

The tested implementation matched explicit Gram contraction for every odd sector

```text
n = 1,3,5,...,13
1 <= d <= n, d odd
```

with four independent random vector pairs per sector modulo `4294967291`.

This probe is deliberately independent of the Grid-FP codec. It fixes the signs/orientation of `partial`, `J`, `Q`, and the `-(d+3)/(d+1)` canonical weight before CUDA work starts.

## 2. Why factorized-authoritative storage matters

The existing B300 file

```text
src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu
```

uses factorized ordering only inside scratch groups. The authoritative HBM arrays are still in canonical Motzkin rank order; group I/O reconstructs a `MateID` and calls `factor_global_rank_*()` before loading/storing the sharded arrays.

The RAM backend already contains a better authoritative layout in

```text
src/cuda/gridfp/ramstream32_factorized_storage.hpp
```

For each fixed intermediate height it orders both segment code lists by

```text
(occupancy mask, rank inside that occupancy mask).
```

The main state is then stored as row-major HIGH x LOW rectangles for each `(high_end_height, center_symbol)` block.

For a fixed HIGH occupancy group, every factor block becomes one consecutive authoritative interval. For a fixed LOW occupancy group, every HIGH row contributes one consecutive LOW slice. No per-state canonical rank is required.

## 3. W=28 storage geometry

`src/cpp/probes/factorized_storage_geometry_probe.cpp` recomputes the W=28 geometry without CUDA or the existing codec implementation.

Expected output is:

```text
W=28 H=13 center=1 L=14
main_states=385719506620
high_all_codes=787333 low_all_codes=1201917
LOW14 groups=16384 max_states=961466716 max_row_runs=1171380 total_row_runs=17867517828 avg_run=21.588 warp_row_eff=50.672%
HIGH13 groups=8192 max_states=1471935235 max_fblock_runs=22 total_fblock_runs=106495 best_owner_weighted=30.849%
mask_begin_mib=2.812 removable_canonical_prefix_mib=12.014 net_metadata_delta_mib=-9.201
```

The state partitions are checked against the known W=28 main-state count.

Interpretation:

- HIGH13 group I/O is extremely regular: at most 22 consecutive main-state ranges per group.
- LOW14 group I/O is row-sliced. The state-weighted average consecutive run is 21.588 uint32 values.
- A naive one-warp-per-row kernel uses about 50.7% of lane slots. This is acceptable for a first implementation but leaves room for a packed-row kernel.
- If HIGH13 groups are scheduled on the GPU owning the largest fraction of their factorized ranges, the weighted local-HBM fraction is about 30.85%, versus 12.5% for random assignment over eight GPUs.
- Storage occupancy-start metadata is smaller than the canonical HIGH prefix-base tables it makes unnecessary, so factorized-authoritative layout does not create a new large GPU-memory cost.

## 4. Proposed B300 storage-major I/O

Do not precompute one `PeerInterval` per LOW row: W=28 has about 17.9 billion such rows across all LOW14 groups.

Use a compact rectangle descriptor instead:

```cpp
struct StorageRect {
    uint64_t global;
    uint64_t local;
    uint32_t rows;
    uint32_t global_stride;
    uint32_t width;
};
```

There are only O(H) descriptors per group.

### Fixed HIGH

`width == global_stride`, so each nonempty factor block is one fully consecutive range. Reuse the existing interval-copy path or a simple linear peer-copy kernel.

### Fixed LOW

Each row is

```text
auth[global + row * global_stride ... + width)
    <->
scratch[local + row * width ... + width).
```

A first CUDA implementation can assign one warp to one row. The shard owner should be computed once per row, not once per state. A later version can pack short rows using precomputed fast-divide/magic constants.

When a Mate cache fits, construct the `MateID` during storage gather from the already known row/column coordinates rather than calling a general rank/unrank routine:

- fixed LOW: HIGH all-rank is the row; LOW mask-local rank is the column;
- fixed HIGH: HIGH mask-local rank is the row; LOW all-rank is the column.

This makes Mate generation a few table lookups and bit shifts.

## 5. Authoritative factor tables can be reused, not duplicated

The scratch factor codec currently packs

```text
(all-rank << segment_bits) | mask-local-rank
```

inside `*_packed_rank`.

For storage-major authoritative layout, replace `all-rank` by the occupancy-major storage rank while preserving the same mask-local rank. Replace `*_all_codes` by the corresponding occupancy-major list. The local factor codec remains a bijection and its factor blocks now line up directly with authoritative rectangles.

Therefore the 1-GiB LOW packed-rank table is not duplicated. It is replaced in place.

The canonical `high_main_base` / `high_block_base` arrays are no longer needed once authoritative HBM is storage-major.

## 6. Midpoint contraction

For point-symmetry MITM, stop after 14 completed rows. The existing CPU point-symmetry probe already joins the main half-DP vector at a row boundary; deferred/blocked states are not part of the midpoint vector.

For each exact full-frontier occupancy mask `S` and its reflected mask `R(S)`:

1. read the corresponding factorized storage rectangles;
2. map `(factor rectangle coordinates)` to the dense one-defect TL rank;
3. apply the sparse odd-TL transform to both vectors;
4. contract with the recursively generated weighted involution;
5. reduce to one residue scalar.

The transform never changes occupancy, so exact occupancy sectors are independent.

The earlier operation-count estimate for W=28 is about

```text
2.660e12 sparse transform contributions
+ 1.929e11 weighted leaf products
= 2.852e12 primitive join operations / residue.
```

This is to be compared with 14 additional rows of the full Grid-FP sweep, not with an explicit meander adjacency matrix.

## 7. Implementation order

1. Keep the production source unchanged.
2. Add a storage-major derivative of the experimental B300 factorized backend.
3. Replace canonical group gather/scatter with HIGH interval and LOW rectangle I/O.
4. Validate residues on small widths / one GPU against the canonical backend.
5. Validate multi-GPU sharding and peer I/O.
6. Add the odd-TL midpoint kernel.
7. Only after correctness is fixed, add owner-aware HIGH scheduling and DSM specializations for large occupancy sectors.

Do not run broad GitHub Actions matrices for this branch; use targeted local/B300 regression runs.