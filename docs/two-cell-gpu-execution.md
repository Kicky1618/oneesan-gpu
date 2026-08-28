# Two-cell GPU execution addendum

Status: experimental implementation design, 2026-08-29.

This note complements `two-cell-channel.md`.  The algebraic reduced channel is
unchanged; this file records the execution structure discovered after the first
component-local probes.  CUDA files mentioned here are microprobes/prototypes.
They have not yet been compiled or benchmarked on the target B300-class device.
The small-width structural statements are backed by the exact CPU probes and
independent exhaustive enumeration described below.

## 1. Fixed stationary reduced vector

The sliding bases `Q_i` admit one position-independent global layout

```text
Q~ = A(M_{W-1}) + canonical-C.
```

For a C coordinate, canonicalization rotates the active distinguished occupied
strand to canonical bit zero while preserving the compact L/R sequence.  Thus a
source coordinate and its coordinate-set recoupling on the next cell have the
same stationary rank.

Important distinction:

```text
coordinate recoupling != matrix perfect matching
```

`recouple_coordinate()` identifies the source/destination coordinate sets and
therefore their common stationary addresses.  It is not generally a nonzero
matrix edge.  The actual support-tree matching is a separate object.

At W=28 the stationary uint32 vector has

```text
R_28 = 165,727,043,758 values
size = 617.381 GiB
```

and no second destination vector is required for a component-local in-place
step once all source values of a component have been captured locally.

## 2. Interior component arithmetic is now closed form

`direct_component_sources()` has a canonical source order.  In that order every
interior component belongs to one of five families:

```text
singleton
triple
deep RN
deep LR
deep LN
```

The actual unique perfect matching and all residual edges are closed-form in
this order.  Deep LN has one variable pivot, but the pivot is simply the first
tail source `s >= 4` whose local pair is `LL`.  Therefore the production-shaped
hot path needs

```text
K_step calls for matching = 0
leaf-peeling calls         = 0
global matching table      = 0
```

Leaf peeling remains useful only as an oracle/safety fallback in CPU probes.

The exact arithmetic still uses the minimum

```text
component_size - 1
```

modular additions after the matching permutation.  Summed over W=28 this is

```text
R_28 - M_26 = 118,389,089,432 additions/interior step.
```

### Warp register form

The closed forms allow every destination lane to compute its own value directly
from source values held in registers.  A few `__shfl_sync` operations provide
`x0..x3`; the deep tail uses one warp reduction.  No per-component shared
arrays for values, outputs, ranks, or matching descriptors are needed.

For example the universal triple is

```text
(y0,y1,y2) = (x2, x1+x2, x0+x2).
```

The current helper is `two_cell_component_warp_arithmetic.cuh` and its CUDA
self-check compares the register result with explicit `K_step` edges.

## 3. Deep marked-face reconstruction is warp-parallel

A deep component has at most 17 source coordinates at W=28.  The packed word
uses two uint32 masks, `support` and `left`.

The older reconstruction performed a serial parenthesis partner scan for every
candidate L.  The current CUDA helper instead computes the pre-height of each
physical position once and uses

```text
__match_any_sync
```

to group L endpoints and R endpoints at the same matching level.  An L lane
selects the first later R in the returned mask.  Hence the explicit rightward
partner loop is gone.

Exhaustive CPU enumeration through W=14 additionally found that the
face-generated candidates are already valid and mutually distinct.  Therefore
production-shaped reconstruction also removes

```text
per-candidate valid_word scan = 0
candidate duplicate search    = 0
```

A ballot produces the valid candidate mask and `popc(lower lanes)` gives each
candidate its canonical output index directly.  The resulting order is checked
against `direct_component_sources()` because the closed component arithmetic
relies on that order.

## 4. Primitive connectivity/rank path

The global state still factorizes into occupied support and primitive L/R
connectivity.

`primitive_rank()` no longer scans every physical slot.  It iterates only the L
endpoints.  For the m-th L at occupied ordinal `j_m`, the skipped R-first suffix
count is

```text
primitive[occupied-j_m-1][2*m-j_m].
```

At W=28 this is at most 13 L iterations.

For component labels, rank -> connectivity unrank is removed from the CUDA hot
path by a compact primitive-L LUT.  Width-26 labels need only

```text
1,033,411 uint32 entries
about 3.94 MiB
```

for all occupied sectors.  A warp deposits the compact L mask into physical
support positions with `popc + ballot`.

The label primitive rank is already known during enumeration.  It is reused for
retained source slots 0..2 and also source slot 3 of RN/LN deep components.
Only the remaining deep A coordinates currently require a fresh primitive-rank
calculation.

## 5. Two-step shared-memory fusion

Two adjacent interior steps can be joined by their outer-support invariant.
For `steps=2`, one fusion block consists of only

```text
16 A support sectors
4  C support sectors
```

and each sector is one contiguous primitive-rank interval.

This gives a stronger boundary-copy result than per-state unrank/rank:

```text
local address  = sector.local_base  + primitive
stationary HBM = sector.global_base + primitive
```

The stationary global sector base is identical at the beginning and end of the
fused segment.  Therefore a CTA builds at most 20 sector descriptors once,
loads the corresponding intervals, executes both K phases locally, and stores
the same intervals back.  There is no per-state fusion unrank or stationary-rank
calculation on the block boundary.

Inside the fusion block, component values and ranks are register-resident; the
large dynamic shared allocation contains only the fused uint32 state block.

With a 228 KiB per-CTA budget, W=28 single-CTA fusion currently fits outer
popcount `o <= 15`.  The exact state coverage is approximately

```text
58.6591%
```

so the ideal value-traffic reduction relative to two separate passes is

```text
29.3295%.
```

These are capacity/traffic estimates, not measured speedups.

## 6. Distributed shared memory extension

CUDA thread-block clusters provide a path for fusion blocks larger than one
CTA's shared-memory allocation.  The prototype uses

```text
cooperative_groups::this_cluster()
cluster.sync()
cluster.map_shared_rank()
cudaLaunchKernelEx
cudaLaunchAttributeClusterDimension
```

and the occupancy APIs choose only cluster sizes accepted by the target
runtime/device.

### Primitive-sliced DSM layout

A naive DSM prototype divides the fusion-local rank interval contiguously.
A better layout divides *each of the <=20 primitive sectors* among cluster CTAs.
For a sector with `P` primitive states and `C` CTAs, owner `r` stores

```text
[floor(P*r/C), floor(P*(r+1)/C)).
```

Each slice remains contiguous in the stationary global vector.  A component is
assigned to the CTA owning its label primitive rank.  Since the first three
retained coordinates share the label primitive rank, and RN/LN source 3 does as
well, most component accesses remain local even though the full fusion block is
distributed.

Small-width exhaustive locality measurements give the fraction of source
accesses served by the component owner's CTA:

```text
W=10: cluster2 93.66%, cluster4 84.13%, cluster8 77.08%
W=12: cluster2 95.11%, cluster4 86.54%, cluster8 78.76%
W=13: cluster2 95.59%, cluster4 87.59%, cluster8 79.89%
```

These are small-width structural measurements, not W=28 hardware measurements.

### W=28 capacity model

Using primitive-sector slicing, representative maximum local state counts are

```text
o=16, 2 CTAs: 54,145 states = 216,580 B
o=17, 4 CTAs: 49,505 states = 198,020 B
o=18, 8 CTAs: 47,245 states = 188,980 B
```

With a roughly 228 KiB budget and a few KiB of static workspace, the
capacity-only cumulative fusion coverage becomes approximately

```text
1 CTA : 58.66%   -> ideal two-pass traffic reduction 29.33%
2 CTA : 74.68%   -> 37.34%
4 CTA : 86.74%   -> 43.37%
8 CTA : 94.41%   -> 47.21%
```

The 8-block figure is a portable planning ceiling, not an assumption about a
specific GPU.  Runtime mode queries the actual maximum potential cluster size
and active-cluster count before launching a bucket.

## 7. Current prototype ladder

The main implementation checkpoints are:

```text
two_cell_component_stationary_register_microprobe.cu
    stationary one-vector + register component arithmetic

two_cell_fusion2_register_reuse_microprobe.cu
    two-step fusion + primitive-rank reuse

two_cell_fusion2_sectorcache_microprobe.cu
    <=20 cached contiguous boundary sectors

two_cell_fusion2_cluster_dsm_microprobe.cu
    first cluster/DSM correctness prototype

two_cell_fusion2_cluster_sliced_microprobe.cu
    primitive-sliced distributed block

two_cell_fusion2_cluster_sliced_lut_microprobe.cu
    primitive LUT label decode + collective capacity guard

two_cell_fusion2_cluster_lut_runtime_microprobe.cu
    occupancy-checked runtime cluster selection

two_cell_fusion2_cluster_lut_forced_microprobe.cu
    small-width forced 2/4/8-CTA remote-DSM correctness path
```

The forced-cluster microprobe exists because an automatic small-width planner
normally selects cluster size one and would otherwise fail to exercise remote
DSM reads/writes.

## 8. What is still unproven/unmeasured

The following remain gates before this becomes the production backend:

1. Compile the CUDA prototype chain with the target toolkit and remove any
   translation-unit/include-chain issues found by nvcc.
2. Run small-width arithmetic checks on compute capability >=9.0 hardware,
   including forced 2/4/8-block cluster paths.
3. Query actual B300/Hopper/Blackwell shared-memory and cluster limits rather
   than using planning budgets.
4. Measure DSM local/remote bandwidth and cluster-barrier cost.  Capacity alone
   does not imply that a larger cluster is faster.
5. Compare single-CTA fusion, 2-CTA DSM, 4-CTA DSM, and fallback one-step kernels
   on identical support buckets.
6. Integrate the physical row-turn boundary fusion with the same latest register
   and sector-copy machinery.
7. Preserve the existing exact solver as oracle until a complete forward-turn-
   reverse-turn snake cycle agrees on all small widths.

No GitHub Actions run is needed for these exploratory probes; explicit local or
target-GPU execution is the intended validation path.
