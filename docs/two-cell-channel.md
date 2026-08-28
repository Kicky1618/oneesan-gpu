# Two-cell minimal channel

Status: experimental exact reformulation, updated 2026-08-29.

This note records the reduced `N/R/L` two-cell channel obtained from the
`beta=0` seven-tile transfer, together with the current stationary one-vector
execution model.  The algebraic reduction is exact.  Several structural laws
below are still supported by exhaustive small-width probes rather than a final
combinatorial proof; those cases are identified explicitly.

## Main-space notation

Let `M_W` be the number of one-defect Motzkin words of width `W`: words in
`N,R,L` whose height starts at one, never becomes negative, and ends at zero.
Equivalently,

```text
M_W = sum_p binom(W,2p+1) Catalan(p+1).
```

For width 28,

```text
M_28 = 385,719,506,620
M_27 = 135,015,505,407
M_26 =  47,337,954,326
M_25 =  16,626,415,975
```

## One-cell factorization

For active physical sites `i,i+1`, let `T_i` be the ordinary local seven-tile
operator at loop fugacity zero.  It factors exactly as

```text
T_i = E_i R_i.
```

The one-cell image has two pieces:

- `A_i(w)`, `w in M_{W-1}`: collapse the two active sites to one site;
- `C_i(u)`, `u in M_{W-2}`: insert a local `LR` cup.

Hence

```text
rank(T_i) = M_{W-1} + M_{W-2}.
```

At `W=28` this is `182,353,459,733` coordinates, or `679.320 GiB` for one
`uint32` residue.

## Exact two-cell quotient

Before processing the next forward cell `(i+1,i+2)`, the only redundant
one-cell-channel direction is

```text
C_i(N q) - A_i(L R q).
```

Both representatives have the same image under `T_{i+1}`, so quotient by

```text
C_i(N q) == A_i(L R q).
```

A concrete reduced basis is

```text
Q_i = { A_i(w) : w in M_{W-1} }
    + { C_i(u) : u in M_{W-2}, u[i] != N }.
```

Exactly `M_{W-3}` C words are removed.  Therefore

```text
R_W := dim Q_i
     = M_{W-1} + M_{W-2} - M_{W-3}.
```

For width 28,

```text
R_28 = 165,727,043,758
```

or `617.381 GiB` for one `uint32` residue.

## Reduced transition

Write

```text
K_i = P_{i+1} R_{i+1} E_i : Q_i -> Q_{i+1},
```

where `P` applies the quotient above.  Every matrix coefficient is `+1` and
every source has fan-out one, two, or three.

The source fan-out counts are

```text
n1 = 2 M_{W-2} - M_{W-3}
n2 = M_{W-1} - 2 M_{W-2} + M_{W-3}
n3 = M_{W-2} - M_{W-3}
```

and

```text
nnz(K_i) = 2 M_{W-1} + M_{W-2} - 2 M_{W-3}.
```

For `W=28`,

```text
n1  = 78,049,492,677
n2  = 56,966,012,730
n3  = 30,711,538,351
nnz = 284,116,133,190
average source fan-out = 1.71436192155
```

A table-free inverse formula exists, but the component decomposition below is a
better execution model than a global destination gather.

## Interior support forest

Form the bipartite support graph of one interior reduced step: source
coordinates on one side, destination coordinates on the other, and one edge
per nonzero of `K_i`.

Exhaustive enumeration through the explored widths gives

```text
vertices             = 2 R_W
edges                = nnz(K_i)
connected components = M_{W-2}
each component       = balanced bipartite tree
```

The edge count is exactly consistent with

```text
nnz(K_i) = 2 R_W - M_{W-2}.
```

If the forest property holds generally, every component has a unique perfect
matching: two distinct perfect matchings in a tree would produce an alternating
cycle.  Thus every component multiply can be written as

```text
matching permutation copy
+ exactly (component_size - 1) residual additions.
```

Summed over a whole interior step,

```text
residual additions
  = nnz(K_i) - R_W
  = R_W - M_{W-2}.
```

For `W=28`,

```text
components       = 47,337,954,326
residual adds    = 118,389,089,432
average sources/component = 3.50093378807
```

The matching permutation is generally **not identity** in the stationary
coordinate order described below.

## Table-free component labels

For every unrestricted `u in M_{W-2}`, define

```text
S_i(u) = P_i C_i(u).
```

If `u[i] != N`, this is the retained C coordinate.  If `u[i] == N`, the
quotient maps it to the corresponding A coordinate.  Exhaustive enumeration
finds exactly one such seed in every support-tree component, hence

```text
component_count       = M_{W-2}
component_table_bytes = 0.
```

The label itself can be ranked/unranked by occupied support and primitive
connectivity using only small `O(W^2)` rank tables.

## Packed component reconstruction

Represent a word by two 32-bit masks:

```text
support : occupied physical positions
left    : occupied L endpoints; other occupied positions are R
```

For `W=28`, every A/C word fits in the masks.  No strings or mate arrays are
required.

There are three component-size classes.

### Singleton

If

```text
u[i] == N,
```

then the eliminated C direction is represented by one A coordinate and the
component has one source.

### Triple

If `u[i] != N` and the local pair is not one of

```text
RN, LN, LR,
```

then the source list is exactly

```text
0: C(u)
1: A(insert N before u[i])
2: A(insert N after  u[i])
```

and has size three.

### Deep component

For local `RN`, `LN`, or `LR`, collapse

```text
RN -> R
LN -> L
LR -> N
```

to a width-`W-3` word `v`.  Construct the central destination by inserting two
vacancies before the collapsed symbol.  Then

```text
component_sources
  = { the three fixed coordinates }
    union inverse_K(Dcenter).
```

The inverse contributes one fixed coordinate plus one source per strand exposed
to the marked face.  Thus only one marked-face inverse scan is needed.

If `r_i(v)` is the number of strands bordering that face,

```text
deep_component_size = 4 + r_i(v).
```

The face cannot contain more strands than the one-defect diagram contains, so

```text
component_size <= floor(W/2) + 3.
```

At `W=28`,

```text
max source coordinates/component = 17
max uint32 source payload         = 68 bytes.
```

The W=28 component-size distribution produced by the marked-face generating
function is

```text
pairs  components
1      16,626,415,975
3      14,085,122,376
5       4,945,087,837
6       4,945,087,812
7       3,437,713,026
8       1,930,340,540
9         898,275,650
10        341,465,226
11        102,042,591
12         22,621,768
13          3,441,735
14            323,450
15             16,027
16                312
17                  1
```

The weighted sum is exactly `R_28`.

## Stationary coordinate layout

The reduced bases `Q_i` change syntactically as the active position moves, but
the change is only a C-coordinate vacancy swap.  A forward coordinate mapping
is

```text
A(w)       -> A(w)
C(... x N ...) -> C(... N x ...),  x in {L,R},
```

when the active position advances across that vacancy.

Normalize C support by rotating the occupied active strand through the prefix
so it sits at the fixed canonical position.  This produces one stationary
basis

```text
Q_tilde
  = A(M_{W-1})
    + { canonical C(u) : canonical bit 0 is occupied }.
```

Exhaustive probes verify that every `Q_i` maps bijectively to this same basis and
that corresponding source/destination coordinates receive the same stationary
rank.

Important distinction:

```text
stationary recoupling = coordinate-set / address bijection
matrix matching       = unique perfect matching of actual K_i edges
```

These are **not the same map in general**.  Earlier experimental code that
identified them was corrected.  The stationary layout remains valid because it
only needs equal source/destination coordinate sets, not a diagonal matrix
edge.

Consequences:

```text
one global value vector is sufficient
no destination rank calculation is required after the local coordinate index is known
no global permutation table is required
```

For `W=28`, the one stationary `uint32` vector is `617.381 GiB` instead of a
simple two-vector allocation of about `1.235 TiB`.

Reverse rows use the exact reflected coordinate bijection.  The reverse
recoupling is likewise an address bijection, not an asserted matching edge.

## Matching without a global table

`direct_component_sources()` has a canonical source order.  In this order the
actual matching/support graph is much more rigid than a generic tree matching.

### Singleton and triple

Singleton is `[1]`.

Every three-source component has source adjacency masks

```text
[100, 010, 111]
```

against stationary destination indices.  Its unique matching is

```text
src_to_dst = [2,1,0]
```

and the exact arithmetic is

```text
y0 = x2
y1 = x1 + x2
y2 = x0 + x2.
```

No `K_step()` call or matching reconstruction is required.

### Deep LR

For a deep LR component of size `n`,

```text
src 0 -> {2}
src 1 -> {1}
src 2 -> {0,1,2}
src 3 -> {1,3}
src s -> {3,s},  s >= 4.
```

The matching is

```text
[2,1,0,3,4,...,n-1].
```

Again no `K_step()` is required for matching.

### Deep RN

For deep RN,

```text
src 0 -> {2}
src 1 -> {1}
src 2 -> {1,2,n-1}
src 3 -> {0,3}
src s -> {3,s},  s >= 4.
```

The matching is

```text
[2,1,n-1,0,4,5,...,n-2,3].
```

No matching search is required.

### Deep LN: one pivot

Deep LN has the same tail graph.  The only topology-dependent datum is one
pivot index `k >= 4`:

```text
src 0 -> {2}
src 1 -> {1}
src 2 -> {1,2,k}
src 3 -> {0,3}
src s -> {3,s},  s >= 4.
```

Thus the matching is

```text
0 -> 2
1 -> 1
2 -> k
3 -> 0
k -> 3
all other s >= 4 -> s.
```

The pivot is obtained from **one** `K_step(src[2])`; all other component edges
are then known algebraically.  Exhaustive probes through W=14 find no need for
leaf peeling on any valid component.  Generic leaf peeling remains only as a
safety/reference fallback.

Across all 25 forward interior positions at `W=28`, exact path counting gives

```text
total components                = 1,183,448,858,150
LN pivot components             =   126,383,557,900
matching K_step/component       = 0.106792580879
matching leaf-peeling calls     = 0  (for the observed closed laws)
descriptor-free component share = 89.3207419%
```

The residual arithmetic remains exactly `n-1` additions per component.

## Primitive and support ranking

The global stationary state still factorizes by occupied support and primitive
connectivity.  Primitive rank ignores vacancy positions and depends only on the
occupied L/R sequence.

The old physical-slot scan can be reduced to the positions of L endpoints.  If
`j_m` is the occupied-sequence index of the `m`-th L endpoint, then

```text
primitive_rank
  = sum_m P[occupied-j_m-1][2m-j_m].
```

At `W=28` this uses at most 13 endpoint iterations instead of scanning all 27
physical slots.

For component enumeration, labels have length 26, so all primitive L masks for
all odd occupied counts through 25 require only

```text
1,033,411 uint32 entries
~= 3.94 MiB.
```

The small combinatorial rank tables themselves are only on the order of tens of
KiB.

A support/primitive tile can amortize support unrank and support-base generation
over many primitive connectivities.  The current CUDA probes use this as the
canonical enumeration direction rather than a flat component-rank decode.

## Physical row turns

A snake row turn sees the boundary pair twice.  At `beta=0`,

```text
T_edge^2 = I + 2e.
```

The direct reduced turn has

```text
turn_nnz = 3 (M_{W-1} - M_{W-3})
```

with coefficients only `1` and `2`.

The turn support graph has small cycles rather than being a forest, but it still
has exactly `M_{W-2}` components and the same unrestricted width-`W-2` label.
Its component arithmetic has a closed `alpha / beta / passive` form, so no turn
CSR or graph traversal is needed.

At `W=28`,

```text
turn_nnz = 355,167,268,296.
```

Right and left turns share one executor by geometric reflection.

## Two-step shared-memory fusion

The stationary basis makes it possible to keep a union block resident across
multiple transfers.  For two consecutive transfers, fix the support bits
outside the local four-bit A window.  All coordinates in that outer-support
block remain closed under both `K_i` and `K_{i+1}`.

The block-local coordinate rank is only

```text
A/C local support code + primitive rank.
```

For two steps this means 16 A local-support sectors and 4 C sectors.  No global
block dictionary is needed.

The execution pattern is

```text
global stationary vector
  -> load one union block once into shared memory
  -> K_i component-local matching/residual adds
  -> K_{i+1} component-local matching/residual adds
  -> store the union block once
```

The large block data uses one shared `uint32` array.  A second full shared
buffer is unnecessary; only tiny per-component input/output scratch is needed
because the actual matching can be nonidentity.

For blocks that do not fit, the hybrid executor falls back to two stationary
component passes on the same global vector.  Thus no second global value vector
is required in either path.

Planner probes currently indicate that, for a roughly 228 KiB shared-memory
budget at `W=28`, two-step fusion is the useful depth: substantially more states
fit than with deeper union windows.  If a fraction `f` of states is fused, value
traffic for a pair of transfers is reduced by

```text
HBM reduction = f / 2
```

relative to two separate load/store passes.  Current planning estimates put the
ideal reduction near 29% for the two-step interior pairing; this is a model,
not a measured B300 result.

## Boundary fusion with row turn

A useful special case fuses the final interior transfer with the physical row
turn.  The turn does not advance the active coordinate window, so the union
block has the smaller `steps=1` fusion footprint while carrying two operators:

```text
last interior K
+ physical turn.
```

The same construction works at the left boundary by reflection.  Therefore a
width-28 row can be organized conceptually as

```text
12 x fusion2 interior pairs
+ 1 x (last interior transfer + row turn)
```

for the forward direction, with the reflected organization on the reverse row.
The current 228-KiB planning model gives a row-core value-traffic reduction of
about 29.5%.  Again this is a capacity/traffic model pending real CUDA timing.

## HBM models

A destination gather would read one source per nonzero and write one value per
destination:

```text
4 * (nnz(K_i) + R_28)
  = 1.799372707792 TB per interior step.
```

A non-fused stationary component pass needs one value load and one value store:

```text
8 * R_28
  = 1.325816350064 TB per interior step.
```

This removes `118,389,089,432` global `uint32` source loads, about `441.034 GiB`
per interior step, versus the destination-gather model.

Two-step fusion further removes the intermediate global load/store for the
fraction of union blocks that fit in shared memory.

All quoted HBM figures are logical count-array traffic, not measured hardware
bandwidth.  They exclude rank generation, communication, kernel launch costs,
cache effects, and occupancy losses.

## Current probe family

The main validation/prototype files now include:

- `two_cell_channel_probe.cpp`: factorization, quotient, dimensions, fan-out,
  coefficients, delayed exactness;
- `two_cell_inverse_gather_probe.cpp`: table-free inverse preimages;
- `two_cell_direct_component_probe.cpp`: one marked-face component scan;
- `two_cell_component_size_distribution_probe.cpp`: exact size buckets;
- `two_cell_component_matching_probe.cpp`: actual local perfect matching and
  residual arithmetic;
- `two_cell_component_matching_fastpath_probe.cpp`: singleton/triple closed
  matching;
- `two_cell_component_matching_deep_fastpath_probe.cpp`: RN/LR deep closed
  matching;
- `two_cell_component_matching_ln_pivot_probe.cpp`: LN one-pivot law;
- `two_cell_reverse_stationary_probe.cpp`: reverse stationary address bijection,
  explicitly separated from matrix matching;
- `two_cell_turn_closed_block_probe.cpp`: closed physical-turn components;
- `two_cell_fusion_component_probe.cpp`: union-block component codec;
- `two_cell_fusion_unrank_probe.cpp`: local fusion rank/unrank;
- `two_cell_fusion2_shared_microprobe.cu`: two-buffer correctness prototype;
- `two_cell_fusion2_bucketed_microprobe.cu`: outer-popcount shared buckets;
- `two_cell_fusion2_singlebuffer_microprobe.cu`: one large shared value buffer;
- `two_cell_fusion2_hybrid_microprobe.cu`: fused/fallback mixed executor;
- `two_cell_boundary_fusion_singlebuffer_microprobe.cu`: right boundary
  last-transfer + turn fusion;
- `two_cell_boundary_fusion_left_microprobe.cu`: reflected left boundary fusion.

The probes are intended for explicit local execution.  They should not be used
to justify repeated GitHub Actions runs.

## Remaining gates

1. Compile and run the new C++/CUDA probes on a real repository checkout.  The
   connector commits above have not themselves executed `nvcc`.
2. Turn the observed support-forest and one-label-per-component properties into
   a clean combinatorial proof.
3. Prove the closed RN/LR/LN-pivot matching laws directly from the local
   seven-tile relations; currently they are exhaustively checked through the
   explored small widths.
4. Replace the serial deep marked-face reconstruction in the production-shaped
   kernel with the warp-parallel face helper and measure register pressure.
5. Measure fusion2 bucket occupancy and dynamic-shared limits on the target GPU;
   choose the actual shared-memory threshold from hardware rather than the
   current 228-KiB planning model.
6. Compare a complete stationary reduced snake cycle against the existing exact
   grid solver on small n before making the reduced backend authoritative.
7. Only after the above, integrate the winning component/fusion kernel into the
   B300 backend while retaining the current exact path as reference.
