# Two-cell minimal channel

Status: experimental exact reformulation, 2026-08-28.

This note records the canonical `N/R/L` two-cell channel obtained from the
`beta=0` seven-tile transfer. Unlike the existing GGCount-style production
backend, this formulation has no authoritative blocked/deferred vector.

## Main-space notation

Let `M_W` be the number of one-defect Motzkin words of width `W`, i.e. words in
`N,R,L` whose height starts at one, never goes negative, and ends at zero.
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

## One-cell rank factorization

For active physical sites `i,i+1`, let `T_i` be the ordinary local seven-tile
operator at loop fugacity zero. It factors exactly as

```text
T_i = E_i R_i.
```

The image channel has two pieces.

- `A_i(w)`, `w in M_{W-1}`: collapse the two active sites to one site. If the
  collapsed symbol is occupied, `E_i` expands it to the sum of the two physical
  orientations.
- `C_i(u)`, `u in M_{W-2}`: insert a local `LR` cup at the two active sites.

Hence

```text
rank(T_i) = M_{W-1} + M_{W-2}.
```

At `W=28` this is `182,353,459,733` coordinates, or `679.320 GiB` for one
`uint32` residue.

## The only two-cell kernel direction

Before processing the next forward cell `(i+1,i+2)`, the only redundant
one-cell-channel direction is

```text
C_i(N q) - A_i(L R q).
```

Indeed both representatives have the same image under `T_{i+1}`. Therefore
we quotient by

```text
C_i(N q) == A_i(L R q).
```

A concrete reduced basis is

```text
Q_i = { A_i(w) : w in M_{W-1} }
    + { C_i(u) : u in M_{W-2}, u[i] != N }.
```

Deleting or inserting a fixed `N` preserves Motzkin validity, so exactly
`M_{W-3}` C words are removed. Thus

```text
R_W := dim Q_i
     = M_{W-1} + M_{W-2} - M_{W-3}.
```

For width 28,

```text
R_28 = 165,727,043,758
```

which is `42.9656890%` of the full main space. One `uint32` residue occupies
`617.381 GiB`; on eight GPUs this is `77.173 GiB/GPU`, or
`154.345 GiB/GPU` for a simple double buffer.

The difference between the one-cell and two-cell channels is exactly
`M_25 = 16,626,415,975` values at W=28, only `61.938 GiB` total or
`7.742 GiB/GPU`.

## Reduced transition

Let

```text
Kbar_i = P_{i+1} R_{i+1} E_i : Q_i -> Q_{i+1},
```

where `P` applies the quotient relation above.

Every matrix coefficient is `+1`, and every source has fan-out one, two, or
three. The local classification is:

```text
source class       fan-out
A00                2
A01                1
A10                3
A11, local LR loop 1
A11, otherwise     2
C                  1
```

Write

```text
c = M_{W-2} - M_{W-3}.
```

There are `c` states of type `A01`, `c` of type `A10`, and `c` retained C
states. There are `M_{W-3}` `A00` states, and exactly `M_{W-3}` locally paired
`A11` states. This gives

```text
n1 = 2 M_{W-2} - M_{W-3}
n2 = M_{W-1} - 2 M_{W-2} + M_{W-3}
n3 = M_{W-2} - M_{W-3}
```

and therefore

```text
nnz(Kbar_i) = 2 M_{W-1} + M_{W-2} - 2 M_{W-3}.
```

For width 28,

```text
n1  = 78,049,492,677
n2  = 56,966,012,730
n3  = 30,711,538,351
nnz = 284,116,133,190
average source fan-out = 1.71436192155
```

Destination C channels have exactly one predecessor. A table-free inverse
formula for the A destinations is implemented by the inverse-gather probe, but
the component decomposition below makes a global destination gather less
attractive as the final GPU execution model.

## Support forest

Form the bipartite support graph of one interior reduced step. The left side is
the `R_W` destination coordinates, the right side is the `R_W` source
coordinates, and every nonzero of `Kbar_i` is an edge.

Exact enumeration through the currently explored small widths finds a much
stronger structure than sparse invertibility:

```text
number of vertices   = 2 R_W
number of edges      = nnz(Kbar_i)
number of components = M_{W-2}
every component      = balanced bipartite tree
```

The edge formula is consistent with the tree identity:

```text
nnz(Kbar_i)
  = 2 R_W - M_{W-2}.
```

Every component has a unique perfect matching, because a tree cannot contain
two different perfect matchings. Leaf peeling constructs this matching.
Contracting the matching edges leaves an oriented forest. If destination row
`r` is stored in the slot of its matched source column, the remaining operation
is an in-place lifting DAG with exactly

```text
nnz(Kbar_i) - R_W
  = R_W - M_{W-2}
```

additions and no multiplication.

For W=28 this is

```text
components    = 47,337,954,326
lifting adds  = 118,389,089,432
average source coordinates/component = 3.50093378807
average source+destination vertices/component = 7.00186757614
```

### Table-free component labels

For every unrestricted word `u in M_{W-2}`, define the source seed

```text
S_i(u) = P_i C_i(u).
```

If `u[i] != N` this is the retained C state itself. If `u[i] == N`, the quotient
maps it to its A representative. Exact enumeration finds exactly one such seed
in every support-tree component. Therefore all components can be enumerated by
ordinary width-`W-2` Motzkin ranks:

```text
component_count = M_{W-2}
component_table_bytes = 0
```

No global CSR, component table, or atomic accumulation is required for an
interior step.

## Exact component size from the marked face

The component size is much more rigid than the forest observation alone
suggests. There are three classes of component labels.

If

```text
u[i] == N
```

the eliminated C direction is represented by a single A coordinate, so the
component has one source coordinate.

If `u[i] != N` and the local pair is not one of

```text
RN, LN, LR
```

the component has exactly three source coordinates.

The remaining three patterns are the deep components. Collapse the local pair
by

```text
RN -> R
LN -> L
LR -> N
```

to obtain a width-`W-3` one-defect word `v`. This collapse is a bijection from
deep component labels to `M_{W-3}`.

Let `r_i(v)` be the number of connectivity strands bordering the face incident
to the boundary interval immediately before position `i`. In height language,
let `h=h[i]`, extend left and right maximally while the path remains at or above
`h`, count the level-`h` top-level excursions in that interval, and include the
enclosing/root strand. Then exact enumeration gives

```text
deep_component_size = 4 + r_i(v).
```

A face cannot meet more strands than exist in the whole one-defect diagram. If
`v` has `2p+1` occupied positions, it has `p` matched arcs plus the distinguished
root strand, hence

```text
r_i(v) <= p + 1 <= floor((W-2)/2)
```

and therefore

```text
component_size <= floor(W/2) + 3.
```

For W=28 the exact bound is

```text
max face strands = 13
max source coordinates/component = 17
max uint32 component payload = 68 bytes
```

so one warp has enough lanes for every source coordinate of every interior
component.

At the boundary, one-defect words have the unique decomposition

```text
(N | L a R)* R b
```

where every `a` and `b` is an ordinary Motzkin path. The marked-face strand
count is one plus the number of `L a R` atoms before the root. If `M(x)` is the
ordinary Motzkin generating function and `D[n][r]` counts width-`n` one-defect
words with `r` marked-face strands, then

```text
D(x,y) = x y M(x) / (1 - x - x^2 y M(x)).
```

Small-width enumeration shows the same component-size distribution at every
interior position. For W=28 the resulting component buckets are:

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

The weighted sum of these buckets is exactly `R_28`.

## One-scan packed component reconstruction

The component does not need a three-round graph search in the GPU hot path.
Represent a word by two 32-bit masks:

```text
support : occupied positions
left    : occupied positions carrying L; the other occupied positions are R
```

For W=28 both masks fit in 27 bits for A words. Mate lookup is replaced by a
bounded parenthesis scan over these masks; no mate array or string state is
needed.

For a retained component label `u`, three source coordinates are immediate:

```text
C(u)
A(insert N before u[i])
A(insert N after  u[i])
```

For a three-coordinate component this is already complete. For a deep
component, collapse `u` to `v` as above and construct the single central
A-destination

```text
Dcenter = A(insert N,N immediately before v[i]).
```

Then

```text
component_sources
  = { the three coordinates above }
    union inverse_K(Dcenter).
```

The inverse of `Dcenter` contains the fourth fixed coordinate plus exactly one
coordinate per marked-face strand. Thus deep reconstruction needs only one
`inverse_K` call, which is one O(W) face scan. `two_cell_direct_component_probe`
checks this formula against the full component closure.

This gives the intended interior GPU path:

```text
component label
  -> classify 1 / 3 / deep
  -> construct fixed packed coordinates
  -> deep only: one marked-face inverse scan
  -> load each source value once
  -> evaluate the local tree in registers/shared memory
  -> store each destination value once
```

## Reverse snake rows and physical row turns

Reflection gives the reverse quotient exactly. The reverse reduced transfer is
conjugate to the forward transfer by the geometric involution `J`, so reverse
rows do not require an independent state theory.

A physical snake row turn sees the boundary pair twice: once as the final cell
of the old row and once as the first cell of the new row. The direct reduced
turn therefore contains

```text
T_edge^2 = I + 2 e
```

at `beta=0`. Its exact reduced nonzero count is

```text
turn_nnz = 3 (M_{W-1} - M_{W-3}).
```

Turn coefficients are only `1` and `2`. The turn support graph is not a forest;
it contains small cycles. However exact enumeration still gives

```text
turn component count = M_{W-2}
```

with the same unrestricted `C`-word source seed appearing exactly once per
component. Thus the component-local execution model survives row turns even
though the in-place forest-lifting proof does not directly apply there.

For W=28,

```text
turn_nnz = 355,167,268,296
```

and a component-local turn would still need only one global load per reduced
source and one global store per reduced destination; the extra turn edges are
local arithmetic once the component is resident.

## Sector/support factorization

The reduced state still factorizes by occupied support size. In sector `p`,
primitive connectivity has dimension `Catalan(p+1)`. Hence

```text
R_{W,p} = Catalan(p+1)
          * [ binom(W-1,2p+1) + binom(W-3,2p) ].
```

The second term is the retained C block: its lookahead support bit is fixed to
one, leaving `2p` occupied positions among the other `W-3` sites.

For W=28 the largest reduced sector is p=8:

```text
R_28,8 = 50,950,162,120 values = 189.804 GiB/residue total.
```

`two_cell_factorized_gather_probe.cpp` verifies a support/primitive codec whose
tables are only polynomial in W. This is the current canonical global GPU
layout; a production component kernel should generate component members
directly in this rank space.

## HBM roofline and physical locality

The original destination-gather estimate reads one source value per matrix
nonzero and writes one destination value:

```text
4 * (nnz(Kbar) + R_28) = 1.799372707792 TB/cell.
```

A component-local interior kernel loads every source coordinate once and stores
every destination coordinate once:

```text
8 * R_28 = 1.325816350064 TB/cell.
```

This removes `118,389,089,432` global `uint32` source loads, or about
`441.034 GiB` of HBM traffic per interior cell. The logical count-array traffic
is `26.32%` below the destination-gather model.

Across 756 updates per residue at n=27, this idealized count-array traffic is
about `1.002 PB/residue`. At an aggregate 64 TB/s HBM roofline this is about
`15.7 s/residue` before communication, rank generation, local component
reconstruction, and kernel inefficiencies. This is a roofline, not a hardware
measurement.

The canonical support/primitive ranks are not component-contiguous. The
component-locality probe shows that, by W=14, an average component contains only
about 3.36 values but touches about three distinct 128-byte rank sectors. A
component-major dynamic layout would place almost every component in one
128-byte sector. This makes a component-major or size-bucketed layout worth
investigating after the first canonical-rank CUDA prototype; it is not yet the
authoritative layout.

A component-local physical row turn has more local edges, but the same ideal
count-array HBM traffic `8 * R_28` if its component members can be reconstructed
without extra global tables.

## Exact CPU validation

The current probe family separates the algebraic claims:

- `two_cell_channel_probe.cpp`: factorization, quotient, dimensions, fan-out,
  coefficients, and delayed exactness.
- `two_cell_inverse_gather_probe.cpp`: table-free inverse preimages and compact
  rank/unrank.
- `two_cell_factorized_gather_probe.cpp`: support/primitive factorized layout.
- `two_cell_reverse_channel_probe.cpp`: reflection and reverse quotient.
- `two_cell_row_turn_probe.cpp`: exact direct physical snake turns.
- `two_cell_forest_lifting_probe.cpp`: support-tree decomposition, unique
  matching, and constructive in-place lifting.
- `two_cell_component_kernel_probe.cpp`: table-free local component partition.
- `two_cell_turn_component_probe.cpp`: persistence of the component partition
  across physical row turns.
- `two_cell_component_depth_probe.cpp`: fixed three-round closure bound observed
  through the exhaustive widths.
- `two_cell_packed_component_probe.cpp`: packed forward/inverse operators and
  fixed-round component reconstruction without strings or mate arrays.
- `two_cell_component_size_distribution_probe.cpp`: exact small-width component
  size distribution and boundary generating function.
- `two_cell_component_face_probe.cpp`: marked-face size formula and W=28
  seventeen-source bound.
- `two_cell_direct_component_probe.cpp`: one-scan direct source reconstruction.
- `two_cell_component_locality_probe.cpp`: canonical-rank transaction locality.

The probes are intended for explicit local execution, not repeated GitHub
Actions runs.

## Remaining implementation gates

- Turn the observed interior support-forest and one-seed-per-component
  properties into a clean combinatorial proof independent of exhaustive widths.
- Prove the marked-face component formula directly from the local seven-tile
  relations; the current probes verify it exhaustively on the explored widths.
- Convert the one-scan packed reconstruction into a CUDA device header using
  fixed arrays of at most 17 source coordinates and no dynamic allocation.
- Derive the analogous one-scan reconstruction for physical row-turn
  components, which are small but cyclic and have coefficients `1`/`2`.
- Benchmark canonical factorized ranks first, then test a component-major or
  component-size-bucketed layout if scattered HBM transactions dominate.
- Compare a complete CPU reduced snake solver against the existing exact grid
  solver on small n before treating the new representation as authoritative.
- Integrate only the winning CUDA microkernel into the B300 backend; keep the
  existing exact path as the reference until the full-snake comparison passes.
