# Fusion-basis Grid-FP backend

Research branch: `research/odd-tl-midpoint`.

This note records a new exact Grid-FP backend in which dense one-defect TL topology is stored in the finite beta=0 two-step fusion basis rather than as raw `N/L/R` mate strings.

## State space

For an occupancy mask `S` with odd population `m=2r+1`, the dense topology basis is a two-colour Motzkin excursion of length `r`:

```text
A : height -1
B : height  0
C : height  0
D : height +1
```

The number of such words is `Catalan(r+1)`, exactly the dense one-defect TL dimension.  Hence the dilute state count is unchanged:

```text
sum_{m odd} binom(W,m) Catalan((m+1)/2).
```

At `W=28` this is `385719506620` main states.  The width-27 blocked space has `135015505407` states.

A natural authoritative ID is

```text
ID = base[m]
   + colex_rank(occupancy_mask) * Catalan((m+1)/2)
   + fusion_rank(path).
```

`src/cpp/probes/fusion_id_codec_probe.cpp` checks a compact codec:

- occupancy colex rank: four 7-bit automaton lookups at width 28;
- fusion rank: at most four 4-symbol automaton lookups for `r<=13`;
- fusion unrank table for every `r<=13`: `3707851` uint32 entries = about 14.14 MiB.

Thus the earlier idea of a 1-GiB mask-rank LUT and 256-MiB raw fusion-rank LUT is unnecessary.  The complete forward-rank automata are sub-MiB; only the 14-MiB `fusion rank -> packed fusion word` table is substantial.

`src/cpp/probes/fusion_dense_gridfp_probe.cpp` is the dense-ID CPU version of the backend.  It uses ordinary vectors indexed by this ID instead of an `unordered_map<(mask,path)>` state map.

## Local arc insertion

Let `i` be the number of occupied dense terminals below the physical pair being updated.

For `i=0`, append `C`.

For odd `i`, insert `B` at the corresponding fusion position.

For even `i>0`, split one fusion symbol.  If the entering fusion height is `h`:

```text
A -> AC + CA
B -> BC + CB + DA - h/(h+1) AD
C -> CC
D -> CD + DC
```

The `AD` term is absent at `h=0`.

The important collision property is stronger than the source fanout suggests.  For fixed insertion position, every destination fusion word has at most one source.  The inverse local patterns are disjoint:

```text
AC, CA       <- A
BC, CB, DA   <- B
AD           <- B, coefficient -h/(h+1), h>0
CC           <- C
CD, DC       <- D
```

Boundary/odd-position insertions are also one-to-one.  Therefore NN insertion is an injective sparse map in destination space and never requires an atomic between different NN sources.

`src/cpp/probes/fusion_collision_probe.cpp` exhaustively checks the insertion indegree and cap indegree on small fusion spaces.

## Local cap contraction

The reverse dense-terminal deletion is at most one output per source.

Boundary and within-pair cases are deterministic.  At a cross-pair position:

```text
AB, BA -> A
BB     -> B
AD     -> C
BC, CB -> C
DA     -> -(h+2)/(h+1) C
BD, DB -> D
```

All other local pairs vanish.

Viewed destination-first, the cap indegree is at most four:

```text
A <- AB, BA
B <- BB
C <- AD, BC, CB, DA
D <- BD, DB
```

Only the `DA` source carries the non-unit coefficient.

## Nilpotent cup-cap identity

Let `I_i` be fusion-basis adjacent-arc insertion and `C_i` the cap at the same dense position.  Exact rational tests on all small fusion paths give

```text
C_i I_i = 0.
```

This is not accidental: diagrammatically the composite creates one closed Temperley-Lieb loop, and the present algebra has loop fugacity `beta=0`.

This identity is the key to the fully in-place main/blocked schedule below.

## Complete Grid-FP semantics

For a physical pair with occupancy bits `(lo,hi)`:

```text
00 : local arc insertion
01 : move dense terminal through vacancy; fusion word unchanged
10 : p=1 -> move to 01; p>1 -> blocked shrink; fusion word unchanged
11 : local cap; p=1 -> main, p>1 -> blocked
```

The excluded main branch is identity.  A blocked state's excluded branch reinserts a vacancy and returns to main.

`src/cpp/probes/fusion_gridfp_probe.cpp` implements the full sweep without MateID topology.  It agrees with the original GGCount `PathCounter` through W=9, and the research checks also compared main/deferred vectors after every update through small widths.

The blocked indexing lines up exactly with the current physical pair.  For a current main state whose high bit `p` is zero,

```text
insert_zero(remove_bit(main_mask,p),p) == main_mask.
```

For the `11 -> blocked` cap branch, clearing the pair, deleting `p-1`, then reinserting a zero at the next blocked-release position also reconstructs the same `00` rest-mask block.  Thus main and blocked states decompose into stable rest-mask orbits.

## Exact transition work at W=28

`src/cpp/probes/fusion_work_model.cpp` sums over occupancy masks analytically using binomial coefficients and over fusion words using Motzkin prefix/suffix DP.  Over all 27 update positions in one complete width-28 row, on the full main state universe:

```text
raw NN included edges       = 1,278,124,766,802
fusion NN included edges    = 1,906,681,380,330
vacancy-motion edges        = 4,734,587,758,374
cap edges                   = 1,906,681,380,330

raw main-included total     = 7,919,393,905,506
fusion main-included total  = 8,547,950,519,034
included ratio              = 1.079369...
```

The extra work comes only from NN insertion fanout.  Its average fanout, weighted over a whole row, is about `1.49178`.

Main identity contributes

```text
385719506620 * 27 = 10,414,426,678,740
```

edges.  The width-27 blocked excluded branch contributes

```text
135015505407 * 27 = 3,645,418,645,989
```

more edges.  Including both unchanged branches, the total edge ratio becomes only

```text
22,607,795,843,763 / 21,979,239,230,235
= 1.028597742...
```

so fusion locality costs only about 2.86% more recurrence edges.

Only `229,692,461,466` fusion included edges over the complete width-28 row carry a genuinely non-unit rational coefficient.  That is about 2.69% of fusion main-included edges and about 1.02% after the identity/blocked-release edges are included.  Almost all arithmetic is therefore modular addition/subtraction; the height-dependent constants are a tiny table per CRT prime.

## Rest-mask orbit decomposition

Fix an update position `p` and erase the two active occupancy bits.  Let the remaining `W-2`-bit mask have population `q`.

Because the total occupied population must be odd, only two of the four local occupancy classes exist for a given rest mask.

### q odd: 00 / 11 / blocked orbit

Write

```text
A = main block with local occupancy 00
B = main block with local occupancy 11
D = associated blocked block
```

where `A` and `D` have the same fusion dimension and `B` has the next Catalan dimension.  For `p>1` the exact recurrence is

```text
A' = A + D
B' = B + I A
D' = C B
```

with the local insertion/cap operators above.

Because `C I = 0`, this recurrence can be executed in-place in the order

```text
1. B += I A
2. A += D
3. D  = C B
```

Step 3 may use the already-updated `B`, since

```text
C(B + I A) = C B + C I A = C B.
```

Insertion is destination-injective, step 2 is one-to-one, and cap is a destination gather of at most four sources.  No atomic operation or temporary blocked buffer is required.

### q even: 01 / 10 / blocked orbit

All three blocks have the same fusion dimension.  Let

```text
X = main 01
Y = main 10
D = blocked
```

For `p>1`:

```text
X' = X
Y' = Y + X + D
D' = Y
```

so one thread per fusion rank can load `(X,Y,D)` into registers and write `(Y',D')` directly.  Again no atomic or temporary vector is needed.

### p = 1

There is no new blocked output.  The same rest-mask split gives

```text
q odd:
    B' = B + I A
    A' = A + D + C B
    D' = 0

q even:
    X' = X + Y
    Y' = Y + X + D
    D' = 0
```

For the odd sector, compute `B += I A` first and then use the updated `B` in `A += D + C B`; `C I=0` again makes this exact.

Random full-vector tests over every main and blocked state, all update positions, and widths 3 through 7 matched the ordinary out-of-place fusion recurrence exactly, including the `p=1` special case.

## M+D atomic-free CUDA schedule

The orbit equations mean the entire scratch recurrence can use one main buffer and one blocked buffer.

For each `p>1`, three global phases are enough:

```text
phase 1:
    q odd  -> B += I A              (injective scatter / plain add)
    q even -> may be left for phase 2

phase 2:
    q odd  -> A += D                (rankwise one-to-one)
    q even -> (Y,D) = (Y+X+D, Y)    (rankwise register orbit)

phase 3:
    q odd  -> D = C B               (<=4-source destination gather)
```

`p=1` uses the two formulas above and clears/overwrites `D` as part of the same orbit pass.

Thus the hot path needs:

- no MateID reconstruction;
- no LL/RR remote-mate search;
- no destination Motzkin rank scan;
- no CAS/atomic modular add;
- no main identity copy;
- no blocked clear;
- no second main buffer;
- no second blocked buffer.

The scratch shape is exactly `M+D`.

For the current LOW14/HIGH13 n=27 schedule, the existing B300 memory model gives

```text
LOW14  M+D ~= 4.823 GiB
HIGH13 M+D ~= 7.385 GiB
```

instead of the old `2M+2D` maxima `9.646 / 14.770 GiB`.  Unlike the current MateID backend, fusion also does not need the optional multi-GiB per-state full-Mate cache.

With the current authoritative state estimate of about `242.49 GiB/GPU`, fusion metadata around only a few tens of MiB, and a `<=7.385 GiB` scratch arena, the n=27 footprint would be roughly 250 GiB/GPU before small runtime allocations, leaving substantially more headroom than the current production configuration.

## Occupancy-major group I/O

The authoritative ID groups every exact occupancy mask into one contiguous Catalan block.  Across all width-28 odd masks the average block contains about `2873.83` residues, or `11.23 KiB` at uint32.  Width 27 averages about `2011.89` residues, or `7.86 KiB`.

A LOW14 transition-closed group contains at most `2^14` exact-mask runs; a HIGH13 group at most `2^15` runs.  Run length is exactly `Catalan((popcount(mask)+1)/2)`.

Inside group scratch, masks can be ordered by the free submask `z`:

```text
local_offset[k_fixed][z]
 = sum_{z'<z} Catalan((k_fixed+popcount(z')+1)/2)
```

for odd total occupancy.  This table depends only on the number of fixed occupied bits, not on the fixed pattern itself, so all groups share a tiny set of offset tables.  Persistent CTA/warp work queues can process Catalan blocks without per-state group unranking.

This layout is particularly attractive because the present B300 solver spends significant memory and time on LOW/HIGH rank/unrank LUTs and optional MateID caches.  Fusion replaces them with a roughly 14-MiB topology unrank table plus sub-MiB rank automata.

## Relation to current B300 hot path

The production B300 documentation lists the current transition ingredients as LOW/HIGH frontier LUTs, opportunistic full-MateID caching, local rank deltas, known rank ranges, and relaxed atomic load + CAS modular addition.  The H100 measurements also showed that deeper ranking LUTs alone gave a large speedup, while the implementation was not primarily HBM-bandwidth bound.

The fusion backend attacks exactly this remaining compute/control cost: topology is already a packed local word, all topology edits are O(1), ranking is at most four small-table lookups, and the orbit form removes CAS entirely.  The price is only the roughly 2.86% increase in total recurrence edges quantified above.

## Next implementation

The immediate correctness target is `src/cpp/probes/fusion_dense_gridfp_probe.cpp`, followed by a research-only CUDA scratch kernel using the `M+D` rest-mask orbit equations.  Production authoritative storage should not be changed until the dense CPU codec and CUDA scratch recurrence both match the existing `gridfp_transition.hpp` path step-by-step.
