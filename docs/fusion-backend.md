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

Thus the earlier idea of a 1-GiB mask-rank LUT and 256-MiB raw fusion-rank LUT is unnecessary.

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
AD            <- B, coefficient -h/(h+1), h>0
CC            <- C
CD, DC       <- D
```

Boundary/odd-position insertions are also one-to-one.  Therefore NN insertion is an injective sparse map in destination space and never requires an atomic between different NN sources.

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

so fusion locality costs only about 2.86% more graph edges over the full recurrence.

Only a small minority of fusion edges need a general modular multiplication.  The non-unit coefficients are the insertion `AD` term and cap `DA` term; all remaining local coefficients are `0`, `+1`, or can be handled by trivial sign/add operations.  The height-dependent constants are tiny tables per CRT prime.

## Atomic-free destination structure

The local maps make a CAS-free implementation possible.

For main destinations:

- identity: one source (itself);
- NN inverse: at most one source;
- vacancy motion: one source;
- blocked release: one source.

For blocked destinations at `p>1`, the two source classes are disjoint by destination occupancy:

- destination bit `p-1 = 1`: unique `10` shrink source;
- destination bit `p-1 = 0`: cap inverse, at most four `11` sources.

Therefore a fully out-of-place destination-gather kernel needs at most three source terms for a main destination and at most four for a blocked destination, and every destination is written exactly once.

A lower-traffic in-place schedule is possible for `p>1` with one main buffer and two blocked buffers:

```text
1. Dnew = gather_cap_or_drop(main_old)
   - destination blocked state reads <=4 main sources
   - one ordinary store, no clear and no atomic

2. main_forward(main)
   - process source classes 00 and 01
   - NN targets 11 and motion targets 10 are disjoint
   - insertion destinations are injective
   - ordinary read/add/write only

3. blocked_release(Dold -> main)
   - one-to-one mapping
   - ordinary read/add/write only

4. swap(Dold,Dnew)
```

The ordering preserves all old-main values needed by step 1.  `p=1` contains the two local occupancy orbits `00<->11` and `01<->10`; the already-tested p=1 orbit technique is the natural special case.

This gives scratch `M+2D`, matching the current main-only-in-place memory shape, but removes MateID reconstruction, long mate scans, destination Motzkin ranking, CAS modular addition, main identity copy, and blocked clear from the hot path.

For the current LOW14/HIGH13 windows the existing documentation gives `M+2D` count scratch of about 6.065 GiB and 9.287 GiB respectively.  Unlike the current MateID backend, fusion does not need a per-state full-Mate cache.

## Occupancy-major group I/O

The authoritative ID groups every exact occupancy mask into one contiguous Catalan block.  A LOW14 transition-closed group contains at most `2^14` such runs; a HIGH13 group at most `2^15` runs.  Run length is exactly `Catalan((popcount(mask)+1)/2)`.

Inside group scratch, masks can be ordered by the free submask `z`:

```text
local_offset[k_fixed][z]
 = sum_{z'<z} Catalan((k_fixed+popcount(z')+1)/2)
```

for odd total occupancy.  This table depends only on the number of fixed occupied bits, not on the fixed pattern itself, so all groups share a tiny set of offset tables.  Persistent CTA/warp work queues can process Catalan blocks without per-state group unranking.

This layout is particularly attractive because the present B300 solver spends significant memory and time on LOW/HIGH rank/unrank LUTs and optional MateID caches.  Fusion replaces them with a roughly 14-MiB topology unrank table plus sub-MiB rank automata.

## Next implementation

The shortest production experiment is a research-only dense CPU backend using the authoritative ID above, followed by a CUDA scratch kernel with the three `p>1` phases.  It should be compared step-by-step against the current `gridfp_transition.hpp` recurrence before changing production storage.
