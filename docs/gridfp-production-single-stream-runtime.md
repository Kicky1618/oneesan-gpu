# Production Grid-FP single-stream runtime plan

This note records the production-correct reduced Grid-FP runtime path.  It is
separate from the older abstract two-cell model.

## W=28 state size

The exact reduced quotient is

```text
D28 = M28 + M27 - M26
    = 473,397,057,701 states.
```

A uint32 count stream is 1,763.541466 GiB in aggregate.  With the current
whole-outer-support weighted owner on 8 GPUs and K=13, each GPU owns roughly
220.06--220.85 GiB.  The production design therefore cannot use a full second
state buffer on a 288 GiB-class GPU.

Interior component updates are in-place: every production component has the
same grouped physical slot set on the source and destination sides.  A warp
loads the complete component before writing any destination.  No global CSR,
component table, inverse table, or second state stream is required.

## Why K=13

For even W=28 the runtime uses K=(W-2)/2=13, so each grouped physical window has
K+2=15 sites.

The high and low windows are

```text
high = [13,27]
low  = [0,14]
```

and overlap in two physical sites.  This is the smallest K for which one high
segment plus one low segment covers every interior step without a gap.

An exact W=28 sweep over K=13..20 shows K=13 also minimizes peer redistribution
bytes for the current whole-group owner.  One K=13 redistribution changes owner
for 409,769,189,454 reduced states, or 1.490731627 TiB of ideal uint32 payload.
There are two such redistributions in a forward+reverse two-row snake period,
so the ideal logical payload floor is 2.981463254 TiB per two-row period.

## Steady-state two-row schedule

The current runtime schedule is intentionally asymmetric by one step inside the
same 15-site owner window:

```text
high edge: main -> expanded high quotient
forward high interior: p=26..15   (12 steps)
redistribute high -> low
forward low interior:  p=14..2    (13 steps)
low edge: compress -> main -> expand reverse
reverse low interior:  p=2..13    (12 steps)
redistribute low -> high
reverse high interior: p=14..26   (13 steps)
high edge: compress -> main -> expand forward
```

This is why the earlier K=12 main-only one-cell shift is not needed in the
production runtime.  K=13 keeps the low-edge compression and expansion in the
same physical window [0,14], and the first reverse segment remains inside that
window for p=2..13.

## Row-turn component maps

The edge maps are rectangular but table-free.

### Q1 -> main compression

There is one component for every width-(W-1) Motzkin label `v`.

```text
if v[0] != N: seed = B(v)
else:         seed = M(blocked_exclude(v,1))
```

The exact incoming list for each main destination is reconstructed from:

* main excluded identity,
* local p=1 include inverses,
* `ordinary_closure_preimages_partial(..., p=1)`, and
* the retained blocked excluded preimage.

Finite-width checks through W=12 give the component-size bound pattern
`floor((W+3)/2)`; the W=28 candidate is 15 source/destination states.

### main -> Q2 reverse expansion

Expansion is the production reverse reduced p=1 map restricted to main
sources.  Labels are width-(W-1) Motzkin words for which bits 0 and 1 are not
both N.

```text
seed = M(blocked_exclude_reverse(v,W,1))
```

The incoming list is `inverse_reduced_reverse(...,p=1)` with blocked sources
filtered out.  The component count is `M_{W-1}-M_{W-3}`.  Finite-width checks
through W=12 give W=28 candidate maxima of 17 main sources and 18 Q2
destinations.

## Redistribution cycle kernel

The state layout change is an in-place permutation of primitive-contiguous
runs.  For W=28,K=13:

* main support cycles have order 28,
* blocked support cycles have order 2,
* no run table or visited bitset is needed,
* one primitive value is held temporarily in a register while a cycle rotates.

The original P2P kernel recomputes the same grouped rank independently in every
lane and assigns the complete cycle to the owner of the lexicographically
smallest support.

The experimental shared-modal kernel changes two things without changing the
permutation:

1. lane 0 computes owner/local bases for the at-most-28 runs once and stores
   them in about 3 KiB of shared metadata per block;
2. the cycle is executed by the GPU owning the largest number of runs in that
   cycle, minimizing remote run reads/writes among all single-executor choices.

The optimized kernel is an A/B experiment only until it is compiled and timed
on a real multi-GPU CUDA host.

## Validation status

CPU structural probes cover production transition semantics, quotient rank,
factorized codecs, table-free inverse, components, grouped owner locality,
row-turn seeds, and window planning.

CUDA probes now exist for:

* grouped interior updates,
* single-buffer interior updates,
* P2P cycle redistribution,
* row-turn compression/expansion,
* the K=12 diagnostic row-turn pipeline, and
* shared-metadata/modal P2P redistribution.

The current execution environment has no `nvcc`, so the newest CUDA probes have
not been compiled or executed here.  GitHub Actions are intentionally not used
for this validation path.

## Next production gates

1. Compile the dedicated probes on a local CUDA machine and inspect ptxas
   register/local-memory usage.
2. Run baseline vs shared-modal P2P A/B on at least 2 GPUs, then 8 NVLink GPUs.
3. If the shared-modal kernel wins, replace the redistribution kernel in the
   two-row runtime microprobe.
4. Validate the complete two-row residue against the CPU production reference.
5. Only after those gates, move the reduced single-stream path into the B300
   production backend.
