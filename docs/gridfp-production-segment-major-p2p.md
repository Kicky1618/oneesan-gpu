# Production Grid-FP segment-major P2P redistribution

This note records the current exact multi-GPU redistribution candidate for the
production reduced Grid-FP solver. It is deliberately separate from the
abstract two-cell channel. All state counts and transitions here refer to the
production quotient.

## Fixed production geometry for W=28

The current steady-state layout uses K=13, hence a 15-site local window.
The two owner layouts are

- high window: physical sites `[13,27]`,
- low window: physical sites `[0,14]`.

They overlap in exactly two physical sites. Interior reduced transitions stay
inside one owner layout. Only the high/low layout change requires inter-GPU
redistribution.

The production reduced state count is

`D_28 = 473,397,057,701`.

With eight whole-outer-support owners, the largest uint32 state shard is about
220.85 GiB. A second full state stream therefore must not be used on a 288 GB
B300. Interior transitions, row turns, and the redistribution described here
are all in-place apart from bounded scratch.

## Run permutation

At a fixed redistribution boundary, primitive rank is the innermost coordinate.
All states with the same physical support form one contiguous primitive run.
The high/low layout change is a permutation of these runs:

- main support: cyclic rotation of the 28 physical support bits by 13 sites
  (cycle order divides 28),
- blocked support: exchange the two 13-site exclusive blocks while keeping the
  canonical middle two bits fixed (cycle order divides 2).

No value arithmetic is performed during redistribution.

## Why single-executor cycles are not the final form

A table-free in-place implementation can assign each complete run cycle to one
GPU. That needs no second state stream, but the executor repeatedly reads and
writes remote runs. Even when the executor is chosen from the modal owner of
the cycle, this performs more NVLink traffic than the ownership migration
itself requires.

The exact lower bound is obtained by splitting a cycle into maximal consecutive
same-owner segments.

For every owner boundary:

1. the destination owner reads the predecessor run once into local scratch,
2. all GPUs synchronize before any network cycle in that batch is modified,
3. every owner shifts its segment backwards using only local HBM,
4. the first local run receives the saved predecessor value.

Cycles are assigned atomically to batches. Individual owner boundaries of one
cycle must never be placed in different batches, because an earlier batch could
otherwise overwrite an old value required by a later batch.

For W=28, K=13, eight GPUs the exact network payload of one redistribution is

`409,769,189,454 uint32 values`

or about

`1.4907316 TiB`.

This equals the exact number of coordinates whose owner changes. The segmented
algorithm therefore performs one peer read per necessary cross-owner value and
no avoidable peer write.

## Segment-major metadata

The runtime does not keep a global cycle table on every GPU. Metadata is
transposed into destination-owner-major form.

For each destination GPU, batch, and primitive-count class, a network segment
stores:

- one packed remote predecessor run,
- its concatenated destination-local run sequence,
- one uint32 `run_begin` entry.

A run base needs only

- owner: 3 bits,
- owner-local rank: 36 bits.

The implementation stores this as SoA `uint32 low + uint8 high`, i.e. five
bytes per run.

There are only 14 primitive-count classes, `Catalan(1)..Catalan(14)`. Therefore
primitive count and scratch offset do not need to be stored per segment.
For group metadata `(begin,end,scratch_base,pc)`, segment index `i` uses

`scratch_base + (i - begin) * pc`.

The segment length is recovered from consecutive `run_begin` values. Cycle ID,
segment length, and per-segment scratch offset are absent from the hot metadata.

For W=28 the analytic run counts are

- non-fixed run records: `167,763,968`,
- network owner-boundary segments: `117,118,478`.

The dominant one-direction cluster metadata is approximately

`5 * 167,763,968 + 9 * 117,118,478` bytes,

about 1.763 GiB over all eight GPUs, or about 225.65 MiB/GPU on average.
Forward and reverse schedules can both remain resident; scratch dominates the
extra HBM footprint.

## Scratch batching and memory planning

The builder count pass produces an exact segment count for every
`gpu × batch × primitive-class` group. Thus scratch is known before the state
stream is allocated:

`batch_scratch(g,b) = sum_class segment_count(g,b,class) * pc(class)`.

The production setup should:

1. run the count pass on GPU 0,
2. choose a batch count (and, if needed, a hash salt) that satisfies the scratch
   cap,
3. compile and distribute both direction schedules,
4. release builder-global temporary metadata,
5. only then allocate the large per-GPU state stream and shared scratch.

`scripts/run/gridfp-p2p-select-segment-major-batches.sh` implements the initial
batch-count search. The planner also reports state + scratch + both schedule
metadata against a decimal 288 GB B300 budget without allocating the state.

## GPU builder

`gridfp_reduced_production_p2p_segment_major_builder.cuh` contains two setup
kernels.

The count pass sees only physical support and owner. It does not materialize
MateIDs and does not call grouped local rank.

The fill pass materializes exact grouped run bases once. Within each metadata
group it uses a paired uint64 atomic cursor:

`(item_count << 32) | run_count`.

An item of length `len` reserves both fields with one

`atomicAdd(cursor, (1ULL << 32) | len)`.

This prevents arbitrary CUDA scheduling from assigning a run interval to the
wrong header. W=28 has fewer than `2^28` non-fixed runs, so the low 32-bit run
counter cannot carry into the item counter.

The production-order integration probe builds forward and reverse metadata on
GPU 0, copies owner slices to the destination GPUs, frees the global builder
arrays, then allocates the large state arrays.

## Runtime kernels

The production-facing kernels are isolated in

`src/cuda/gridfp/gridfp_reduced_production_p2p_segment_major_runtime.cuh`.

They are:

- `p2p_major_local_cycle_kernel`,
- `p2p_major_gather_kernel`,
- `p2p_major_rotate_kernel`.

The full two-row candidate leaves the existing counter-free interior and turn
kernels unchanged and replaces only the two high/low redistributions.

## Validation ladder

CPU proofs, requiring no CUDA:

```bash
scripts/bench/gridfp-p2p-segment-major-source-preflight.sh
```

On a CUDA host, build individual stages with

```bash
MODE=segment-major-count scripts/build/gridfp-reduced-p2p-schedule-probe.sh
MODE=segment-major-fill scripts/build/gridfp-reduced-p2p-schedule-probe.sh
MODE=segment-major scripts/build/gridfp-reduced-p2p-schedule-probe.sh
MODE=two-row-segment-major scripts/build/gridfp-reduced-p2p-schedule-probe.sh
MODE=two-row-segment-major-built scripts/build/gridfp-reduced-p2p-schedule-probe.sh
```

The intended first multi-GPU execution width is W=10 with two GPUs and four
batches. The final full two-row probe compares against the raw production
operator and includes the projected entry into the next row.

Before a W=28 allocation, run the state-free planner or selector:

```bash
MODE=segment-major-plan scripts/build/gridfp-reduced-p2p-schedule-probe.sh
build/gridfp_reduced_p2p_segment-major-plan 28 8 8 4096 32

scripts/run/gridfp-p2p-select-segment-major-batches.sh
```

## Current validation status

The CPU structural probes were designed to exhaustively validate small widths.
The new CUDA segment-major execution, builder, memory planner, and full two-row
integration have not been compiled or executed in the current development
environment because `nvcc` is unavailable there. GitHub Actions have not been
used. Do not promote this path to the authoritative B300 runtime until the
small-width CUDA ladder passes and ptxas register/local-memory output has been
reviewed.
