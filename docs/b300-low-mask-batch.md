# B300 LOW mask-batched executor research

## Motivation

The v0.35+ LOW executor waits only once per HIGH-mask group, but it still launches
one orbit kernel and one closure kernel for every LOW position of every mask.
For n=27 (W=28, LOW=14, HIGH=13):

- HIGH occupancy masks: 8192
- rows: 28
- LOW stages per group: 14 orbit + 14 closure = 28
- current LOW kernel launches/residue: 6,422,528

A device-level mask-batched executor would launch one kernel per
`(device,row,p,stage)`, reducing this to 6,272 launches/residue on eight GPUs,
exactly 1024x fewer launch calls.

## Scheduling shape

Do not flatten every warp task across every mask and binary-search a giant mask
prefix.  That would trade host launch overhead for trillions of extra mask
lookups.

Instead build a small CTA descriptor list per device and stage:

```text
{ mask, replica, replicas }
```

A descriptor owns one CTA.  Its warps process the selected mask's local task
space as:

```text
warp_task = replica * warps_per_block + warp_in_block
step      = replicas * warps_per_block
```

The host chooses `replicas` from a small range such as 1..16 according to the
mask's exact orbit/closure task count.  This preserves direct group-local task
mapping, gives large masks more CTAs, and requires at most about 16 descriptors
per mask.  At n=27 that is at most 131,072 descriptors across all GPUs; with an
8-byte descriptor the total scheduling table is only 1 MiB.

## Metadata placement

v0.42 already packages the mutable LOW group metadata into a compact structure.
For batching, constant memory can no longer hold one current group.  Split it
into:

1. static per-mask metadata: MAIN/BLOCKED FBlocks and mask identity;
2. dynamic per-(mask,cap) metadata: orbit warp prefix/counts and closure
   prefix/begin/selected fields.

With current v0.42 fields, storing every cap for every owner mask costs about
154.9 MiB/GPU at n=27.  This is small relative to the roughly 19 GiB modeled HBM
headroom of the current B300 plan, and removes all per-row config uploads.
A lower-memory first implementation can keep only one row's dynamic table and
bulk-upload about 11 MiB/GPU per row.

## Kernel structure

For each row and LOW position `p`:

1. launch one mask-batched LOW orbit kernel per GPU;
2. launch one mask-batched LOW closure kernel per GPU;
3. rely on default-stream ordering between the two launches and the next `p`.

Each CTA reads its descriptor, selects the corresponding authoritative
`main_base/block_base`, loads that mask's metadata, and then reuses the existing
v0.37/v0.42 group-local task math.

A later cooperative/persistent variant can fuse all 28 LOW stages for a row,
but that should be a separate experiment because it introduces grid-wide
barriers and cooperative-launch occupancy constraints.

## Correctness invariants

- LOW transitions never change the HIGH occupancy mask; CROSS descriptors may
  flip HIGH L/R state but preserve occupancy, so every write stays owner-local.
- Orbit must globally finish before closure for one `p`.
- Closure must globally finish before orbit for `p-1`.
- Different HIGH occupancy masks are disjoint authoritative ranges and may run
  concurrently.
- Batched and per-group executors must produce identical residues for small-width
  exhaustive tests before B300 benchmarking.

## n=27 model

`src/cpp/probes/factor_low_maskbatch_launch.cpp` pins:

- 6,422,528 -> 6,272 kernel launches/residue;
- 229,376 -> 224 group-config/update calls if using one bulk row update/device;
- 162,422,784 bytes (~154.9 MiB) per GPU to retain all 14 cap-specific v0.42
  configs under the conservative full-struct model;
- <= 1 MiB total descriptor memory with a 16-CTA/mask cap.

The next implementation step is a small-width batched semantic kernel using the
replica descriptor mapping, followed by integration into the full B300 row loop.
