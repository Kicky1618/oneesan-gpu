# RAMstream32 CPU LOW flat exact-page deltas

v5.29 is an implementation optimization of the v5.27 exact global-unique worker objective. It does not change the legal move set, objective order, load cap, or accepted-move semantics.

The provenance remains:

```text
objective=global-unique-neighbor-coalesce-v5.27-plan
```

with an additional implementation marker:

```text
implementation=flat-page-delta-v5.29
```

## Previous candidate hot path

A v5.27 candidate moves one ordered HIGH-mask job to the owner of its immediate left or right neighbour. Only the left and right worker boundaries of that job can change.

The first exact implementation represented the page-reference delta of each candidate with two temporary hash maps:

```text
unordered_map<page_id, delta> for 2 MiB pages
unordered_map<page_id, delta> for 4 KiB pages
```

Those maps were constructed for every legal candidate. At n=27 the scheduler can inspect many thousands of mask jobs over several alternating passes, while every candidate touches only at most two small boundary signatures. General-purpose hash-table construction is therefore unnecessary work in the candidate hot path.

## Flat representation

v5.29 replaces those temporary candidate maps with reusable vectors:

```text
vector<pair<page_id, delta>>
```

For each affected boundary signature, `(page,+1)` or `(page,-1)` entries are appended. The vector is then sorted by page ID and adjacent equal IDs are merged.

Because one job affects at most two boundaries, a page delta is normally in:

```text
{-2, -1, 0, +1, +2}
```

and a `+1/-1` pair for the same page disappears completely after normalization.

The global reference tables remain hash maps because they represent the persistent reference count of every currently exposed cross-worker page. v5.29 removes only the per-candidate temporary hash maps.

## Exact unique-count query

For a normalized flat delta `(page,d)`, v5.29 looks up the current global reference count `old` and forms:

```text
new = old + d
```

The unique page count changes only on:

```text
old = 0, new > 0   -> +1 unique page
old > 0, new = 0   -> -1 unique page
```

A negative `new` is an immediate error.

The selected candidate delta is then applied to the persistent reference table. The same full rebuild check from v5.27 remains after the search, so incremental flat-delta accounting must still exactly equal a from-scratch final page-reference map.

## Allocation behavior

Four vectors are allocated once per v5.27 invocation:

```text
candidate 2 MiB delta
candidate 4 KiB delta
best 2 MiB delta
best 4 KiB delta
```

They are cleared and reused for every job and candidate. Capacity is reserved for two complete factor-block boundary signatures, which is the maximum number of raw entries a one-job move can generate.

The new diagnostics are:

```text
flat_delta_normalizations=...
flat_delta_peak_entries=...
```

`flat_delta_peak_entries` is measured before duplicate-page merging and therefore exposes the largest raw temporary delta actually encountered.

## Algebra selftest

`ramstream32_cpu_low_worker_flat_delta_selftest.cu` checks three cases independently of the full scheduler:

```text
opposite-sign duplicate page cancels across two boundaries
a doubly referenced page can receive a normalized -2 removal
a new page can receive a normalized +2 insertion but adds only one unique ID
```

The regular W=10 v5.27/v5.28 exact-recurrence tests then exercise the optimized implementation inside the full scheduler. The final exact page-reference rebuild remains an additional independent accounting check.

## Expected benefit

v5.29 is aimed at schedule-build time, not LOW recurrence execution time. It should reduce allocator and hash-table overhead during v5.27 candidate evaluation, especially inside the v5.28 two-branch experiment where the exact planner runs twice.

No speedup is claimed until n=27 plan timings are measured. The relevant comparison is the exact planner `build_s` before and after this implementation change; structural page results must remain identical for a fixed search order.

If exact planner build time remains material after v5.29, the next targets are the persistent page-reference lookup structure and repeated sorting of the tiny flat deltas, not the recurrence kernel itself.
