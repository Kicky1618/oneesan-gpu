# RAMstream32 CPU LOW dense worker-page substrate

v5.30 is a research-only representation and exact-coalescer implementation for
the cross-worker page objective.

v5.29 already removed candidate-local `unordered_map` construction by using a
reusable sorted flat page delta. Persistent exact page reference counts still
use 64-bit VM page IDs in hash tables. v5.30 observes that every page the
search can ever expose is present in one of the finite ordered worker-boundary
signatures.

The v5.30 representation therefore:

1. constructs every ordered boundary signature once;
2. unions the 2 MiB and 4 KiB page-ID universes independently;
3. assigns each page a dense `uint32_t` ID;
4. rewrites every boundary signature to dense IDs;
5. stores persistent reference counts in flat `uint32_t` vectors;
6. evaluates candidate deltas entirely in dense IDs.

The exact objective is unchanged:

```text
(global unique 2 MiB pages,
 global unique 4 KiB pages,
 weighted StorageBlock owner transitions)
```

Candidate order, legal neighbour moves, domain load caps and score tie-breaking
mirror v5.29 intentionally.

## Algebra checks

`ramstream32_cpu_low_worker_dense_page_selftest.cu` verifies:

- 64-bit page IDs are sorted and compressed to deterministic dense IDs;
- a page shared by two boundaries gets one universe ID and refcount 2;
- replacing one boundary by another cancels shared page deltas exactly;
- duplicate `+2` refcount updates preserve the unique-page count;
- delta application and predicted unique counts agree.

## Representation plan probe

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-dense-page-plan.sh
```

Run:

```bash
./build/ramstream32_cpu_low_worker_dense_page_plan_n27 27
```

Important fields are:

```text
objective=dense-page-id-substrate-v5.30-plan
ordered_jobs
boundaries
universe_2m
universe_4k
signature_entries_2m
signature_entries_4k
max_boundary_entries_2m
max_boundary_entries_4k
candidate_peak_delta_entries
raw_signature_payload_mib
dense_signature_payload_mib
dense_universe_mib
dense_ref_mib
dense_total_index_mib
mask_index_build_s
dense_build_s
```

`raw_signature_payload_mib` counts signature entries as 64-bit page IDs.
`dense_signature_payload_mib` counts the same entries as 32-bit IDs. The dense
representation additionally stores one 64-bit universe entry and one 32-bit
persistent refcount per distinct page.

## Exact v5.29-v5.30 equivalence

`ramstream32_cpu_low_worker_dense_equivalence_selftest.cu` starts two pools from
the same deterministic:

```text
refined domain -> v5.23 page -> v5.25 worker-locality
```

parent. One pool runs v5.29, the other runs v5.30. The test requires exact
equality of:

```text
sticky_worker_jobs
sticky_worker_cells
before/after exact page tuple
candidate_evaluations
cap_rejections
accepted_moves
page-improving/transition-only move counts
moved_cells
delta-normalization count
```

The dense final schedule then executes two real W=10 LOW generations and every
main/blocked state is compared with the independent reference recurrence.

## n=27 flat-vs-dense build-time comparison

Build:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-low-worker-dense-compare-plan.sh
```

Sweep the intended two-domain worker layouts:

```bash
N=27 \
CONFIGS='32:16 64:32 96:48 128:64' \
bash scripts/bench/ramstream32-cpu-low-worker-dense-compare-plan-sweep.sh
```

Each topology hard-checks:

```text
identical_schedule=1
identical_trace=1
```

and reports:

```text
flat_build_s
dense_build_s
dense_vs_flat_speedup
dense_index_mib
dense_index_build_s
candidate_evaluations
accepted_moves
flat_delta_peak_entries
dense_delta_peak_entries
```

The sweep labels a topology `dense_faster` only above 1.02x, `dense_slower`
below 0.98x, and otherwise `near_tie`.

## Promotion rule

Do not replace v5.29 solely because dense IDs reduce signature payload memory.
Promotion to the default research exact planner requires:

```text
W=10 identical final schedule and search trace
W=10 two-generation exact recurrence passes
n=27 dense index memory is modest
n=27 median dense_vs_flat_speedup is useful or at least not materially slower
```

If dense build time loses despite faster candidate refcount access, retain v5.29
or share/prebuild the dense index across the direct/hybrid v5.28 branches before
considering production integration.
