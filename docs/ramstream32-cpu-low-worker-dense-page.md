# RAMstream32 CPU LOW dense worker-page substrate

v5.30 is a research-only substrate for the exact cross-worker page objective.
It does not change the v5.28/v5.29 selected schedule yet.

v5.29 already removed candidate-local `unordered_map` construction by using a
reusable sorted flat page delta.  Persistent exact page reference counts still
use 64-bit VM page IDs in hash tables.  v5.30 observes that every page the
search can ever expose is present in one of the finite ordered worker-boundary
signatures.

The v5.30 plan therefore:

1. constructs every ordered boundary signature once;
2. unions the 2 MiB and 4 KiB page-ID universes independently;
3. assigns each page a dense `uint32_t` ID;
4. rewrites every boundary signature to dense IDs;
5. permits persistent reference counts to use flat `uint32_t` vectors.

The exact objective is unchanged:

```text
(global unique 2 MiB pages,
 global unique 4 KiB pages,
 weighted StorageBlock owner transitions)
```

Only the representation changes.

## Algebra checks

`ramstream32_cpu_low_worker_dense_page_selftest.cu` verifies:

- 64-bit page IDs are sorted and compressed to deterministic dense IDs;
- a page shared by two boundaries gets one universe ID and refcount 2;
- replacing one boundary by another cancels shared page deltas exactly;
- duplicate `+2` refcount updates preserve the unique-page count;
- delta application and predicted unique counts agree.

## n=27 plan probe

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

`raw_signature_payload_mib` counts the signature entries as 64-bit page IDs.
`dense_signature_payload_mib` counts the same entries as 32-bit IDs.  The dense
representation also needs one 64-bit universe entry per distinct page plus one
32-bit persistent refcount per page.

## Promotion rule

Do not replace the v5.29 implementation solely because dense IDs use less
payload memory.  Promote the dense representation into the exact optimizer only
when the n=27 plan shows both:

```text
dense_total_index_mib is modest relative to solver metadata
dense_build_s is small enough to amortize against exact planner build time
```

After integration, W=10 must run the v5.29 and dense implementations from the
same v5.25 parent and assert identical final schedule, exact tuple, and two
independent LOW recurrence generations before the dense implementation becomes
the default research path.
