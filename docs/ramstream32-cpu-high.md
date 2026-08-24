# RAMstream32 CPU HIGH offload

This note isolates the CPU/GPU partition experiment introduced after the hybrid-sparse LOW backend.

## Motivation

For `n=27` (`W=28`, `LOW=14`, `HIGH=13`), sending every fixed-LOW-occupancy HIGH group to the GPU costs exactly

```text
106.087684520 TiB / CRT residue
```

of H2D+D2H traffic. The group-size distribution is highly skewed, so assigning many small groups to CPU can remove substantial PCIe traffic while leaving the GPU with the large groups.

Use

```bash
python3 scripts/tools/profile_high_group_transfer.py 27
```

to reproduce the distribution without allocating the state arrays.

## Why the partition is exact

A HIGH-window group fixes the occupancy mask of LOW positions `[0, LOW_LUT_K)`.

Every HIGH transition preserves that mask:

- normal transitions leave the exact LOW code unchanged;
- the only boundary CROSS transition flips one occupied LOW symbol `R -> L`;
- therefore occupancy does not change.

The groups are transition-closed. CPU and GPU subsets can be evaluated independently and then the ordinary LOW window can run after both subsets finish.

The W=10 full-state regression checks this implementation against the reference Grid-FP recurrence. It also partitions HIGH groups into two disjoint sets and runs the two sets concurrently to catch accidental cross-group writes.

## v4.8 scratch executor

`ramstream32_cpu_high.hpp` uses the same packed factorized group layout as the GPU backend. A selected group is copied from authoritative System RAM into four mmap scratch arrays:

```text
main A / main B / blocked A / blocked B
```

The complete HIGH window is evaluated out-of-place and copied back.

This is the simplest correctness baseline, but it adds substantial DRAM traffic from identity copies and blocked-array clears.

Enable it with:

```bash
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_MODE=scratch \
CPU_HIGH_WORKERS=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32
```

## v4.9 zero-scratch direct executor

`ramstream32_cpu_high_direct.hpp` removes the temporary count arrays entirely.

At each active HIGH position, states with local pair `NN`, `NR`, or `NL` are unique representatives of three-state in-place orbits. Since every HIGH position satisfies `p > 1`, there is no bottom-edge special case:

```text
NN orbit:
  main[partner] += main[source]
  main[source]   = old_main[source] + old_blocked[drop]
  blocked[drop]  = 0

NR/NL orbit:
  main[source]   = old_main[source] + old_main[partner] + old_blocked[drop]
  blocked[drop]  = old_main[source]
  main[partner]  keeps its identity value
```

Closure states (`LL`, `RR`, `RL`) are processed after the complete orbit pass for that position. Their HIGH-side destination is always blocked because all HIGH positions have `p > 1`.

The only transition that changes the exact LOW code is CROSS. Before authoritative dense rank tables are released, the backend builds

```text
(source LOW mask-code index, matching depth)
    -> destination mask-local LOW rank
```

as a `uint16_t` table. At `LOW_LUT_K=14`, the table is about 32.1 MiB.

Enable direct mode with:

```bash
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_MODE=direct \
CPU_HIGH_WORKERS=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32
```

`CPU_HIGH_MAX_MIB=0` disables CPU HIGH offload regardless of the mode setting.

## v5.0 CPU/GPU HIGH overlap

CPU and GPU HIGH subsets touch disjoint LOW-occupancy groups, so they can execute concurrently. `CPU_HIGH_OVERLAP=1` runs CPU HIGH on a host thread while the main thread drives the GPU HIGH groups. The LOW pass begins only after both subsets have completed.

```bash
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=1 \
CPU_HIGH_WORKERS=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32
```

Overlap is not assumed to be faster. PCIe DMA and CPU direct execution both consume System-RAM bandwidth, so the sweep must compare `CPU_HIGH_OVERLAP=0` and `1` on the target machine.

## v5.1 64-bit HIGH orbit metadata

The first direct executor stored each orbit operation in 16 bytes:

- source HIGH rank;
- partner HIGH rank;
- drop HIGH rank;
- orbit kind;
- partner block ID;
- drop block ID.

The two block IDs are redundant.

For a source main factor block and active HIGH position `p`:

1. if `p != LOW_LUT_K+1`, the partner transformation is entirely inside HIGH and preserves the HIGH endpoint height and center, so the partner remains in the source main block;
2. at `p = LOW_LUT_K+1`, the pair crosses the HIGH/center boundary. `NN -> LR` leaves center `R`, while `NR -> RN` and `NL -> LN` leave center `N`; the unchanged LOW start height `source.hs` therefore determines the destination HIGH endpoint height and partner block;
3. dropping the leading `N` preserves the combined HIGH+center endpoint height, so the blocked destination block is always `source.hs`.

v5.1 stores only

```text
20-bit source HIGH rank
20-bit partner HIGH rank
20-bit drop HIGH rank
```

in one `uint64_t`. Four high bits remain unused. The orbit portion of direct metadata is therefore halved from 16 bytes/op to 8 bytes/op.

This is guarded during metadata construction: the builder still computes partner/drop states with the full topology operations and ranks them normally, then asserts that the derived block IDs equal the explicit block IDs. A mismatch aborts before the compact stream is accepted. Rank overflow beyond 20 bits also aborts.

The production `--plan-only` output computes `cpu_high_direct_meta_mib` through the compatibility stream wrappers, so it reports the complete compact footprint including all split offset tables.

## v5.2 split NN and NR/NL orbit streams

After v5.1, the direct hot loop still decoded an orbit kind and branched between the NN algebra and the common NR/NL algebra. NR and NL have identical count updates, so three streams are unnecessary.

v5.2 uses two streams per `(p, source factor block)`:

- `NN`;
- `NR/NL`.

The 64-bit record now contains only the three 20-bit HIGH ranks; stream identity supplies the algebra. Destination block reconstruction is also hoisted outside the per-operation loop because it depends only on `(p, source block, stream)`.

The reordering is exact. Every blocked drop state uniquely reconstructs its source by inserting the dropped `N`; the symbol immediately below that `N` determines whether the orbit belongs to NN, NR, or NL. Therefore two distinct N* orbit representatives cannot share the same blocked member, and partner states (`LR`, `RN`, `LN`) are never N* representatives at the same position.

The operation size remains 8 bytes. v5.2 is therefore a CPU front-end / branch-prediction optimization, not a metadata-size reduction. The extra cost is one additional offset table.

## v5.3 split BLOCK and CROSS closure streams

The remaining descriptor-kind branch in direct mode was closure dispatch. Closure operations read main source values and only add into blocked destinations; they never modify another closure source. Reordering ordinary blocked closures and boundary CROSS closures is therefore safe, and additions into the same destination are commutative modulo the CRT prime.

v5.3 builds two closure streams:

- ordinary `BLOCK` closures;
- boundary `CROSS` closures.

Runtime no longer decodes `HIGHDESC_BLOCK` versus `HIGHDESC_CROSS` inside the hot loop. The LOW mask-code base used by CROSS is computed only when the CROSS stream for a source block is non-empty.

The n=27 direct plan log reports all four stream populations:

```text
nn_orbit_ops=...
nrnl_orbit_ops=...
block_closure_ops=...
cross_closure_ops=...
```

CI requires these fields to be present, while the W=10 exhaustive reference comparison continues to validate the complete direct result.

## n=27 partition sizes

| threshold | CPU groups | PCIe removed | PCIe remaining |
|---:|---:|---:|---:|
| 64 MiB | 3,473 | 2.6766 TiB | 103.4111 TiB |
| 128 MiB | 9,908 | 19.6619 TiB | 86.4258 TiB |
| 256 MiB | 12,911 | 38.2452 TiB | 67.8425 TiB |
| 512 MiB | 14,913 | 61.2090 TiB | 44.8787 TiB |
| 1024 MiB | 15,914 | 82.5714 TiB | 23.5163 TiB |

These numbers are transfer-volume savings, not predicted wall-time improvements. CPU work must beat the removed PCIe time.

## Threshold sweep and break-even analysis

Build once, then run the same backend across several thresholds:

```bash
N=27 \
CPU_HIGH_MODE=direct \
CPU_HIGH_OVERLAP=0 \
CPU_WORKERS=32 \
CPU_HIGH_WORKERS=32 \
THRESHOLDS='0 64 128 256 512 1024' \
REPEATS=2 \
bash scripts/bench/ramstream32-cpu-high-sweep.sh
```

Repeat with `CPU_HIGH_OVERLAP=1`. Odd repeats use ascending thresholds and even repeats use descending thresholds to reduce order bias. Every run must produce the same residue.

The TSV records total wall time, H2D/GPU/D2H time, CPU HIGH and LOW time, removed/remaining PCIe volume, group count, mode, and overlap setting. The sweep automatically runs

```bash
python3 scripts/tools/analyze_cpu_high_sweep.py RESULT.tsv
```

which reports measured DMA seconds saved per TiB, CPU HIGH seconds spent per TiB, sequential break-even margin, offload efficiency, and the best observed threshold.

## Validation

`.github/workflows/ramstream32-sparse-ci.yml` covers:

1. W=22 and W=28 compilation;
2. the normal hybrid-sparse plan;
3. the n=27 256-MiB scratch partition, requiring 12,911 CPU groups;
4. the same n=27 direct metadata plan and all four v5.3 stream classes;
5. W=10 full-state comparison of both CPU HIGH executors and all CPU LOW executors against the reference recurrence;
6. concurrent execution of two disjoint W=10 HIGH group subsets against the same reference result.

The compact orbit builder additionally validates every derived partner/drop block against the full topology result before discarding those block IDs.

## Next experiments

The next large gains are scheduling and memory-placement problems rather than per-operation branches:

1. replace the fixed MiB threshold with a cost model learned from measured CPU-HIGH and DMA cost per group-size band;
2. NUMA-pin CPU HIGH workers and authoritative occupancy slices so CPU offload does not bounce the roughly 1.9-TiB state space across sockets;
3. allocate CPU workers dynamically between HIGH and LOW according to the measured critical path, especially when `CPU_HIGH_OVERLAP=1`;
4. for multi-GPU B300 nodes, keep the largest HIGH occupancy groups resident across more than one local step when dependencies permit, while CPU handles the long tail of small groups.
