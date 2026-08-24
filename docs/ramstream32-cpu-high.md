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

`CPU_HIGH_MAX_MIB=0` disables threshold-based CPU HIGH offload.

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

The first direct executor stored each orbit operation in 16 bytes. v5.1 removes redundant block IDs and stores only three 20-bit HIGH ranks in one `uint64_t`. Four high bits remain unused, so the orbit portion of direct metadata is halved from 16 bytes/op to 8 bytes/op.

For a source main factor block and active HIGH position `p`:

1. if `p != LOW_LUT_K+1`, the partner transformation is entirely inside HIGH and remains in the source main block;
2. at `p = LOW_LUT_K+1`, `NN -> LR` leaves center `R`, while `NR -> RN` and `NL -> LN` leave center `N`, so `source.hs` and the orbit class determine the partner block;
3. dropping the leading `N` always lands in blocked block `source.hs`.

The builder still computes the full partner/drop states and explicit block IDs first, then asserts that the derived IDs match before accepting the compact stream. Rank overflow beyond 20 bits also aborts.

## v5.2 split NN and NR/NL orbit streams

NR and NL have identical count updates. v5.2 therefore uses two streams per `(p, source factor block)`:

- `NN`;
- `NR/NL`.

Stream identity supplies the algebra, so runtime no longer decodes or branches on orbit kind. Partner/drop block reconstruction is also hoisted outside the per-operation loop because it depends only on `(p, source block, stream)`.

The reordering is exact. Every blocked drop state uniquely reconstructs its source by inserting the dropped `N`; the lower symbol determines NN, NR, or NL. Partner states (`LR`, `RN`, `LN`) are never N* representatives at the same position.

## v5.3 split BLOCK and CROSS closure streams

Closure operations read main source values and only add into blocked destinations, so ordinary blocked closures and boundary CROSS closures can be reordered safely. v5.3 builds independent `BLOCK` and `CROSS` streams and removes the last descriptor-kind branch from the CPU HIGH direct hot loops.

The n=27 direct plan log reports all four stream populations:

```text
nn_orbit_ops=...
nrnl_orbit_ops=...
block_closure_ops=...
cross_closure_ops=...
```

CI requires these fields to be present, while the W=10 exhaustive reference comparison validates the complete result.

## v5.4 exact group cost model and file policy

A pure size threshold is only a proxy for CPU cost. Two occupancy groups with similar transfer size can have different NN/NRNL/closure/CROSS densities.

The cost-plan probe computes the exact number of direct inner-loop cell iterations for every HIGH occupancy group:

```bash
N=27 bash scripts/build/gridfp-ramstream32-cpu-high-cost-plan.sh
./build/ramstream32_cpu_high_cost_plan_n27 27 > high-cost.tsv
```

For each group it reports:

- round-trip bytes removed from PCIe if that group moves to CPU;
- main and blocked state counts;
- NN cell iterations;
- NR/NL cell iterations;
- ordinary blocked-closure cell iterations;
- CROSS cell iterations.

The probe also verifies that all LOW-occupancy groups partition the authoritative main and blocked state spaces exactly once.

Measured threshold sweeps can calibrate aggregate PCIe bandwidth and direct-executor weighted Gcells/s:

```bash
python3 scripts/tools/analyze_cpu_high_sweep.py sweep.tsv \
  --cost-plan high-cost.tsv
```

The resulting rates feed the planner:

```bash
python3 scripts/tools/plan_cpu_high_groups.py high-cost.tsv \
  --pcie-gib-s 45 \
  --cpu-gcell-s 3.2 \
  --gpu-target-mib 12288 \
  > cpu-high.groups
```

The planner evaluates each group independently using estimated DMA time saved versus CPU direct time. Groups larger than `--gpu-target-mib` are forced to CPU so the remaining GPU set still fits. The resulting list is not required to be monotone in group size.

Run that exact policy with:

```bash
CPU_HIGH_MODE=direct \
CPU_HIGH_GROUPS_FILE=cpu-high.groups \
CPU_HIGH_WORKERS=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32
```

`CPU_HIGH_GROUPS_FILE` overrides `CPU_HIGH_MAX_MIB`. The backend reports `cpu_high_policy=file` plus an FNV hash of the final selected-group bitset so a benchmark can identify the exact partition without embedding thousands of IDs in its output.

## n=27 size-threshold reference points

| threshold | CPU groups | PCIe removed | PCIe remaining |
|---:|---:|---:|---:|
| 64 MiB | 3,473 | 2.6766 TiB | 103.4111 TiB |
| 128 MiB | 9,908 | 19.6619 TiB | 86.4258 TiB |
| 256 MiB | 12,911 | 38.2452 TiB | 67.8425 TiB |
| 512 MiB | 14,913 | 61.2090 TiB | 44.8787 TiB |
| 1024 MiB | 15,914 | 82.5714 TiB | 23.5163 TiB |

These are transfer-volume reference points, not predicted wall-time improvements.

## One-command threshold calibration

For direct mode, the sweep now builds the exact cost plan automatically by default, calibrates the model, and emits a candidate group policy:

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

Repeat with `CPU_HIGH_OVERLAP=1`, because memory-bandwidth contention changes both measured DMA and CPU rates. Generated artifacts include the raw sweep TSV, metadata, exact group-cost TSV, analysis output, and a candidate `.groups` policy file.

Set `COST_PLAN=none` to disable automatic cost-plan generation or provide `COST_PLAN=/path/to/existing.tsv` to reuse one.

## Validation

`.github/workflows/ramstream32-sparse-ci.yml` covers W=22/W=28 compilation, normal plans, scratch/direct CPU HIGH plans, all four direct stream classes, the W=10 exhaustive reference comparison, concurrent disjoint HIGH execution, and the exact group-cost probe.

`.github/workflows/ramstream32-bench-script-ci.yml` syntax-checks the sweep/build scripts, compiles the Python tools, validates sweep calibration on synthetic data, and checks that the planner can choose a profitable group while rejecting an intentionally expensive one.

## Next experiments

The remaining large opportunities are increasingly memory-topology and scheduling problems:

1. NUMA-pin CPU HIGH workers and authoritative occupancy slices so CPU offload does not bounce the roughly 1.9-TiB state space across sockets;
2. estimate separate NN, NR/NL, BLOCK, and CROSS weights from hardware counters instead of using equal cell weights;
3. allocate CPU workers dynamically between HIGH and LOW according to the measured critical path, especially when `CPU_HIGH_OVERLAP=1`;
4. for multi-GPU B300 nodes, keep the largest HIGH occupancy groups resident across more than one local step when dependencies permit, while CPU handles the long tail of small groups.
