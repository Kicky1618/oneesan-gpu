# RAMstream32 hybrid-sparse CPU LOW/HIGH backend

This note records the low-VRAM / large-System-RAM execution path after the factorized authoritative layout work.

## Architecture

The authoritative main/blocked count matrices live in System RAM / mmap storage in occupancy-major factorized order.

The baseline hybrid-sparse path uses the GPU for the HIGH window and CPU workers for the LOW window. v4.8 additionally allows selected HIGH groups to stay in System RAM and execute on CPU, removing their host-device transfers.

For `n=27` (`W=28`, `LOW=14`, `HIGH=13`), the authoritative state storage is about 1939.89 GiB. With every HIGH group on the GPU, the exact H2D+D2H traffic floor is 106.087684520 TiB per residue. At an ideal 50 GiB/s this alone is about 36.21 minutes.

## v4.4: 64-bit sparse orbit operations

The first sparse CPU LOW implementation stored each N* orbit operation in 12 bytes. The source factor block and active position already determine both destination factor blocks, so v4.4 stores only three 20-bit LOW ranks and the orbit kind in one `uint64_t`.

At `n=27`:

- orbit operations: 16,826,838;
- old 12-byte orbit stream: about 192.57 MiB;
- v4.4 8-byte orbit stream: 128.379 MiB;
- saved: about 64.19 MiB.

The sparse builder verifies every derived destination block against the dense `LowOrbitHost` representation before discarding the redundant fields.

## v4.5: pre-ranked CROSS transitions

A LOW closure can cross the LOW/HIGH boundary when an RR partner search escapes into HIGH. Before v4.5, every active CROSS update scanned the HIGH code and binary-searched the modified code to recover its destination rank.

v4.5 builds

`(source HIGH mask-code index, matching depth) -> destination mask-local HIGH rank`

while the dense HIGH rank table is still available. With `HIGH_LUT_K <= 15`, the result fits in `uint16_t`; `0xffff` is the invalid sentinel.

At `n=27`, the pre-ranked CROSS table is about 19.52 MiB and replaces the runtime topology work with one lookup.

## v4.6: split LOCAL/CROSS closure streams

v4.6 separates closure operations into `local_closure_ops` and `cross_closure_ops`.

For a local closure, destination storage is determined entirely by the active position:

- `p = 1`: destination is a main state;
- `p > 1`: destination is a blocked state.

The builder asserts this invariant against the dense descriptor. Runtime therefore removes the per-operation MAIN/BLOCK/CROSS dispatch and hoists `p == 1` outside the operation loops.

## v4.7: split NN/NR/NL orbit streams

The remaining LOW hot-loop dispatch in v4.6 was the N* orbit kind. The three classes form disjoint local orbits:

- NN pairs with LR;
- NR pairs with RN;
- NL pairs with LN.

None of `LR`, `RN`, or `LN` is another N* representative at the same active position. v4.7 therefore builds independent `nn_orbit_ops`, `nr_orbit_ops`, and `nl_orbit_ops` streams. Runtime no longer decodes orbit kind or branches on `kind == NN`; the `p == 1` split for NR/NL is also outside the operation loop.

This is an empirical CPU microarchitecture optimization. It can lose if the additional row traversals cost more than the removed branches. Use `scripts/bench/ramstream32-hybrid-sparse-v46-v47.sh` to compare the v4.6 baseline commit `c8de8d73af1c44f075aee937bc8f37e8b7b79d27` against the current implementation under identical conditions.

## v4.8: CPU offload for small HIGH groups

v4.8 attacks the dominant PCIe term rather than only CPU instruction overhead.

A HIGH-window group fixes the occupancy mask of the entire LOW half. Every HIGH transition preserves this mask:

- ordinary HIGH transitions leave the exact LOW code unchanged;
- the only boundary-crossing transition flips one occupied LOW symbol `R -> L`, so occupancy is unchanged.

Therefore fixed-LOW-occupancy HIGH groups are transition-closed. Different groups can be assigned independently to CPU or GPU without changing the recurrence.

`src/cuda/gridfp/ramstream32_cpu_high.hpp` implements an ordinary-System-RAM CPU HIGH executor using the same factorized local block layout as the GPU path. It packs one selected occupancy group into mmap scratch, executes the complete HIGH window with two main and two blocked buffers, and writes the group back. No PCIe transfer occurs for that group.

The existing `HighDescHost` metadata is reused. For the one boundary-crossing case, v4.8 builds the LOW-side symmetric table

`(source LOW mask-code index, matching depth) -> destination mask-local LOW rank`.

This removes runtime LOW bracket scans and rank searches from the CPU HIGH path.

### Group-size distribution at n=27

`scripts/tools/profile_high_group_transfer.py` reproduces the exact forced-two-window group sizes without allocating the state arrays. It verifies that all 16,384 fixed-LOW groups partition the authoritative main and blocked spaces exactly.

For `n=27`:

- HIGH groups: 16,384;
- total HIGH H2D+D2H: 106.087684520 TiB/residue;
- median group roundtrip: about 125.55 MiB;
- 90th percentile: about 429.56 MiB;
- 99th percentile: about 1492.00 MiB;
- maximum: about 9877.92 MiB.

Candidate CPU thresholds are strongly non-linear because group sizes are discrete:

| CPU HIGH threshold | groups on CPU | fraction of groups | PCIe removed | PCIe remaining |
|---:|---:|---:|---:|---:|
| 64 MiB | 3,473 | 21.20% | 2.6766 TiB (2.52%) | 103.4111 TiB |
| 128 MiB | 9,908 | 60.47% | 19.6619 TiB (18.53%) | 86.4258 TiB |
| 256 MiB | 12,911 | 78.80% | 38.2452 TiB (36.05%) | 67.8425 TiB |
| 512 MiB | 14,913 | 91.02% | 61.2090 TiB (57.70%) | 44.8787 TiB |
| 1024 MiB | 15,914 | 97.13% | 82.5714 TiB (77.83%) | 23.5163 TiB |

The important point is that group count and transferred-state count are very different. At 256 MiB, the CPU takes 78.8% of the groups but only 36.05% of the HIGH state traffic; the GPU keeps the much larger groups where launch and transfer overheads are better amortized.

### Running the partition

The feature is disabled by default:

```bash
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32
```

Set a threshold in MiB to move small HIGH groups to CPU:

```bash
CPU_HIGH_MAX_MIB=256 \
CPU_HIGH_WORKERS=32 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32
```

A no-state-allocation plan can be inspected with:

```bash
CPU_HIGH_MAX_MIB=256 \
./build/oneesan_cuda_gridfp_ramstream32_factorized_hybrid_sparse_n27 \
  27 4294967291 12288 32 --plan-only
```

The v4.8 output reports:

- selected CPU/GPU HIGH group counts;
- CPU HIGH pack/kernel/unpack sums and wall time;
- per-worker CPU HIGH scratch peak;
- baseline, removed, and remaining PCIe TiB/residue;
- the ordinary CPU LOW and GPU timings.

The current implementation deliberately executes GPU HIGH groups and CPU HIGH groups sequentially. This isolates the cost/benefit of CPU offload. Once a winning threshold is measured, the two disjoint group sets can be overlapped in a later version; that overlap must be benchmarked because PCIe DMA and CPU HIGH work both consume System-RAM bandwidth.

### Correctness validation

The W=10 exhaustive-state regression now checks both CPU windows independently against the full transition recurrence:

- LOW out-of-place executor;
- LOW in-place executor;
- LOW direct executor;
- LOW sparse v4.7 executor;
- HIGH CPU executor including pre-ranked boundary CROSS transitions.

The n=27 CI plan with `CPU_HIGH_MAX_MIB=256` also requires exactly 12,911 CPU HIGH groups, independently matching the transfer-distribution profiler.

## Current bottlenecks and next experiments

The immediate experiment is no longer only v4.6 versus v4.7. The main sweep should measure `CPU_HIGH_MAX_MIB = 0, 64, 128, 256, 512, 1024` while holding modulus, CPU worker count, GPU scratch budget, and machine fixed.

For every threshold record:

- full `wall_s`;
- GPU `h2d_s`, `gpu_kernel_s`, and `d2h_s`;
- `cpu_high_wall_s` and CPU HIGH pack/kernel/unpack sums;
- `cpu_low_wall_s` and LOW kernel sum;
- measured host memory bandwidth and CPU utilization;
- remaining PCIe TiB predicted by the partition.

A threshold wins only if CPU HIGH time grows more slowly than the PCIe time it removes. At 50 GiB/s, for example, the 256 MiB partition removes about 38.245 TiB, corresponding to roughly 13.05 minutes of ideal serialized PCIe transfer. That is a substantial budget for CPU work, but real transfer and DRAM contention must be measured.

After choosing a threshold, the next large steps are:

1. overlap disjoint GPU-HIGH and CPU-HIGH group sets;
2. use NUMA-aware worker placement and first-touch allocation on multi-socket high-memory hosts;
3. partition the remaining large HIGH groups across aggregate multi-GPU memory and keep them resident for more local work;
4. batch multiple CRT residues through shared topology metadata and scheduling;
5. investigate a CPU HIGH in-place orbit to cut its four-buffer scratch and memory traffic.

The central shift in v4.8 is that the 106.088 TiB/residue HIGH PCIe figure is no longer a fixed lower bound: it is now a tunable CPU/GPU partition boundary.