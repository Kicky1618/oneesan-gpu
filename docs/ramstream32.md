# RAM-streaming profile (`ramstream32`)

`ramstream32` is the low-VRAM / high-system-RAM execution profile for the GPU frontier DP.

The authoritative `main` and `blocked` arrays live in ordinary anonymous system RAM. Only the current occupancy group is moved through pinned staging into GPU scratch. The profile deliberately avoids registering the full authoritative array with CUDA and avoids GPU random loads over mapped host memory.

## Implementations

### v1: canonical gather/scatter baseline

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32.cu
scripts/build/gridfp-ramstream32.sh
```

v1 performs CPU-side `group unrank -> canonical rank` for every packed/unpacked state. It is primarily a correctness/timing baseline.

### v2: canonical interval packing

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32_intervals.cu
scripts/build/gridfp-ramstream32-intervals.sh
```

v2 reuses the canonical interval construction from the factorized B300 implementation. Long canonical runs become parallel `memcpy`; fragmented groups fall back to v1 rank/unrank.

### v3: occupancy-major factorized authoritative RAM

Production entry point:

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32_factorized_forced2.cu
src/cuda/gridfp/ramstream32_factorized_storage.hpp
scripts/build/gridfp-ramstream32-factorized.sh
```

`oneesan_cuda_gridfp_ramstream32_factorized.cu` is kept as the generic-layout research prototype. The build helper intentionally targets the strict forced-two-window implementation, because the factorized codec requires one whole side of the split to have fixed occupancy.

For `W = H + 1 + L`, main states are stored as

```text
biguplus_(he,c) HIGH_he x LOW_(he + delta(c))
```

and blocked states as

```text
biguplus_h HIGH_h x LOW_h.
```

Within each fixed-height HIGH/LOW dimension, codes are ordered by

```text
(occupancy mask, mask-local rank)
```

instead of canonical code rank. The existing factorized GPU codec is reused with its `all-rank` field redefined to this occupancy-major rank. This changes only the permutation in authoritative memory, not the state set or DP transitions.

The two windows are exactly the same split used by the B300 forced2 profile:

```text
HIGH window: p = W-1 .. L+1   (all LOW occupancy fixed)
LOW  window: p = L   .. 1     (all HIGH occupancy fixed)
```

Consequences:

- fixed HIGH occupancy: the selected HIGH rows are consecutive and the complete LOW dimension is present, so each factor block is one large RAM copy;
- fixed LOW occupancy: each HIGH row contributes one consecutive LOW slice. This removes random 4-byte gathers, but still produces many short row copies. v4 addresses this with LOW-mask batching.

## Build and regression

For v3, `N` is the cell width. The build helper chooses a balanced factor split by default.

First regression (`n=21`, `W=22`, default `LOW=11`, `HIGH=10`):

```bash
N=21 ARCH=native bash scripts/build/gridfp-ramstream32-factorized.sh
./build/oneesan_cuda_gridfp_ramstream32_factorized_n21 \
  21 4294967291 256 16
```

Runtime arguments are

```text
n modulus scratch_target_mib cpu_threads
```

The expected residue is

```text
998035516
```

For `n=27`, the default split is `LOW=14`, `HIGH=13`:

```bash
N=27 ARCH=native bash scripts/build/gridfp-ramstream32-factorized.sh
./build/oneesan_cuda_gridfp_ramstream32_factorized_n27 \
  27 4294967291 16384 32
```

The strict two-window maxima for `n=27` are approximately

```text
LOW occupancy fixed  (HIGH window):  9.65 GiB scratch
HIGH occupancy fixed (LOW window):  14.77 GiB scratch
```

so a 16 GiB scratch target is the practical minimum for the current two-window v3. Factorized lookup tables consume additional VRAM; for `n=27` the dense LOW/HIGH packed-rank tables are about 1 GiB and 256 MiB respectively. In practice this profile therefore targets roughly 20--24 GiB or more of usable VRAM, with 24--32 GiB cards being the comfortable range.

The packed-rank representation also remains valid at `n=27`: the largest LOW all-rank class is 232,323 (< 2^18, because 14 bits are reserved for mask-local rank), and the largest HIGH all-rank class is 149,019 (< 2^19).

## Memory hierarchy

```text
ordinary system RAM
  occupancy-major factorized main authoritative array
  occupancy-major factorized blocked authoritative array
          |
          | rectangular RAM copies
          v
pinned group staging
          |
          | explicit H2D / D2H
          v
GPU factorized group scratch
  main A/B
  blocked A/B
```

The authoritative mappings use `MADV_HUGEPAGE` where available and are not CUDA-pinned. `MAP_NORESERVE` only makes virtual reservation cheap; physical memory still has to exist when pages are touched.

Current v3 still allocates a pinned staging area large enough for the largest resident group. This is much smaller than pinning the full authoritative array, but can still be several GiB at `n=27`. A slab-staging/pipelined variant is therefore another planned improvement alongside mask batching.

## Instrumentation

v3 reports

- `pack_s`: authoritative RAM -> pinned staging;
- `h2d_s`: staging -> GPU;
- `kernel_s`: factorized GPU transitions;
- `d2h_s`: GPU -> staging;
- `unpack_s`: staging -> authoritative RAM;
- `memcpy_runs`: total contiguous RAM copy calls in pack + unpack;
- `avg_memcpy_elems`: average 32-bit states per RAM copy;
- `high_window_max_gib`, `low_window_max_gib`: strict two-window scratch maxima;
- `wall_s`: total DP row-loop time.

The comparison of interest is

```text
v1 rank/unrank pack
v2 canonical interval pack
v3 occupancy-major factorized pack
```

under the same modulus and feasible scratch budget.

## Capacity of the 32-bit authoritative arrays

Approximate `main + blocked` capacity per CRT residue:

| n | authoritative size |
|---:|---:|
| 23 | 29.48 GiB |
| 24 | 83.73 GiB |
| 25 | 238.29 GiB |
| 26 | 679.32 GiB |
| 27 | 1939.89 GiB |

Practical targets:

- 128 GiB RAM + 24--32 GiB VRAM: `n=24`
- 384--512 GiB RAM + 24--32 GiB VRAM: `n=25`
- 1 TiB RAM + 24--32 GiB VRAM: `n=26`
- comfortably above 2 TiB RAM + 24--32 GiB VRAM: `n=27`

For `n=27`, the exact factorized count check gives

```text
main    = 385719506620
blocked = 135015505407
total   = 520735012027 states
```

which is identical to the canonical authoritative state count.

## v4 direction: LOW-mask batching

An occupancy-major layout makes fixed-HIGH groups excellent, but fixed-LOW groups still have narrow slices. For `n=27`, `L=14`, positive LOW occupancy/height classes have approximately

```text
median  = 7 states
mean    = 17.26 states
maximum = 1001 states
```

per HIGH row. A sequence of tiny row copies wastes CPU cycles even though each individual access is contiguous.

The proposed v4 solution is to process adjacent LOW masks in batches:

```text
system RAM row
  [mask m][mask m+1][mask m+2] ... [mask m+k]
             | one wider RAM copy per HIGH row
             v
        batch staging / VRAM
             |
             | HBM-only extract/repack kernels
             v
       individual factorized groups
```

Adjacent occupancy classes are adjacent in v3 storage, so batching need not transfer unrelated authoritative states. It moves group separation/repacking from system RAM to HBM, where bandwidth is much higher. Batch size should be selected from available VRAM and measured RAM/PCIe bandwidth.

After mask batching, the other major optimization is bounded slab staging plus double/triple buffering to overlap RAM copies, H2D, GPU transitions, D2H and RAM writeback without pinning multi-GiB staging buffers.
