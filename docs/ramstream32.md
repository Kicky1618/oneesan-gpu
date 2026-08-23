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

### v3: factorized authoritative RAM layout

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32_factorized.cu
scripts/build/gridfp-ramstream32-factorized.sh
```

v3 changes the physical order of the authoritative RAM array while preserving the exact state set and transitions.

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

instead of canonical code rank. The existing factorized device codec is reused with its `all-rank` field redefined to this occupancy-major rank. This preserves the local factorized transition kernels but makes occupancy classes contiguous in system RAM.

Consequences:

- fixed HIGH occupancy: the selected rows are consecutive and the complete LOW dimension is present, so each factor block can be packed with one large `memcpy`;
- fixed LOW occupancy: each HIGH row contributes one consecutive LOW slice, so accesses are no longer random 4-byte gathers, but v3 still needs one copy per HIGH row.

The latter is the remaining weakness addressed by the planned v4 mask-batching scheme below.

## Build and regression

For v3, `N` is the cell width. The build helper chooses a balanced factor split by default.

First regression (`n=21`, `W=22`, default `LOW=11`, `HIGH=10`):

```bash
N=21 ARCH=native bash scripts/build/gridfp-ramstream32-factorized.sh
./build/oneesan_cuda_gridfp_ramstream32_factorized_n21 \
  21 4294967291 2048 11 16
```

Arguments are

```text
n modulus scratch_target_mib max_window cpu_threads
```

The expected residue is

```text
998035516
```

For `n=27` the default split is `LOW=14`, `HIGH=13`:

```bash
N=27 ARCH=native bash scripts/build/gridfp-ramstream32-factorized.sh
./build/oneesan_cuda_gridfp_ramstream32_factorized_n27 \
  27 4294967291 4096 14 32
```

The `scratch_target_mib` limit covers the main/blocked GPU work buffers. Factorized lookup tables consume additional VRAM; for `n=27` the dense LOW/HIGH packed-rank tables are about 1 GiB and 256 MiB respectively.

## Memory hierarchy

```text
ordinary system RAM
  factorized main authoritative array
  factorized blocked authoritative array
          |
          | occupancy-major rectangular copies
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

## Instrumentation

v3 reports

- `pack_s`: authoritative RAM -> pinned staging;
- `h2d_s`: staging -> GPU;
- `kernel_s`: factorized GPU transitions;
- `d2h_s`: GPU -> staging;
- `unpack_s`: staging -> authoritative RAM;
- `memcpy_runs`: total contiguous RAM copy calls in pack + unpack;
- `avg_memcpy_elems`: average 32-bit states per RAM copy;
- `wall_s`: total DP row-loop time.

The comparison of interest is

```text
v1 rank/unrank pack
v2 canonical interval pack
v3 occupancy-major factorized pack
```

under the same modulus, scratch target and window size.

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
- comfortably above 2 TiB RAM: `n=27`

For `n=27`, the exact factorized count check gives

```text
main    = 385719506620
blocked = 135015505407
total   = 520735012027 states
```

which is the same state count as the canonical authoritative representation; v3 changes only the permutation in memory.

## v4 direction: LOW-mask batching

An occupancy-major layout makes fixed-HIGH groups excellent, but fixed-LOW groups still have narrow slices. For `n=27`, `L=14`, positive LOW occupancy/height classes have approximately

```text
median  = 7 states
mean    = 17.26 states
maximum = 1001 states
```

per HIGH row. A sequence of tiny row copies would waste CPU cycles even though the accesses are contiguous.

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

Adjacent occupancy classes are adjacent in v3 storage, so batching does not transfer extra authoritative states. It only moves the group-separation/repacking step from system RAM to HBM, where bandwidth is much higher. Batch size should be chosen from the available VRAM and measured PCIe/DRAM bandwidth.

After mask batching, the next optimization is a double/triple-buffered pipeline overlapping CPU RAM copies, H2D, GPU transitions, D2H and RAM writeback.
