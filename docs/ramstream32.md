# RAM-streaming profile (`ramstream32`)

`ramstream32` is the low-VRAM / high-system-RAM execution profile for the GPU frontier DP.

The authoritative `main` and `blocked` arrays live in ordinary anonymous system RAM. Only the current occupancy group is packed into a small pinned host staging buffer and copied explicitly to GPU scratch. This avoids registering the full authoritative array with CUDA and avoids GPU random loads over mapped host memory.

## Implementations

### v1: canonical gather/scatter baseline

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32.cu
scripts/build/gridfp-ramstream32.sh
```

v1 performs CPU-side `group unrank -> canonical rank` for every packed/unpacked state. It exists as a correctness baseline and exposes separate `pack_s`, `h2d_s`, `kernel_s`, `d2h_s`, and `unpack_s` timings.

### v2: canonical interval packing

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32_intervals.cu
scripts/build/gridfp-ramstream32-intervals.sh
```

v2 reuses the canonical interval construction developed in the factorized B300 implementation. Before materializing intervals it computes an upper bound on interval count. If the predicted/actual average run is sufficiently long, pack/unpack becomes parallel `memcpy` over canonical runs. Fragmented groups fall back to the v1 rank/unrank path.

The default threshold is 16,384 states = 64 KiB per interval for 32-bit counts. Override it with `MIN_INTERVAL_ELEMS` at build time.

## Build and regression

`TARGET_W` is the vertex/frontier width, so `TARGET_W=22` builds `n=21`. The largest current target is `TARGET_W=28` (`n=27`).

Build v2 for the first regression:

```bash
TARGET_W=22 ARCH=native bash scripts/build/gridfp-ramstream32-intervals.sh
```

Run:

```bash
./build/oneesan_cuda_gridfp_ramstream32_intervals_w22 21 4294967291 2048 14 16
```

Arguments are:

```text
n modulus scratch_target_mib max_window cpu_threads
```

The known factorized result for `n=21`, modulus `4294967291`, is residue `998035516`.

Useful threshold sweep:

```bash
for x in 4096 16384 65536 262144; do
  MIN_INTERVAL_ELEMS=$x TARGET_W=22 \
    bash scripts/build/gridfp-ramstream32-intervals.sh
  ./build/oneesan_cuda_gridfp_ramstream32_intervals_w22 \
    21 4294967291 2048 14 16
done
```

## Memory hierarchy

```text
ordinary system RAM
  main authoritative array
  blocked authoritative array
          |
          | interval memcpy or fallback rank/unrank
          v
small pinned staging buffers
          |
          | explicit H2D / D2H
          v
GPU scratch
  main A
  main B
  blocked
```

The anonymous authoritative allocation uses `MADV_HUGEPAGE` where available and is not CUDA-pinned. Linux zero-page/lazy allocation makes the initial virtual mapping cheap; physical RAM is still required as pages are touched. Avoid relying on `MAP_NORESERVE` as a substitute for capacity planning: insufficient RAM can still lead to an OOM kill during the run.

## Instrumentation

Both versions report:

- `pack_s`: authoritative RAM -> pinned staging
- `h2d_s`: pinned staging -> GPU scratch
- `kernel_s`: GPU transitions while the group is resident
- `d2h_s`: GPU scratch -> pinned staging
- `unpack_s`: pinned staging -> authoritative RAM
- `wall_s`: end-to-end row loop

v2 additionally reports:

- `interval_arrays`: main/blocked group arrays handled with interval copies
- `fallback_arrays`: arrays handled with rank/unrank
- `interval_runs`: total contiguous canonical runs copied
- `avg_interval_elems`: average states per selected interval

These counters tell us whether canonical storage is already sufficiently contiguous or whether changing the authoritative layout is necessary.

## Capacity of the 32-bit authoritative arrays

Approximate `main + blocked` capacity per CRT residue:

| n | authoritative size |
|---:|---:|
| 23 | 29.48 GiB |
| 24 | 83.73 GiB |
| 25 | 238.29 GiB |
| 26 | 679.32 GiB |
| 27 | 1939.89 GiB |

Practical first targets:

- 128 GiB RAM + 24--32 GiB VRAM: `n=24`
- 384--512 GiB RAM + 24--32 GiB VRAM: `n=25`
- 1 TiB RAM + 24--32 GiB VRAM: `n=26`
- comfortably above 2 TiB RAM: `n=27`

## v3 criterion: factorized authoritative layout

Do not change the global layout merely because VRAM is small. First measure v2.

If `fallback_arrays` remains high or `pack_s + unpack_s` dominates even when PCIe transfer time is small, change system-RAM storage to the exact factorized decomposition

```text
biguplus_(h,c) HIGH_h x LOW_(h + delta(c))
```

and tile each block row-major. The factorized topology codec already supplies an exact bijection, so this changes storage order rather than DP semantics. In that layout, occupancy groups should map to large tiles/runs instead of canonical random gather.

After v3, the next step is a double/triple-buffered pipeline that overlaps CPU packing, H2D, GPU transitions, D2H, and CPU unpacking.
