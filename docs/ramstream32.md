# RAM-streaming profile (`ramstream32`)

`ramstream32` is the low-VRAM / high-system-RAM execution profile for the GPU frontier DP.

The authoritative `main` and `blocked` arrays live in ordinary anonymous system RAM. Only the current occupancy group is packed into a small pinned host staging buffer and copied explicitly to GPU scratch. This deliberately avoids registering the full authoritative array with CUDA and avoids GPU random loads over mapped host memory.

## Current implementation

Source:

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32.cu
```

Build helper:

```bash
TARGET_W=22 ARCH=native bash scripts/build/gridfp-ramstream32.sh
```

`TARGET_W` is the vertex/frontier width, so `TARGET_W=22` builds the `n=21` regression binary. For the largest target, use `TARGET_W=28` (`n=27`).

Run syntax:

```text
oneesan_cuda_gridfp_ramstream32_w<W> n modulus scratch_target_mib max_window cpu_threads
```

Example regression run:

```bash
TARGET_W=22 bash scripts/build/gridfp-ramstream32.sh
./build/oneesan_cuda_gridfp_ramstream32_w22 21 4294967291 2048 14 16
```

The known factorized implementation gives residue `998035516` for `n=21`, modulus `4294967291`; use this as the first correctness check.

## Memory hierarchy

```text
ordinary system RAM
  main authoritative array
  blocked authoritative array
          |
          | CPU pack / unpack
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

The anonymous authoritative allocation uses `MADV_HUGEPAGE` where available and is not CUDA-pinned. Linux zero-page/lazy allocation means creating a very large virtual mapping itself is cheap; the machine must still have enough physical memory (or configured swap) once the DP touches the pages.

The current version is intentionally a correctness/performance baseline. It performs CPU-side group `unrank -> canonical rank` during pack and unpack, then runs the same group-local GPU transition structure as the existing mapped-host prototype.

## Instrumentation

Each run reports:

- `pack_s`: ordinary RAM -> pinned group staging
- `h2d_s`: pinned staging -> GPU scratch
- `kernel_s`: GPU transitions inside the resident window
- `d2h_s`: GPU scratch -> pinned staging
- `unpack_s`: pinned staging -> authoritative RAM
- `wall_s`: end-to-end row loop

This is the main purpose of v1: identify whether the next bottleneck is CPU topology work, PCIe, or GPU transition work on the target host.

## Capacity of the 32-bit authoritative arrays

Approximate `main + blocked` capacity per CRT residue:

| n | authoritative size |
|---:|---:|
| 23 | 29.48 GiB |
| 24 | 83.73 GiB |
| 25 | 238.29 GiB |
| 26 | 679.32 GiB |
| 27 | 1939.89 GiB |

For `n=27`, system RAM should be comfortably above 2 TiB if the machine also needs normal OS/page-cache/headroom.

## Next optimization: interval packing

The existing factorized B300 source already contains canonical-layout interval construction (`Interval`, `intervals_rec`, `make_intervals`, and tiled peer intervals). That machinery proves useful here as well: when a group maps to long canonical runs, pack/unpack can become large `memcpy` operations instead of one host `unrank/rank` per state.

The intended v2 policy is:

```text
if average canonical interval is large:
    memcpy interval pack/unpack
else:
    factorized-code pack/unpack
```

This should be implemented before attempting full 2D factorized authoritative storage, because it preserves the existing canonical authoritative format and gives a direct measurement of how much of the PCIe/RAM problem can be solved by layout-aware streaming alone.

## Planned v3

If interval packing remains too fragmented, change the authoritative system-RAM layout itself to the exact factorized decomposition

```text
biguplus_(h,c) HIGH_h x LOW_(h + delta(c))
```

and store each block as a tiled row-major matrix. That removes canonical random gather from the steady-state path and turns group transfer into contiguous/tiled RAM traffic. The factorized topology codec already supplies the bijection needed to preserve exact DP semantics.
