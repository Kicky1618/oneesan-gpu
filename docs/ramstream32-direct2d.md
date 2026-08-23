# `ramstream32` direct-2D transfer path

This note covers the `v3.1` transport variant:

```text
src/cuda/gridfp/oneesan_cuda_gridfp_ramstream32_factorized_direct2d.cu
scripts/build/gridfp-ramstream32-factorized-direct2d.sh
```

It uses the same occupancy-major factorized authoritative layout as v3, but removes the multi-GiB pinned group staging buffers.

## Transfer mapping

For the strict two-window split `W = H + 1 + L`:

- LOW window (`p=L..1`): HIGH occupancy is fixed. The selected HIGH rows are contiguous in authoritative RAM and every LOW column is present. Each factor block is copied with one ordinary `cudaMemcpy`.
- HIGH window (`p=W-1..L+1`): LOW occupancy is fixed. Each HIGH row contributes one contiguous LOW slice. The whole strided rectangle is expressed by one `cudaMemcpy2D`, with source pitch equal to the complete LOW dimension and destination pitch equal to the selected LOW-mask width.

Thus the driver receives one 1-D or 2-D transfer per non-empty factor block instead of one CPU `memcpy` per HIGH row. Host memory remains pageable; CUDA may internally stage pages, but the application no longer pins the full authoritative array or a largest-group-sized staging area.

## Build

Regression width:

```bash
N=21 ARCH=native bash scripts/build/gridfp-ramstream32-factorized-direct2d.sh
./build/oneesan_cuda_gridfp_ramstream32_factorized_direct2d_n21 \
  21 4294967291 256
```

Expected residue:

```text
998035516
```

Large target:

```bash
N=27 ARCH=native bash scripts/build/gridfp-ramstream32-factorized-direct2d.sh
./build/oneesan_cuda_gridfp_ramstream32_factorized_direct2d_n27 \
  27 4294967291 16384
```

The source is compiled in CI for both `(W,L,H)=(22,11,10)` and `(28,14,13)` using CUDA 12.8.

## Fundamental PCIe traffic

Direct 2-D DMA removes copy-call fragmentation, but it does not remove the external-memory traffic itself.

For each row, the strict profile performs two windows. Each window partitions the full authoritative state space into groups and transfers each group H2D and D2H once. Ignoring metadata and all implementation overhead, the lower bound is therefore

```text
traffic_per_residue = 4 * W * authoritative_bytes
```

Approximate values:

| n | authoritative | minimum PCIe traffic / residue | ideal time at 50 GiB/s |
|---:|---:|---:|---:|
| 23 | 29.48 GiB | 2.76 TiB | 0.94 min |
| 24 | 83.73 GiB | 8.18 TiB | 2.79 min |
| 25 | 238.29 GiB | 24.20 TiB | 8.26 min |
| 26 | 679.32 GiB | 71.65 TiB | 24.46 min |
| 27 | 1939.89 GiB | 212.18 TiB | 72.42 min |

These are transport-only lower bounds for one CRT residue. They assume 50 GiB/s sustained useful payload bandwidth and no kernel, paging, synchronization or copy-engine overhead.

This changes the optimization priorities by size:

- `n<=25`: direct2d can plausibly make a low-VRAM GPU useful without exotic machinery.
- `n=26`: PCIe is already a major component; overlap and CPU/GPU group scheduling matter.
- `n=27`: even perfect copy-call coalescing leaves a roughly hour-scale PCIe floor per residue, so a production profile needs to reduce the number of states crossing PCIe, not merely make each transfer faster.

## Next directions

### Adjacent-mask batching

A single occupancy mask remains one logical factorized group, but adjacent occupancy masks are adjacent in authoritative RAM. Several masks can be transferred as one wider rectangle and separated/repacked only in HBM. This reduces CUDA copy-call count while preserving the same payload volume.

### CPU/GPU hybrid window execution

The stronger optimization is to leave selected groups or a whole window in system RAM and execute its transition on CPU. A scheduler should compare

```text
GPU cost = H2D + GPU transition + D2H
CPU cost = RAM-local transition
```

per group class. On high-bandwidth multi-socket EPYC/Xeon hosts, avoiding PCIe can outweigh the GPU arithmetic advantage for memory-heavy groups.

### Bounded slab staging and overlap

For platforms where pageable `cudaMemcpy2D` performs poorly, use a bounded pinned slab (for example 128--512 MiB) rather than largest-group-sized pinning, then overlap RAM pack, H2D, transition, D2H and RAM writeback with multiple streams.

The direct2d implementation is therefore both a usable candidate and a measurement baseline: its `h2d_s`, `kernel_s`, `d2h_s`, `copy1d` and `copy2d` counters expose whether the next machine is limited by PCIe payload, transfer-call overhead, or GPU transition time.
