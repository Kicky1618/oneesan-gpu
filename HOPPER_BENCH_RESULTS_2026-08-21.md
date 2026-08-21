# Hopper benchmark results — 2026-08-21

Current production kernel under test:

`oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu`

Key optimizations enabled in this source:

- authoritative HBM state sharded across GPUs
- transition-closed groups with precomputed schedule
- hybrid canonical interval I/O vs rank/unrank I/O
- LOW/HIGH LUT paths
- double-buffered blocked state with two CUDA streams
- main->main rank-delta
- opportunistic full main MateID cache
- NR/NL drop-N rank delta

All compared runs produced the same regression residues.

## Single-GPU comparison

### n=18, target scratch 16 MiB

Build configuration: `TARGET_W=19`, `LOW_LUT_K=7`, `HIGH_LUT_K=7`, modulus `2147483647`.

| GPU | wall time |
|---|---:|
| RTX 3070 8GB | 2.796 s warm mean |
| H100 SXM 80GB | 1.802 s mean |
| H200 141GB | 1.842 s mean |

Speedup vs RTX 3070:

- H100: 1.55x
- H200: 1.52x

This small case contains substantial group/kernel-launch overhead, so H200's extra HBM bandwidth does not help yet.

### n=20, target scratch 256 MiB

Build configuration: `TARGET_W=21`, `LOW_LUT_K=6`, `HIGH_LUT_K=6`, modulus `2147483647`.

| GPU | wall time |
|---|---:|
| RTX 3070 8GB | 23.435 s |
| H100 SXM 80GB | 9.544 s mean |
| H200 141GB | 9.306 s mean |

Speedup vs RTX 3070:

- H100: 2.46x
- H200: 2.52x

H200 is about 2.56% faster than H100 on this larger workload.

## H200 x2 NVLink scaling

Vast.ai 2x H200 instance topology:

- `nvidia-smi topo -m`: `NV18`
- `cudaDeviceCanAccessPeer`: both directions enabled
- `cudaMemcpyPeerAsync`, 256 MiB x8:
  - GPU0 -> GPU1: 349.56 GiB/s
  - GPU1 -> GPU0: 360.32 GiB/s

### n=20

| GPUs | wall time | scaling |
|---|---:|---:|
| 1x H200 | 9.302 s mean on the same 2-GPU host |
| 2x H200 | 5.138 s mean | 1.81x |

### n=18, four-prime batch

Mean per-residue wall time:

| GPUs | wall / residue | scaling |
|---|---:|---:|
| 1x H200 | 1.765 s |
| 2x H200 | 0.912 s | 1.93x |

The smaller n=18 workload scales closer to ideal because it has 128 groups/window, enough work to balance the two workers while authoritative-state traffic remains small.

## H100 x2 topology probe

A 2x H100 SXM instance was obtained long enough to probe the fabric before that provider's container exited unexpectedly.

- topology: `NV6`
- peer access: enabled both directions
- `cudaMemcpyPeerAsync`, 256 MiB x8:
  - GPU0 -> GPU1: 122.45 GiB/s
  - GPU1 -> GPU0: 123.01 GiB/s

This is only about 0.35x the measured H200 x2 peer-copy bandwidth on the tested machines. The H100 provider instance exited immediately after the P2P test, so no reliable 1-vs-2-GPU solver timing was recorded on that host.

A second H100 x2 rental attempt was blocked by insufficient Vast.ai account credit, so no further paid instances were started.

## Exact CRT batch path

The production exact runner uses 48 near-2^32 primes, giving a 1536-bit CRT modulus. For n=27 the safe bound is 1513 bits.

Batch mode keeps the following resident across residues:

- authoritative HBM allocation
- scratch allocations
- LOW/HIGH LUTs
- precomputed group schedule

Each residue is printed immediately so the Python runner can checkpoint incrementally.

## Production build

The n=27 binaries were rebuilt after the latest source promotion:

- `oneesan_cuda_gridfp_b300_hbm32_n27_sm103`
- `oneesan_cuda_gridfp_b300_hbm32_batch_n27_sm103`

Both use:

- `TARGET_W=28`
- `LOW_LUT_K=13`
- `HIGH_LUT_K=13`
- `sm_103`

## Cloud cleanup

All `oneesan-*` Vast.ai benchmark instances created in this session were destroyed after use. No benchmark instance was intentionally left running.
