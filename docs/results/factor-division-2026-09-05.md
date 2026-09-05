# Removing per-state division from the row6 CRT20 factor codec

Follow-up: GPU access became available during the subsequent
[algorithm optimization](reverse-algorithm-2026-09-05.md). GPU correctness
checks now cover this division change as part of the solver; its isolated
runtime improvement and bandwidth saturation are still unverified.

The default n=27 batch source computes `r / x.stride` in both
`factor_unrank_main` and `factor_unrank_block`. `r` is 64-bit; the divisor
is fixed for a factor block but not known at compile time. This operation
occurs before the dependent LUT loads in gather/scatter and uncached reverse
transitions. It is a concrete instruction cost, not yet a measured dominant
bottleneck. Earlier H100/H200 and LUT-depth measurements in `docs/b300-hbm32.md`
already indicate that rank/unrank work matters before bandwidth saturation.

Each host-built factor block now stores `floor((2^64-1)/stride)`. Device unranking
uses multiply-high, subtraction and one correction to obtain the exact quotient
and remainder. The proof and shared implementation are in
`src/common/invariant_division.hpp`. No floating-point arithmetic is used.
Empty blocks retain their previous zero result. The extra constant storage is
768 bytes per GPU (96 block slots, eight additional bytes each); there is no
per-state allocation. Block metadata uploads also grow accordingly.

This change applies only to
`oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu`.
Other experimental variants and the default n<27 fullmate source are untouched.

## Static code generation

CUDA 13.3.73, `-O3 -std=c++17 -lineinfo -arch=sm_103`, `TARGET_W=28`,
`LOW_LUT_K=14`, `HIGH_LUT_K=13`. Counts are disassembled static SASS instructions
(including padding), **not executed instruction counts or measured speedups**.
The baseline is the source before this edit, including its original block layout.

| Kernel | Instructions before → after | Registers before → after |
|---|---:|---:|
| gather_main | 864 → 784 | 32 → 30 |
| gather_block | 760 → 680 | 32 → 28 |
| scatter_main | 848 → 760 | 34 → 30 |
| scatter_block | 768 → 680 | 32 → 30 |
| reverse2_main_group | 952 → 872 | 30 → 30 |
| reverse2_block_group | 1088 → 1008 | 32 → 32 |

All six have zero stack and zero register spills before and after.

## Validation and reproduction

Host arithmetic tests passed 3,412,417 cases under UBSan, including UINT64_MAX,
divisor 1, powers of two, quotient boundaries, random full-width inputs, and
51 distinct strides generated from all LOW14 occupancy masks and free suffixes.
The actual solver's host block constructors passed 9,926 factor positions at
width 10, exhaustively covering both partition directions and main/blocked
blocks. Run both without requiring a GPU:

```bash
ARCH=sm_103 bash scripts/test/factor-division.sh
```

GPU driver/device access was unavailable on the editing host. End-to-end GPU
correctness, runtime improvement, and bandwidth saturation remain unverified.
The comparison harness compiles the same source with
`FACTOR_RECIPROCAL_DIV=0/1`, warms both, alternates order, checks residues, and
reports median wall/gather/transition/scatter times. Both comparison builds
retain the enlarged metadata layout to isolate the arithmetic change.

```bash
python3 scripts/bench/bench_factor_division.py --n 9 --repeats 2
python3 scripts/bench/bench_factor_division.py --n 20 --scratch-mib 512 --repeats 5
# B300 x8, only on a machine with enough VRAM for the existing production plan:
python3 scripts/bench/bench_factor_division.py --n 27 --gpus 8 --scratch-mib 16384
# Build both branches without a GPU:
python3 scripts/bench/bench_factor_division.py --n 9 --arch sm_103 --compile-only
```

Logs, binaries, flags, relevant environment settings and JSON timing results
are retained in the printed temporary directory. For n=9, 20 and 23 the harness
also checks an existing expected residue; other sizes check agreement with the
baseline. Gather/transition/scatter counters are existing solver host timings,
not DRAM bandwidth measurements. A GPU run and kernel-level memory/issue/stall
profiling are still needed to identify the remaining limiting resource.
