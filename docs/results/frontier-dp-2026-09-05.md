# Frontier DP setup and state decoding optimization (2026-09-05)

Target: `oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu`.
This change applies to the main frontier DP, independently of the strip upper-bound computation.

## Changes and correctness argument

- Build each group's factor blocks during schedule preparation. A group's fixed positions,
  occupancy mask, partition direction and state order do not change between rows or CRT moduli.
  Consequently its offsets, strides and reciprocal divisors can be reused exactly.
- Upload the six factor configuration fields as one structure. Remove eight legacy group
  table/mask/width transfers from the active factorized path. The active kernels obtain their
  configuration from factor tables; the legacy `rank_drop_n_t` helper is not instantiated.
  The upload still completes before gathering; each group's scatter completes before another
  configuration is installed on that device. Multi-GPU constants remain device-local.
- Experimental (disabled by default): reuse gathered main-state encodings during scatter. If `U_g(i)` is the group's unranking
  map, gather saves `C[i] = U_g(i)`. Transitions modify only count vectors, so scatter can
  read `C[i]` in place of recomputing `U_g(i)`. The global destination rank is unchanged.
  This was slightly slower in the isolation benchmark.
- Cache blocked-state encodings (enabled by default when space permits) using the same invariant. This also avoids
  repeated target unranking in one-step and fused two-step reverse transitions. Cache
  admission includes both count buffers, the existing main cache, the blocked cache and
  alignment allowance within the configured scratch budget. Uncached execution remains
  available. No count, modular operation, predecessor or multiplicity is omitted.
- An experimental single-stream schedule serializes the two output kernels and removes
  cross-stream event waits. Both kernels read old vectors and write distinct next vectors;
  the special first-column copies/clears occur first. Thus serialization preserves the
  recurrence, but can reduce GPU overlap. It is disabled by default after measurement.

These are exact representation and execution changes, not a new path-counting recurrence.
They require no unproved pruning or probabilistic equivalence assumption.

## Reproduction

`python scripts/bench/bench_factor_division.py --optimization frontier --n 20 --arch sm_86 --repeats 5`

The comparison keeps host-side factor precomputation in both builds; the original frozen
baseline additionally recomputed those blocks for every group execution. Use `--optimization config`, `--optimization scatter`, or `--optimization block-cache` to isolate the components. Warmup is excluded and run
order alternates. Every measured n=20 result is checked against residue 2308006916 modulo
4294967291.

The optional `--optimization factor-memory` compares constant versus global factor configuration
storage. Global storage was slower and `GRIDFP_GLOBAL_FACTOR_CONFIG=0` remains the default.

Runtime controls: `GRIDFP_CACHE_BLOCK_MATES=0` disables the blocked cache;
`GRIDFP_SINGLE_STREAM_MAX_STATES=0` retains dual streams (default), while a positive value
serializes groups whose main plus blocked state count is at most that value.

## Scope

Performance measurements use a desktop RTX 3070 (sm_86), one GPU, n=20 and 512 MiB scratch.
They do not establish B300 x8 speedups, bandwidth saturation, or n=27 runtime. The desktop
remains active, so timings have drift; compare balanced repeated runs rather than one run.

## Isolation trials

Each row below is a median of three measured executions after warmup. Compare variants
within a trial only: desktop load changed between trials. All residues matched.

| Trial | Variant | Wall seconds | Transition seconds | Default decision |
|---|---|---:|---:|---|
| Setup / streams | Original | 10.28060 | 5.14959 | Reference |
| Setup / streams | Batched config, dual streams | 8.61693 | 5.19584 | Adopt |
| Setup / streams | Single stream for <= 1,048,576 states | 9.53726 | 6.09149 | Disable |
| Setup / streams | Single stream for all groups | 9.60175 | 6.15317 | Disable |
| State cache | Batched config only | 8.59426 | 5.19833 | Reference |
| State cache | Plus main scatter cache | 8.65306 | 5.19761 | Disable main scatter cache |
| State cache | Plus blocked target cache | 8.55717 | 5.08127 | Retain blocked cache; small gain |
| Factor storage | Constant | 9.06966 | 5.21978 | Keep constant |
| Factor storage | Global | 9.38331 | 5.43897 | Disable global storage |

The blocked-cache isolation trial included the main-scatter experiment in both cache
variants; the final default disables main-scatter reuse. Global-memory trials also included
both caches. The final comparison below measures the actual selected defaults.

## Final comparison

Six measured runs per variant, after one warmup each, alternating order. The original
reference is the frozen pre-change solver (reciprocal division, rank reuse and shard search
already enabled). The final variant uses selected defaults, including blocked caching and
dual streams; main-scatter reuse and global factor storage are disabled.

| Phase (host wall timer) | Before, seconds | Final, seconds |
|---|---:|---:|
| Full solve | 10.231150 | 8.506240 |
| Group setup + gather | 3.233980 | 1.549180 |
| Transitions + event orchestration | 5.138495 | 5.087255 |
| Scatter | 1.597410 | 1.610930 |

Full-solve median speedup: **1.203x**, **16.86%** less elapsed time.
Median of the six paired speedup ratios: 1.203x.
Most improvement is in group setup, not device transition throughput. These phase timers
include host orchestration and are not isolated CUDA-event kernel timings.

Raw measurements, warmups and source fingerprints are in
[frontier-dp-2026-09-05.json](frontier-dp-2026-09-05.json).

## Validation

- Independent CPU predecessor oracle: 145,752 targets / 193,312 edges, widths 3–12;
  another 260,000 width-28 samples and boundary cases passed under UBSan.
- Shard addressing: 1,297,727 CPU cases under UBSan and 1,000,374 GPU cases across
  1–8 emulated shards passed. This is not a physical multi-GPU execution test.
- Actual main reverse kernel: 12 cached/uncached rank-policy comparisons passed with
  the new factor configuration upload helper.
- Final n=9 binary: two exact residues matched across all four combinations of blocked
  caching on/off and dual/single streams. Reproduce with
  `python scripts/test/frontier-runtime.py /path/to/n9-binary`.
- Compute Sanitizer memcheck on the final n=9 binary, two CRT moduli: zero errors.
- Final n=20, 40 MiB scratch (cache admission can fall back): residue 2308006916
  modulo 4294967291, matching the 512 MiB runs.
- Final n=21: residue 2124618149 modulo 4294966997, matching the known reference.
- Updated comparison script: legacy and selected-default n=9 builds and runs passed.
- Production width 28, LUT split 14/13: sm_103 cubin compilation succeeded.

No B300 execution or physical multi-GPU regression was possible on this one-GPU machine.
The unrelated row8 Python-suite failures recorded in the earlier optimization report are
outside this change; this iteration ran the targeted frontier checks above.
