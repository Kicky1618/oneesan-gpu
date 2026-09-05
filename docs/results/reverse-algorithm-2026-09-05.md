# Reverse-transition algorithm optimization

The subsequent [symmetry-quotient optimization](../research/strip-orbit-quotient.md)
adds a proved mathematical state reduction to the CRT upper-bound transfer.

The default n=27 row6 CRT20 batch solver now reuses the target's known factor
rank in the fused two-step main transition. Previously, each main output called
`factor_rank_main(t)` to recover the index `i` from which `t` was obtained.
That repeated a factor-block lookup, height calculation and two packed-rank LUT
loads. The first main evaluation now reads `in[i]` directly. Other predecessors
still compute their own ranks. This removes one redundant rank conversion per
main output of every fused two-step update, without another cache or changing
the recurrence, state count or modular arithmetic.

`reverse2_main_group_kernel<false/true>` permits testing both policies against
the same input within one GPU process. The normal launch uses the default
`GRIDFP_REUSE_MAIN_RANK=1`; setting it to zero restores recomputation.

## Closure-scan experiment: not enabled by default

Inverse blocked transitions were factored into `src/common/gridfp_reverse.hpp`
so the production kernels and the independent host verification exercise the
same enumeration. An optional sparse traversal skips empty symbols using a
bit mask. Empty symbols cannot change closure depth or emit a predecessor,
so skipping them preserves enumeration order and multiplicity.

However, fewer visited positions did not mean faster GPU execution. On RTX
3070, a CUDA-event microbenchmark over 1,048,576 legal width-28 blocked targets
gave these medians (10 measured batches, 10 launches per batch, alternating
order, one warm-up batch):

| Position | Scalar scan (ms) | Sparse scan (ms) |
|---|---:|---:|
| 2 | 0.160862 | 0.220979 |
| 14 | 0.151654 | 0.188109 |
| 27 | 0.155546 | 0.190822 |

This benchmark isolates enumeration with a digest callback; it is not a solver
throughput measurement. All output digests agreed. The extra bit manipulation
was counterproductive here. **`GRIDFP_SPARSE_REVERSE` therefore defaults to 0**;
the sparse implementation remains explicitly selectable for experiments.

## Validation

- Host UBSan test: all 145,752 blocked targets at widths 3..12, with 193,312
  predecessor edges, agree with an independently inverted **forward** transition.
  Scalar and sparse enumeration also preserve order exactly.
- Width 28: long-empty-run boundary cases plus 260,000 sampled target/position
  pairs agree, and every emitted predecessor maps forward to its target.
- GPU n=9: two CRT primes reproduce the known exact count
  `41044208702632496804`; Compute Sanitizer memcheck reports zero errors for the
  candidate with both rank reuse and sparse traversal enabled.
- GPU n=18: the final scalar-scan/rank-reuse variant returns `503411004` modulo
  `4294967291`, matching `work/b300_exact_n18/exact.txt` and the baseline.
- A separate GPU test compares the actual main kernel with/without rank reuse
  on deterministic nonzero modular coefficients, both partition directions,
  three occupancy masks, and cached/uncached MateIDs. All 12 cases passed,
  covering 2,672,048 output comparisons.

The final default also compiled for `sm_103`, width 28 / LOW14 / HIGH13 using
CUDA 13.3.73. The main two-step kernel uses 28 registers (previously 30), with
zero stack and zero spills. Its static SASS count is 864 (previously 872 after
the reciprocal-division change); static counts are not executed instruction
counts or speedup estimates.

The full n=18 timing comparison was interrupted by another GPU workload. Before
that interference, three measured baseline/reuse pairs were
`2.36155/2.39470`, `2.32290/2.26426`, and `2.26611/2.22947` seconds. A later reuse
run took **84.3305 seconds**, and the next run failed the solver's free-VRAM
check. The incomplete raw report is retained beside this document. These
measurements do **not** establish a reliable end-to-end speedup. B300 x8 timing
and memory-bandwidth saturation are also unverified.

## Follow-up with the background compute workload stopped

At n=20 / 512 MiB scratch on RTX 3070, six measured runs per variant after
warm-up, covering all six variant orders, all returned the known residue
`2308006916`. Medians were:

| Variant | Wall seconds | Transition seconds |
|---|---:|---:|
| Original divisions, repeated rank | 10.85940 | 5.274690 |
| Reciprocal factor division only | 10.88260 | 5.280145 |
| Reciprocal division + rank reuse | 10.83715 | 5.197250 |

The transition counter improves modestly with rank reuse, but there is no
convincing overall speedup in this shared-desktop measurement. Raw samples
and binary hashes are in `optimization-isolation-2026-09-05.json`.

Global shard addressing now also uses a bounded binary search instead of
general division: at most three comparisons suffice for 1..8 shards, and the
one-GPU case uses the global index directly. Comparing shifted global indices
avoids threshold overflow. `src/common/shard_address.hpp` states the valid-index
preconditions; no new buffers are required. `GRIDFP_SHARD_SEARCH=0` restores the
old code. Host UBSan checked 1,297,727 addresses, and a single-GPU test emulating
1..8 address spaces checked 1,000,374 addresses with zero memcheck errors.
The final n=9 solver also passed both known residues under memcheck.

A separate six-pair n=20 comparison gave wall medians of 10.89335 seconds
(legacy shard division) and 10.89055 seconds (binary search): again, no
meaningful full-run gain was established. Every residue matched. Samples are
in `shard-search-2026-09-05.json`. Real eight-GPU P2P timing remains untested;
neither arithmetic optimization establishes bandwidth saturation.

## Reproduce

```bash
# Exhaustive and width-28 host checks (no GPU required):
bash scripts/test/gridfp-reverse.sh
# Actual production-kernel GPU comparison, no timing loop:
ARCH=sm_86 bash scripts/test/gridfp-reverse.sh --gpu
# End-to-end comparison, on an otherwise idle GPU:
python3 scripts/bench/bench_factor_division.py --optimization reverse \
  --n 18 --scratch-mib 64 --repeats 6
# Isolated rejected sparse-scan experiment:
nvcc -O3 -std=c++17 -arch=sm_86 src/cuda/b300/reverse_scan_bench.cu -o /tmp/reverse-scan
/tmp/reverse-scan
```

The GPU rank test executable also accepts `--bench` for alternating CUDA-event
measurements of the actual main kernel. `--optimization sparse` in the full
solver benchmark compares scalar/sparse traversal while keeping rank reuse on.
The default `--optimization division` continues to test the earlier reciprocal
division change.
