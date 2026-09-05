# B300 8x8 bucket snake backend

Status: implementation/research note, 2026-08-25.

## Placement model

For occupancy-owner functions `owner_H(mH)` and `owner_L(mL)`, define
`B[a,b]` as the authoritative states with owner pair `(a,b)`.

Two physical placements are used:

- H-major: GPU `a` owns `B[a,*]`. A LOW window is GPU-local because LOW
  transitions preserve HIGH occupancy.
- L-major: GPU `b` owns `B[*,b]`. A HIGH window is GPU-local because HIGH
  transitions preserve LOW occupancy.

Changing placement is a raw transpose of the 8x8 bucket matrix. Bucket-internal
`HIGH-row x LOW-column` offsets do not change.

## Snake/reflected row scan

Let `J` reverse frontier positions and exchange `L <-> R`. The reverse cell
operator is the conjugate

```
T_rev(p) = J^-1 T_fwd(W-p) J.
```

Alternating row direction gives

```
row 0:  HIGH_fwd -> transpose -> LOW_fwd     (L-major -> H-major)
row 1:  LOW_rev  -> transpose -> HIGH_rev    (H-major -> L-major)
row 2:  HIGH_fwd -> transpose -> LOW_fwd
...
```

The GPU-free dictionary-DP probe checks equality of the complete main and
blocked vectors at every row boundary for small widths. Reverse LOW preserves
HIGH occupancy and reverse HIGH preserves LOW occupancy, so the same 8x8
ownership remains valid.

## One-transpose-per-row lower bound

Assume both of the following requirements:

1. every LOW window is executed entirely from local HBM, hence must use an
   H-major placement;
2. every HIGH window is executed entirely from local HBM, hence must use an
   L-major placement.

Every row contains one complete LOW window and one complete HIGH window.
Because H-major and L-major are different placements for non-diagonal buckets,
at least one ownership change is necessary inside each row. Therefore any
schedule satisfying the two locality requirements needs at least `W`
transposes for `W` rows.

The ordinary fixed-direction schedule needs `2W-1` transposes when the first
placement is chosen to match the first window. The alternating snake schedule
ends each row in the placement required by the first window of the next row,
so it uses exactly `W` transposes and attains the lower bound.

For `n=27`, `W=28`:

```
ordinary: 55 transposes / residue
snake:    28 transposes / residue
reduction: 49.09%
```

Using the previously derived logical off-diagonal payload
`1697.400389 GiB / transpose`, the logical communication volume is
approximately

```
ordinary: 91.169 TiB / residue
snake:    46.413 TiB / residue
```

At 14.4 TB/s aggregate NVLink this corresponds to ideal payload-only floors of
about 6.96 s and 3.54 s per residue respectively. These are lower bounds, not
runtime predictions. Production swaps fixed-capacity paired slots, so
`peer_gib_per_transpose` printed by the exact planner is the authoritative raw
byte count.

## Reverse direct transition

Reverse descriptors are generated once on the host from the conjugate
transition. Runtime does not mirror MateID values.

- reverse LOW CROSS uses the same inactive-HIGH `L -> R` depth map as forward
  LOW;
- reverse HIGH CROSS uses the same inactive-LOW `R -> L` depth map as forward
  HIGH.

Reverse orbit operations use the same in-place three-state orbit algebra as the
forward direct executor. The first correctness backend used source-oriented
CAS closure updates. The current fused backend reverses closure edges by
active-half destination and reuses the forward factorized CROSS preimage
walkers, eliminating closure atomics.

Production metadata construction validates:

- source-edge conservation from source-oriented closure metadata to fused
  destination records;
- destination-reference conservation;
- monotone per-step/per-block offsets;
- owner-local source and destination rank bounds;
- maximum local and CROSS destination indegrees.

The validation line starts with `reverse_bucket_fused_validate`.

## Pseudo-Mersenne closure accumulation

For CRT moduli `p = 2^32-c`, destination-gather closure can accumulate its
uint32 contributors in a uint64 accumulator and reduce once at the destination.
The reducer uses four base-`2^32` folds; three folds are insufficient for all
uint64 inputs because a residual bit 32 can still contribute `c` modulo `p`.

`PM_ACCUM=1` now routes both forward and reverse fused closures through the
uint64 accumulator. Orbit kernels remain on the proven uint32 modular-add path.
A forward PM CROSS-address bug found during this integration was fixed:
`bkf_sum_high_preimages_u64` must use the LOW bucket view (`bkf_low_main`) for
its active-LOW source locator.

## Transpose implementations

Three implementations are available for A/B measurement:

- `sync`: one staging buffer/GPU, host barriers for every chunk stage;
- `events`: one staging buffer/GPU, cross-device CUDA event dependencies;
- `pipeline`: two staging buffers/GPU, overlaps local commit of chunk `k` with
  bidirectional peer fetch of chunk `k+1`.

The production planner reports `transpose_staging_multiplier`; it is 1 for
`sync/events` and 2 for `pipeline`, and HBM preflight includes the exact
staging allocation.

Build examples:

```bash
N=27 REVERSE_MODE=fused TRANSPOSE_MODE=events PM_ACCUM=0 \
  bash scripts/build/b300-bucket-snake.sh

N=27 REVERSE_MODE=fused TRANSPOSE_MODE=pipeline PM_ACCUM=1 \
  bash scripts/build/b300-bucket-snake.sh

N=27 TRANSPOSE_MODE=events PM_ACCUM=0 \
  bash scripts/build/b300-bucket-snake-batch.sh
```

The exact CRT wrapper is:

```bash
TRANSPOSE_MODE=events PM_ACCUM=0 \
  bash scripts/run/b300x8-bucket-snake-exact.sh 27
```

Its work directory is separate from the older backend and the exact runner
binds checkpoints to the solver binary SHA-256.

## Hardware validation gates

The intended order is:

1. W=10 reverse fused GPU selftest, `PM_ACCUM=0`;
2. W=10 reverse fused GPU selftest, `PM_ACCUM=1`;
3. n=21 / modulus 4294967291 snake fused, expected residue `998035516`;
4. events versus pipeline transpose A/B on 8 GPUs;
5. n=27 one-residue smoke;
6. reusable multi-prime exact CRT run.

Current GitHub Actions runs for these CUDA workflows are failing before job
steps are created (`steps=null`), so they do not currently constitute either a
successful compile or a compile failure. Real nvcc/GPU validation remains a
required gate.
