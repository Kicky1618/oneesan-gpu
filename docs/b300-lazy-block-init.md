# B300 v0.13: lazy BLOCKED scratch initialization

v0.12 proves that the authoritative BLOCKED vector is zero at every complete
row boundary and replaces its P2P gather with local zero stores. v0.13 removes
those local stores as well.

## Why the scratch can stay uninitialized

The first HIGH position is `p=TARGET_W-1`. The v0.6+ executor uses the exact
blocked-domain orbit: one iteration per BLOCKED coordinate `di`.

For each `di`, `blocked_exclude(di,p)` gives the corresponding MAIN orbit
representative. The orbit then writes a new value to `blockv[di]`:

- NN orbit: `blockv[di] = 0`;
- NR/NL pair orbit: `blockv[di] = c`.

At a row boundary the old BLOCKED value is known to be zero, so the first HIGH
orbit can use `d=0` directly instead of reading `blockv[di]`. By the time the
following closure kernel runs, every BLOCKED coordinate has been defined.

The CUDA specialization also defensively writes zero on the first HIGH position
before any unexpected invalid-descriptor/aux `continue`, so an anomalous table
entry cannot expose uninitialized scratch to the closure pass.

## Semantic validation

`factor_lazyblock_firsthigh_semantics.cpp` deliberately fills BLOCKED scratch
with nonzero poison values, then compares:

1. the canonical first-HIGH-position transition with row-boundary BLOCKED=0;
2. blocked-domain orbit + closure with `d=0` and poisoned physical scratch.

It tests arbitrary seeded MAIN values for W<=12 and requires exact vector
equality. It also verifies every BLOCKED coordinate is overwritten exactly once
by the first blocked-domain orbit before closure.

## Traffic and local-memory effect

Peer/network payload is unchanged from v0.12 because both versions already
eliminate the authoritative BLOCKED gather:

```text
v0.12/v0.13 logical HIGH I/O = 92.334545196 TiB/residue
balanced 7/8 peer model      = 80.792727047 TiB/residue
```

The difference is local scratch initialization. v0.12 writes zero to all D
BLOCKED words once per DP row:

```text
D = 135,015,505,407
D * 4 * 28 = 13.753139324 TiB/residue local writes
```

If work is balanced across eight GPUs, that is about 1.719 TiB/GPU/residue of
local HBM writes avoided by v0.13.

This is not a wall-time prediction. The shared batch host now compiles out the
gather-side BLOCKED kernel launch entirely when
`MASKSHARD_LAZY_ZERO_BLOCK_INIT` is enabled, so v0.13 performs neither the P2P
BLOCKED gather nor a zero-fill/no-op replacement launch. v0.12 and earlier keep
their original gather behavior through the compile-time guard.

## Files

- `src/cuda/b300/maskshard_lazy_block_init.cuh`
- `src/cuda/b300/maskshard_zero_block_gather.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch.cu`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_lazyblockinit_batch_guarded.cu`
- `src/cpp/probes/factor_lazyblock_firsthigh_semantics.cpp`
- `scripts/bench/b300_maskshard_lazyblock_ab.py`
- `.github/workflows/b300-lazyblock-v13.yml`

Backend alias:

```text
b300-factorized-maskshard-v0.13-highrowpack16-lazyblockinit-batch
```

## B300 A/B

```bash
python3 scripts/bench/b300_maskshard_lazyblock_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 \
  --vram-reserve-mib 1024
```

v0.12 and v0.13 must return identical residues. The relevant phase is
`high_io_sum_s`; `wall_s` decides whether eliminating the local zero write and
launch is worth keeping.

Do not merge/promote before fresh nvcc and full-P2P correctness validation.
