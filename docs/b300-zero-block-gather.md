# B300 v0.12: eliminate the row-boundary BLOCKED gather

## Invariant

A complete Grid-FP row executes horizontal positions down to `p=1`.
At `p=1`, `include_horizontal()` never returns a BLOCKED destination: all valid
MAIN included branches stay MAIN, while every pre-existing BLOCKED state has
only `blocked_exclude()`, which inserts `N` and returns to the MAIN-width state
space.

Therefore, after every complete row, the authoritative BLOCKED vector is
**identically zero for arbitrary input values**. This is a transition-semantic
invariant, not a property of one modulus or one initial seed.

`factor_rowboundary_blockzero.cpp` exhaustively checks the p=1 statement through
W=13 and pins the n=27 state counts used by the traffic model.

## Waste in v0.4-v0.11

At the next row, the HIGH-window executor builds local `(MAIN,BLOCKED)` scratch.
The old path gathers both vectors from the HIGH-mask-sharded authoritative HBM.
Thus it performs P2P reads of the entire BLOCKED vector even though every word
is known to be zero.

For n=27:

```text
M = 385,719,506,620
D = 135,015,505,407
rows = 28
```

Ignoring local-vs-peer placement, old HIGH gather+scatter traffic is

```text
2(M+D) * 4 * 28 = 106.087684520 TiB/residue
```

v0.12 removes only the BLOCKED gather, leaving MAIN gather/scatter and BLOCKED
scatter unchanged:

```text
(2M+D) * 4 * 28 = 92.334545196 TiB/residue
saved                 13.753139324 TiB/residue
ratio                   0.870360642
reduction               12.9639358%
```

Under the simple balanced model where 7/8 of these logical transfers are peer
traffic, the corresponding peer estimate changes from

```text
92.826723955 -> 80.792727047 TiB/residue
```

The previously quoted ~92.81 TiB model was rounded; the exact 7/8 model above is
used here only for apples-to-apples structural accounting.

## Runtime implementation

`MASKSHARD_SKIP_ZERO_BLOCK_GATHER` is enabled only by the v0.12 wrapper.
The include hook introduces `maskshard_high_block_io_skipzero_kernel` after the
ordinary HIGH-I/O kernel definitions.

- gather (`SCATTER=false`): write zero to local BLOCKED scratch;
- scatter (`SCATTER=true`): preserve the ordinary authoritative BLOCKED write.

The local zero stores are still necessary because the HIGH orbit reads the
BLOCKED scratch. The optimization removes the remote authoritative read; it does
not claim that zero initialization is free.

Files:

- `src/cuda/b300/maskshard_zero_block_gather.cuh`
- `src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_fullclosure_highrowpack16_zeroblockgather_batch_guarded.cu`
- `src/cpp/probes/factor_rowboundary_blockzero.cpp`
- `scripts/bench/b300_maskshard_zeroblock_ab.py`
- `.github/workflows/b300-zeroblock-v12.yml`

Backend alias:

```text
b300-factorized-maskshard-v0.12-highrowpack16-zeroblockgather-batch
```

## Required measurement

Compare v0.11 and v0.12 on the same B300x8 modulus:

```bash
python3 scripts/bench/b300_maskshard_zeroblock_ab.py \
  --n 27 --gpus 8 --threads 256 \
  --modulus 4294967291 \
  --vram-reserve-mib 1024
```

Residues must match exactly. The primary phase metric is `high_io_sum_s`; total
`wall_s` determines whether the removed P2P read matters enough to keep the
variant.

No merge/promotion should occur before fresh nvcc and real full-P2P validation.
