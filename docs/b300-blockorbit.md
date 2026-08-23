# B300 blocked-domain orbit research

Status: experimental follow-up to `docs/b300-mask-shard.md`. v0.4 remains the correctness baseline; v0.5 adds main-coordinate orbit aux; v0.6 changes the orbit iteration domain; v0.7 also compacts aux into blocked coordinates.

## Delete-N bijection

Fix one horizontal position `p`. A main orbit representative is exactly a main state whose upper active symbol at `p` is `N`, i.e. one of `NN`, `NR`, `NL`.

Deleting that `N` produces a valid blocked state. Conversely `blocked_exclude(blocked, p)` inserts `N` at the same position and produces exactly one representative main state. These maps are inverse.

Therefore, for every `p`,

```
# orbit representatives = # blocked states.
```

At n=27 / W=28:

```
M = main states    = 385,719,506,620
D = blocked states = 135,015,505,407
D / M              = 0.3500354612348228
```

The representative count is independent of `p`.

## v0.6 blocked-domain executor

v0.5 scans every main state in the orbit kernel, loads an aux descriptor, and immediately rejects about 65% of those states. v0.6 instead scans the blocked vector.

Existing descriptor tables already contain the inverse map:

- `HighDesc.block_desc`: blocked HIGH coordinate -> representative main coordinate;
- `LowDesc.block_desc`: blocked LOW coordinate -> representative main coordinate.

For each blocked state the kernel obtains the representative main state without reconstructing a MateID. v0.6 reuses the v0.5 main-coordinate aux table to identify `NN` vs `NR/NL` and, for the pair case, to obtain the companion main target.

The orbit update algebra is unchanged:

- NN: companion gets `+c`; representative becomes `c + old_block`; blocked becomes 0;
- NR/NL, p>1: representative becomes `c + companion + old_block`; blocked becomes `c`;
- NR/NL, p=1: representative becomes `c + companion + old_block`; companion becomes `c + companion`; blocked becomes 0.

A pure-C++ exhaustive seeded-vector probe compares this blocked-domain pass plus the existing closure pass against the ordinary out-of-place Grid-FP transition for every p. W=10 and W=12 pass. It also verifies that blocked->representative is a bijection onto all NN/NR/NL sources.

## Exact work reduction

One residue has W rows and W-1 horizontal positions per row.

Representative state-steps at n=27:

```
HIGH positions (13): D * 13 * 28 = 49,145,643,968,148
LOW positions  (14): D * 14 * 28 = 52,926,078,119,544
total orbit reps    :             = 102,071,722,087,692
```

For kernel source scanning per p:

```
v0.5: orbit M + closure M = 2M = 771,439,013,240
v0.6: orbit D + closure M = M+D = 520,735,012,027
ratio v0.6/v0.5            = 0.6750177306
```

Thus v0.6 removes about 32.50% of total orbit+closure source-thread iterations, and about 64.996% of the orbit-pass iterations specifically.

## v0.7 compact blocked-coordinate aux

Once iteration is over blocked states, a main-coordinate aux table is wasteful. `block_desc` already supplies the representative main coordinate. The only extra information needed is:

- whether the representative is NN or NR/NL;
- for NR/NL with p>1, the companion main block/rank;
- for LOW p=1, only the PAIR kind is needed because `LowDesc.main_desc` is already the companion target.

v0.7 therefore stores one 32-bit word in exactly the same coordinate system as the descriptor block table:

```
HIGH aux index = [p][HighDesc blocked active coordinate]
LOW  aux index = [p][LowDesc  blocked active coordinate]
```

The blocked kernel has already computed the descriptor index `bdi`, so the compact aux lookup reuses that exact index. No additional base table or index arithmetic is introduced.

Exact n=27 auxiliary memory:

```
v0.5/v0.6 HIGH main aux : 113.573406 MiB/GPU
v0.5/v0.6 LOW  main aux : 186.499115 MiB/GPU
v0.5/v0.6 total         : 300.072521 MiB/GPU

v0.7 HIGH block aux     :  39.044682 MiB/GPU
v0.7 LOW  block aux     :  64.189293 MiB/GPU
v0.7 total              : 103.233974 MiB/GPU
```

The exact HBM model is now:

```
v0.4 peak                : 248.980810 GiB/GPU
v0.5/v0.6 peak           : 249.273850 GiB/GPU
v0.7 compact-aux peak    : 249.081625 GiB/GPU
planning usable HBM      : 268.590000 GiB/GPU
v0.7 headroom            :  19.508375 GiB/GPU
```

Compared with v0.6, v0.7 returns about 196.84 MiB/GPU while preserving the same blocked-domain execution count.

## v0.7 validation layers

The branch contains three independent checks around this transformation:

1. `factor_blockorbit_semantics.cpp`: blocked-domain orbit + closure equals canonical out-of-place update for every p at W=10/W=12.
2. `factor_blockorbit_compactaux_semantics.cpp`: one compact aux word per blocked state is sufficient to reproduce the same update algebra, including LOW p=1.
3. `maskshard_blockorbitaux_hostplan.cu`: builds the actual CUDA-side compact host tables and checks the block descriptor -> representative -> compact aux path against exact MateID transitions.

The CUDA runtime candidate is:

`oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_batch_guarded.cu`

It defines `MASKSHARD_BLOCK_ORBIT_AUX`, so the existing aux allocation/pointer plumbing is reused but uploads blocked-coordinate tables instead of main-coordinate tables.

## Main-state partition and closure follow-up

For fixed p, main states partition into three disjoint classes:

1. orbit representatives `NN/NR/NL`: D states;
2. their companions `LR/RN/LN`: D states;
3. closure sources `LL/RR/RL`: `M - 2D` states.

At n=27:

```
closure sources = 115,688,495,806
fraction of M   = 0.2999290775303543
```

If both HIGH and LOW closure could be iterated compactly, the two transition passes would scan

```
D + (M - 2D) = M - D = 250,704,001,213
```

source states per p, 32.498% of v0.5's `2M` scan count.

HIGH closure compaction is naturally row-oriented in factorized storage, so it is the next plausible algorithmic step. LOW closure compaction is harder because a naive one-active-column traversal gives strided HIGH-row memory accesses. It should be treated separately rather than assuming the same representation is optimal for both windows.

## Runtime candidates

- v0.4 A/B baseline: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu`
- v0.5 main-domain aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_aux_batch_guarded.cu`
- v0.6 blocked-domain/full-aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_aux_batch_guarded.cu`
- v0.7 blocked-domain/compact-aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_compactaux_batch_guarded.cu`

GitHub Actions is still failing before job steps start (`steps=null`), so fresh nvcc confirmation remains blocked by CI infrastructure. Do not merge any candidate to the production path before fresh nvcc CI and real full-P2P multi-GPU residue checks.
