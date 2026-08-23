# B300 blocked-domain orbit research

Status: experimental follow-up to `docs/b300-mask-shard.md`. v0.4 remains the correctness baseline; v0.5 is descriptor+aux main-domain orbit; v0.6 changes only the orbit iteration domain.

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

For each blocked state the kernel therefore obtains the representative main state without reconstructing a MateID. The v0.5 aux table is reused temporarily to identify `NN` vs `NR/NL` and, for the pair case, to obtain the companion main target.

The orbit update algebra is unchanged. The old blocked cell being iterated is exactly the blocked member of the orbit:

- NN: companion gets `+c`; representative becomes `c + old_block`; blocked becomes 0;
- NR/NL, p>1: representative becomes `c + companion + old_block`; blocked becomes `c`;
- NR/NL, p=1: representative becomes `c + companion + old_block`; companion becomes `c + companion`; blocked becomes 0.

A pure-C++ exhaustive seeded-vector probe compares this blocked-domain pass plus the existing closure pass against the ordinary out-of-place Grid-FP transition for every p. W=10 and W=12 pass. It also verifies that blocked->representative is a bijection onto all NN/NR/NL sources.

## Exact work reduction

One residue has W rows and W-1 horizontal positions per row.

Representative state-steps:

```
HIGH positions (13): D * 13 * 28 = 49,145,643,968,148
LOW positions  (14): D * 14 * 28 = 52,926,078,119,544
total orbit reps    :             = 102,071,722,087,692
```

v0.5 removes two dense packed-rank lookups per representative relative to v0.4, i.e. exactly 204,143,444,175,384 dense-rank lookups/residue. In the simple logical-load model, v0.5 saves one uint32 load per representative relative to v0.4, about 371.335 TiB/residue. This is not a physical-HBM prediction because broadcasts and caches can collapse many HIGH-side accesses.

For kernel source scanning per p:

```
v0.5: orbit M + closure M = 2M = 771,439,013,240
v0.6: orbit D + closure M = M+D = 520,735,012,027
ratio v0.6/v0.5            = 0.6750177306
```

Thus v0.6 removes about 32.50% of total orbit+closure source-thread iterations, and about 64.996% of the orbit-pass iterations specifically.

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

If a future backend can iterate only closure sources, the two transition passes would scan

```
D + (M - 2D) = M - D = 250,704,001,213
```

source states per p, 32.498% of v0.5's `2M` scan count. HIGH closure compaction is naturally row-oriented in factorized storage. LOW closure compaction is harder because a naive one-block-per-LOW-column scheme produces strided HIGH-row memory accesses; it should not be implemented blindly before B300 profiling.

## Aux memory follow-up

v0.6 currently reuses the v0.5 main-coordinate aux tables (300.073 MiB/GPU), so its HBM peak is unchanged at about 249.274 GiB/GPU.

Because blocked-domain iteration needs one aux entry only per blocked descriptor coordinate, a compact follow-up can reduce aux storage to

```
HIGH block aux: 39.044682 MiB/GPU
LOW block aux : 64.189293 MiB/GPU
total         : 103.233974 MiB/GPU
```

Projected peak with compact block aux is about 249.081625 GiB/GPU, leaving about 19.508375 GiB against the 268.59 GiB planning figure. Keep this as a separate optimization until v0.6 compiles and is benchmarked.

## Runtime candidates

- v0.4 A/B baseline: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu`
- v0.5 main-domain aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_aux_batch_guarded.cu`
- v0.6 blocked-domain aux: `oneesan_cuda_gridfp_b300_hbm32_maskshard_blockorbit_aux_batch_guarded.cu`

Do not merge any of them to the production path before fresh nvcc CI and real full-P2P multi-GPU residue checks.
