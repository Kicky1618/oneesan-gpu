# B300 HIGH-mask-sharded Grid-FP research

Status: experimental. Keep PR #12 draft until an actual full-P2P multi-GPU run validates residues and performance.

## 1. Motivation

The older B300 HBM32 backends shard the authoritative Grid-FP vectors by flat canonical rank. Forced occupancy groups therefore have to be gathered from all GPUs and scattered back after each window. At n=27 this makes group I/O a first-order cost even though the authoritative vectors fit in aggregate B300 HBM.

The key observation is that one of the two forced windows can be made exactly shard-local by choosing the authoritative shard key to be the HIGH occupancy mask rather than global rank.

## 2. Exact n=27 state mass

For W=28, LOW=14 and HIGH=13:

- main states: 385,719,506,620
- blocked states: 135,015,505,407
- total: 520,735,012,027
- uint32 authoritative storage: about 1939.889 GiB

For one fixed HIGH occupancy mask with popcount `kh` and one fixed LOW occupancy mask with popcount `kl`, the exact number of main+blocked states is

```
pair[kh][kl] = sum_he high[kh][he] * (
    2 * low[kl][he]
    + low[kl][he + 1]
    + (he > 0 ? low[kl][he - 1] : 0)
)
```

The binomially weighted sum reproduces 520,735,012,027 exactly.

Greedy longest-processing-time placement of the 8192 HIGH masks over 8 GPUs gives roughly 242.485--242.488 GiB/GPU with relative imbalance around 1.17e-5. HIGH-mask and LOW-mask popcount have only about 0.001119 bit mutual information, so there is very little locality/load-balance tradeoff to exploit with a more complicated scheduler.

## 3. Why HIGH-mask ownership localizes the LOW window

The LOW-active window uses p=14..1. Its active symbols are LOW+center; HIGH symbols are inactive. A transition may alter exact HIGH topology in the boundary-crossing case, but it never changes HIGH occupancy.

This invariant was exhaustively checked at small widths, including cases where HIGH topology itself changes. Therefore every state reached from a fixed HIGH occupancy mask remains in the same authoritative shard.

The authoritative group for one HIGH mask is packed in exactly the FBlock order used by `make_factor_*_blocks(false, mask)`. Consequently:

- LOW group local rank is authoritative owner-local rank;
- LOW gather/scatter is unnecessary;
- no canonical MateID rank conversion is required for LOW group I/O.

## 4. HIGH window

The HIGH-active window uses p=27..15 and fixes a LOW occupancy mask. It does change HIGH occupancy, so a fixed LOW group spans multiple authoritative owners.

The retained design is bulk factorized gather/scatter:

- scratch row: HIGH all-rank;
- scratch column: LOW mask-local rank;
- authoritative owner: HIGH occupancy mask;
- authoritative row: HIGH mask-local rank;
- authoritative column: LOW occupancy-major storage rank.

The exact best-local peer-volume model is about 92.81 TiB/residue at n=27. A direct remote-update experiment was rejected: even the best sampled balanced linear shard function produced about 430 TiB/residue, several times worse than bulk I/O.

### LOW-column-base optimization

`StorageFactorHost` orders LOW codes by `(height, occupancy mask, mask-local rank)`. Therefore for a fixed mask and height,

```
low_storage_rank = low_begin[mask, height] + low_mask_rank
```

The current mask-shard metadata reconstructs `low_begin` once from the base mask counts and uploads about 1.875 MiB/GPU at LOW=14. This removes the per-element LOW-code lookup and the large dense packed-rank lookup from HIGH peer I/O.

The contiguity identity was independently checked for all 1,201,917 LOW factor codes at LOW=14.

## 5. Executor evolution

### v0.1

HIGH-mask-sharded authoritative HBM. LOW window is owner-local, but transition windows still use ping-pong buffers.

### v0.2: HIGH descriptors

`HighDesc` precomputes the HIGH transition result by HIGH all-rank and active p. Normal transitions become a destination block/rank lookup. A boundary-crossing LL case stores a small matching depth and changes LOW topology by flipping the corresponding unmatched R to L.

This removes HIGH factor unrank/rank from the transition kernel. n=27 HighDesc storage is about 152.62 MiB/GPU.

### v0.3: HIGH in-place orbit

The identity branch and included branch form small local orbits. NN/NR/NL are chosen as one representative per orbit. The companion main state and old blocked state can be obtained from the compact HIGH+center code by bit operations and dense HIGH rank lookup.

The orbit kernel consumes old main/blocked values and writes the new orbit values in place. LL/RR/RL closure contributions are then added atomically from HighDesc.

Effects:

- HIGH identity copy: zero
- HIGH scratch: one M+D buffer instead of 2M+2D
- maximum HIGH scratch at n=27: about 4.823 GiB

For production groups the offset within one FBlock is below 2^32. Device kernels therefore use 32-bit quotient/remainder on the fast path and keep a 64-bit fallback for generic correctness.

### v0.4: full in-place orbit

The same orbit decomposition applies to the LOW window. Since the LOW group is already authoritative owner-local HBM, it can be updated directly in place. `LowDesc` handles boundary-crossing RR by storing the depth of the unmatched HIGH L that must be flipped to R.

Effects:

- LOW scratch: zero
- LOW identity copy: zero
- LOW copyback: zero
- only HIGH M+D scratch remains

The current exact memory model, including the 1.875 MiB LOW-column-base table, is:

```
max authoritative GPU : 242.487631 GiB
HIGH M+D scratch       :   4.823201 GiB
factor tables          : 1297.989273 MiB
mask-shard metadata    :    8.761250 MiB
HighDesc               :  152.618088 MiB
LowDesc                :  250.688408 MiB
v0.4 peak              : 248.980810 GiB/GPU
planning usable HBM    : 268.590000 GiB/GPU
headroom                :  19.609190 GiB/GPU
```

This model intentionally leaves CUDA context/allocator overhead outside the analytical sum. The production wrapper therefore also performs runtime HBM admission checks.

## 6. CRT batch backend

`oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch.cu` keeps the expensive structures alive across CRT residues:

- authoritative HBM allocation: once
- factor tables: once
- HighDesc/LowDesc: once
- mask-shard peer metadata: once
- scratch arena: retained and reused

For each modulus it only changes `D_MOD`, clears authoritative vectors, initializes the start state and runs the DP.

`oneesan_cuda_gridfp_b300_hbm32_maskshard_fullorbit_batch_guarded.cu` wraps allocations with an HBM admission rule. By default every allocation must leave `clamp(total_HBM/32, 256 MiB, 8 GiB)` free; `GRIDFP_VRAM_RESERVE_MIB` overrides that reserve.

## 7. Correctness evidence

The research branch currently has independent CPU checks for:

- exact n=27 state mass and mask balance;
- HIGH-mask closure of every LOW-active transition at small widths;
- authoritative address / HIGH-route bijection;
- HighDesc semantics against `gridfp_transition.hpp`;
- HIGH orbit vs ordinary out-of-place update on seeded vectors;
- LOW orbit vs ordinary out-of-place update, including p=1;
- non-atomic orbit footprint disjointness and absence of same-kernel closure cascades;
- complete grouped full-orbit schedule vs canonical out-of-place DP for full W=8, W=10 and W=12 runs;
- occupancy-major LOW storage contiguity at LOW=14;
- n=27 HBM accounting.

The race-freedom probe checks that each NN/NR/NL representative owns a disjoint `{source main, companion main, old blocked}` footprint. Closure destinations use atomics, and a main closure destination is verified not to be another closure source in the same kernel pass.

## 8. Validation still required

GitHub Actions is currently failing before job steps start (`steps=null`), including older known-good CPU jobs. v0.1 compiled for W=22 and W=28 before this runner outage. The newer v0.2/v0.3/v0.4/batch/guarded variants still need nvcc CI once runners recover.

Before merge, run on a real full-P2P multi-GPU node and require:

1. small-width residues equal a canonical/reference solver;
2. multiple moduli in one batch equal separate single-residue runs;
3. HBM admission succeeds with useful safety margin;
4. no peer-access/runtime errors;
5. profile HIGH gather/scatter, HIGH orbit, HIGH closure, LOW orbit and LOW closure separately;
6. measure effective peer bandwidth and final wall time on B300 x8.

## 9. Current production candidate

The current research candidate is the guarded v0.4 batch backend. It should not replace the main production backend until the GPU checks above pass.
