# RAMstream32 hybrid-sparse CPU LOW backend

This note records the low-VRAM / large-System-RAM execution path after the factorized authoritative layout work.

## Architecture

The authoritative main/blocked count matrices live in System RAM / mmap storage in occupancy-major factorized order.

For each Grid-FP row:

- the HIGH window is transferred to the GPU and executed with the factorized compact bi-descriptor backend;
- the LOW window is executed directly by CPU workers against the authoritative System-RAM arrays;
- the LOW side therefore uses no gather/scatter scratch buffer and does not cross PCIe;
- only the HIGH window contributes host-device traffic.

This changes the direct2d traffic floor from approximately `4 * W * authoritative_bytes` per residue to `2 * W * authoritative_bytes`.

For `n=27` (`W=28`, `LOW=14`, `HIGH=13`), the current plan reports:

- authoritative state storage: about 1939.89 GiB;
- GPU HIGH-window maximum: 9.6464 GiB;
- PCIe traffic: 106.088 TiB per residue;
- ideal PCIe time at 50 GiB/s: 2172.68 s = 36.21 min;
- CPU LOW scratch: 0 bytes.

The remaining hard lower bound is therefore the HIGH-window PCIe traffic plus direct System-RAM updates on the CPU side.

## v4.4: 64-bit sparse orbit operations

The first sparse CPU LOW implementation stored each N* orbit operation in 12 bytes:

- source LOW rank;
- orbit kind;
- partner factor block;
- partner LOW rank;
- blocked factor block;
- blocked LOW rank.

The two factor-block IDs are redundant.

For a source factor block and LOW position `p`:

1. dropping N never changes HIGH, so the blocked factor block is exactly the source HIGH endpoint height;
2. below the LOW/center boundary, the partner keeps the center unchanged, so the partner factor block is the source block;
3. at `p=LOW_LUT_K`, the source center is N and the orbit kind uniquely determines the partner center: NN/NL -> L, NR -> R.

The operation can therefore be represented by only three 20-bit LOW ranks plus a 2-bit orbit kind, i.e. 62 bits total. The production stream uses one `uint64_t` per operation.

At `n=27`:

- orbit operations: 16,826,838;
- old 12-byte orbit stream: about 192.57 MiB;
- v4.4 8-byte orbit stream: 128.379 MiB;
- saved: about 64.19 MiB (33.3% of orbit-stream storage);
- closure stream: 116.241 MiB;
- sparse orbit + closure metadata falls from about 308.81 MiB to 244.62 MiB.

The sparse builder verifies every derived destination block against the dense `LowOrbitHost` representation before discarding the redundant fields.

Validation:

- W=22 / n=21 plan build: pass;
- W=28 / n=27 plan build: pass;
- exhaustive W=10 CPU LOW comparison against the full reference transition: pass.

Merged as commit `71a40bd401b44e30de937bbc31ab88e12e37d827`.

## v4.5: pre-ranked CROSS transitions

A LOW closure can cross the LOW/HIGH boundary only when an RR partner search escapes into HIGH. The descriptor stores a small matching depth.

Before v4.5, every active CROSS update performed two topology operations at runtime:

1. scan the HIGH code to find the `depth`-th unmatched L and flip it to R;
2. binary-search the resulting HIGH code inside the destination `(occupancy mask, endpoint height)` slice to recover its mask-local rank.

Both operations depend only on topology, not on counts or modulus.

v4.5 builds a table

`(source HIGH mask-code index, matching depth) -> destination mask-local HIGH rank`

while the dense HIGH rank table is still available. Flipping L to R preserves the occupancy mask. With `HIGH_LUT_K <= 15`, every mask-local rank fits in `uint16_t`; `0xffff` is reserved as the invalid sentinel.

At `n=27`:

- LOW descriptor CROSS fraction: 0.0732474;
- pre-ranked CROSS table: 19.5223 MiB;
- metadata build time: about 3.70 s;
- runtime CROSS topology work becomes one `uint16_t` lookup instead of an O(H) bracket scan plus binary search.

Validation:

- W=22 / n=21 sparse plan: pass;
- W=28 / n=27 sparse plan: pass;
- exhaustive W=10 sparse CPU LOW comparison: pass;
- complete RAMstream W=22/W=28 compile regression workflow: pass.

Merged as commit `8bab5b17b4eb7e608d2d142cecbfea5d7bfda59a`.

## v4.6: split LOCAL/CROSS closure streams

The v4.5 sparse runtime still decoded the descriptor kind for every closure operation. That branch is unnecessary once the topology has been classified during metadata construction.

For a closure that does not cross into HIGH, destination storage is fixed by the active LOW position:

- `p = 1`: destination is a main state;
- `p > 1`: destination is a blocked state.

v4.6 therefore builds two independent 64-bit streams per `(p, factor block)`:

- `local_closure_ops`: source LOW rank, destination factor block, destination LOW rank;
- `cross_closure_ops`: the same fields plus the HIGH matching depth.

The dense descriptor is still available while the sparse stream is built. The builder asserts that every non-CROSS closure obeys the `p=1 -> MAIN`, `p>1 -> BLOCK` invariant. A mismatch aborts metadata construction rather than silently entering the optimized executor.

Runtime changes:

- LOCAL and CROSS operations are traversed in separate loops;
- the per-operation `LOWDESC_MAIN/BLOCK/CROSS` dispatch is removed;
- `p == 1` is hoisted outside the operation loops;
- CROSS continues to use the v4.5 pre-ranked `uint16_t` HIGH destination table;
- both streams remain 8 bytes/op, so the change is intended as a CPU front-end/branch-prediction optimization rather than a metadata-size reduction.

The backend identifies itself as `gridfp-ramstream32-factorized-hybrid-sparse-v4.6` and reports LOCAL/CROSS closure MiB separately so a high-memory host benchmark can correlate runtime changes with the actual stream mix.

Validation is wired into `.github/workflows/ramstream32-sparse-ci.yml`:

- compile and `--plan-only` for W=22 / n=21;
- compile and `--plan-only` for W=28 / n=27;
- exhaustive W=10 comparison of out-of-place, in-place, direct, and sparse CPU LOW executors against the full reference recurrence.

## v4.7: split NN/NR/NL orbit streams

The remaining hot-loop dispatch in v4.6 was the orbit kind. Every N* representative carried two kind bits and runtime executed a per-operation branch between NN and NR/NL behavior.

The three N* classes form disjoint local orbits:

- NN pairs with LR;
- NR pairs with RN;
- NL pairs with LN.

None of `LR`, `RN`, or `LN` is itself an N* representative at the same active position, and dropping the N produces distinct blocked states because the lower symbol is retained. The in-place orbit updates can therefore be reordered by class without changing the algebra.

v4.7 builds three independent 64-bit orbit streams per `(p, factor block)`:

- `nn_orbit_ops`;
- `nr_orbit_ops`;
- `nl_orbit_ops`.

The operation record remains 8 bytes to keep the representation simple and preserve the existing rank fields. Runtime no longer decodes `kind` and no longer branches on `kind == NN`. For NR/NL, `p == 1` versus `p > 1` is also hoisted outside the operation loop.

The cost is two additional small offset tables and potentially more row passes when all three classes are present. Consequently v4.7 is explicitly an empirical CPU microarchitecture experiment rather than an assumed improvement. The relevant comparison is v4.6 versus v4.7 on the same host, with `cpu_wall_s`, `cpu_kernel_sum_s`, instructions, branch misses, LLC misses, and memory bandwidth recorded.

The backend identifies itself as `gridfp-ramstream32-factorized-hybrid-sparse-v4.7` and reports NN/NR/NL orbit MiB separately. The W=10 full-state selftest also checks that the three stream counts sum to the total sparse orbit count before comparing the final main/blocked arrays against the reference recurrence.

## Current bottlenecks and next experiments

v4.7 removes the remaining orbit-kind dispatch, but does not alter the dominant data movement bound. It may improve or regress CPU LOW time depending on whether branch savings outweigh the extra row traversals.

The immediate measurement should compare the v4.6 commit `c8de8d73af1c44f075aee937bc8f37e8b7b79d27` with v4.7 under identical conditions. At minimum record:

- `cpu_wall_s` and `cpu_kernel_sum_s`;
- CPU cycles/instructions and branch misses;
- LLC misses and memory bandwidth;
- NN/NR/NL orbit operation counts;
- LOCAL/CROSS closure operation counts;
- full `wall_s`, to determine whether LOW-side savings remain visible behind HIGH-side PCIe traffic.

If v4.7 does not win, the likely better compromise is a two-stream layout: NN versus NR+NL, retaining only one predictable branch or one small center-selection operation while avoiding a third row traversal.

After the microarchitectural choice is settled, the more consequential directions are:

1. increase the CPU/System-RAM fraction so fewer HIGH states cross PCIe;
2. partition HIGH work across aggregate multi-GPU memory and keep partitions resident for more than one local step;
3. batch multiple CRT residues through the same topology metadata and scheduling decisions;
4. investigate whether a legal wider in-place orbit can merge part of the HIGH boundary work into the CPU LOW pass without introducing cross-group write races.

For `n=27`, copy-call coalescing alone cannot remove the remaining 106.088 TiB/residue HIGH-window traffic. Any large wall-time reduction must reduce transferred bytes, hide them behind useful work, or increase the effective host-device bandwidth.