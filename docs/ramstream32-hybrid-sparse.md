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

## Current bottlenecks and next experiments

The two changes above deliberately target topology/metadata work that can be removed exactly. They do not change the DP recurrence.

The next CPU LOW experiment should be measured before promotion:

- split closure operations into local and CROSS streams;
- for local closures, the destination kind is determined by the outer `p` (`p=1` -> main, `p>1` -> blocked), eliminating the per-operation kind branch;
- keep CROSS operations in a separate branch-free loop using the v4.5 pre-ranked HIGH table;
- benchmark on a high-memory, many-core host before merging, because this is a microarchitectural optimization rather than an asymptotic/topological reduction.

For `n=27`, further copy-call coalescing alone cannot remove the remaining 106.088 TiB/residue HIGH-window traffic. Larger gains require either reducing how many HIGH states cross PCIe, increasing the fraction executed directly on the CPU/System-RAM side, or moving more of the authoritative representation into aggregate GPU memory.
