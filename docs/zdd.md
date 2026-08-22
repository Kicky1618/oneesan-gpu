# ZDD output

There are now two ZDD builders.

1. `src/cpp/oneesan_zdd_replay.cpp` is the preferred **Grid-FP replay exporter**. It starts from the same initial `MateID`, uses the same main/blocked state representation, and calls the same horizontal-inclusion transition function as the factorized CUDA kernel through `src/common/gridfp_transition.hpp`.
2. `src/cpp/oneesan_zdd.cpp` is an independent SIMPATH-style frontier implementation based on vertex degrees and connected components. It is kept as a correctness oracle.

Both produce a ZDD whose members are exactly the full edge sets of simple paths from the upper-left corner to the lower-right corner of an `(n+1) x (n+1)` grid graph. Every real horizontal and vertical grid edge is a ZDD variable.

## How replay reconstruction works

The counting solver discards history when it sums integer/residue values, so a final count alone is insufficient to reconstruct a ZDD. The transition system is sufficient, however.

The grid-specialized DP processes horizontal edges as the binary branch. For a main `MateID` state:

- the excluded horizontal-edge branch is the identity state;
- the included branch is exactly `gridfp::include_horizontal(mate, width, p)`;
- an included branch may lead to either a main state or a blocked state.

A blocked state has no included-horizontal branch. Its excluded branch is
`gridfp::blocked_exclude(compressed, p)`.

Vertical grid edges are not branching decisions in the Iwashita grid DP. Their inclusion is forced by the endpoint symbol in the frontier. The replay exporter inserts these forced vertical edges as ZDD nodes with `low=0, high=child`. Thus the exported family contains **all real grid edges**, not merely the horizontal decision edges.

The replay begins at the same state used by the CUDA solver,

```text
R at position W-1
```

and accepts only the same final state,

```text
R at position 0
```

where `W=n+1`.

## Build the replay exporter

```bash
./scripts/build/zdd-replay.sh
```

By default RAPiDD is read from `/home/kicky/ダウンロード/rapidd`. Override it with:

```bash
RAPIDD_ROOT=/path/to/rapidd ./scripts/build/zdd-replay.sh
```

## Generate from the Grid-FP transition system

Portable ONEESAN format:

```bash
./scripts/run/zdd-replay.sh 9 work/zdd/replay_n9.zdd
```

Also emit a SAPPOROBDD `ZBDD_Import()` compatible file:

```bash
./scripts/run/zdd-replay.sh 9 work/zdd/replay_n9.zdd \
  --sapporo work/zdd/replay_n9.sapporo.zbdd
```

Increase the RAPiDD node arena when needed:

```bash
MAX_NODES=67108864 \
  ./scripts/run/zdd-replay.sh 12 work/zdd/replay_n12.zdd
```

## Independent builder

The original independent builder remains available:

```bash
./scripts/build/zdd.sh
./scripts/run/zdd.sh 9 work/zdd/independent_n9.zdd
```

This builder does not use the Grid-FP state representation. It independently enforces degree, connectivity, endpoint, and no-cycle conditions and is therefore useful for differential testing.

## Validate and count exactly

RAPiDD's built-in `cardinality()` saturates at `uint64_t`. The validator reads the exported DAG and uses Python arbitrary-precision integers:

```bash
./scripts/tools/validate_zdd.py work/zdd/replay_n9.zdd
```

For replay n=9:

```text
variables=180
nodes=413244
exact_cardinality=41044208702632496804
valid=1
```

The same replay export was re-imported by SAPPOROBDD `ZBDD_Import()`. Its `CardMP16()` is

```text
2399A525A7F680EA4
```

which is the same exact decimal cardinality `41044208702632496804`.

## Differential correctness tests

For n=1..6, replay cardinalities agree with the existing exact counts:

| n | exact paths | replay ZDD nodes |
|---:|---:|---:|
| 1 | 2 | 4 |
| 2 | 12 | 29 |
| 3 | 184 | 149 |
| 4 | 8,512 | 650 |
| 5 | 1,262,816 | 2,583 |
| 6 | 575,780,564 | 9,668 |

For n=1..4 the replay and independent ZDDs were fully enumerated, translated back to canonical real grid-edge IDs, and compared as set families:

```text
n=1  2/2       MATCH
n=2  12/12     MATCH
n=3  184/184   MATCH
n=4  8512/8512 MATCH
```

The reusable check for small instances is:

```bash
./scripts/tools/compare_zdd_families.py \
  work/zdd/independent_n4.zdd \
  work/zdd/replay_n4.zdd
```

## CUDA transition sharing

`src/common/gridfp_transition.hpp` contains the semantic main/blocked transition functions. The factorized CUDA kernel and `oneesan_zdd_replay.cpp` both call these functions. After the extraction, the CUDA n=21 regression still produced

```text
residue=998035516  (mod 4294967291)
```

and ran in 49.281 s on the local RTX 3070 test.

The older `forced2window_opt` kernel keeps its specialized rank-delta/drop-N code path for performance; it was not mechanically replaced by the generic result-Mate transition because doing so would throw away those rank optimizations.

## Format

`ONEESAN_ZDD_V1` stores:

- grid metadata;
- ZDD level to canonical real-grid-edge mapping;
- root node ID;
- reduced ZDD nodes `(id, level, low, high)`.

The text format is intended to be easy to inspect and validate rather than space-optimal. `--sapporo` emits SAPPOROBDD's importable `_i/_o/_n` text format for interoperability.
