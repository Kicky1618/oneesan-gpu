# HIGH-mask graph partition study

This note revisits the direct owner-compute alternative to the current HIGH
bulk gather/scatter transpose.

## Why revisit it?

`factor_highmask_edge_cut.cpp` previously searched only 3-row GF(2) linear
hashes from the 13-bit HIGH occupancy mask to 8 GPUs. That family produced a
best sampled direct-peer estimate around 430 TiB/residue under a 2.5% load
constraint. It was useful evidence, but not a general graph-partition result.

The exact n=27 HIGH-mask transition graph was therefore reconstructed from the
same `include_horizontal` / `blocked_exclude` semantics:

```text
HIGH occupancy-mask vertices       8,192
nonzero undirected mask pairs      139,267
authoritative state weight         520,735,012,027
transition updates / HIGH window   6,154,161,750,113
same-mask updates                     73,007,659,168
cuttable cross-mask updates        6,081,154,090,945
```

Every edge weight is the exact number of state-level HIGH transition updates
between its two occupancy masks, including the complete LOW multiplicity.

## Baselines

Using 4 bytes per crossing update and 28 DP rows per residue:

```text
all updates remote                 626.884 TiB/residue
plain state-weight LPT shard       553.455 TiB/residue
sampled balanced GF(2) linear      429.807 TiB/residue
current bulk transpose              92.81  TiB/residue
```

The LPT result is expected to have a very poor transition cut because it only
optimizes HBM balance.

## Nonlinear exploratory partitioning

An offline sparse-graph experiment used recursive normalized spectral bisection
followed by capacity-constrained single-node moves and pair swaps. This is a
heuristic upper bound, not an optimality proof.

With maximum authoritative load constrained to 1.025x average, a constructed
partition reached approximately:

```text
remote update fraction             0.342477
raw direct peer traffic            214.693 TiB/residue
max authoritative load ratio       1.024990
modeled max authoritative HBM      248.546 GiB/GPU
modeled peak incl. v0.9 overhead   255.231 GiB/GPU
```

Allowing a much more aggressive 1.075x authoritative imbalance reached:

```text
remote update fraction             0.332365
raw direct peer traffic            208.354 TiB/residue
max authoritative load ratio       1.074998
modeled max authoritative HBM      260.672 GiB/GPU
modeled peak incl. v0.9 overhead   267.358 GiB/GPU
headroom vs 268.59 GiB plan          1.232 GiB/GPU
```

The latter is too close to the planning HBM limit to be attractive without real
allocator measurements.

## Can destination aggregation rescue direct owner-compute?

For one fixed horizontal HIGH position, let `S(k)` be the exact canonical state
count at width `k`. The non-identity update destinations have a useful image
decomposition:

- blocked destinations cover exactly `S(W-1)` states;
- `blocked_exclude` contributes an injective `S(W-1)`-state image in main;
- the NN -> LR branch contributes a disjoint `S(W-2)`-state main image;
- RN/LN -> NR/NL main destinations are already contained in the
  `blocked_exclude` image.

The corresponding update and distinct-destination counts are therefore

```text
updates / HIGH position              S(W) + S(W-1) - S(W-2)
distinct destinations / position     2*S(W-1) + S(W-2)
```

`factor_highupdate_aggregation.cpp` exhaustively checks the image decomposition
at every position for W=4..13 and then evaluates the count recurrence directly
at n=27.

For W=28 / LOW=14 / HIGH=13:

```text
S(28)                               385,719,506,620
S(27)                               135,015,505,407
S(26)                                47,337,954,326
updates / HIGH position             473,397,057,701
distinct destinations / position   317,368,965,140
ideal destination/update ratio            0.670407557
ideal global update reduction             32.9592443%
```

If every update were remote, replacing every set of colliding updates by one
4-byte destination payload would reduce the raw model only from

```text
626.884 TiB/residue -> 420.268 TiB/residue.
```

So even globally perfect per-destination aggregation has only a 1.4916x ceiling
on this workload. Against the current 92.81 TiB transpose, an aggregated direct
scheme would still need fewer than about

```text
92.81 / 420.268 ~= 22.08%
```

of all distinct destination payloads to cross GPUs.

This 22.08% is not a proof against owner-aware aggregation: a good partition may
place high-collision destinations non-randomly. However, exact reduced-width
experiments show the expected adverse correlation. For example, at W=14/LOW=7,
a balanced cut's remote update subset retained about 80% of its payloads even
after grouping by `(source GPU, destination state)`, which is weaker aggregation
than the 67.04% global ratio. This reduced-width result is diagnostic evidence,
not an n=27 regression claim.

## Break-even against the current transpose

Without aggregation, the all-remote direct-update model is about
626.884 TiB/residue. To beat ~92.81 TiB/residue using the same 4-byte accounting,
an 8-way owner partition would need

```text
remote update fraction < 92.81 / 626.884 ~= 0.14805
```

The nonlinear heuristic reaches about 0.33-0.34. Destination aggregation relaxes
the theoretical target, but its total collision budget is too small to make the
current 0.33-0.34 cut obviously competitive. Real remote atomics, metadata and
fine-grained messages also cost more than the idealized 4-byte payload model.

This does **not** prove that no better partition or aggregation exists. It does
show that direct owner-compute needs a qualitatively stronger compression or
communication-avoiding transformation, not just a better 8-way graph cut.

## Reproduce/export the exact graph

Build the exporter:

```bash
g++ -O3 -std=c++17 -Wall -Wextra -Werror \
  src/cpp/probes/factor_highmask_graph_export.cpp \
  -o build/factor_highmask_graph_export
```

Validate only:

```bash
build/factor_highmask_graph_export 28 14
```

Export the graph:

```bash
build/factor_highmask_graph_export 28 14 build/highmask28
```

This writes:

- `build/highmask28.meta.tsv`: width/split/count totals;
- `build/highmask28.nodes.tsv`: HIGH occupancy mask and authoritative state weight;
- `build/highmask28.edges.tsv`: unordered mask pair and exact transition weight.

The n=27 exporter pins all counts listed at the top of this note, so changes to
transition semantics or graph construction fail loudly.

The aggregation image decomposition can be checked independently with:

```bash
g++ -O3 -std=c++17 -Wall -Wextra -Werror \
  src/cpp/probes/factor_highupdate_aggregation.cpp \
  -o build/factor_highupdate_aggregation
build/factor_highupdate_aggregation 28 14 13
```

## Reproduce the nonlinear heuristic

`scripts/research/highmask_spectral_partition.py` consumes the exported TSVs.
It requires NumPy and SciPy and currently targets exactly 8 shards.

```bash
python3 scripts/research/highmask_spectral_partition.py \
  build/highmask28 \
  --max-load-ratio 1.025 \
  --output build/highmask28.partition.json
```

The output JSON contains shard loads, cut update weight, remote-update fraction,
direct-peer TiB/residue, and the owner id for every HIGH occupancy mask.

The script validates node/edge/state/update totals against `meta.tsv` before it
partitions. The eigensolver and local search are heuristic, so a run is not
required to reproduce the exact 214.693 TiB exploratory number bit-for-bit to be
valid. Compare the reported cut and load ratio instead.

The TSV format also makes the same exact graph available to METIS/KaHIP-style
external experiments without changing the Grid-FP counting code.

## Consequence for current work

Keep HIGH gather/scatter as the primary v0.9-v0.11 execution model. Focus B300
measurement on:

1. `high_io_sum_s` to see the realized cost of the ~92.81 TiB transpose;
2. `high_closure_sum_s` for v0.9/v0.10/v0.11 row-packing effects;
3. only revisit owner-compute if a scheme aggregates/compresses crossing updates
   substantially beyond ordinary destination collision folding.

The exact graph exporter and destination-image probe are now in-tree, while the
spectral partition numbers remain exploratory research measurements rather than
CI performance regressions.
