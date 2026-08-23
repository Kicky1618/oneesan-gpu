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
followed by weighted single-node moves and pair swaps. This is a heuristic upper
bound, not an optimality proof.

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

## Break-even against the current transpose

The all-remote direct-update model is about 626.884 TiB/residue. To beat the
current ~92.81 TiB/residue bulk transpose using the same 4-byte accounting, an
8-way owner partition would need

```text
remote update fraction < 92.81 / 626.884 ~= 0.14805
```

The nonlinear heuristic reaches about 0.33-0.34, still more than twice the
break-even fraction. Real direct remote atomics or fine-grained messages would
normally cost more than the idealized 4-byte payload model, strengthening the
case for the existing transpose.

This does **not** prove that no better partition exists. It does show that the
old ~430 TiB result was not merely an artifact worth fixing into production:
even a substantially better nonlinear cut remains far behind bulk transfer.

## Consequence for current work

Keep HIGH gather/scatter as the primary v0.9-v0.11 execution model. Focus B300
measurement on:

1. `high_io_sum_s` to see the realized cost of the ~92.81 TiB transpose;
2. `high_closure_sum_s` for v0.9/v0.10/v0.11 row-packing effects;
3. only revisit owner-compute if a communication scheme aggregates multiple
   crossing updates into substantially fewer payload transfers than the simple
   state-update edge model.

A useful future reproducibility task is to add an exact HIGH-mask weighted graph
exporter for METIS/KaHIP-style external partitioners. Until then, the spectral
numbers above should be treated as exploratory research measurements rather than
CI regression values.
