# Two-cell cluster-aware row plan

Status: capacity/traffic model plus CUDA microprobe plan, 2026-08-29.

This note records the current W=28 row-level execution target.  None of the
percentages below are measured B300 speedups; they are exact state-weighted
capacity/traffic calculations under a per-CTA shared-memory budget.

## Row decomposition

For a width-28 forward snake row, organize the reduced operations as

```text
12 x (interior K + interior K)
 1 x (last interior K + physical row turn)
```

so the 26 row-core operators are completely covered by 13 two-operator pairs.
The reflected reverse row has the same state-count/capacity model once the
reverse fusion executor is brought to the same implementation generation.

The stationary reduced vector contains

```text
R_28 = 165,727,043,758 uint32 values
     ~= 617.381 GiB.
```

A normal one-step stationary pass loads and stores every value once.  A fused
pair loads/stores a resident state only once across two operators.  Therefore,
if a pair fuses a state fraction `f`, its ideal value-traffic reduction against
two separate passes is `f/2`.

## Primitive-sliced cluster DSM

A fused block is split into stationary support sectors, and each sector is
split by primitive-rank interval across cluster CTAs.  For `steps=2` there are
at most 20 sectors; for the boundary `steps=1` footprint there are at most 10.
Each CTA still performs contiguous global interval copies.

A component is assigned to the CTA owning the label primitive rank.  This keeps
most source accesses local while allowing larger outer-support blocks to use
distributed shared memory.

## W=28, 228 KiB/CTA model

The current planner uses

```text
per-CTA shared budget = 228 KiB
workspace reserve     = 4096 B
allowed cluster sizes = 1, 2, 4, 8
```

and selects the smallest cluster size that fits each outer-popcount bucket.

State-weighted coverage is

```text
maximum cluster      interior fusion2      boundary K+turn
1 CTA                58.6590942%           64.0206032%
2 CTAs               74.6782369%           79.0666742%
4 CTAs               86.7397090%           89.2973872%
8 CTAs               94.4127437%           98.6002946%
```

The corresponding 26-operation row-core value-traffic reduction is

```text
maximum cluster      row traffic reduction
1 CTA                 29.5357590%
2 CTAs                37.5079045%
4 CTAs                43.4682267%
8 CTAs                47.3674315%
```

The formula is

```text
row_reduction = (12 * f_interior + f_boundary) / 26.
```

Thus the capacity ceiling approaches a factor

```text
1 / (1 - 0.473674315) ~= 1.90x
```

on the value-traffic part of the row core.  This is not a total-kernel speedup:
DSM remote accesses, synchronization, rank work, component reconstruction, and
modular arithmetic remain.

## Runtime selection

Capacity alone must not choose the cluster size in production.  The runtime
microprobes use

```text
cudaOccupancyMaxPotentialClusterSize
cudaOccupancyMaxActiveClusters
cudaLaunchKernelEx
cudaLaunchAttributeClusterDimension
```

and reject a bucket if the target kernel/configuration is not launchable.

The current right-boundary ladder is

```text
two_cell_boundary_register_sector_microprobe.cu
    single CTA, 10 stationary sectors, register K/turn arithmetic

two_cell_boundary_cluster_sliced_lut_microprobe.cu
    primitive-sliced DSM boundary kernel

two_cell_boundary_cluster_forced_microprobe.cu
    force cluster=2/4/8 at W<=10 and compare with exact CPU boundary reference

two_cell_boundary_cluster_runtime_microprobe.cu
    choose the smallest accepted cluster per bucket using occupancy APIs
```

The interior fusion2 path has the analogous single-CTA, sliced-DSM, forced, and
runtime probes.

## Validation order on target GPU

1. Compile all CPU probes and CUDA microprobes with the target toolkit.
2. Run the stationary one-step register kernel at small width.
3. Run forced fusion2 DSM with cluster 2, then 4, then 8.
4. Run forced boundary DSM with cluster 2, then 4, then 8.
5. Compare every result with the existing exact CPU reference.
6. Measure local-vs-remote DSM bandwidth and cluster barrier cost.
7. Benchmark each outer-popcount bucket independently; do not assume the
   largest fitting cluster is fastest.
8. Implement and validate the reflected reverse fusion path and left boundary.
9. Only then compose a full forward-turn-reverse-turn snake cycle.

GitHub Actions are intentionally not part of this exploratory validation loop.
