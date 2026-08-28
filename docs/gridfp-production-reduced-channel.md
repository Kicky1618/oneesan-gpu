# Exact reduced channel for production Grid-FP

Status: experimental exact reformulation, 2026-08-28.

This note is deliberately separate from `two-cell-channel.md`.
The earlier two-cell `A/C` channel is internally exact for its beta=0 local
operator, but that operator does not model the production Grid-FP deferred
semantics: a production main state always has an excluded identity branch,
while a blocked state has no included branch and is consumed one cell later.
The construction below starts from `gridfp_transition.hpp` directly and is the
candidate that should be compared with the authoritative Grid-FP solver.

## Production state space

At an interior forward position `p`, the authoritative state is

```text
main    : M_W
blocked : M_{W-1}
```

where `M_k` is the number of one-defect Motzkin words of width `k`.
The full direct-sum dimension is therefore

```text
M_W + M_{W-1}.
```

For `W=28`:

```text
M_28 = 385,719,506,620
M_27 = 135,015,505,407
full  = 520,735,012,027
```

## Exact three-term kernel

For every width-`W-1` blocked word whose lookahead symbol is `N`, production
has the exact local kernel relation

```text
B_p(N q) + M_p(LR q) - M_p(NN q) = 0.
```

The reason is immediate from the production transition:

- `M(NN q)` keeps its excluded `M(NN q)` branch and its included branch creates
  `M(LR q)`;
- `M(LR q)` has only the excluded identity branch for this local case;
- `B(N q)` has no included branch, and its excluded branch expands to
  `M(NN q)`.

Thus the combination above maps to zero.

There are exactly `M_{W-2}` independent local directions of this form.  A
canonical quotient basis is

```text
Q_p = all main states
    + blocked states with compressed bit (p-1) != N.
```

Its dimension is

```text
D_W = M_W + M_{W-1} - M_{W-2}.
```

Small-width finite-field rank checks give the same rank for the production
interior step, so the quotient is experimentally minimal as a linear channel.

At `W=28`:

```text
eliminated = M_26 = 47,337,954,326
dimension  = 473,397,057,701
```

This removes `9.0906033%` of the authoritative `main+blocked` coordinates.
One `uint32` stream is `1763.541 GiB`, or `220.443 GiB/GPU` striped over eight
GPUs.  A naive full double buffer would be `440.885 GiB/GPU`, so production
integration still needs streaming, in-place scheduling, or another memory
reduction.

## Signed reduced transition

For a forward interior transition `Q_p -> Q_{p-1}`, run the ordinary production
step and project every destination blocked coordinate whose next lookahead is
`N` by

```text
B_{p-1}(N q) -> M_{p-1}(NN q) - M_{p-1}(LR q).
```

The reduced source fan-out is at most three in exhaustive small-width checks.
All coefficients are `+1` or `-1`; subtraction is therefore the only new
arithmetic compared with the authoritative nonnegative transfer.

The reverse-scan quotient is the exact horizontal-reflection conjugate and has
the same dimension, coefficient set, and observed fan-out bound.

## Physical row boundaries

The reduced blocked channel is needed only in the row interior.  On a forward
row, keep `Q_p` down to `p=2`, then fuse production steps `S_2` and `S_1`.
The result is main-only because the edge step cannot leave a deferred state.
The reverse row is the reflected construction: keep the reduced channel up to
`p=W-2`, then fuse the last two reverse steps.

`gridfp_reduced_production_channel_probe.cpp` checks equality of the complete
forward and reverse row operators on every main basis state at the tested
widths.  Thus snake row boundaries close naturally in the ordinary main space;
no special boundary blocked buffer is required.

## Table-free inverse

A destination-oriented inverse is also table-free.

For a blocked destination, restore the removed site and enumerate:

1. the local deferred `NR/NL` source when present;
2. `LL/RR/RL` closure sources using the existing
   `ordinary_closure_preimages_partial()` implementation.

For a main destination, add:

1. the excluded identity source;
2. the local nonblocked included inverses;
3. a retained blocked source when its excluded expansion equals the destination;
4. signed preimages of a projected noncanonical blocked result.

The reverse inverse is obtained by reflection.  Exhaustive comparison with the
full incoming edge lists passes in the dedicated inverse probe.  The observed
maximum destination indegree is

```text
floor(W/2) + 2
```

through the tested widths, giving `16` at `W=28` if the pattern continues.
This is an observed bound, not yet a proof.

## Component-local execution

The bipartite support graph of one reduced interior step again splits into tiny
balanced components.  Exhaustive enumeration gives

```text
component_count = M_{W-1} - M_{W-3}.
```

A component has a table-free width-`W-1` label `v`: require that the two local
mark positions are not both `N`.

For a forward step:

```text
if v[p-1] != N:
    seed = B(v)
else:
    seed = M(blocked_exclude(v,p))
```

where the second case necessarily has `v[p-2] != N`.  The reverse seed is the
reflected analogue using the next mark on the other side.

Starting at the seed and alternating

```text
source --reduced forward edges--> destination
       <--table-free inverse------
```

reconstructs the complete component without a global adjacency table, CSR, or
component table.  The component-kernel probe verifies that every source and
destination coordinate is covered exactly once and that signed arithmetic
matches the ordinary reduced scatter.

Observed worst-component sizes through `W=12` are bounded by

```text
source/destination pairs <= floor(W/2) + 4
edges                    <= 3 floor(W/2) + 5.
```

If those patterns continue, `W=28` needs at most about `18` source/destination
pairs and `47` signed edges per component.  These are implementation planning
bounds only until a combinatorial proof is added.

For `W=28`:

```text
components = M_27 - M_25
           = 118,389,089,432
average states/component = 3.99865443659
```

## Sector/support factorization

The canonical reduced production space still factorizes by occupied support.
In connectivity sector `p`, the main part contributes

```text
Catalan(p+1) * binom(W, 2p+1),
```

and the retained blocked part has one fixed occupied mark, hence

```text
Catalan(p+1) * binom(W-2, 2p).
```

Therefore

```text
D_{W,p} = Catalan(p+1)
          * [ binom(W,2p+1) + binom(W-2,2p) ].
```

At `W=28` the largest sector is `p=9`:

```text
D_28,9 = 142,248,263,300 values
       = 529.916 GiB per uint32 residue.
```

A production GPU codec should use this support/primitive factorization rather
than generic Motzkin strings in the hot path.

## HBM roofline

If a component-local kernel loads every source count once and stores every
destination count once, the ideal count-array traffic is

```text
8 * D_28 = 3.787176461608 TB / interior step.
```

Across `756` updates per residue at `n=27`, this is about

```text
2.863 PB / residue.
```

At an aggregate `64 TB/s` HBM roofline this is about `44.7 s/residue` before
rank generation, communication, component reconstruction, and kernel
inefficiency.  It is a logical lower-bound model, not a hardware measurement.

## Validation probes

The production-specific probes are:

- `gridfp_reduced_production_channel_probe.cpp`: kernel relation, signed quotient,
  forward/reverse row exactness, and row-edge main-only closure;
- `gridfp_reduced_production_inverse_probe.cpp`: table-free destination inverse;
- `gridfp_reduced_production_component_probe.cpp`: global support-graph structure
  and unique component labels;
- `gridfp_reduced_production_component_kernel_probe.cpp`: local reconstruction
  and one-load/one-store arithmetic check.

These probes are intended to be run locally.  They do not require adding or
repeatedly triggering GitHub Actions workflows.

## Remaining gates

1. Add a small finite-field rank probe to turn the observed minimality into an
   explicit regression test.
2. Prove the component-count formula and the observed component-size bounds.
3. Replace validation `std::map` ranking with the production support/primitive
   rank codec.
4. Implement a CUDA microkernel for one reduced production component and measure
   register pressure, rank-generation cost, and HBM efficiency.
5. Compare complete small-n path counts against the authoritative Grid-FP solver
   before enabling any exact B300 mode.
