# Occupancy-block layout for the fusion backend

The fusion backend naturally separates an occupancy mask from the dense one-defect
fusion topology.  Use the authoritative ordering

```text
ID = base[popcount(mask)]
   + colex_rank(mask) * Catalan((popcount(mask)+1)/2)
   + fusion_rank.
```

Every exact occupancy mask is therefore one contiguous Catalan block.  This is useful
both for the normal Grid-FP sweep and for the point-symmetry midpoint closure.

## Transition orbit

At physical update position `p`, remove the two occupancy bits `(p-1,p)` and call the
remaining pattern the rest mask.  One rest mask gives one independent three-block
orbit.

If the rest population is odd:

```text
A = main 00
B = main 11
D = blocked compressed state

B += Insert(A)
A += D
D  = Cap(B)
```

`Cap * Insert = 0` at beta=0, so this ordering is exactly in-place even though `B`
has already been updated.  The three vectors have dimensions

```text
dim(A) = dim(D) = Catalan((m+1)/2)
dim(B)          = Catalan((m+3)/2)
```

for odd rest population `m`.

If the rest population is even:

```text
X = main 01
Y = main 10
D = blocked compressed state

oldY = Y
Y += X + D
D  = oldY
```

and all three topology vectors have the same Catalan dimension.  The `p=1` endpoint
uses the analogous two local occupancy orbits and has already been checked separately
in the research probes.

Thus the hot transition kernel can work one occupancy orbit at a time.  The dense
insertion position `i = popcount(mask below p-1)` is constant for the whole block, so
there is no per-state occupancy decoding, MateID construction, or distant mate search.

## LOW14 / HIGH13 group I/O at width 28

A LOW14 coarse group fixes 14 occupancy bits and leaves 14 free.  Exactly 8192 free
masks have odd total occupancy.  A HIGH13 group fixes 13 bits and leaves 15 free, so it
contains 16384 odd masks.

The largest main groups occur when all fixed bits are occupied.

```text
LOW14, k_fixed=14
  main states       = 961,466,716
  exact blocks       = 8,192
  average block      = 117,366.54 states = about 458 KiB of uint32
  smallest block     = 1,430 states
  largest block      = 2,674,440 states
  total main bytes   = 3.582 GiB

HIGH13, k_fixed=13
  main states       = 1,471,935,235
  exact blocks       = 16,384
  average block      = 89,839.80 states = about 351 KiB of uint32
  smallest block     = 429 states
  largest block      = 2,674,440 states
  total main bytes   = 5.483 GiB
```

Even moderate fixed populations give long runs.  For example a LOW14 group with seven
fixed occupied bits averages about 1,486 states per exact block; the worst low-density
groups are small in total volume.

A gather/scatter implementation therefore needs only a small block descriptor such as

```text
{remote_global_base, local_orbit_base, length, owner_gpu}
```

per exact occupancy block.  A warp or CTA can copy each run coalesced.  This removes the
current need to recover a MateID and canonical Motzkin rank for every individual state.

Because an authoritative exact block is contiguous, a global 8-GPU shard boundary can
split at most one such block at each boundary.  The same block descriptors can be reused
for all CRT primes.

## Scratch

The beta=0 orbit identity `Cap*Insert=0` permits one main and one blocked scratch array,
`M+D`.  Using the current n=27 LOW14/HIGH13 maximum group sizes, the research estimates
are approximately

```text
LOW14  : 4.823 GiB count scratch
HIGH13 : 7.385 GiB count scratch
```

No full-Mate cache is needed.  The topology metadata is instead a roughly 14-MiB
fusion-word unrank table plus small rank automata and local height-dependent coefficient
tables.
