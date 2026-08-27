# Two-cell minimal channel

Status: experimental exact reformulation, 2026-08-28.

This note records the canonical `N/R/L` two-cell channel obtained from the
`beta=0` seven-tile transfer.  Unlike the existing GGCount-style production
backend, this formulation has no authoritative blocked/deferred vector.

## Main-space notation

Let `M_W` be the number of one-defect Motzkin words of width `W`, i.e. words in
`N,R,L` whose height starts at one, never goes negative, and ends at zero.
Equivalently,

```text
M_W = sum_p binom(W,2p+1) Catalan(p+1).
```

For width 28,

```text
M_28 = 385,719,506,620
M_27 = 135,015,505,407
M_26 =  47,337,954,326
M_25 =  16,626,415,975
```

## One-cell rank factorization

For active physical sites `i,i+1`, let `T_i` be the ordinary local seven-tile
operator at loop fugacity zero.

It factors exactly as

```text
T_i = E_i R_i.
```

The image channel has two pieces.

- `A_i(w)`, `w in M_{W-1}`: collapse the two active sites to one site.  If the
  collapsed symbol is occupied, `E_i` expands it to the sum of the two physical
  orientations.
- `C_i(u)`, `u in M_{W-2}`: insert a local `LR` cup at the two active sites.

Hence

```text
rank(T_i) = M_{W-1} + M_{W-2}.
```

At `W=28` this is `182,353,459,733` coordinates, or `679.320 GiB` for one
`uint32` residue.

## The only two-cell kernel direction

Before processing the next forward cell `(i+1,i+2)`, the only redundant
one-cell-channel direction is

```text
C_i(N q) - A_i(L R q).
```

Indeed both representatives have the same image under `T_{i+1}`.  Therefore
we quotient by

```text
C_i(N q) == A_i(L R q).
```

A concrete reduced basis is

```text
Q_i = { A_i(w) : w in M_{W-1} }
    + { C_i(u) : u in M_{W-2}, u[i] != N }.
```

Deleting or inserting a fixed `N` preserves Motzkin validity, so exactly
`M_{W-3}` C words are removed.  Thus

```text
R_W := dim Q_i
     = M_{W-1} + M_{W-2} - M_{W-3}.
```

For width 28,

```text
R_28 = 165,727,043,758
```

which is `42.9656890%` of the full main space.  One `uint32` residue occupies
`617.381 GiB`; on eight GPUs this is `77.173 GiB/GPU`, or
`154.345 GiB/GPU` for a simple double buffer.

The difference between the one-cell and two-cell channels is exactly
`M_25 = 16,626,415,975` values at W=28, only `61.938 GiB` total or
`7.742 GiB/GPU`.  This makes a one-cell-channel row-boundary fallback cheap if
needed by the first implementation.

## Reduced transition

Let

```text
Kbar_i = P_{i+1} R_{i+1} E_i : Q_i -> Q_{i+1},
```

where `P` applies the quotient relation above.

Every matrix coefficient is `+1`, and every source has fan-out one, two, or
three.  The local classification is:

```text
source class       fan-out
A00                2
A01                1
A10                3
A11, local LR loop 1
A11, otherwise     2
C                  1
```

Write

```text
c = M_{W-2} - M_{W-3}.
```

There are `c` states of type `A01`, `c` of type `A10`, and `c` retained C
states.  There are `M_{W-3}` `A00` states, and exactly `M_{W-3}` locally paired
`A11` states (insert an adjacent `LR` into a width `W-3` word).  This gives

```text
n1 = 2 M_{W-2} - M_{W-3}
n2 = M_{W-1} - 2 M_{W-2} + M_{W-3}
n3 = M_{W-2} - M_{W-3}
```

for the number of source columns with fan-out 1, 2, and 3.  Therefore

```text
nnz(Kbar_i) = 2 M_{W-1} + M_{W-2} - 2 M_{W-3}.
```

For width 28,

```text
n1  = 78,049,492,677
n2  = 56,966,012,730
n3  = 30,711,538,351
nnz = 284,116,133,190
average source fan-out = 1.71436192155
```

Destination C channels have exactly one predecessor.  The A block therefore
has average indegree

```text
2 - M_{W-3}/M_{W-1},
```

which is `1.87685550689` at W=28.  Exact small-width enumeration currently
shows the maximum destination indegree growing slowly (`7` at W=14); a general
closed-form maximum has not yet been proved and should not be hard-coded from
this observation.

## Sector/support factorization

The reduced state still factorizes by occupied support size.  In sector `p`,
primitive connectivity has dimension `Catalan(p+1)`.  Hence

```text
R_{W,p} = Catalan(p+1)
          * [ binom(W-1,2p+1) + binom(W-3,2p) ].
```

The second term is the retained C block: its lookahead support bit is fixed to
one, leaving `2p` occupied positions among the other `W-3` sites.

For W=28 the largest reduced sector is p=8:

```text
R_28,8 = 50,950,162,120 values = 189.804 GiB/residue total.
```

This support/primitive split is the intended GPU layout; no full Motzkin
rank/unrank is required in the hot path.

## HBM roofline

If destination-owned gather reads each nonzero source once and writes each
reduced destination once, the count-array traffic per cell is

```text
4 * (nnz(Kbar) + R_28) = 1.79937 TB/cell.
```

Relative to the old full-state count this is

```text
4.66497721 bytes / full-state-step.
```

Across 756 updates per residue at n=27, the logical count-array traffic is
about `1.360 PB/residue`.  At an aggregate 64 TB/s HBM roofline this is about
`21.3 s/residue` before communication and kernel inefficiencies.  This is a
roofline, not a hardware measurement.

## Exact CPU validation

`src/cpp/probes/two_cell_channel_probe.cpp` implements the seven-tile operator
directly on canonical link patterns and exhaustively verifies:

1. `T_i = E_i R_i` on every full basis state and every position;
2. `C_i(Nq) == A_i(LRq)` after the next full transfer;
3. the reduced dimension, fan-out and nnz formulae;
4. all reduced coefficients are `+1`;
5. delayed exactness: after a reduced step, applying the following full
   transfer gives exactly the same full vector as consecutive full transfers.

Example:

```bash
g++ -std=c++20 -O2 src/cpp/probes/two_cell_channel_probe.cpp -o /tmp/two_cell_channel_probe
/tmp/two_cell_channel_probe 14
```

The exhaustive CPU check passes through W=14.  It does not use GitHub Actions.

## Remaining implementation gates

- Define the reverse/reflected quotient explicitly for snake rows and verify it
  against the current `J^-1 T_fwd J` convention.
- Decide whether to carry the two-cell quotient across row boundaries or use
  the cheap one-cell-channel boundary fallback.
- Implement destination preimages in support/primitive-factorized form.  C
  destinations are direct copies; only A destinations need variable gather.
- Prove or safely bound the maximum A indegree for W=28 before choosing a fully
  unrolled gather width.
- Compare the CPU reduced solver against the existing exact grid solver on
  complete small-n path counts before starting the B300 CUDA backend.
