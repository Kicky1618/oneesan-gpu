# Generalized strip research

Date: 2026-08-22

This note records generalizations that can be studied on a home machine without pushing the square instance size beyond roughly `n=20`.

## 1. Rectangular strips

Let `W` be the number of grid vertices in the frontier direction and `H` the number of processed rows.  For fixed `W`, the Grid-FP row transfer is a finite linear operator.  Therefore

\[
a_H = \#\{\text{simple paths between opposite corners of the }W\times H\text{ vertex grid}\}
\]

satisfies a finite linear recurrence in `H` over every field.

`src/cpp/probes/rect_sequence_bm.cpp` generates the modular sequence by repeatedly applying one row transfer, recovers the minimal scalar recurrence with Berlekamp--Massey, verifies it on held-out terms, and can evaluate an arbitrarily large `H` by Kitamasa-style polynomial reduction.

Measured over `GF(1,000,000,007)`:

| vertex width W | recurrence degree |
|---:|---:|
| 2 | 1 |
| 3 | 4 |
| 4 | 12 |
| 5 | 27 |
| 6 | 75 |
| 7 | 186 |
| 8 | 472 |
| 9 | 1324 |

For `W=8`, 1400 generated terms give degree 472 and 928 held-out recurrence checks.  After the recurrence is known,

```text
H = 10^18
mod 1,000,000,007
answer = 692391588
query time ~= 82 ms
```

For `W=9`, 5000 terms stabilize at degree 1324 with 3676 recurrence checks, and

```text
H = 10^18
mod 1,000,000,007
answer = 660245003
query time ~= 625 ms
```

The `W=8` degree is also 472 over `GF(998244353)`, so the observed degree is not a special rank drop modulo `1,000,000,007`.

Small-width characteristic polynomials factor nontrivially.  For example,

\[
P_3(x)=(x^2-3x-1)(x^2-x+1),
\]

and the degree-12 `W=4` polynomial factors into two degree-6 factors.  `W=5` factors as degree 13 times degree 14.  A reflection-sector explanation is a natural next target.

## 2. Bounded-height support in the first rows

After `r` complete Grid-FP rows, experiments show that the nonzero main states are exactly the defect-Motzkin words whose maximum height is at most `r`.  The same height bound is preserved at every position inside row `r`.

Thus the support size is

\[
B(W,r)=(A_r^W)_{1,0},
\]

where `A_r` is the `(r+1) x (r+1)` tridiagonal matrix with ones on the diagonal and the two adjacent diagonals.  Its largest eigenvalue is

\[
1+2\cos\frac{\pi}{r+2}.
\]

For `W=28`:

| height cap r | states | fraction of full main space |
|---:|---:|---:|
| 1 | 134,217,728 | 0.035% |
| 2 | 18,457,556,052 | 4.79% |
| 3 | 112,925,875,764 | 29.27% |
| 4 | 240,539,369,472 | 62.36% |
| 5 | 329,056,985,516 | 85.31% |

This supports a compact-prefix/full-factorized hybrid for B300: the first three rows can be processed without allocating the full main state.

## 3. r-row boundary coefficient Hankel rank

Let `F_r(w)` be the coefficient attached to frontier word `w` after exactly `r` rows, extended by zero to invalid words.  Split a word as `w=uv`.  The matrix

\[
H_r(u,v)=F_r(uv)
\]

is block diagonal by the Motzkin height `h` at the cut.

### Experimental rank identity

For sufficiently long words and an interior cut, the measured rank of the height-`h` block is

\[
\rho(r,h)=[x^{r-h}](1-2x-3x^2)^{-(h+1)/2}.
\]

These are exactly the entries `T(r,h)` of OEIS A111960, the renewal/convolution triangle for the central trinomial coefficients.  Consequently,

\[
\rho(r)=\sum_{h=0}^r \rho(r,h)
=[x^r]\frac{1}{\sqrt{1-2x-3x^2}-x},
\]

which is OEIS A111961.

Measured height-block ranks:

```text
r=1:   1, 1                                      sum=2
r=2:   3, 2, 1                                   sum=6
r=3:   7, 7, 3, 1                                sum=18
r=4:  19,20,12,4,1                               sum=56
r=5:  51,61,40,18,5,1                            sum=176
r=6: 141,182,135,68,25,6,1                       sum=558
r=7: 393,547,441,251,105,33,7,1                  sum=1778
```

For `r=7`, the coefficient set at `W=17` contains 4,179,451 nonzero words; the rank calculation still gives the row above exactly over `GF(1,000,000,007)`.

References:

- OEIS A111960: <https://oeis.org/A111960>
- OEIS A111961: <https://oeis.org/A111961>
- Goy--Shattuck, *Determinants of Some Hessenberg-Toeplitz Matrices with Motzkin Number Entries*, JIS 26 (2023).  The paper gives a combinatorial interpretation of A111961 as Motzkin paths in which a subset of steps terminating on the x-axis is marked.



### Generic weights, obstacles, and a second factorization clue

The rank triangle is not an artifact of unit weights.  `rowr_coeff_weighted.cpp`
assigns independent-looking position-dependent nonzero weights to included
horizontal transitions.  Sparse Gaussian elimination still gives exactly the
same height-block ranks:

```text
r=4, W=11: 19,20,12,4,1       sum=56
r=5, W=13: 51,61,40,18,5,1    sum=176
r=6, W=15: 141,182,135,68,25,6,1  sum=558
```

By contrast, a deterministic obstacle mask at `r=4,W=11` lowered the blocks to
`8,8,4,1` (sum 21).  This is strong evidence for the more robust statement:

> `T(r,h)` is the universal separator-rank upper bound for arbitrary local
> weights, and generic nonzero weights attain the bound.  Specializations such
> as obstacles can only lower rank.

`row_hankel_pivots.cpp` now emits a deterministic set of pivot prefix/suffix
pairs for each height block.  For `r=2,3`, increasing both sides of the cut
simply pads the pivot words by `N`, so the observed nonzero minor is visibly a
finite local object.  At `r=4`, most pivots stabilize in the same way; a few
lexicographically last pivots move with the far boundary, so a boundary-free
canonical choice still needs to be constructed rather than read directly from
naive elimination.

There is also a useful algebraic factorization already recorded for A111960:

\[
T = \text{Pascal}\cdot U,
\]

where `U=A111959` is the renewal array for the aerated central binomial
coefficients.  In Riordan-array notation,

\[
\left(\frac1{\sqrt{1-2x-3x^2}},
      \frac{x}{\sqrt{1-2x-3x^2}}\right)
=
\left(\frac1{1-x},\frac{x}{1-x}\right)
\left(\frac1{\sqrt{1-4x^2}},
      \frac{x}{\sqrt{1-4x^2}}\right).
\]

Equivalently,

\[
T(r,h)=\sum_{j=h}^{r}\binom rj U(j,h),
\]

with

\[
U(h+2d,h)=4^d\frac{((h+1)/2)_d}{d!},\qquad
U(h+2d+1,h)=0.
\]

This suggests a more concrete proof model: first choose the rows that are
`idle` across the separator (the Pascal factor), then encode the remaining
creation/annihilation events by the aerated-central-binomial basis.  It is a
promising route to an explicit invertible minor because it separates harmless
flat rows from the genuinely topological events.

References:

- OEIS A111960: https://oeis.org/A111960
- OEIS A111959: https://oeis.org/A111959



### GPU endcap compilation from the Hankel ranks

The fixed-row rank calculation is now directly useful as a GPU optimization.
Instead of running ordinary frontier DP for the first few rows, compile the
exact `r`-row transfer into its minimal separator state space and evaluate it
with a meet-in-the-middle lookup table.  The dimensions used here are exactly
the observed A111961 totals:

```text
r=2:   6 states
r=3:  18 states
r=4:  56 states
r=5: 176 states
```

On the local RTX 3070, with one modulus `4294967291`, identical factorization
parameters, and checking that every variant returns the same residue, the
progression is:

```text
n=20 (W=21)
row2 direct                 16.0105 s   residue 2308006916
row3 compact-LUT            15.2813 s   residue 2308006916
row4 compact-LUT            14.5239 s   residue 2308006916
row5 compact-LUT            13.9956 s   residue 2308006916

n=21 (W=22)
row2 direct                 44.8589 s   residue 998035516
row3 compact-LUT            43.1793 s   residue 998035516
row4 compact-LUT            39.8964 s   residue 998035516
row5 compact-LUT            39.4684 s   residue 998035516
```

The exact timings vary with GPU clocks, so adjacent variants should be treated
as approximate unless measured in one alternating run.  The robust conclusion
is structural: skipping more initial row transfers continues to pay through
`r=5`, provided the separator automaton itself is evaluated by a split lookup
rather than scanning all `W` symbols independently for every bounded state.

A plain rank-18 `r=3` automaton demonstrated the failure mode at `n=20`: it
reduced the later transfer work by about one second but spent almost the same
time in initialization.  Splitting the word into prefix and suffix tables
removed almost all of that overhead.  The same idea scales to `r=4` and `r=5`.
For `r=5`, the full separator dimension is 176, while the compact table path
uses 61 relevant intermediate components, keeping table construction practical.
A four-lane evaluation prototype was slower (`39.8427 s` at `n=21`) than the
ordinary compact evaluator (`39.4684 s`) and is therefore not the current
choice.

This suggests an optimization principle stronger than merely "use a deeper
bounded prefix":

> Compute the finite-rank Hankel representation of the first `r` rows, minimize
> its separator basis, then compile that representation into a two-sided GPU
> lookup table.  Increase `r` only while the saved full-frontier row transfers
> dominate the growth of the separator representation.

The `r=6` experiment was then run with the existing packed height-block
representation.  It stores at most the largest height block (182 components)
per half-word rather than a dense 558-vector.  On `n=20`, modulus `1000000007`,
the comparison in the same reverse2 prototype was:

```text
r=5 packed endcap: active 10.9551 s, wall 11.4414 s
r=6 packed endcap: active 10.3186 s, wall 11.8387 s
```

Thus `r=6` does save another two full transfer windows, but its initializer is
currently too expensive: about `0.86 s` host LUT construction plus `0.64 s`
GPU initialization.  Increasing host construction to 20 threads reduced LUT
build time to about `0.56 s`; the best tested configuration (4 lanes) reached
`active=10.2821 s, wall=11.4985 s`, only about 0.5% slower than the `r=5` wall
time.  Sixteen lanes was clearly worse.  So `r=6` is very close to crossing
the practical threshold, and the remaining target is specifically the endcap
initializer rather than the subsequent frontier DP.

The conclusion above was superseded by the later row-6 implementation work.
The bounded frontier splits exactly into height-graded Cartesian products
`P_h x S_h`; for n=20 their total pair count is 256,377,982, exactly the bounded
row-6 state count.  A first exact INT8/cuBLAS realization proved that the blocks
can be evaluated as matrix products, but on the RTX 3070 the existing Cartesian
modular-dot kernel is faster for these small separator dimensions.

The larger win was in table construction.  Replacing the old `one child = one
thread` expansion by `one child = one block`, with separator components spread
over threads, reduced the GPU row-6 LUT construction substantially.  The final
production CRT path keeps exact uint32 modular arithmetic, uses a true-packed
height-level representation rather than 182 padded components per ternary word,
and supports per-height Cartesian lane counts.  Alternating n=20 measurements
moved the practical fold point from r=5 to r=6: representative runs were roughly
10.55--10.63 s for r=5 versus 9.84--9.92 s for the optimized r=6 path, with the
same residue.

For B300-scale n=27, the true-packed representation is also a memory optimization:
the row-6 intermediate estimate falls from about 5.05 GiB/GPU to about 1.26 GiB/GPU.
The factorized DP rank tables are a separate memory issue.  A CUDA-VMM variant now
keeps the original direct `rank[4ary_code]` device access while physically mapping
only 2-MiB pages that contain valid Motzkin codes.  At LOW14/HIGH13 this maps about
324 MiB + 88 MiB instead of 1024 MiB + 256 MiB, with no measurable RTX-3070
throughput loss in n=20/21 regressions.  The host rank construction was made sparse
as well, reducing n=27 PLAN-process RSS from about 1.40 GiB to about 167 MiB.

A later sparse-orbit formulation attacks the four-Count transfer scratch as well.
The upper factor window retains the fast reverse2 kernels; the lower window is an
in-place orbit over one main and one blocked Count array. Enumerating only the LOW
ranks whose pair classes can act as orbit owners/closures reduces the logical scan
from two full main-state passes per cell to roughly 65% of one pass. The decisive
optimization is to precompute LOW-factor destination ranks: all owner transitions
and LOW/center-local closures become packed rank-table operations, while only RR
closures crossing into the HIGH factor keep the generic rank path.

On the RTX 3070 at `n=21, p=4294966997`, the previous VMM reverse2 path measured
`wall=23.783 s, transition=14.006 s`; the HIGH-RR pre-ranked sparse orbit measured
`wall=20.815 s, transition=11.133 s`, with the same residue `2124618149`. At `n=20`
the paths are approximately tied, suggesting that the pre-ranking payoff grows with
frontier width. The n=27 x8 PLAN uses about 244.62 MiB/GPU of pre-rank tables; removing the lower-window MateID cache
reduces actual ordinary-DP peak from about 257.69 GiB/GPU to 252.80 GiB/GPU.


#### Full pre-ranked two-window orbit

The lower-window pre-ranked orbit was generalized symmetrically to the upper
window.  Upper owner transitions and closures contained in HIGH are pre-ranked;
only LL closures crossing from HIGH into LOW require a short LOW-code scan.  A
GPU validation probe compared every upper pre-ranked owner/fast-closure/cross-LL
record against the generic `include_horizontal` + factor-rank path at n=20.  This
probe caught and fixed an enum-shadowing bug in the first cross-LL kernel and now
passes completely.

With both windows in-place, ordinary Count scratch at n=27 falls to about
7.385 GiB/GPU.  The complete pre-rank tables are about 393.7 MiB/GPU and the
2-MiB-VMM PLAN gives an ordinary-DP peak of about 250.687 GiB/GPU.  RTX 3070
regressions reproduce the known CRT residues; representative n=21 timing is
`wall=20.815 s, transition=11.133 s`, versus roughly 24 s wall for the earlier
reverse2 path under comparable conditions.

#### Row-7 dense Tensor-Core endcap

The `r=7` separator has height-block dimensions
`393,547,441,251,105,33,7,1`, total dimension `1778`.  A `W=17` cut at `8|9`
realizes all 1778 dimensions, and `W=18` exact boundary coefficients are enough
to construct one-symbol transitions.  The generated transitions are almost
fully dense (`N: 723176/723304`, `R: 596926/596947`,
`L: 596563/596947` nonzero entries), so extending the sparse row-6 kernels is
not appropriate.  The natural implementation is dense matrix multiplication
grouped by height and symbol.

The exact `W=17/18` boundary coefficients fit in unsigned 128-bit integers
(maximum observed width 83 bits).  The production representation therefore
stores only the exact values needed to rebuild the separator basis and
transitions in `row7_exact_compact_u128.bin` (about 41 MiB).  For an arbitrary
32-bit CRT prime the 1778-dimensional modular representation is reconstructed
from this table.  On the RTX 3070 this reconstruction takes about 3.0 s, mostly
in the eight finite-field basis inversions.  The resulting dense modular table
is cached by exact-table fingerprint and modulus; one cache entry is about
7.67 MiB.  Across a CRT batch, construction of the next uncached modulus is
also pipelined on the CPU while the current modulus runs on the GPU.

Dense finite-field multiplication is implemented exactly on INT8 Tensor Cores.
Each uint32 coefficient is decomposed into four base-256 digits, each digit is
centered to signed int8 by subtracting 128, and the 16 digit-pair GEMMs are
combined by exponent.  Row/column correction terms undo the centering, and a
base-256 Horner reconstruction with modular reductions after exponents 3 and 0
returns the exact result modulo the current CRT prime.  This works unchanged
for the production primes immediately below `2^32`.

For `n=21` (`W=22`, split `11|11`), the packed prefix and suffix tables occupy
about 104.91 MiB and 59.46 MiB.  The endcap represents 728,148,564 bounded
states.  With a warm modular-table cache, the integrated reverse2 solver on an
RTX 3070 measured approximately:

```text
row-5 baseline, p=4294967279: wall 27.2299 s, residue 1882054188
row-7 Tensor,  p=4294967279: wall 24.3312 s, residue 1882054188
row-7 Tensor,  p=4294967291: wall 24.2784 s, residue 998035516
```

Thus the row-7 path is about 10.5% faster per warm/pipelined residue in this
configuration.  A cold modular table costs about three seconds only once; in a
multi-prime run, the next table was ready with only a few microseconds of wait.
The full `b300x8-exact.sh` -> CRT runner -> checkpoint path was also smoke-tested
at `n=21`, preserving the same residue.

For multi-GPU initialization, the full rank factors as
`prefix_base + suffix_rank`.  Each GPU therefore selects only prefix rows whose
possible suffix-rank interval intersects its authoritative full-rank shard, and
runs Tensor GEMMs only for those rows.  A two-shard emulation on one RTX 3070
reconstructed the same final residue.  At `n=27` (`W=28`, split `14|14`) there
are 382,187,801,740 cap-7 states and the unsharded row-7 join contains about
118.96 trillion logical finite-field MACs.  With eight equal full-rank shards,
the selected-row calculation gives about 13.09--15.86 trillion logical MACs per
GPU, close to ideal 1/8 partitioning and without inter-GPU communication during
the endcap construction.

#### Row-8 runtime-generated Tensor-Core endcap

For `r=8`, the measured W=19 Hankel blocks at cut `9|10` have dimensions

```text
[1107, 1640, 1428, 888, 420, 152, 42, 8, 1]
```

with total dimension `5686`, matching the A111960/Riordan prediction.  W=18 is
too small to contain all of these ranks, so W=19 is the first viable cut.  The
one-symbol automaton has 18,682,094 uint32 transition coefficients; 18,680,504
were nonzero in the `p=1000000007` construction, a density of about 99.9915%.
This makes dense Tensor-Core linear algebra even more strongly preferred than at
row 7.

Unlike the row-7 implementation, row 8 does not keep an exact or pre-generated
large modular transition table in the production dependency chain.  The only
persistent basis artifact is `row8_pivots_w19.bin` (45,536 bytes), containing the
selected prefix/suffix words.  For each modulus, GPU cap-8 frontier DP generates
four small-width coefficient arrays:

- W=9 for the terminal vector;
- W=10 for the initial residual vector;
- W=19 for the square basis blocks B_h;
- W=20 for the one-symbol extension blocks V_{a,h}.

The modular representation is then computed as
`M_{a,h} = V_{a,h} B_{h+delta(a)}^{-1}`.  Gaussian elimination and the same exact
base-256 INT8 Tensor-Core multiplication used by row 7 are runtime-modulus
aware, including primes immediately below 2^32.  At `p=4294967291`, all nine
basis matrices were invertible and all 25 checks `M B = V`, plus the initial
vector check, passed.  On the RTX 3070 the complete uncached modular automaton
build took about 4.3--4.7 s; the dominant work was the W=19/W=20 cap-8 frontier
DP rather than the 1640-dimensional finite-field inversion.

The resulting modular tables are cached by modulus, pivot fingerprint, and a
semantic builder ABI.  The cache payload is hashed and is about 74,739,672 bytes
per prime.  Warm cache loading removes the 4-second construction cost.  A test
that flipped one byte in a cache entry caused a cache miss, full regeneration,
and the same final residue.

At n=21, the integrated row-8 path was compared against row 7 at
`p=1000000007`:

```text
row-7 warm: active 23.2108 s, wall 24.1407 s, residue 745080216
row-8:      active 21.4357 s, wall 23.2118 s, residue 745080216
```

Thus skipping the eighth ordinary row saves about 2.09 s of subsequent DP and
still wins by about 0.93 s wall-clock despite the larger row-8 initializer.  At
the production prime `4294967291`, a warm row-8 run produced the previously
cross-checked residue `998035516` with wall time around 23.6 s on the RTX 3070.
A separate n=9 regression compares the production row-8 solver against the CPU
exact value `41044208702632496804`, whose residue modulo `4294967291` is
`2674633373`; cold build, warm cache, and corrupted-cache rebuild all reproduce
that residue.

The finite-window closure check was also extended one full symbol beyond the
transition-construction window.  A W=21 cap-8 array has 258,205,711 states.  For
every valid height block, every ordered symbol pair `(a,b)`, and every pair of
selected basis words, the directly generated coefficient
`f(prefix_i a b suffix_j)` was compared against `M_a M_b B`.  This checks
50,664,348 coefficients rather than samples.  All entries agreed for both
`4294967291` and `4294967279`; the raw/predicted FNV hashes were respectively
`7b13c84631d79141` and `fc090d3a92dbc852`.  On the RTX 3070 the W=21 compact
coefficient array itself took about 7.51 s to generate and 0.26 s to copy.

This depth-2 result is strong evidence for closure but is deliberately not
promoted to a proof: a finite number of extension depths does not by itself
establish the all-word Hankel-rank upper bound.

For eight-way rank sharding, prefix generation is pruned from the final shard
requirements backwards through every prefix level.  In an n=21 fake-eight-shard
run the final prefix sets contained about 10.9k--13.4k words per shard and the
largest per-shard row-8 initializer took about 0.92 s on one RTX 3070.  For
n=27, static counting predicts roughly 0.82--0.98 GiB of pruned prefix vectors
and about 4.09 GiB of suffix vectors per GPU; the dense join is roughly
43--51 trillion logical finite-field MACs per GPU.  The current W=28 production
source compiles successfully for `sm_100a`, but real B300 x8 measurements are
still pending.

The important exactness caveat is that these finite-field checks establish a
consistent 5686-state realization on the tested finite Hankel windows, not the
all-word Hankel-rank upper bound.  Until the closure/rank theorem below, or an
equivalent independently checkable certificate, is completed, row 8 remains a
modular benchmark path and is rejected by `b300x8-exact.sh`.

### Conjecture / proof target

The observed identity should be promoted to a theorem:

> For fixed `r,h`, once both sides of the word cut are sufficiently long, the height-`h` Hankel block of the `r`-row Grid-FP boundary coefficient function has rank `T(r,h)` over characteristic zero and over all but finitely many primes.

The Riordan-array formula is

\[
T(r,h)=[x^{r-h}] C(x)^{h+1},\qquad
C(x)=\frac{1}{\sqrt{1-2x-3x^2}},
\]

so `T(r,h)` is an `(h+1)`-fold convolution of the central trinomial coefficients.  This strongly suggests a cut decomposition into `h+1` independent grand-Motzkin sectors, followed by a nondegeneracy argument for the lower bound.

## 4. Algorithmic meaning

The Hankel rank is the minimum bond dimension of an exact linear representation of the fixed-`r` boundary coefficient tensor.  Therefore the first few complete rows can be compiled into small weighted automata instead of being executed as ordinary frontier-DP sweeps.

Observed minimal dimensions:

```text
r=1:   2
r=2:   6
r=3:  18
r=4:  56
r=5: 176
r=6: 558
r=7: 1778
r=8: 5686
```

The `r=2` six-state automaton has already been converted into a CUDA direct initializer and is roughly four times faster than executing the first two bounded-DP rows on the RTX 3070 development machine.

The dimensions grow as A111961, asymptotically about `(1+sqrt(5))^r/sqrt(5)`, so direct compilation is most attractive for small endcaps (`r=2,3,4`) rather than for a macroscopic fraction of the grid.

## 5. Next proof and implementation tasks

1. Prove the bounded-height support invariant, ideally in Lean.
2. Construct an explicit `T(r,h)`-dimensional cut representation and prove the A111960 Hankel-rank upper bound.
3. Exhibit a nonzero `T(r,h) x T(r,h)` minor (or an invertible pairing) for the matching lower bound.
4. Derive the reflection-sector decomposition of the fixed-width row transfer and explain the factorization of the scalar recurrence polynomial.
5. Turn the fixed-width recurrence probe into a reusable rectangular-strip solver with recurrence serialization and CRT/exact reconstruction for moderate widths.


### Group runtime constant batching (2026-08-27)

The full-pre-rank path originally issued many synchronous `cudaMemcpyToSymbol` calls for every forced group (DP tables, masks, widths, factor blocks, and factor mode).  On RTX 3070 / n=21, profiling split the old gather-side time into roughly 2.73 s of group setup and 2.84 s of actual gather I/O.

The production `ROW6_PRERANK=1` source now packs all group-dependent constant state into one `GroupRuntimeCfg` and uploads it with a single `cudaMemcpyToSymbol` per group.  The compact inverse-closure representation remains unchanged.  An alternating same-GPU A/B against the immediately preceding compact-inverse source gave 22.48--22.61 s vs 20.15--20.23 s wall time at n=21 (about 10.7% faster overall); absolute times remain clock/thermal dependent.  The n=20/n=21 known-residue regression and a second CRT prime still pass.

The n=27 / 8-GPU memory plan is unchanged by this optimization: 355 MiB/GPU pre-rank LUT, 0.779 GiB/GPU factor data, 7.385 GiB/GPU Count scratch, and about 250.650 GiB/GPU ordinary-DP peak with 2 MiB VMM granularity.

