# OneesanFormal

Lean 4 formalization for the correctness-critical parts of the Oneesan Grid-FP / exact-counting pipeline.
The project is pinned to `leanprover/lean4:v4.34.0-rc2`; `lakefile.toml` and `lake-manifest.json`
pin the same mathlib revision.

## Proved modules

- `OrbitTransfer.lean`: exact weighted quotient-transfer recurrence and total-count
  preservation for finite symmetry partitions; includes the four-map
  complement/reflection recipe. Its application assumptions and the Python
  implementation correspondence are detailed in `docs/research/strip-orbit-quotient.md`.

- `ReverseMainCore.lean`: inverse/main transition tables used by the reverse gather kernel, including
  soundness/completeness and injectivity of the main-to-main included branch table.
- `BoundedHeight.lean`: local height/peak properties used by bounded-state optimizations.
- `RankSplit.lean`, `GapHistory.lean`, `Cotree.lean`: supporting combinatorial/rank facts used by the
  optimized frontier representation and reverse analysis.
- `MmapResume.lean`: four-phase durable transaction model for one transition-closed mmap group.
  `resume_eq_uninterrupted` proves restart equivalence at every crash point and
  `resumeAll_eq_uninterrupted` lifts it to arbitrary out-of-order independent groups. It also proves
  the fixed-occupancy group partition is disjoint and covering.
- `ExactCRT.lean`: uniqueness of a CRT representative below a rigorous bound once the modulus product
  exceeds that bound. This justifies the final `exact <= path_bound` acceptance rule.
- `CheckerboardBound.lean`: abstract checkerboard obstruction and the cardinality/injection step used by
  the path-count upper bound. It also proves the strip relaxation: restricting a globally valid face
  assignment to independent row strips is injective, so dropping cross-strip constraints gives an upper
  bound equal to the product of the per-strip valid counts.
- `GridFaceBoundary.lean`: constructive rectangular-grid cycle-space theorem over GF(2). It recovers a
  bounded-face potential by prefix XOR/discrete integration, proves every even edge assignment with zero
  ghost boundary is exactly a bounded-face boundary, and proves that two T-joins with the same terminal
  parity have such a representation after XOR. This supplies the previously missing `P XOR P0` topology step.
- `CheckerboardGridBound.lean`: connects the concrete `RectBoundaryWitness` indexing to the local 2x2
  checkerboard obstruction. `outer_reference_xor_has_checkerboard_free_face_encoding` packages the chain
  from equal terminal parity + an outer-boundary reference + path degree at most two to a checkerboard-free
  bounded-face encoding.
- `PathDegree.lean`: uses mathlib's `SimpleGraph.Walk.IsPath` theory to prove every vertex of the subgraph
  traced by a simple path has neighbor-set cardinality at most two.
- `GridGraph.lean`: defines the concrete `(n+1) × (n+1)` rectangular grid graph and proves both the explicit
  four-direction adjacency law and the exact four-neighbor set of every interior vertex.
- `GridBoundaryNeighbors.lean`, `PathGridEncoding.lean`, `PathGridParity.lean`, and `OuterReference.lean`:
  prove the complete coordinate bridge from mathlib simple paths to the production H/V ghost-edge indexing,
  including all four corners, four boundary sides, interior vertices, and the fixed outer reference path.
- `CornerPathBound.lean`: proves every simple northwest-to-southeast grid path has a checkerboard-free bounded-face
  encoding, with no residual topology/parity hypothesis.
- `PathEdgeSet.lean`, `PathHVInjective.lean`, `RectBoundaryInjective.lean`, and `FacePathInjective.lean`: prove the
  inverse side of the encoding. A fixed-endpoint simple path is determined by its undirected edge set; H/V determines
  that edge set; and a bounded-face witness determines its real H/V boundary. Hence equal chosen face encodings imply
  equal paths.
- `CornerPathCounting.lean`: packages the above as an actual finite-cardinality theorem
  `#(NW→SE simple paths) ≤ #(checkerboard-free n×n face matrices)`.
- `StripBound27.lean`: specializes the strip relaxation used by the production n=27 run, proving an injection from
  checkerboard-free `27×27` face matrices into three checkerboard-free `9×27` strips.
- `StripDP.lean`: gives the semantic transfer-DP for checkerboard-free strips and proves its recursive state counts
  equal the finite cardinalities of valid column chains. In particular, valid `9×27` strips are bounded by the
  26-transition DP total.
- `BitColumnDP.lean`, `TableDP.lean`, `LoopDP.lean`, and `RawDP.lean`: refinement chain from the semantic column DP to
  increasingly execution-oriented implementations. `RawDP` uses only integer state numbers, `Nat.testBit`, and
  allocation-free finite folds, while proving cellwise and total equality back to the semantic DP.
- `FastDP9.lean`, `CompressedDP9.lean`, and `CompressedTableDP9.lean`: execution refinements specialized to the
  production 9-row strip. A finite 512² proof validates an O(1) bitwise compatibility formula, complement symmetry
  reduces 512 states to 256, and a precomputed 256×256 weight table makes native evaluation cheap. The evaluated
  9×27 strip count is
  `3165928478117342768922265826341920493835329849417470440184018662`.
- `ProductionBound27.lean`: composes all preceding injections/refinements and proves the end-to-end numerical theorem
  `card_corner_paths_27_le_pathBound27`. The resulting production upper bound is the cube of the verified strip count,
  a 633-bit integer:
  `31732427633797389964407887052573851640105323179333844527763421102211579188310190597412900756550874123129342585261840138964010190312364508107168927006232277067783279927977876779029754107293528`.

Together these modules close the mathematical topology/local-obstruction chain behind the production
checkerboard upper bound and formally connect the n=27 strip computation to the same semantics and exact numerical
bound. There are no `sorry`, `admit`, or custom axioms in the formal source.

## Trust boundary for concrete evaluation

The topology, injectivity, strip-relaxation, and DP-refinement theorems are ordinary Lean proofs checked by the
kernel. A small number of finite specialization facts and the large 9×27 DP constant use `native_decide` so that
512²/256² finite computations finish quickly. Those particular proof steps additionally trust Lean's native-code
compiler/execution path; `native_decide` is therefore not described here as pure kernel reduction. The final cube
identity is proved with `norm_num`, and the Python correctness suite independently recomputes the same 9×27 strip
count, `[9,9,9]` partition, and 633-bit n=27 bound and checks that the constants in `ProductionBound27.lean` match.

## Filesystem refinement obligations

The mmap theorems deliberately separate the pure transaction state machine from OS durability. The C++
layer must establish that the undo journal is durable before in-place writes, an uncommitted journal
restores the exact old bytes, touched mmap ranges are durable before the completion bit, and different
transition-closed groups touch disjoint authoritative ranges. These obligations are exercised by:

- `tests/mmap_resume_test.cpp`
- `tests/mmap_resume_corruption_test.cpp`
- `tests/mmap_resume_e2e_test.cpp`
- the production Grid-FP partition self-test (`824` cases / `14,196` groups)
- the GPU fault-injection integration script when an NVIDIA GPU is available.

From `formal/`, run:

```bash
lake build
```
