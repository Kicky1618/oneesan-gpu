import OneesanFormal.GridGraph
import OneesanFormal.PathDegree
import OneesanFormal.CheckerboardGridBound
import Mathlib.Data.Set.Card

namespace OneesanFormal.PathGridEncoding

open SimpleGraph
open OneesanFormal.GridGraph
open OneesanFormal.PathDegree
open OneesanFormal.GridFaceBoundary
open OneesanFormal.CheckerboardGridBound

/-- GF(2) membership bit for one undirected edge of a walk's traced subgraph. -/
noncomputable def pathEdgeBit {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (a b : GridVertex n) : F2 := by
  classical
  exact if p.toSubgraph.Adj a b then 1 else 0


/-- At an interior grid vertex, the four directional edge bits count exactly
its neighbors in the subgraph traced by the walk. -/
theorem interior_four_bits_eq_neighbor_ncard
    {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v)
    (r c : Nat) (hr : r + 1 < n) (hc : c + 1 < n) :
    let center : GridVertex n :=
      (⟨r + 1, by omega⟩, ⟨c + 1, by omega⟩)
    let left : GridVertex n :=
      (⟨r + 1, by omega⟩, ⟨c, by omega⟩)
    let right : GridVertex n :=
      (⟨r + 1, by omega⟩, ⟨c + 2, by omega⟩)
    let up : GridVertex n :=
      (⟨r, by omega⟩, ⟨c + 1, by omega⟩)
    let down : GridVertex n :=
      (⟨r + 2, by omega⟩, ⟨c + 1, by omega⟩)
    bitDegree (pathEdgeBit p center left) +
      bitDegree (pathEdgeBit p center right) +
      bitDegree (pathEdgeBit p center up) +
      bitDegree (pathEdgeBit p center down) =
      (p.toSubgraph.neighborSet center).ncard := by
  classical
  dsimp
  let center : GridVertex n :=
    (⟨r + 1, by omega⟩, ⟨c + 1, by omega⟩)
  let left : GridVertex n :=
    (⟨r + 1, by omega⟩, ⟨c, by omega⟩)
  let right : GridVertex n :=
    (⟨r + 1, by omega⟩, ⟨c + 2, by omega⟩)
  let up : GridVertex n :=
    (⟨r, by omega⟩, ⟨c + 1, by omega⟩)
  let down : GridVertex n :=
    (⟨r + 2, by omega⟩, ⟨c + 1, by omega⟩)
  have hfull : (gridGraph n).neighborSet center = {left, right, up, down} := by
    simpa [center, left, right, up, down] using neighborSet_interior hr hc
  have hsub : p.toSubgraph.neighborSet center ⊆ {left, right, up, down} := by
    intro x hx
    have hg : (gridGraph n).Adj center x := p.toSubgraph.adj_sub hx
    have : x ∈ (gridGraph n).neighborSet center := hg
    rw [hfull] at this
    exact this
  have hfinite : (p.toSubgraph.neighborSet center).Finite :=
    Set.Finite.subset (Set.toFinite {left, right, up, down}) hsub
  rw [Set.ncard_eq_toFinset_card _ hfinite]
  have hfinset : hfinite.toFinset =
      ({left, right, up, down} : Finset (GridVertex n)).filter
        (fun x => p.toSubgraph.Adj center x) := by
    ext x
    simp only [Set.Finite.mem_toFinset, Finset.mem_filter, Finset.mem_insert,
      Finset.mem_singleton, Subgraph.mem_neighborSet]
    constructor
    · intro hx
      exact ⟨hsub hx, hx⟩
    · rintro ⟨_, hx⟩
      exact hx
  rw [hfinset]
  have hbit (a b : GridVertex n) :
      bitDegree (pathEdgeBit p a b) = if p.toSubgraph.Adj a b then 1 else 0 := by
    classical
    by_cases h : p.toSubgraph.Adj a b <;> simp [pathEdgeBit, bitDegree, h]
  rw [hbit center left, hbit center right, hbit center up, hbit center down]
  have hlr : left ≠ right := by
    intro h
    have hh := congrArg (fun z : GridVertex n => z.2.val) h
    dsimp [left, right] at hh
    omega
  have hlu : left ≠ up := by
    intro h
    have hh := congrArg (fun z : GridVertex n => z.1.val) h
    dsimp [left, up] at hh
    omega
  have hld : left ≠ down := by
    intro h
    have hh := congrArg (fun z : GridVertex n => z.1.val) h
    dsimp [left, down] at hh
    omega
  have hru : right ≠ up := by
    intro h
    have hh := congrArg (fun z : GridVertex n => z.1.val) h
    dsimp [right, up] at hh
    omega
  have hrd : right ≠ down := by
    intro h
    have hh := congrArg (fun z : GridVertex n => z.1.val) h
    dsimp [right, down] at hh
    omega
  have hud : up ≠ down := by
    intro h
    have hh := congrArg (fun z : GridVertex n => z.1.val) h
    dsimp [up, down] at hh
    omega
  have hsum := Finset.sum_boole (R := Nat)
    (fun x : GridVertex n => p.toSubgraph.Adj center x)
    ({left, right, up, down} : Finset (GridVertex n))
  have hsumNat :
      (∑ x ∈ ({left, right, up, down} : Finset (GridVertex n)),
        if p.toSubgraph.Adj center x then 1 else 0 : Nat) =
      (({left, right, up, down} : Finset (GridVertex n)).filter
        (fun x => p.toSubgraph.Adj center x)).card := by
    simpa only [Nat.cast_id] using hsum
  rw [← hsumNat]
  rw [Finset.sum_insert (by simp [hlr, hlu, hld])]
  rw [Finset.sum_insert (by simp [hru, hrd])]
  rw [Finset.sum_insert (by simp [hud])]
  rw [Finset.sum_singleton]
  simp only [Nat.add_assoc]

/-- Therefore a mathlib simple path has at most two of the four incident grid
edges at every interior vertex. -/
theorem isPath_interior_four_bits_le_two
    {n : Nat} {u v : GridVertex n}
    {p : (gridGraph n).Walk u v} (hp : p.IsPath)
    (r c : Nat) (hr : r + 1 < n) (hc : c + 1 < n) :
    let center : GridVertex n :=
      (⟨r + 1, by omega⟩, ⟨c + 1, by omega⟩)
    let left : GridVertex n :=
      (⟨r + 1, by omega⟩, ⟨c, by omega⟩)
    let right : GridVertex n :=
      (⟨r + 1, by omega⟩, ⟨c + 2, by omega⟩)
    let up : GridVertex n :=
      (⟨r, by omega⟩, ⟨c + 1, by omega⟩)
    let down : GridVertex n :=
      (⟨r + 2, by omega⟩, ⟨c + 1, by omega⟩)
    bitDegree (pathEdgeBit p center left) +
      bitDegree (pathEdgeBit p center right) +
      bitDegree (pathEdgeBit p center up) +
      bitDegree (pathEdgeBit p center down) ≤ 2 := by
  dsimp
  rw [interior_four_bits_eq_neighbor_ncard p r c hr hc]
  exact isPath_neighborSet_ncard_le_two hp


/-- The edge membership bit is symmetric because `toSubgraph.Adj` is an
undirected relation. -/
theorem pathEdgeBit_symm {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (a b : GridVertex n) :
    pathEdgeBit p a b = pathEdgeBit p b a := by
  classical
  by_cases h : p.toSubgraph.Adj a b
  · have hb : p.toSubgraph.Adj b a := h.symm
    simp [pathEdgeBit, h, hb]
  · have hb : ¬ p.toSubgraph.Adj b a := by
      intro hba
      exact h hba.symm
    simp [pathEdgeBit, h, hb]

/-- Horizontal path-edge encoding in the same ghost-column convention as
`GridFaceBoundary`: real horizontal edges are indexed by `c=1..n`, while
`c=0` and `c=n+1` are zero ghost edges. -/
noncomputable def pathHorizontal {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r c : Nat) : F2 := by
  classical
  by_cases h : r ≤ n ∧ 0 < c ∧ c ≤ n
  · exact pathEdgeBit p
      (⟨r, by omega⟩, ⟨c, by omega⟩)
      (⟨r, by omega⟩, ⟨c - 1, by omega⟩)
  · exact 0

/-- Vertical path-edge encoding in the same ghost-row convention as
`GridFaceBoundary`: real vertical edges are indexed by `r=1..n`, while
`r=0` and `r=n+1` are zero ghost edges. -/
noncomputable def pathVertical {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r c : Nat) : F2 := by
  classical
  by_cases h : 0 < r ∧ r ≤ n ∧ c ≤ n
  · exact pathEdgeBit p
      (⟨r, by omega⟩, ⟨c, by omega⟩)
      (⟨r - 1, by omega⟩, ⟨c, by omega⟩)
  · exact 0

@[simp] theorem pathHorizontal_left_ghost {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r : Nat) : pathHorizontal p r 0 = 0 := by
  simp [pathHorizontal]

@[simp] theorem pathHorizontal_right_ghost {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r : Nat) : pathHorizontal p r (n + 1) = 0 := by
  simp [pathHorizontal]

@[simp] theorem pathVertical_top_ghost {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (c : Nat) : pathVertical p 0 c = 0 := by
  simp [pathVertical]

@[simp] theorem pathVertical_bottom_ghost {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (c : Nat) : pathVertical p (n + 1) c = 0 := by
  simp [pathVertical]

/-- The H/V encoding of a mathlib simple path has ordinary degree at most two
at every interior grid vertex, exactly in the indexing consumed by the
checkerboard-bound proof. -/
theorem isPath_encoded_interior_degree_le_two
    {n : Nat} {u v : GridVertex n}
    {p : (gridGraph n).Walk u v} (hp : p.IsPath)
    (r c : Nat) (hr : r + 1 < n) (hc : c + 1 < n) :
    bitDegree (pathHorizontal p (r + 1) (c + 1)) +
      bitDegree (pathHorizontal p (r + 1) (c + 2)) +
      bitDegree (pathVertical p (r + 1) (c + 1)) +
      bitDegree (pathVertical p (r + 2) (c + 1)) ≤ 2 := by
  have h := isPath_interior_four_bits_le_two hp r c hr hc
  dsimp at h
  let center : GridVertex n :=
    (⟨r + 1, by omega⟩, ⟨c + 1, by omega⟩)
  let right : GridVertex n :=
    (⟨r + 1, by omega⟩, ⟨c + 2, by omega⟩)
  let down : GridVertex n :=
    (⟨r + 2, by omega⟩, ⟨c + 1, by omega⟩)
  have hright := pathEdgeBit_symm p center right
  have hdown := pathEdgeBit_symm p center down
  dsimp [center, right, down] at hright hdown
  rw [hright, hdown] at h
  simpa [pathHorizontal, pathVertical,
    show r + 1 ≤ n by omega, show r + 2 ≤ n by omega,
    show c + 1 ≤ n by omega, show c + 2 ≤ n by omega,
    show 0 < r + 1 by omega, show 0 < r + 2 by omega,
    show 0 < c + 1 by omega, show 0 < c + 2 by omega] using h


/-- Coordinate-complete bridge from a mathlib simple grid path to the
checkerboard-free face encoding theorem.  The candidate path's H/V arrays are
now concrete; the only remaining reference-side hypothesis is equality of
vertex parity with the chosen outer-boundary reference assignment. -/
theorem isPath_with_reference_has_checkerboard_free_face_encoding
    {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    {p : (gridGraph n).Walk u v} (hp : p.IsPath)
    (H₀ V₀ : Nat → Nat → F2)
    (hHleft₀ : ∀ r, r ≤ n → H₀ r 0 = 0)
    (hHright₀ : ∀ r, r ≤ n → H₀ r (n + 1) = 0)
    (hVtop₀ : ∀ c, c ≤ n → V₀ 0 c = 0)
    (hVbottom₀ : ∀ c, c ≤ n → V₀ (n + 1) c = 0)
    (hsame : ∀ r c, r ≤ n → c ≤ n →
      vertexParity (pathHorizontal p) (pathVertical p) r c =
        vertexParity H₀ V₀ r c)
    (hbaseInterior : ∀ r c, r + 1 < n → c + 1 < n →
      H₀ (r + 1) (c + 1) = 0 ∧ H₀ (r + 1) (c + 2) = 0 ∧
      V₀ (r + 1) (c + 1) = 0 ∧ V₀ (r + 2) (c + 1) = 0) :
    ∃ F : Nat → Nat → F2,
      RectBoundaryWitness n (edgeXor (pathHorizontal p) H₀)
        (edgeXor (pathVertical p) V₀) F ∧
      noCheckerboardGrid n F := by
  apply outer_reference_xor_has_checkerboard_free_face_encoding
    n hn (pathHorizontal p) (pathVertical p) H₀ V₀
  · intro r _
    exact pathHorizontal_left_ghost p r
  · intro r _
    exact pathHorizontal_right_ghost p r
  · intro c _
    exact pathVertical_top_ghost p c
  · intro c _
    exact pathVertical_bottom_ghost p c
  · exact hHleft₀
  · exact hHright₀
  · exact hVtop₀
  · exact hVbottom₀
  · exact hsame
  · exact hbaseInterior
  · intro r c hr hc
    exact isPath_encoded_interior_degree_le_two hp r c hr hc


/-- Casting the ordinary 0/1 degree contribution of a path-edge bit back to
GF(2) recovers the bit itself. -/
theorem cast_bitDegree_pathEdgeBit {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (a b : GridVertex n) :
    ((bitDegree (pathEdgeBit p a b) : Nat) : F2) = pathEdgeBit p a b := by
  classical
  by_cases h : p.toSubgraph.Adj a b <;> simp [pathEdgeBit, bitDegree, h]

/-- For any explicit finite list of all grid neighbors of a vertex, summing
path-edge membership bits counts exactly the neighbors retained by the path
subgraph. -/
theorem edgeBit_sum_eq_neighbor_ncard
    {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (center : GridVertex n)
    (s : Finset (GridVertex n))
    (hfull : (gridGraph n).neighborSet center = (s : Set (GridVertex n))) :
    (∑ x ∈ s, bitDegree (pathEdgeBit p center x)) =
      (p.toSubgraph.neighborSet center).ncard := by
  classical
  have hsub : p.toSubgraph.neighborSet center ⊆ (s : Set (GridVertex n)) := by
    intro x hx
    have hg : (gridGraph n).Adj center x := p.toSubgraph.adj_sub hx
    have hxg : x ∈ (gridGraph n).neighborSet center := hg
    rw [hfull] at hxg
    exact hxg
  have hfinite : (p.toSubgraph.neighborSet center).Finite :=
    Set.Finite.subset s.finite_toSet hsub
  rw [Set.ncard_eq_toFinset_card _ hfinite]
  have hfinset : hfinite.toFinset = s.filter (fun x => p.toSubgraph.Adj center x) := by
    ext x
    simp only [Set.Finite.mem_toFinset, Finset.mem_filter, Subgraph.mem_neighborSet]
    constructor
    · intro hx
      exact ⟨hsub hx, hx⟩
    · rintro ⟨_, hx⟩
      exact hx
  rw [hfinset]
  calc
    (∑ x ∈ s, bitDegree (pathEdgeBit p center x)) =
        ∑ x ∈ s, if p.toSubgraph.Adj center x then 1 else 0 := by
          apply Finset.sum_congr rfl
          intro x _
          by_cases h : p.toSubgraph.Adj center x <;>
            simp [pathEdgeBit, bitDegree, h]
    _ = (s.filter (fun x => p.toSubgraph.Adj center x)).card := by
      have hsum := Finset.sum_boole (R := Nat)
        (fun x : GridVertex n => p.toSubgraph.Adj center x) s
      simpa only [Nat.cast_id] using hsum


/-- GF(2) version of `edgeBit_sum_eq_neighbor_ncard`: the sum of incident
path-edge bits is the path-subgraph degree reduced modulo two. -/
theorem edgeBit_f2sum_eq_neighbor_ncard_cast
    {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (center : GridVertex n)
    (s : Finset (GridVertex n))
    (hfull : (gridGraph n).neighborSet center = (s : Set (GridVertex n))) :
    (∑ x ∈ s, pathEdgeBit p center x) =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  have hnat := edgeBit_sum_eq_neighbor_ncard p center s hfull
  have hcast := congrArg (fun k : Nat => (k : F2)) hnat
  simpa [cast_bitDegree_pathEdgeBit] using hcast


/-- Expansion of a real horizontal H-entry to the corresponding undirected
path-subgraph edge bit. -/
theorem pathHorizontal_of_valid {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r c : Nat)
    (hr : r ≤ n) (hc0 : 0 < c) (hcn : c ≤ n) :
    pathHorizontal p r c = pathEdgeBit p
      (⟨r, by omega⟩, ⟨c, by omega⟩)
      (⟨r, by omega⟩, ⟨c - 1, by omega⟩) := by
  simp [pathHorizontal, hr, hc0, hcn]

/-- Expansion of a real vertical V-entry to the corresponding undirected
path-subgraph edge bit. -/
theorem pathVertical_of_valid {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r c : Nat)
    (hr0 : 0 < r) (hrn : r ≤ n) (hc : c ≤ n) :
    pathVertical p r c = pathEdgeBit p
      (⟨r, by omega⟩, ⟨c, by omega⟩)
      (⟨r - 1, by omega⟩, ⟨c, by omega⟩) := by
  simp [pathVertical, hr0, hrn, hc]


/-- Two-neighbor specialization of the GF(2) incidence sum. -/
theorem edgeBit_pair_eq_neighbor_ncard_cast
    {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (center a b : GridVertex n)
    (hab : a ≠ b)
    (hfull : (gridGraph n).neighborSet center = ({a, b} : Set (GridVertex n))) :
    pathEdgeBit p center a + pathEdgeBit p center b =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  have hfull' : (gridGraph n).neighborSet center =
      (({a, b} : Finset (GridVertex n)) : Set (GridVertex n)) := by
    simpa using hfull
  have h := edgeBit_f2sum_eq_neighbor_ncard_cast p center
    ({a, b} : Finset (GridVertex n)) hfull'
  simpa [hab] using h

/-- Three-neighbor specialization of the GF(2) incidence sum. -/
theorem edgeBit_triple_eq_neighbor_ncard_cast
    {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (center a b c : GridVertex n)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c)
    (hfull : (gridGraph n).neighborSet center = ({a, b, c} : Set (GridVertex n))) :
    pathEdgeBit p center a + pathEdgeBit p center b + pathEdgeBit p center c =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  have hfull' : (gridGraph n).neighborSet center =
      (({a, b, c} : Finset (GridVertex n)) : Set (GridVertex n)) := by
    simpa using hfull
  have h := edgeBit_f2sum_eq_neighbor_ncard_cast p center
    ({a, b, c} : Finset (GridVertex n)) hfull'
  simpa [hab, hac, hbc, add_assoc] using h

/-- Four-neighbor specialization of the GF(2) incidence sum. -/
theorem edgeBit_quad_eq_neighbor_ncard_cast
    {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (center a b c d : GridVertex n)
    (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d)
    (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
    (hfull : (gridGraph n).neighborSet center = ({a, b, c, d} : Set (GridVertex n))) :
    pathEdgeBit p center a + pathEdgeBit p center b +
      pathEdgeBit p center c + pathEdgeBit p center d =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  have hfull' : (gridGraph n).neighborSet center =
      (({a, b, c, d} : Finset (GridVertex n)) : Set (GridVertex n)) := by
    simpa using hfull
  have h := edgeBit_f2sum_eq_neighbor_ncard_cast p center
    ({a, b, c, d} : Finset (GridVertex n)) hfull'
  simpa [hab, hac, had, hbc, hbd, hcd, add_assoc] using h


end OneesanFormal.PathGridEncoding
