import OneesanFormal.PathGridEncoding
import OneesanFormal.GridBoundaryNeighbors
import OneesanFormal.OuterReference

namespace OneesanFormal.PathGridParity

open OneesanFormal.GridGraph
open OneesanFormal.GridFaceBoundary
open OneesanFormal.PathGridEncoding
open OneesanFormal.GridBoundaryNeighbors
open OneesanFormal.PathDegree
open OneesanFormal.OuterReference

private theorem vertex_ne_of_row_ne {n : Nat} {a b : GridVertex n}
    (h : a.1.val ≠ b.1.val) : a ≠ b := by
  intro hab
  exact h (congrArg (fun z : GridVertex n => z.1.val) hab)

private theorem vertex_ne_of_col_ne {n : Nat} {a b : GridVertex n}
    (h : a.2.val ≠ b.2.val) : a ≠ b := by
  intro hab
  exact h (congrArg (fun z : GridVertex n => z.2.val) hab)

/-- H/V incidence parity equals path-subgraph degree mod 2 at the northwest corner. -/
theorem encodedParity_northwest {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) :
    vertexParity (pathHorizontal p) (pathVertical p) 0 0 =
      ((p.toSubgraph.neighborSet (northwestCorner n)).ncard : F2) := by
  let right : GridVertex n := (⟨0, by omega⟩, ⟨1, by omega⟩)
  let down : GridVertex n := (⟨1, by omega⟩, ⟨0, by omega⟩)
  have hfull : (gridGraph n).neighborSet (northwestCorner n) =
      ({right, down} : Set (GridVertex n)) := by
    simpa [right, down] using neighborSet_northwest (n := n) hn
  have hne : right ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [right, down]
    omega
  have hsum := edgeBit_pair_eq_neighbor_ncard_cast p
    (northwestCorner n) right down hne hfull
  rw [pathEdgeBit_symm p (northwestCorner n) right,
      pathEdgeBit_symm p (northwestCorner n) down] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, northwestCorner, right, down,
    show 1 ≤ n by omega] using hsum

/-- H/V incidence parity equals path-subgraph degree mod 2 at the northeast corner. -/
theorem encodedParity_northeast {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨0, by omega⟩, ⟨n, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) 0 n =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨0, by omega⟩, ⟨n, by omega⟩)
  let left : GridVertex n := (⟨0, by omega⟩, ⟨n - 1, by omega⟩)
  let down : GridVertex n := (⟨1, by omega⟩, ⟨n, by omega⟩)
  have hfull : (gridGraph n).neighborSet center = ({left, down} : Set (GridVertex n)) := by
    simpa [center, left, down] using neighborSet_northeast (n := n) hn
  have hne : left ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [left, down]
    omega
  have hsum := edgeBit_pair_eq_neighbor_ncard_cast p center left down hne hfull
  rw [pathEdgeBit_symm p center down] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, left, down,
    show 1 ≤ n by omega, hn] using hsum

/-- H/V incidence parity equals path-subgraph degree mod 2 at the southwest corner. -/
theorem encodedParity_southwest {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨n, by omega⟩, ⟨0, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) n 0 =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨n, by omega⟩, ⟨0, by omega⟩)
  let right : GridVertex n := (⟨n, by omega⟩, ⟨1, by omega⟩)
  let up : GridVertex n := (⟨n - 1, by omega⟩, ⟨0, by omega⟩)
  have hfull : (gridGraph n).neighborSet center = ({right, up} : Set (GridVertex n)) := by
    simpa [center, right, up] using neighborSet_southwest (n := n) hn
  have hne : right ≠ up := by
    apply vertex_ne_of_row_ne
    dsimp [right, up]
    omega
  have hsum := edgeBit_pair_eq_neighbor_ncard_cast p center right up hne hfull
  rw [pathEdgeBit_symm p center right] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, right, up,
    show 1 ≤ n by omega, hn] using hsum

/-- H/V incidence parity equals path-subgraph degree mod 2 at the southeast corner. -/
theorem encodedParity_southeast {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) :
    vertexParity (pathHorizontal p) (pathVertical p) n n =
      ((p.toSubgraph.neighborSet (southeastCorner n)).ncard : F2) := by
  let left : GridVertex n := (⟨n, by omega⟩, ⟨n - 1, by omega⟩)
  let up : GridVertex n := (⟨n - 1, by omega⟩, ⟨n, by omega⟩)
  have hfull : (gridGraph n).neighborSet (southeastCorner n) =
      ({left, up} : Set (GridVertex n)) := by
    simpa [left, up] using neighborSet_southeast (n := n) hn
  have hne : left ≠ up := by
    apply vertex_ne_of_row_ne
    dsimp [left, up]
    omega
  have hsum := edgeBit_pair_eq_neighbor_ncard_cast p
    (southeastCorner n) left up hne hfull
  simpa [vertexParity, pathHorizontal, pathVertical, southeastCorner, left, up,
    show 1 ≤ n by omega, hn] using hsum


/-- H/V incidence parity equals path-subgraph degree mod 2 on the top edge. -/
theorem encodedParity_top {n c : Nat} (hc0 : 0 < c) (hcn : c < n)
    {u v : GridVertex n} (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨0, by omega⟩, ⟨c, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) 0 c =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨0, by omega⟩, ⟨c, by omega⟩)
  let left : GridVertex n := (⟨0, by omega⟩, ⟨c - 1, by omega⟩)
  let right : GridVertex n := (⟨0, by omega⟩, ⟨c + 1, by omega⟩)
  let down : GridVertex n := (⟨1, by omega⟩, ⟨c, by omega⟩)
  have hfull : (gridGraph n).neighborSet center =
      ({left, right, down} : Set (GridVertex n)) := by
    simpa [center, left, right, down] using neighborSet_top hc0 hcn
  have hlr : left ≠ right := by
    apply vertex_ne_of_col_ne
    dsimp [left, right]
    omega
  have hld : left ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [left, down]
    omega
  have hrd : right ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [right, down]
    omega
  have hsum := edgeBit_triple_eq_neighbor_ncard_cast p center left right down
    hlr hld hrd hfull
  rw [pathEdgeBit_symm p center right, pathEdgeBit_symm p center down] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, left, right, down,
    hc0, show c ≤ n by omega, show c + 1 ≤ n by omega,
    show 1 ≤ n by omega] using hsum

/-- H/V incidence parity equals path-subgraph degree mod 2 on the bottom edge. -/
theorem encodedParity_bottom {n c : Nat} (hc0 : 0 < c) (hcn : c < n)
    {u v : GridVertex n} (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨n, by omega⟩, ⟨c, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) n c =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨n, by omega⟩, ⟨c, by omega⟩)
  let left : GridVertex n := (⟨n, by omega⟩, ⟨c - 1, by omega⟩)
  let right : GridVertex n := (⟨n, by omega⟩, ⟨c + 1, by omega⟩)
  let up : GridVertex n := (⟨n - 1, by omega⟩, ⟨c, by omega⟩)
  have hfull : (gridGraph n).neighborSet center =
      ({left, right, up} : Set (GridVertex n)) := by
    simpa [center, left, right, up] using neighborSet_bottom hc0 hcn
  have hlr : left ≠ right := by
    apply vertex_ne_of_col_ne
    dsimp [left, right]
    omega
  have hlu : left ≠ up := by
    apply vertex_ne_of_row_ne
    dsimp [left, up]
    omega
  have hru : right ≠ up := by
    apply vertex_ne_of_row_ne
    dsimp [right, up]
    omega
  have hsum := edgeBit_triple_eq_neighbor_ncard_cast p center left right up
    hlr hlu hru hfull
  rw [pathEdgeBit_symm p center right] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, left, right, up,
    hc0, show c ≤ n by omega, show c + 1 ≤ n by omega,
    show 0 < n by omega] using hsum

/-- H/V incidence parity equals path-subgraph degree mod 2 on the left edge. -/
theorem encodedParity_left {n r : Nat} (hr0 : 0 < r) (hrn : r < n)
    {u v : GridVertex n} (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨r, by omega⟩, ⟨0, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) r 0 =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨r, by omega⟩, ⟨0, by omega⟩)
  let right : GridVertex n := (⟨r, by omega⟩, ⟨1, by omega⟩)
  let up : GridVertex n := (⟨r - 1, by omega⟩, ⟨0, by omega⟩)
  let down : GridVertex n := (⟨r + 1, by omega⟩, ⟨0, by omega⟩)
  have hfull : (gridGraph n).neighborSet center =
      ({right, up, down} : Set (GridVertex n)) := by
    simpa [center, right, up, down] using neighborSet_left hr0 hrn
  have hru : right ≠ up := by
    apply vertex_ne_of_col_ne
    dsimp [right, up]
    omega
  have hrd : right ≠ down := by
    apply vertex_ne_of_col_ne
    dsimp [right, down]
    omega
  have hud : up ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [up, down]
    omega
  have hsum := edgeBit_triple_eq_neighbor_ncard_cast p center right up down
    hru hrd hud hfull
  rw [pathEdgeBit_symm p center right, pathEdgeBit_symm p center down] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, right, up, down,
    hr0, show r ≤ n by omega, show r + 1 ≤ n by omega,
    show 1 ≤ n by omega] using hsum

/-- H/V incidence parity equals path-subgraph degree mod 2 on the right edge. -/
theorem encodedParity_right {n r : Nat} (hr0 : 0 < r) (hrn : r < n)
    {u v : GridVertex n} (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨r, by omega⟩, ⟨n, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) r n =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨r, by omega⟩, ⟨n, by omega⟩)
  let left : GridVertex n := (⟨r, by omega⟩, ⟨n - 1, by omega⟩)
  let up : GridVertex n := (⟨r - 1, by omega⟩, ⟨n, by omega⟩)
  let down : GridVertex n := (⟨r + 1, by omega⟩, ⟨n, by omega⟩)
  have hfull : (gridGraph n).neighborSet center =
      ({left, up, down} : Set (GridVertex n)) := by
    simpa [center, left, up, down] using neighborSet_right hr0 hrn
  have hlu : left ≠ up := by
    apply vertex_ne_of_col_ne
    dsimp [left, up]
    omega
  have hld : left ≠ down := by
    apply vertex_ne_of_col_ne
    dsimp [left, down]
    omega
  have hud : up ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [up, down]
    omega
  have hsum := edgeBit_triple_eq_neighbor_ncard_cast p center left up down
    hlu hld hud hfull
  rw [pathEdgeBit_symm p center down] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, left, up, down,
    hr0, show r ≤ n by omega, show r + 1 ≤ n by omega,
    show 0 < n by omega] using hsum


/-- H/V incidence parity equals path-subgraph degree mod 2 at a strictly
interior vertex. -/
theorem encodedParity_interior {n r c : Nat}
    (hr0 : 0 < r) (hrn : r < n) (hc0 : 0 < c) (hcn : c < n)
    {u v : GridVertex n} (p : (gridGraph n).Walk u v) :
    let center : GridVertex n := (⟨r, by omega⟩, ⟨c, by omega⟩)
    vertexParity (pathHorizontal p) (pathVertical p) r c =
      ((p.toSubgraph.neighborSet center).ncard : F2) := by
  dsimp
  let center : GridVertex n := (⟨r, by omega⟩, ⟨c, by omega⟩)
  let left : GridVertex n := (⟨r, by omega⟩, ⟨c - 1, by omega⟩)
  let right : GridVertex n := (⟨r, by omega⟩, ⟨c + 1, by omega⟩)
  let up : GridVertex n := (⟨r - 1, by omega⟩, ⟨c, by omega⟩)
  let down : GridVertex n := (⟨r + 1, by omega⟩, ⟨c, by omega⟩)
  have hfull : (gridGraph n).neighborSet center =
      ({left, right, up, down} : Set (GridVertex n)) := by
    simpa [center, left, right, up, down,
      show r - 1 + 1 = r by omega, show c - 1 + 1 = c by omega,
      show r - 1 + 2 = r + 1 by omega, show c - 1 + 2 = c + 1 by omega] using
      (neighborSet_interior (n := n) (r := r - 1) (c := c - 1)
        (by omega) (by omega))
  have hlr : left ≠ right := by
    apply vertex_ne_of_col_ne
    dsimp [left, right]
    omega
  have hlu : left ≠ up := by
    apply vertex_ne_of_row_ne
    dsimp [left, up]
    omega
  have hld : left ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [left, down]
    omega
  have hru : right ≠ up := by
    apply vertex_ne_of_row_ne
    dsimp [right, up]
    omega
  have hrd : right ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [right, down]
    omega
  have hud : up ≠ down := by
    apply vertex_ne_of_row_ne
    dsimp [up, down]
    omega
  have hsum := edgeBit_quad_eq_neighbor_ncard_cast p center left right up down
    hlr hlu hld hru hrd hud hfull
  rw [pathEdgeBit_symm p center right, pathEdgeBit_symm p center down] at hsum
  simpa [vertexParity, pathHorizontal, pathVertical, center, left, right, up, down,
    hr0, hc0, show r ≤ n by omega, show r + 1 ≤ n by omega,
    show c ≤ n by omega, show c + 1 ≤ n by omega] using hsum

/-- Complete coordinate bridge: at every vertex of the finite rectangular
GridGraph, the H/V encoding used by the production checkerboard proof has
vertex parity equal to the path-subgraph degree modulo two. -/
theorem encoded_vertexParity_eq_neighbor_ncard_cast
    {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (r c : Nat)
    (hr : r ≤ n) (hc : c ≤ n) :
    vertexParity (pathHorizontal p) (pathVertical p) r c =
      ((p.toSubgraph.neighborSet
        ((⟨r, by omega⟩, ⟨c, by omega⟩) : GridVertex n)).ncard : F2) := by
  by_cases hr0 : r = 0
  · subst r
    by_cases hc0 : c = 0
    · subst c
      simpa [northwestCorner] using encodedParity_northwest hn p
    · by_cases hcn : c = n
      · subst c
        simpa using encodedParity_northeast hn p
      · have hcpos : 0 < c := Nat.pos_of_ne_zero hc0
        have hclt : c < n := by omega
        simpa using encodedParity_top hcpos hclt p
  · by_cases hrn : r = n
    · subst r
      by_cases hc0 : c = 0
      · subst c
        simpa using encodedParity_southwest hn p
      · by_cases hcn : c = n
        · subst c
          simpa [southeastCorner] using encodedParity_southeast hn p
        · have hcpos : 0 < c := Nat.pos_of_ne_zero hc0
          have hclt : c < n := by omega
          simpa using encodedParity_bottom hcpos hclt p
    · have hrpos : 0 < r := Nat.pos_of_ne_zero hr0
      have hrlt : r < n := by omega
      by_cases hc0 : c = 0
      · subst c
        simpa using encodedParity_left hrpos hrlt p
      · by_cases hcn : c = n
        · subst c
          simpa using encodedParity_right hrpos hrlt p
        · have hcpos : 0 < c := Nat.pos_of_ne_zero hc0
          have hclt : c < n := by omega
          simpa using encodedParity_interior hrpos hrlt hcpos hclt p



/-- A concrete mathlib simple path from NW to SE has the same vertex-parity
vector as the fixed outer-boundary reference: odd exactly at the two corners. -/
theorem corner_path_vertexParity
    {n : Nat} (hn : 0 < n)
    {p : (gridGraph n).Walk (northwestCorner n) (southeastCorner n)}
    (hp : p.IsPath) :
    ∀ r c, r ≤ n → c ≤ n →
      vertexParity (pathHorizontal p) (pathVertical p) r c = cornerParity n r c := by
  intro r c hr hc
  let center : GridVertex n :=
    (⟨r, by omega⟩, ⟨c, by omega⟩)
  have hnw : center = northwestCorner n ↔ r = 0 ∧ c = 0 := by
    simpa [center] using vertex_eq_northwest_iff hr hc
  have hse : center = southeastCorner n ↔ r = n ∧ c = n := by
    simpa [center] using vertex_eq_southeast_iff hr hc
  have hend :
      (center = northwestCorner n ∨ center = southeastCorner n) ↔
        ((r = 0 ∧ c = 0) ∨ (r = n ∧ c = n)) := or_congr hnw hse
  calc
    vertexParity (pathHorizontal p) (pathVertical p) r c =
        ((p.toSubgraph.neighborSet center).ncard : F2) := by
          simpa [center] using
            encoded_vertexParity_eq_neighbor_ncard_cast hn p r c hr hc
    _ = (expectedPathDegree p center : F2) := by
          rw [isPath_neighborSet_ncard_eq hp (corner_ne hn)]
    _ = cornerParity n r c := by
          classical
          by_cases h : center = northwestCorner n ∨ center = southeastCorner n
          · have hcnd := hend.mp h
            simp [expectedPathDegree, cornerParity, h, hcnd]
          · have hcnd : ¬ ((r = 0 ∧ c = 0) ∨ (r = n ∧ c = n)) := by
              intro hcnd
              exact h (hend.mpr hcnd)
            simp [expectedPathDegree, cornerParity, h, hcnd, CharTwo.two_eq_zero]

end OneesanFormal.PathGridParity
