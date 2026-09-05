import OneesanFormal.GridGraph
import OneesanFormal.CutGridCrossing
import OneesanFormal.PathEdgeSet
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace OneesanFormal.PathCutCrossing

open SimpleGraph
open OneesanFormal.GridGraph
open OneesanFormal.CutGridCrossing
open OneesanFormal.CutHeight

/-- The concrete horizontal grid edge in row `i` crossing the vertical cut
between columns `c` and `c+1`.  This is the same edge as
`CutGridCrossing.crossingEdge`, specialized to the square `r × r` strip. -/
def gridCrossingEdge {r : Nat} (c : Fin r) (i : Fin r) :
    Sym2 (GridVertex r) :=
  crossingEdge (r := r) (w := r) c i

/-- Rows whose physical crossing edge belongs to a given walk. -/
noncomputable def crossingRows {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) : Finset (Fin r) := by
  classical
  exact Finset.univ.filter fun i => gridCrossingEdge c i ∈ p.edgeSet

@[simp] theorem mem_crossingRows_iff {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) (i : Fin r) :
    i ∈ crossingRows p c ↔ gridCrossingEdge c i ∈ p.edgeSet := by
  classical
  simp [crossingRows]

/-- A vertical cut in an `r`-row grid has at most `r` path edges crossing it. -/
theorem crossingRows_card_le_height {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) :
    (crossingRows p c).card ≤ r := by
  classical
  exact le_trans (Finset.card_le_univ (s := crossingRows p c)) (by simp)

/-- The actual crossing edges used by a path are in bijection with their row
indices, hence there are at most `r` of them. -/
noncomputable def crossingPathEdges {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) : Finset (Sym2 (GridVertex r)) := by
  classical
  exact (crossingRows p c).image (gridCrossingEdge c)

theorem crossingPathEdges_card_eq {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) :
    (crossingPathEdges p c).card = (crossingRows p c).card := by
  classical
  apply Finset.card_image_iff.mpr
  intro a ha b hb hab
  exact crossingEdge_injective c hab

theorem crossingPathEdges_card_le_height {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) :
    (crossingPathEdges p c).card ≤ r := by
  rw [crossingPathEdges_card_eq]
  exact crossingRows_card_le_height p c

/-- A semantic frontier stack for a path at a cut: every live component stores
one concrete path edge crossing that cut.  The witness is injective because
one physical edge belongs to only one live path component. -/
structure PathCutStackWitness {r : Nat} {u v : GridVertex r}
    (p : (gridGraph r).Walk u v) (c : Fin r) (Component : Type*) where
  stack : List Component
  nodup : stack.Nodup
  crossingRow : Component → Fin r
  row_mem : ∀ q, q ∈ stack → crossingRow q ∈ crossingRows p c
  injective_on_stack : Set.InjOn crossingRow {q | q ∈ stack}

/-- Any frontier stack admitting the concrete path-cut witness has height at
most the number of physical rows. -/
theorem stack_length_le_height_of_path_cut_witness
    {r : Nat} {u v : GridVertex r} {Component : Type*}
    (p : (gridGraph r).Walk u v) (c : Fin r)
    (W : PathCutStackWitness p c Component) :
    W.stack.length ≤ r := by
  exact stack_length_le_height W.stack W.nodup W.crossingRow W.injective_on_stack

/-- Edge-valued version: if every stack component owns a distinct actual path
edge crossing the cut, then stack height is at most `r`. -/
theorem stack_length_le_height_of_distinct_crossing_edges
    {r : Nat} {u v : GridVertex r} {Component : Type*}
    (p : (gridGraph r).Walk u v) (c : Fin r)
    (stack : List Component) (hnodup : stack.Nodup)
    (rowOf : Component → Fin r)
    (hmem : ∀ q, q ∈ stack → rowOf q ∈ crossingRows p c)
    (hinj : Set.InjOn rowOf {q | q ∈ stack}) :
    stack.length ≤ r := by
  let W : PathCutStackWitness p c Component := {
    stack := stack
    nodup := hnodup
    crossingRow := rowOf
    row_mem := hmem
    injective_on_stack := hinj
  }
  exact stack_length_le_height_of_path_cut_witness p c W

end OneesanFormal.PathCutCrossing
