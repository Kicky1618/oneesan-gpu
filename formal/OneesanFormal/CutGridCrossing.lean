import OneesanFormal.CutHeight
import Mathlib.Data.Sym.Sym2
import Mathlib.Tactic

namespace OneesanFormal.CutGridCrossing

open Sym2
open OneesanFormal.CutHeight

/-- Vertices of an `r`-row, `w`-column rectangular strip. -/
abbrev StripVertex (r w : Nat) := Fin (r + 1) × Fin (w + 1)

/-- The physical horizontal edge crossing the vertical cut between columns
`c` and `c+1` in row `i`. -/
def crossingEdge {r w : Nat} (c : Fin w) (i : Fin r) : Sym2 (StripVertex r w) :=
  s((⟨i, by omega⟩, ⟨c, by omega⟩),
    (⟨i, by omega⟩, ⟨c.val + 1, by omega⟩))

/-- Distinct strip rows give distinct physical edges crossing the same cut. -/
theorem crossingEdge_injective {r w : Nat} (c : Fin w) :
    Function.Injective (crossingEdge (r := r) c) := by
  intro i j hij
  unfold crossingEdge at hij
  rw [Sym2.eq_iff] at hij
  rcases hij with hsame | hswap
  · have hrow := congrArg (fun v : StripVertex r w => v.1.val) hsame.1
    exact Fin.ext hrow
  · have hcol := congrArg (fun v : StripVertex r w => v.2.val) hswap.1
    simp [crossingEdge] at hcol

/-- Therefore a vertical cut of an `r`-row strip has exactly `r` distinct
physical crossing edges, canonically indexed by `Fin r`. -/
theorem card_crossingEdges {r w : Nat} (c : Fin w) :
    Fintype.card (Fin r) = r := by
  simp

/-- Concrete edge-valued version of the cut-height argument.  If every open
component is assigned one crossing edge and that assignment factors through a
row witness, distinct components are bounded by the number of strip rows. -/
theorem card_openComponents_le_of_crossing_edges
    {r w : Nat} {OpenComponent : Type*} [Fintype OpenComponent]
    (c : Fin w)
    (rowOf : OpenComponent → Fin r)
    (edgeOf : OpenComponent → Sym2 (StripVertex r w))
    (hedge : ∀ q, edgeOf q = crossingEdge c (rowOf q))
    (hinjEdge : Function.Injective edgeOf) :
    Fintype.card OpenComponent ≤ r := by
  have hinjRow : Function.Injective rowOf := by
    intro a b hab
    apply hinjEdge
    rw [hedge a, hedge b, hab]
  exact card_openComponents_le_height rowOf hinjRow

end OneesanFormal.CutGridCrossing
