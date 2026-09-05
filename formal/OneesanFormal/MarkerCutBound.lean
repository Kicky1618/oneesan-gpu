import OneesanFormal.PathCutCrossing
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace OneesanFormal.MarkerCutBound

open OneesanFormal.GridGraph
open OneesanFormal.PathCutCrossing

/-- Abstract semantic data carried by the unmatched `L` markers at one
vertical cut.  `Marker` denotes *marker occurrences*, not physical-component
IDs: the same physical component may own several markers.  Each occurrence is
witnessed by one horizontal path edge crossing the current cut. -/
structure MarkerCrossingWitness (r w : Nat) where
  Marker : Type*
  [markerFintype : Fintype Marker]
  cut : Fin w
  crossingRow : Marker → Fin r
  /-- Different unmatched marker occurrences require different physical
  crossing edges, hence different crossing rows. -/
  injective_crossingRow : Function.Injective crossingRow

attribute [instance] MarkerCrossingWitness.markerFintype

/-- A strip with `r` horizontal crossing slots cannot carry more than `r`
unmatched marker occurrences at one vertical cut. -/
theorem marker_card_le_height {r w : Nat} (W : MarkerCrossingWitness r w) :
    Fintype.card W.Marker ≤ r := by
  let e : W.Marker ↪ Fin r := ⟨W.crossingRow, W.injective_crossingRow⟩
  simpa using Fintype.card_le_of_injective e e.injective

/-- List-position version matching the implementation better.  Component IDs
may repeat in `stack`; what matters is that the `n` marker *positions* have
distinct crossing rows. -/
structure StackOccurrenceWitness (r n : Nat) where
  crossingRow : Fin n → Fin r
  injective_crossingRow : Function.Injective crossingRow

/-- The number of stack occurrences is at most the physical strip height. -/
theorem stack_occurrences_le_height {r n : Nat} (W : StackOccurrenceWitness r n) :
    n ≤ r := by
  let e : Fin n ↪ Fin r := ⟨W.crossingRow, W.injective_crossingRow⟩
  simpa using Fintype.card_le_of_injective e e.injective

/-- A convenient constructor from an explicit family of pairwise-distinct
crossing rows. -/
def StackOccurrenceWitness.ofRows {r : Nat} {rows : List (Fin r)}
    (h : rows.Nodup) : StackOccurrenceWitness r rows.length where
  crossingRow i := rows.get i
  injective_crossingRow := h.injective_get

end OneesanFormal.MarkerCutBound
