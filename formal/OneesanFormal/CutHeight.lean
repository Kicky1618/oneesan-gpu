import Mathlib.Data.Fintype.Card
import Mathlib.Tactic

namespace OneesanFormal.CutHeight

/-- At a vertical cut through an `r`-row strip there are exactly `r` physical
horizontal edge slots.  We use `Fin r` as their canonical row index. -/
abbrev CrossingSlot (r : Nat) := Fin r

@[simp] theorem card_crossingSlot (r : Nat) :
    Fintype.card (CrossingSlot r) = r := by
  simp [CrossingSlot]

/-- Abstract form of the geometric height argument.  If every live/open
component crossing a vertical separator can be assigned a physical crossing
edge, and distinct components use distinct edges, then at most `r` components
can be live at once. -/
theorem card_openComponents_le_height
    {r : Nat} {OpenComponent : Type*} [Fintype OpenComponent]
    (crossingRow : OpenComponent → CrossingSlot r)
    (hinj : Function.Injective crossingRow) :
    Fintype.card OpenComponent ≤ r := by
  rw [← card_crossingSlot r]
  exact Fintype.card_le_of_injective crossingRow hinj

/-- Set-valued version convenient when open components are represented as a
finite subset of a larger component type. -/
theorem ncard_openComponents_le_height
    {r : Nat} {Component : Type*}
    (openComponents : Finset Component)
    (crossingRow : Component → CrossingSlot r)
    (hinj : Set.InjOn crossingRow ↑openComponents) :
    openComponents.card ≤ r := by
  classical
  let f : {c // c ∈ openComponents} → CrossingSlot r := fun c => crossingRow c.1
  have hf : Function.Injective f := by
    intro a b hab
    apply Subtype.ext
    exact hinj a.2 b.2 hab
  simpa using
    (card_openComponents_le_height (r := r)
      (OpenComponent := {c // c ∈ openComponents}) f hf)

/-- A useful equivalent formulation for frontier stacks: if every stack entry
has a distinct crossing-row witness, its length is at most the strip height. -/
theorem stack_length_le_height
    {r : Nat} {Component : Type*}
    (stack : List Component)
    (hnodup : stack.Nodup)
    (crossingRow : Component → CrossingSlot r)
    (hinj : Set.InjOn crossingRow {c | c ∈ stack}) :
    stack.length ≤ r := by
  classical
  have hcard : stack.toFinset.card = stack.length := by
    simpa using List.toFinset_card_of_nodup hnodup
  rw [← hcard]
  apply ncard_openComponents_le_height stack.toFinset crossingRow
  intro a ha b hb hab
  apply hinj
  · simpa using ha
  · simpa using hb
  · exact hab

end OneesanFormal.CutHeight
