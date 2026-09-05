import Mathlib.Combinatorics.SimpleGraph.Basic

namespace OneesanFormal.GridGraph

open SimpleGraph

abbrev GridVertex (n : Nat) := Fin (n + 1) × Fin (n + 1)

/-- One directed positive-coordinate grid step; `SimpleGraph.fromRel` below
adds the reverse direction automatically. -/
def forwardStep {n : Nat} (a b : GridVertex n) : Prop :=
  (a.1 = b.1 ∧ a.2.val + 1 = b.2.val) ∨
  (a.2 = b.2 ∧ a.1.val + 1 = b.1.val)

/-- The `(n+1) × (n+1)` vertex grid surrounding an `n × n` bounded-face
matrix. -/
def gridGraph (n : Nat) : SimpleGraph (GridVertex n) :=
  SimpleGraph.fromRel forwardStep

/-- Explicit four-direction adjacency characterization. -/
theorem gridGraph_adj_iff {n : Nat} {a b : GridVertex n} :
    (gridGraph n).Adj a b ↔
      (a.1 = b.1 ∧
        (a.2.val + 1 = b.2.val ∨ b.2.val + 1 = a.2.val)) ∨
      (a.2 = b.2 ∧
        (a.1.val + 1 = b.1.val ∨ b.1.val + 1 = a.1.val)) := by
  rw [gridGraph, SimpleGraph.fromRel_adj]
  constructor
  · rintro ⟨_, h | h⟩
    · rcases h with h | h
      · exact Or.inl ⟨h.1, Or.inl h.2⟩
      · exact Or.inr ⟨h.1, Or.inl h.2⟩
    · rcases h with h | h
      · exact Or.inl ⟨h.1.symm, Or.inr h.2⟩
      · exact Or.inr ⟨h.1.symm, Or.inr h.2⟩
  · intro h
    refine ⟨?_, ?_⟩
    · intro hab
      subst b
      rcases h with ⟨_, hstep⟩ | ⟨_, hstep⟩ <;> omega
    · rcases h with h | h
      · rcases h.2 with hright | hleft
        · exact Or.inl (Or.inl ⟨h.1, hright⟩)
        · exact Or.inr (Or.inl ⟨h.1.symm, hleft⟩)
      · rcases h.2 with hdown | hup
        · exact Or.inl (Or.inr ⟨h.1, hdown⟩)
        · exact Or.inr (Or.inr ⟨h.1.symm, hup⟩)


/-- The four concrete neighbors of an interior grid vertex. -/
theorem neighborSet_interior
    {n r c : Nat} (hr : r + 1 < n) (hc : c + 1 < n) :
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
    (gridGraph n).neighborSet center = {left, right, up, down} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · intro h
    rcases h with ⟨hrow, hcol⟩ | ⟨hcol, hrow⟩
    · rcases hcol with hright | hleft
      · right; left
        apply Prod.ext
        · exact hrow.symm
        · apply Fin.ext
          simpa using hright.symm
      · left
        apply Prod.ext
        · exact hrow.symm
        · apply Fin.ext
          simpa using hleft
    · rcases hrow with hdown | hup
      · right; right; right
        apply Prod.ext
        · apply Fin.ext
          simpa using hdown.symm
        · exact hcol.symm
      · right; right; left
        apply Prod.ext
        · apply Fin.ext
          simpa using hup
        · exact hcol.symm
  · intro h
    rcases h with rfl | rfl | rfl | rfl
    · exact Or.inl ⟨rfl, Or.inr (by simp)⟩
    · exact Or.inl ⟨rfl, Or.inl (by simp [Nat.add_assoc])⟩
    · exact Or.inr ⟨rfl, Or.inr (by simp)⟩
    · exact Or.inr ⟨rfl, Or.inl (by simp [Nat.add_assoc])⟩


/-- Northwest and southeast corner vertices used by the counting problem. -/
def northwestCorner (n : Nat) : GridVertex n :=
  (⟨0, by omega⟩, ⟨0, by omega⟩)

def southeastCorner (n : Nat) : GridVertex n :=
  (⟨n, by omega⟩, ⟨n, by omega⟩)

theorem corner_ne {n : Nat} (hn : 0 < n) :
    northwestCorner n ≠ southeastCorner n := by
  intro h
  have hh := congrArg (fun z : GridVertex n => z.1.val) h
  dsimp [northwestCorner, southeastCorner] at hh
  omega


/-- Coordinate characterization of the northwest corner. -/
theorem vertex_eq_northwest_iff {n r c : Nat} (hr : r ≤ n) (hc : c ≤ n) :
    ((⟨r, by omega⟩, ⟨c, by omega⟩) : GridVertex n) = northwestCorner n ↔
      r = 0 ∧ c = 0 := by
  constructor
  · intro h
    constructor
    · have hh := congrArg (fun z : GridVertex n => z.1.val) h
      simpa [northwestCorner] using hh
    · have hh := congrArg (fun z : GridVertex n => z.2.val) h
      simpa [northwestCorner] using hh
  · rintro ⟨rfl, rfl⟩
    rfl

/-- Coordinate characterization of the southeast corner. -/
theorem vertex_eq_southeast_iff {n r c : Nat} (hr : r ≤ n) (hc : c ≤ n) :
    ((⟨r, by omega⟩, ⟨c, by omega⟩) : GridVertex n) = southeastCorner n ↔
      r = n ∧ c = n := by
  constructor
  · intro h
    constructor
    · have hh := congrArg (fun z : GridVertex n => z.1.val) h
      simpa [southeastCorner] using hh
    · have hh := congrArg (fun z : GridVertex n => z.2.val) h
      simpa [southeastCorner] using hh
  · rintro ⟨rfl, rfl⟩
    rfl

end OneesanFormal.GridGraph
