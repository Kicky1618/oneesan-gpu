import OneesanFormal.GridGraph

namespace OneesanFormal.GridBoundaryNeighbors

open SimpleGraph
open OneesanFormal.GridGraph

/-- Northwest corner has exactly east and south neighbors. -/
theorem neighborSet_northwest {n : Nat} (hn : 0 < n) :
    let right : GridVertex n := (⟨0, by omega⟩, ⟨1, by omega⟩)
    let down : GridVertex n := (⟨1, by omega⟩, ⟨0, by omega⟩)
    (gridGraph n).neighborSet (northwestCorner n) = {right, down} := by
  dsimp [northwestCorner]
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simpa using hright.symm
    · simp at hleft
    · right
      apply Prod.ext
      · apply Fin.ext
        simpa using hdown.symm
      · exact hcol.symm
    · simp at hup
  · rintro (rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inl (by simp)⟩
    · exact Or.inr ⟨rfl, Or.inl (by simp)⟩

/-- Northeast corner has exactly west and south neighbors. -/
theorem neighborSet_northeast {n : Nat} (hn : 0 < n) :
    let center : GridVertex n := (⟨0, by omega⟩, ⟨n, by omega⟩)
    let left : GridVertex n := (⟨0, by omega⟩, ⟨n - 1, by omega⟩)
    let down : GridVertex n := (⟨1, by omega⟩, ⟨n, by omega⟩)
    (gridGraph n).neighborSet center = {left, down} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · have hx : x.2.val ≤ n := Nat.le_of_lt_succ x.2.isLt
      simp at hright
      omega
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simp at hleft
        dsimp
        omega
    · right
      apply Prod.ext
      · apply Fin.ext
        simpa using hdown.symm
      · exact hcol.symm
    · simp at hup
  · rintro (rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inr ⟨rfl, Or.inl (by simp)⟩

/-- Southwest corner has exactly east and north neighbors. -/
theorem neighborSet_southwest {n : Nat} (hn : 0 < n) :
    let center : GridVertex n := (⟨n, by omega⟩, ⟨0, by omega⟩)
    let right : GridVertex n := (⟨n, by omega⟩, ⟨1, by omega⟩)
    let up : GridVertex n := (⟨n - 1, by omega⟩, ⟨0, by omega⟩)
    (gridGraph n).neighborSet center = {right, up} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simpa using hright.symm
    · simp at hleft
    · have hx : x.1.val ≤ n := Nat.le_of_lt_succ x.1.isLt
      simp at hdown
      omega
    · right
      apply Prod.ext
      · apply Fin.ext
        simp at hup
        dsimp
        omega
      · exact hcol.symm
  · rintro (rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inl (by simp)⟩
    · exact Or.inr ⟨rfl, Or.inr (by simp; omega)⟩

/-- Southeast corner has exactly west and north neighbors. -/
theorem neighborSet_southeast {n : Nat} (hn : 0 < n) :
    let left : GridVertex n := (⟨n, by omega⟩, ⟨n - 1, by omega⟩)
    let up : GridVertex n := (⟨n - 1, by omega⟩, ⟨n, by omega⟩)
    (gridGraph n).neighborSet (southeastCorner n) = {left, up} := by
  dsimp [southeastCorner]
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · have hx : x.2.val ≤ n := Nat.le_of_lt_succ x.2.isLt
      simp at hright
      omega
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simp at hleft
        dsimp
        omega
    · have hx : x.1.val ≤ n := Nat.le_of_lt_succ x.1.isLt
      simp at hdown
      omega
    · right
      apply Prod.ext
      · apply Fin.ext
        simp at hup
        dsimp
        omega
      · exact hcol.symm
  · rintro (rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inr ⟨rfl, Or.inr (by simp; omega)⟩

/-- A non-corner top-boundary vertex has west, east, and south neighbors. -/
theorem neighborSet_top {n c : Nat} (hc0 : 0 < c) (hcn : c < n) :
    let center : GridVertex n := (⟨0, by omega⟩, ⟨c, by omega⟩)
    let left : GridVertex n := (⟨0, by omega⟩, ⟨c - 1, by omega⟩)
    let right : GridVertex n := (⟨0, by omega⟩, ⟨c + 1, by omega⟩)
    let down : GridVertex n := (⟨1, by omega⟩, ⟨c, by omega⟩)
    (gridGraph n).neighborSet center = {left, right, down} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · right; left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simpa using hright.symm
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simp at hleft
        dsimp
        omega
    · right; right
      apply Prod.ext
      · apply Fin.ext
        simpa using hdown.symm
      · exact hcol.symm
    · simp at hup
  · rintro (rfl | rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inl ⟨rfl, Or.inl (by simp)⟩
    · exact Or.inr ⟨rfl, Or.inl (by simp)⟩

/-- A non-corner bottom-boundary vertex has west, east, and north neighbors. -/
theorem neighborSet_bottom {n c : Nat} (hc0 : 0 < c) (hcn : c < n) :
    let center : GridVertex n := (⟨n, by omega⟩, ⟨c, by omega⟩)
    let left : GridVertex n := (⟨n, by omega⟩, ⟨c - 1, by omega⟩)
    let right : GridVertex n := (⟨n, by omega⟩, ⟨c + 1, by omega⟩)
    let up : GridVertex n := (⟨n - 1, by omega⟩, ⟨c, by omega⟩)
    (gridGraph n).neighborSet center = {left, right, up} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · right; left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simpa using hright.symm
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simp at hleft
        dsimp
        omega
    · have hx : x.1.val ≤ n := Nat.le_of_lt_succ x.1.isLt
      simp at hdown
      omega
    · right; right
      apply Prod.ext
      · apply Fin.ext
        simp at hup
        dsimp
        omega
      · exact hcol.symm
  · rintro (rfl | rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inl ⟨rfl, Or.inl (by simp)⟩
    · exact Or.inr ⟨rfl, Or.inr (by simp; omega)⟩

/-- A non-corner left-boundary vertex has east, north, and south neighbors. -/
theorem neighborSet_left {n r : Nat} (hr0 : 0 < r) (hrn : r < n) :
    let center : GridVertex n := (⟨r, by omega⟩, ⟨0, by omega⟩)
    let right : GridVertex n := (⟨r, by omega⟩, ⟨1, by omega⟩)
    let up : GridVertex n := (⟨r - 1, by omega⟩, ⟨0, by omega⟩)
    let down : GridVertex n := (⟨r + 1, by omega⟩, ⟨0, by omega⟩)
    (gridGraph n).neighborSet center = {right, up, down} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simpa using hright.symm
    · simp at hleft
    · right; right
      apply Prod.ext
      · apply Fin.ext
        simpa using hdown.symm
      · exact hcol.symm
    · right; left
      apply Prod.ext
      · apply Fin.ext
        simp at hup
        dsimp
        omega
      · exact hcol.symm
  · rintro (rfl | rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inl (by simp)⟩
    · exact Or.inr ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inr ⟨rfl, Or.inl (by simp)⟩

/-- A non-corner right-boundary vertex has west, north, and south neighbors. -/
theorem neighborSet_right {n r : Nat} (hr0 : 0 < r) (hrn : r < n) :
    let center : GridVertex n := (⟨r, by omega⟩, ⟨n, by omega⟩)
    let left : GridVertex n := (⟨r, by omega⟩, ⟨n - 1, by omega⟩)
    let up : GridVertex n := (⟨r - 1, by omega⟩, ⟨n, by omega⟩)
    let down : GridVertex n := (⟨r + 1, by omega⟩, ⟨n, by omega⟩)
    (gridGraph n).neighborSet center = {left, up, down} := by
  dsimp
  ext x
  rw [SimpleGraph.mem_neighborSet, gridGraph_adj_iff]
  constructor
  · rintro (⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩)
    · have hx : x.2.val ≤ n := Nat.le_of_lt_succ x.2.isLt
      simp at hright
      omega
    · left
      apply Prod.ext
      · exact hrow.symm
      · apply Fin.ext
        simp at hleft
        dsimp
        omega
    · right; right
      apply Prod.ext
      · apply Fin.ext
        simpa using hdown.symm
      · exact hcol.symm
    · right; left
      apply Prod.ext
      · apply Fin.ext
        simp at hup
        dsimp
        omega
      · exact hcol.symm
  · rintro (rfl | rfl | rfl)
    · exact Or.inl ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inr ⟨rfl, Or.inr (by simp; omega)⟩
    · exact Or.inr ⟨rfl, Or.inl (by simp)⟩

end OneesanFormal.GridBoundaryNeighbors
