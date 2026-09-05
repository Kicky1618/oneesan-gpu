import OneesanFormal.BitColumnDP

namespace OneesanFormal.TableDP
open OneesanFormal.BitColumnDP

abbrev NStates (h : Nat) := 2 ^ h

/-- Local lightweight replacement for the non-public `Vector.get_ofFn` lemma. -/
theorem vector_get_ofFn {n : Nat} {α : Type} (f : Fin n → α) (i : Fin n) :
    (Vector.ofFn f).get i = f i := by
  simp [Vector.get, Vector.ofFn]
  congr 1


def compatTable (h : Nat) : Vector (Vector Bool (NStates h)) (NStates h) :=
  Vector.ofFn fun y : Fin (NStates h) =>
    Vector.ofFn fun x : Fin (NStates h) =>
      decide (bitCompatible (BitVec.ofFin x) (BitVec.ofFin y))

@[simp] theorem compatTable_get (h : Nat) (y x : Fin (NStates h)) :
    ((compatTable h).get y).get x =
      decide (bitCompatible (BitVec.ofFin x) (BitVec.ofFin y)) := by
  rw [compatTable, vector_get_ofFn, vector_get_ofFn]

def tableStep (h : Nat)
    (table : Vector (Vector Bool (NStates h)) (NStates h))
    (prev : Vector Nat (NStates h)) : Vector Nat (NStates h) :=
  Vector.ofFn fun y : Fin (NStates h) =>
    ∑ x : Fin (NStates h),
      if ((table.get y).get x) then prev.get x else 0

def tableDPWith (h : Nat)
    (table : Vector (Vector Bool (NStates h)) (NStates h)) :
    Nat → Vector Nat (NStates h)
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => tableStep h table (tableDPWith h table k)

def tableDP (h k : Nat) : Vector Nat (NStates h) :=
  tableDPWith h (compatTable h) k

@[simp] theorem tableStep_get (h : Nat)
    (table : Vector (Vector Bool (NStates h)) (NStates h))
    (prev : Vector Nat (NStates h)) (y : Fin (NStates h)) :
    (tableStep h table prev).get y =
      ∑ x : Fin (NStates h), if ((table.get y).get x) then prev.get x else 0 := by
  rw [tableStep, vector_get_ofFn]

@[simp] theorem tableDPWith_zero_get (h : Nat) (y : Fin (NStates h)) :
    (tableDPWith h (compatTable h) 0).get y = 1 := by
  rw [tableDPWith, vector_get_ofFn]

/-- Precomputed-table DP agrees pointwise with the already-refined semantic DP. -/
theorem tableDPWith_get_eq_fastDP_get (h k : Nat) (y : Fin (NStates h)) :
    (tableDPWith h (compatTable h) k).get y = (fastDP h k).get y := by
  induction k generalizing y with
  | zero =>
      rw [tableDPWith_zero_get, fastDP_zero_get]
  | succ k ih =>
      rw [tableDPWith, tableStep_get, fastDP, fastStep_get]
      apply Finset.sum_congr rfl
      intro x _
      rw [compatTable_get, ih x]
      by_cases hc : bitCompatible (BitVec.ofFin x) (BitVec.ofFin y)
      · simp
      · simp

theorem tableDP_get_eq_fastDP_get (h k : Nat) (y : Fin (NStates h)) :
    (tableDP h k).get y = (fastDP h k).get y := by
  exact tableDPWith_get_eq_fastDP_get h k y

/-- Fast total using a table computed only once. -/
def tableTotal (h k : Nat) : Nat :=
  ∑ y : Fin (NStates h), (tableDP h k).get y

theorem tableTotal_eq_fastTotal (h k : Nat) : tableTotal h k = fastTotal h k := by
  unfold tableTotal fastTotal
  apply Finset.sum_congr rfl
  intro y _
  exact tableDP_get_eq_fastDP_get h k y

end OneesanFormal.TableDP
