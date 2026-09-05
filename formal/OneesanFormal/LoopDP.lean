import OneesanFormal.TableDP
import Mathlib.Algebra.BigOperators.Fin

namespace OneesanFormal.LoopDP
open OneesanFormal.TableDP
open OneesanFormal.BitColumnDP

/-- Allocation-free sum over all `Fin n` values. -/
def foldSum (n : Nat) (f : Fin n → Nat) : Nat :=
  Fin.foldl n (fun acc i => acc + f i) 0

theorem foldSum_eq_sum {n : Nat} (f : Fin n → Nat) :
    foldSum n f = ∑ i, f i := by
  induction n with
  | zero => simp [foldSum]
  | succ n ih =>
      rw [foldSum, Fin.foldl_succ_last, Fin.sum_univ_castSucc]
      change Fin.foldl n (fun acc i => acc + f i.castSucc) 0 + f (Fin.last n) = _
      have h := ih (fun i : Fin n => f i.castSucc)
      simpa [foldSum] using congrArg (fun z => z + f (Fin.last n)) h

/-- Same table transfer as `tableStep`, but implemented as a tight `Fin.foldl`. -/
def loopStep (h : Nat)
    (table : Vector (Vector Bool (NStates h)) (NStates h))
    (prev : Vector Nat (NStates h)) : Vector Nat (NStates h) :=
  Vector.ofFn fun y : Fin (NStates h) =>
    foldSum (NStates h) fun x => if ((table.get y).get x) then prev.get x else 0

def loopDPWith (h : Nat)
    (table : Vector (Vector Bool (NStates h)) (NStates h)) :
    Nat → Vector Nat (NStates h)
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => loopStep h table (loopDPWith h table k)

def loopDP (h k : Nat) : Vector Nat (NStates h) :=
  loopDPWith h (compatTable h) k

@[simp] theorem loopStep_get (h : Nat)
    (table : Vector (Vector Bool (NStates h)) (NStates h))
    (prev : Vector Nat (NStates h)) (y : Fin (NStates h)) :
    (loopStep h table prev).get y =
      foldSum (NStates h) fun x => if ((table.get y).get x) then prev.get x else 0 := by
  rw [loopStep, vector_get_ofFn]

@[simp] theorem loopDPWith_zero_get (h : Nat) (y : Fin (NStates h)) :
    (loopDPWith h (compatTable h) 0).get y = 1 := by
  rw [loopDPWith, vector_get_ofFn]

/-- The loop implementation is pointwise equal to the proven table DP. -/
theorem loopDPWith_get_eq_tableDPWith_get (h k : Nat) (y : Fin (NStates h)) :
    (loopDPWith h (compatTable h) k).get y =
      (tableDPWith h (compatTable h) k).get y := by
  induction k generalizing y with
  | zero =>
      rw [loopDPWith_zero_get, tableDPWith_zero_get]
  | succ k ih =>
      rw [loopDPWith, loopStep_get, tableDPWith, tableStep_get, foldSum_eq_sum]
      apply Finset.sum_congr rfl
      intro x _
      rw [ih x]

/-- Allocation-free total. -/
def loopTotal (h k : Nat) : Nat :=
  foldSum (NStates h) fun y => (loopDP h k).get y

theorem loopTotal_eq_tableTotal (h k : Nat) : loopTotal h k = tableTotal h k := by
  unfold loopTotal tableTotal loopDP tableDP
  rw [foldSum_eq_sum]
  apply Finset.sum_congr rfl
  intro y _
  exact loopDPWith_get_eq_tableDPWith_get h k y

end OneesanFormal.LoopDP
