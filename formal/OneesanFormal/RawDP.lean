import OneesanFormal.LoopDP
import Mathlib.Data.Fin.SuccPred

namespace OneesanFormal.RawDP
open OneesanFormal.TableDP
open OneesanFormal.LoopDP
open OneesanFormal.BitColumnDP

/-- `Fin.foldl` Boolean conjunction is true exactly when every entry is true. -/
theorem foldAnd_eq_true {n : Nat} (f : Fin n → Bool) :
    Fin.foldl n (fun acc i => acc && f i) true = true ↔ ∀ i, f i = true := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Fin.foldl_succ_last, Bool.and_eq_true, Fin.forall_fin_succ']
      change
        (Fin.foldl n (fun acc i => acc && f i.castSucc) true = true ∧
          f (Fin.last n) = true) ↔ _
      rw [ih (fun i : Fin n => f i.castSucc)]

/-- Raw checkerboard obstruction at one row boundary. -/
def badBits (x y : Nat) (r : Nat) : Bool :=
  let a := x.testBit r
  let b := x.testBit (r + 1)
  let c := y.testBit r
  let d := y.testBit (r + 1)
  (a != b) && (c != d) && (a != c)

/-- Allocation-free compatibility predicate on the integer state numbers. -/
def rawCompatible (h : Nat) (x y : Nat) : Bool :=
  Fin.foldl (h - 1) (fun acc r => acc && !(badBits x y r.val)) true

/-- The raw integer bit test is exactly the semantic BitVec compatibility. -/
theorem rawCompatible_eq_true_iff_bitCompatible
    {h : Nat} (x y : Fin (2 ^ h)) :
    rawCompatible h x.val y.val = true ↔
      bitCompatible (BitVec.ofFin x) (BitVec.ofFin y) := by
  rw [rawCompatible, foldAnd_eq_true]
  constructor
  · intro hall r
    have hr := hall r
    have hbadFalse : badBits x.val y.val r.val = false := by
      cases hb : badBits x.val y.val r.val <;> simp_all
    intro hcb
    unfold boolCheckerboard at hcb
    rcases hcb with ⟨hse, hsw, hne⟩
    have ha : (BitVec.ofFin x).getLsb ⟨r.val, by omega⟩ = x.val.testBit r.val := rfl
    have hb : (BitVec.ofFin x).getLsb ⟨r.val + 1, by omega⟩ = x.val.testBit (r.val + 1) := rfl
    have hc : (BitVec.ofFin y).getLsb ⟨r.val, by omega⟩ = y.val.testBit r.val := rfl
    have hd : (BitVec.ofFin y).getLsb ⟨r.val + 1, by omega⟩ = y.val.testBit (r.val + 1) := rfl
    simp only [ha, hb, hc, hd] at hse hsw hne
    cases hxa : x.val.testBit r.val <;>
      cases hxb : x.val.testBit (r.val + 1) <;>
      cases hya : y.val.testBit r.val <;>
      cases hyb : y.val.testBit (r.val + 1) <;>
      simp [badBits, hxa, hxb, hya, hyb] at hbadFalse hse hsw hne
  · intro hsem r
    have hr := hsem r
    have ha : (BitVec.ofFin x).getLsb ⟨r.val, by omega⟩ = x.val.testBit r.val := rfl
    have hb : (BitVec.ofFin x).getLsb ⟨r.val + 1, by omega⟩ = x.val.testBit (r.val + 1) := rfl
    have hc : (BitVec.ofFin y).getLsb ⟨r.val, by omega⟩ = y.val.testBit r.val := rfl
    have hd : (BitVec.ofFin y).getLsb ⟨r.val + 1, by omega⟩ = y.val.testBit (r.val + 1) := rfl
    simp only [ha, hb, hc, hd] at hr
    cases hxa : x.val.testBit r.val <;>
      cases hxb : x.val.testBit (r.val + 1) <;>
      cases hya : y.val.testBit r.val <;>
      cases hyb : y.val.testBit (r.val + 1) <;>
      simp [badBits, hxa, hxb, hya, hyb, boolCheckerboard] at hr ⊢

end OneesanFormal.RawDP

namespace OneesanFormal.RawDP
open OneesanFormal.TableDP
open OneesanFormal.LoopDP
open OneesanFormal.BitColumnDP

/-- Tight raw-state transfer using only Nat.testBit and Fin.foldl. -/
def rawStep (h : Nat) (prev : Vector Nat (NStates h)) : Vector Nat (NStates h) :=
  Vector.ofFn fun y : Fin (NStates h) =>
    foldSum (NStates h) fun x =>
      if rawCompatible h x.val y.val then prev.get x else 0

def rawDP (h : Nat) : Nat → Vector Nat (NStates h)
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => rawStep h (rawDP h k)

@[simp] theorem rawStep_get (h : Nat) (prev : Vector Nat (NStates h))
    (y : Fin (NStates h)) :
    (rawStep h prev).get y =
      foldSum (NStates h) fun x =>
        if rawCompatible h x.val y.val then prev.get x else 0 := by
  rw [rawStep, vector_get_ofFn]

@[simp] theorem rawDP_zero_get (h : Nat) (y : Fin (NStates h)) :
    (rawDP h 0).get y = 1 := by
  rw [rawDP, vector_get_ofFn]

/-- Raw integer DP refines the already-proved semantic materialized DP. -/
theorem rawDP_get_eq_fastDP_get (h k : Nat) (y : Fin (NStates h)) :
    (rawDP h k).get y = (fastDP h k).get y := by
  induction k generalizing y with
  | zero =>
      rw [rawDP_zero_get, fastDP_zero_get]
  | succ k ih =>
      rw [rawDP, rawStep_get, foldSum_eq_sum, fastDP, fastStep_get]
      apply Finset.sum_congr rfl
      intro x _
      rw [ih x]
      have hc := rawCompatible_eq_true_iff_bitCompatible x y
      cases hr : rawCompatible h x.val y.val <;> simp_all

/-- Total raw integer DP count. -/
def rawTotal (h k : Nat) : Nat :=
  foldSum (NStates h) fun y => (rawDP h k).get y

theorem rawTotal_eq_fastTotal (h k : Nat) : rawTotal h k = fastTotal h k := by
  unfold rawTotal fastTotal
  rw [foldSum_eq_sum]
  apply Finset.sum_congr rfl
  intro y _
  exact rawDP_get_eq_fastDP_get h k y

end OneesanFormal.RawDP
