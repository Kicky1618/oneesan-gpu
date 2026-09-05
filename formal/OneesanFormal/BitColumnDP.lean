import OneesanFormal.StripDP
import Mathlib.Data.BitVec
import Mathlib.Data.Vector.Basic

namespace OneesanFormal.BitColumnDP
open OneesanFormal.GridFaceBoundary
open OneesanFormal.StripDP

/-- Computable finite enumeration of bitvectors through their underlying Fin. -/
def bitVecEquivFin (h : Nat) : BitVec h ≃ Fin (2 ^ h) where
  toFun := BitVec.toFin
  invFun := BitVec.ofFin
  left_inv := BitVec.ofFin_toFin
  right_inv := BitVec.toFin_ofFin

instance bitVecFintype (h : Nat) : Fintype (BitVec h) :=
  Fintype.ofEquiv (Fin (2 ^ h)) (bitVecEquivFin h).symm

@[simp] theorem card_bitVec (h : Nat) : Fintype.card (BitVec h) = 2 ^ h := by
  rw [Fintype.card_congr (bitVecEquivFin h), Fintype.card_fin]

/-- The two GF(2) values, exposed as a computable Bool map. -/
def boolF2 : Bool → F2
  | false => 0
  | true => 1

@[simp] theorem boolF2_false : boolF2 false = 0 := rfl
@[simp] theorem boolF2_true : boolF2 true = 1 := rfl

theorem boolF2_injective : Function.Injective boolF2 := by
  intro a b h
  cases a <;> cases b <;> simp [boolF2] at h ⊢

/-- Read a bitvector as one semantic GF(2) face column. -/
def bitColumn {h : Nat} (x : BitVec h) : Column h :=
  fun r => boolF2 (x.getLsb r)

theorem bitColumn_injective {h : Nat} : Function.Injective (bitColumn (h := h)) := by
  intro x y hxy
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  let r : Fin h := ⟨i, hi⟩
  have hb : x.getLsb r = y.getLsb r := by
    apply boolF2_injective
    exact congrFun hxy r
  simpa [BitVec.getLsbD, BitVec.getLsb, r] using hb

@[simp] theorem card_column (h : Nat) : Fintype.card (Column h) = 2 ^ h := by
  rw [Fintype.card_fun, ZMod.card, Fintype.card_fin]

/-- Every semantic GF(2) column is represented by exactly one bitvector. -/
theorem bitColumn_bijective {h : Nat} : Function.Bijective (bitColumn (h := h)) := by
  apply (Fintype.bijective_iff_injective_and_card (bitColumn (h := h))).2
  exact ⟨bitColumn_injective, by simp⟩

noncomputable def bitColumnEquiv (h : Nat) : BitVec h ≃ Column h :=
  Equiv.ofBijective bitColumn bitColumn_bijective

end OneesanFormal.BitColumnDP

namespace OneesanFormal.BitColumnDP
open OneesanFormal.StripDP
open OneesanFormal.GridFaceBoundary

/-- Checkerboard predicate on raw Boolean bits. -/
def boolCheckerboard (nw ne se sw : Bool) : Prop :=
  nw = se ∧ ne = sw ∧ nw ≠ ne

instance boolCheckerboardDecidable (nw ne se sw : Bool) :
    Decidable (boolCheckerboard nw ne se sw) := by
  unfold boolCheckerboard
  infer_instance

@[simp] theorem f2Checkerboard_boolF2_iff (nw ne se sw : Bool) :
    OneesanFormal.CheckerboardGridBound.f2Checkerboard
      (boolF2 nw) (boolF2 ne) (boolF2 se) (boolF2 sw) ↔
      boolCheckerboard nw ne se sw := by
  cases nw <;> cases ne <;> cases se <;> cases sw <;>
    simp [boolF2, boolCheckerboard,
      OneesanFormal.CheckerboardGridBound.f2Checkerboard]

/-- Computable compatibility relation directly on bitvector states. -/
def bitCompatible {h : Nat} (x y : BitVec h) : Prop :=
  ∀ r : Fin (h - 1),
    ¬ boolCheckerboard
      (x.getLsb ⟨r.val, by omega⟩)
      (y.getLsb ⟨r.val, by omega⟩)
      (y.getLsb ⟨r.val + 1, by omega⟩)
      (x.getLsb ⟨r.val + 1, by omega⟩)

instance bitCompatibleDecidable (h : Nat) :
    DecidableRel (bitCompatible (h := h)) := fun _x _y =>
  Fintype.decidableForallFintype

@[simp] theorem bitCompatible_iff_columnCompatible {h : Nat} (x y : BitVec h) :
    bitCompatible x y ↔ columnCompatible (bitColumn x) (bitColumn y) := by
  constructor
  · intro hb r
    exact (f2Checkerboard_boolF2_iff _ _ _ _).not.mpr (hb r)
  · intro hc r
    exact (f2Checkerboard_boolF2_iff _ _ _ _).not.mp (hc r)

end OneesanFormal.BitColumnDP

namespace OneesanFormal.BitColumnDP
open OneesanFormal.StripDP

noncomputable def stateColumnEquiv (h : Nat) : Fin (2 ^ h) ≃ Column h :=
  (bitVecEquivFin h).symm.trans (bitColumnEquiv h)

@[simp] theorem stateColumnEquiv_apply (h : Nat) (i : Fin (2 ^ h)) :
    stateColumnEquiv h i = bitColumn (BitVec.ofFin i) := rfl

/-- One materialized transfer step over exactly `2^h` column states. -/
def fastStep (h : Nat) (prev : Vector Nat (2 ^ h)) : Vector Nat (2 ^ h) :=
  Vector.ofFn fun y : Fin (2 ^ h) =>
    ∑ x : Fin (2 ^ h),
      if bitCompatible (BitVec.ofFin x) (BitVec.ofFin y) then prev.get x else 0

/-- Materialized DP after `k` transitions; `k=0` gives one length-one chain per ending column. -/
def fastDP (h : Nat) : Nat → Vector Nat (2 ^ h)
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => fastStep h (fastDP h k)

@[simp] theorem fastDP_zero_get (h : Nat) (y : Fin (2 ^ h)) :
    (fastDP h 0).get y = 1 := by
  simp [fastDP, Vector.get]

@[simp] theorem fastStep_get (h : Nat) (prev : Vector Nat (2 ^ h))
    (y : Fin (2 ^ h)) :
    (fastStep h prev).get y =
      ∑ x : Fin (2 ^ h),
        if bitCompatible (BitVec.ofFin x) (BitVec.ofFin y) then prev.get x else 0 := by
  simp [fastStep, Vector.get]
  congr 1

/-- Cellwise refinement theorem: the efficient materialized state DP computes
exactly the semantic `endCount` recurrence. -/
theorem fastDP_get_eq_endCount (h k : Nat) (y : Fin (2 ^ h)) :
    (fastDP h k).get y = endCount h k (stateColumnEquiv h y) := by
  induction k generalizing y with
  | zero =>
      rw [fastDP_zero_get]
      rfl
  | succ k ih =>
      simp only [fastDP, fastStep_get, endCount]
      have hsum := Fintype.sum_equiv (stateColumnEquiv h)
        (fun x : Fin (2 ^ h) =>
          if bitCompatible (BitVec.ofFin x) (BitVec.ofFin y)
          then (fastDP h k).get x else 0)
        (fun c : Column h =>
          if columnCompatible c (stateColumnEquiv h y) then endCount h k c else 0)
        (by
          intro x
          rw [stateColumnEquiv_apply, stateColumnEquiv_apply]
          rw [ih x]
          by_cases hc : bitCompatible (BitVec.ofFin x) (BitVec.ofFin y)
          · have hc' : columnCompatible (bitColumn (BitVec.ofFin x))
                (bitColumn (BitVec.ofFin y)) :=
              (bitCompatible_iff_columnCompatible _ _).mp hc
            simp [hc']
          · have hc' : ¬ columnCompatible (bitColumn (BitVec.ofFin x))
                (bitColumn (BitVec.ofFin y)) := by
              intro hcol
              exact hc ((bitCompatible_iff_columnCompatible _ _).mpr hcol)
            simp [hc'])
      exact hsum

/-- Total materialized DP count. -/
def fastTotal (h k : Nat) : Nat :=
  ∑ y : Fin (2 ^ h), (fastDP h k).get y

/-- The fast total is exactly the semantic transfer-DP total. -/
theorem fastTotal_eq_endCount_sum (h k : Nat) :
    fastTotal h k = ∑ y : Column h, endCount h k y := by
  unfold fastTotal
  have hsum := Fintype.sum_equiv (stateColumnEquiv h)
    (fun y : Fin (2 ^ h) => (fastDP h k).get y)
    (fun c : Column h => endCount h k c)
    (by intro y; exact fastDP_get_eq_endCount h k y)
  exact hsum
