import OneesanFormal.FastDP9
import Mathlib.Algebra.BigOperators.Fin

namespace OneesanFormal.CompressedDP9
open OneesanFormal.FastDP9
open OneesanFormal.LoopDP

set_option maxRecDepth 4096

def complement9 (x : Fin 512) : Fin 512 := ⟨511 - x.val, by omega⟩

theorem complement9_bijective : Function.Bijective complement9 := by
  native_decide

noncomputable def complement9Equiv : Fin 512 ≃ Fin 512 :=
  Equiv.ofBijective complement9 complement9_bijective

@[simp] theorem complement9Equiv_apply (x : Fin 512) : complement9Equiv x = complement9 x := rfl

theorem fastCompatible9_complement : ∀ x y : Fin 512,
    fastCompatible9 (complement9 x).val (complement9 y).val =
      fastCompatible9 x.val y.val := by
  native_decide

def rep9 (i : Fin 256) : Fin 512 := ⟨2 * i.val, by omega⟩

def pairState (p : Fin 256 × Fin 2) : Fin 512 :=
  if p.2.val = 0 then rep9 p.1 else complement9 (rep9 p.1)

theorem pairState_bijective : Function.Bijective pairState := by
  native_decide

noncomputable def pairStateEquiv : (Fin 256 × Fin 2) ≃ Fin 512 :=
  Equiv.ofBijective pairState pairState_bijective

@[simp] theorem pairStateEquiv_apply (p : Fin 256 × Fin 2) :
    pairStateEquiv p = pairState p := rfl

@[simp] theorem pairState_zero (i : Fin 256) : pairState (i, 0) = rep9 i := by
  simp [pairState]

@[simp] theorem pairState_one (i : Fin 256) : pairState (i, 1) = complement9 (rep9 i) := by
  simp [pairState]

/-- Split a 512-state sum into one representative and its complement for each
of 256 pairs. -/
theorem sum_pairState (f : Fin 512 → Nat) :
    (∑ x : Fin 512, f x) =
      ∑ i : Fin 256, (f (rep9 i) + f (complement9 (rep9 i))) := by
  rw [← Equiv.sum_comp pairStateEquiv f]
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro i _
  rw [Fin.sum_univ_two]
  simp


/-- DP values remain invariant under complementing all nine bits. -/
theorem fast9DP_complement (k : Nat) (y : Fin 512) :
    (fast9DP k).get (complement9 y) = (fast9DP k).get y := by
  induction k generalizing y with
  | zero => simp
  | succ k ih =>
      rw [fast9DP, fast9Step_get, fast9Step_get]
      rw [foldSum_eq_sum, foldSum_eq_sum]
      let f : Fin 512 → Nat := fun x =>
        if fastCompatible9 x.val (complement9 y).val then (fast9DP k).get x else 0
      have hreindex := Equiv.sum_comp complement9Equiv f
      rw [show (∑ x : Fin 512,
          if fastCompatible9 x.val (complement9 y).val then (fast9DP k).get x else 0) =
          ∑ x : Fin 512, f x by rfl]
      rw [← hreindex]
      apply Finset.sum_congr rfl
      intro x _
      simp only [f, complement9Equiv_apply]
      rw [ih x, fastCompatible9_complement x y]


/-- One 256-state step, pairing each column with its bitwise complement. -/
def compressedStep (prev : Vector Nat 256) : Vector Nat 256 :=
  Vector.ofFn fun y : Fin 256 =>
    foldSum 256 fun x =>
      (if fastCompatible9 (rep9 x).val (rep9 y).val then prev.get x else 0) +
      (if fastCompatible9 (complement9 (rep9 x)).val (rep9 y).val then prev.get x else 0)

def compressedDP : Nat → Vector Nat 256
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => compressedStep (compressedDP k)

@[simp] theorem compressedStep_get (prev : Vector Nat 256) (y : Fin 256) :
    (compressedStep prev).get y =
      foldSum 256 fun x =>
        (if fastCompatible9 (rep9 x).val (rep9 y).val then prev.get x else 0) +
        (if fastCompatible9 (complement9 (rep9 x)).val (rep9 y).val then prev.get x else 0) := by
  rw [compressedStep, OneesanFormal.TableDP.vector_get_ofFn]

@[simp] theorem compressedDP_zero_get (y : Fin 256) :
    (compressedDP 0).get y = 1 := by
  rw [compressedDP, OneesanFormal.TableDP.vector_get_ofFn]

/-- Each compressed cell is exactly the value of its even representative in
the original 512-state DP. -/
theorem compressedDP_get_eq_fast9DP_rep (k : Nat) (y : Fin 256) :
    (compressedDP k).get y = (fast9DP k).get (rep9 y) := by
  induction k generalizing y with
  | zero => simp
  | succ k ih =>
      rw [compressedDP, compressedStep_get, fast9DP, fast9Step_get]
      rw [foldSum_eq_sum, foldSum_eq_sum]
      rw [sum_pairState]
      apply Finset.sum_congr rfl
      intro x _
      rw [ih x]
      rw [fast9DP_complement k (rep9 x)]

/-- Total count reconstructed from the 256 complement-pair representatives. -/
def compressedTotal (k : Nat) : Nat :=
  foldSum 256 fun i => (compressedDP k).get i + (compressedDP k).get i

theorem compressedTotal_eq_fast9Total (k : Nat) :
    compressedTotal k = fast9Total k := by
  unfold compressedTotal fast9Total
  rw [foldSum_eq_sum, foldSum_eq_sum]
  rw [sum_pairState]
  apply Finset.sum_congr rfl
  intro i _
  rw [compressedDP_get_eq_fast9DP_rep k i]
  rw [fast9DP_complement k (rep9 i)]

end OneesanFormal.CompressedDP9
