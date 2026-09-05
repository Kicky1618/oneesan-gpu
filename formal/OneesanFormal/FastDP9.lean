import OneesanFormal.RawDP

namespace OneesanFormal.FastDP9

set_option maxRecDepth 4096
open OneesanFormal.RawDP
open OneesanFormal.LoopDP
open OneesanFormal.TableDP

def fastCompatible9 (x y : Nat) : Bool :=
  let d := x ^^^ y
  (((x ^^^ (x >>> 1)) &&& d &&& (d >>> 1)) == 0)

theorem fastCompatible9_correct :
    ∀ x : Fin 512, ∀ y : Fin 512,
      fastCompatible9 x.val y.val = rawCompatible 9 x.val y.val := by
  native_decide

def fast9Step (prev : Vector Nat 512) : Vector Nat 512 :=
  Vector.ofFn fun y : Fin 512 =>
    foldSum 512 fun x => if fastCompatible9 x.val y.val then prev.get x else 0

def fast9DP : Nat → Vector Nat 512
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => fast9Step (fast9DP k)

@[simp] theorem fast9Step_get (prev : Vector Nat 512) (y : Fin 512) :
    (fast9Step prev).get y =
      foldSum 512 fun x => if fastCompatible9 x.val y.val then prev.get x else 0 := by
  rw [fast9Step, OneesanFormal.TableDP.vector_get_ofFn]

@[simp] theorem fast9DP_zero_get (y : Fin 512) : (fast9DP 0).get y = 1 := by
  rw [fast9DP, OneesanFormal.TableDP.vector_get_ofFn]

theorem fast9DP_get_eq_rawDP_get (k : Nat) (y : Fin 512) :
    (fast9DP k).get y = (rawDP 9 k).get y := by
  induction k generalizing y with
  | zero => rw [fast9DP_zero_get, rawDP_zero_get]
  | succ k ih =>
      rw [fast9DP, fast9Step_get, rawDP, rawStep_get]
      rw [foldSum_eq_sum, foldSum_eq_sum]
      apply Finset.sum_congr rfl
      intro x _
      rw [ih x, fastCompatible9_correct x y]

def fast9Total (k : Nat) : Nat := foldSum 512 fun y => (fast9DP k).get y

theorem fast9Total_eq_rawTotal (k : Nat) : fast9Total k = rawTotal 9 k := by
  unfold fast9Total rawTotal
  rw [foldSum_eq_sum, foldSum_eq_sum]
  apply Finset.sum_congr rfl
  intro y _
  exact fast9DP_get_eq_rawDP_get k y


end OneesanFormal.FastDP9
