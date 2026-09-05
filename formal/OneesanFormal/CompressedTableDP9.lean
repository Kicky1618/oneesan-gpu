import OneesanFormal.CompressedDP9

namespace OneesanFormal.CompressedTableDP9
open OneesanFormal.FastDP9
open OneesanFormal.CompressedDP9
open OneesanFormal.LoopDP
open OneesanFormal.TableDP

set_option maxRecDepth 4096

def boolNat (b : Bool) : Nat := if b then 1 else 0

def compressedWeight (x y : Fin 256) : Nat :=
  boolNat (fastCompatible9 (rep9 x).val (rep9 y).val) +
  boolNat (fastCompatible9 (complement9 (rep9 x)).val (rep9 y).val)

def weightTable : Vector (Vector Nat 256) 256 :=
  Vector.ofFn fun y => Vector.ofFn fun x => compressedWeight x y

@[simp] theorem weightTable_get (x y : Fin 256) :
    (weightTable.get y).get x = compressedWeight x y := by
  rw [weightTable, vector_get_ofFn, vector_get_ofFn]

def tableStep (table : Vector (Vector Nat 256) 256)
    (prev : Vector Nat 256) : Vector Nat 256 :=
  Vector.ofFn fun y => foldSum 256 fun x => (table.get y).get x * prev.get x

@[simp] theorem tableStep_get (table : Vector (Vector Nat 256) 256)
    (prev : Vector Nat 256) (y : Fin 256) :
    (tableStep table prev).get y =
      foldSum 256 fun x => (table.get y).get x * prev.get x := by
  rw [tableStep, vector_get_ofFn]

def tableDPWith (table : Vector (Vector Nat 256) 256) : Nat → Vector Nat 256
  | 0 => Vector.ofFn fun _ => 1
  | k + 1 => tableStep table (tableDPWith table k)

@[simp] theorem tableDPWith_zero_get
    (table : Vector (Vector Nat 256) 256) (y : Fin 256) :
    (tableDPWith table 0).get y = 1 := by
  rw [tableDPWith, vector_get_ofFn]

def tableDP : Nat → Vector Nat 256 := tableDPWith weightTable

/-- The precomputed-weight implementation is pointwise equal to the 256-state
semantic complement-pair DP. -/
theorem tableDPWith_get_eq_compressedDP_get (k : Nat) (y : Fin 256) :
    (tableDPWith weightTable k).get y = (compressedDP k).get y := by
  induction k generalizing y with
  | zero => rw [tableDPWith_zero_get, compressedDP_zero_get]
  | succ k ih =>
      rw [tableDPWith, tableStep_get, compressedDP, compressedStep_get]
      rw [foldSum_eq_sum, foldSum_eq_sum]
      apply Finset.sum_congr rfl
      intro x _
      rw [weightTable_get, ih x]
      unfold compressedWeight boolNat
      cases h₁ : fastCompatible9 (rep9 x).val (rep9 y).val <;>
        cases h₂ : fastCompatible9 (complement9 (rep9 x)).val (rep9 y).val <;>
        simp_all [two_mul]

@[simp] theorem tableDP_get_eq_compressedDP_get (k : Nat) (y : Fin 256) :
    (tableDP k).get y = (compressedDP k).get y :=
  tableDPWith_get_eq_compressedDP_get k y

/-- Executable total from the precomputed 256x256 weight table. -/
def tableTotal : Nat := foldSum 256 fun i => 2 * (tableDP 26).get i

theorem tableTotal_eq_compressedTotal : tableTotal = compressedTotal 26 := by
  unfold tableTotal compressedTotal
  rw [foldSum_eq_sum, foldSum_eq_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [tableDP_get_eq_compressedDP_get]
  omega


/-- Compiler-backed `native_decide` evaluation of the production 9x27
strip transfer count. -/
theorem tableTotal_value : tableTotal =
    3165928478117342768922265826341920493835329849417470440184018662 := by
  native_decide

end OneesanFormal.CompressedTableDP9
