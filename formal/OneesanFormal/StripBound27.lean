import OneesanFormal.CornerPathCounting
import OneesanFormal.CheckerboardBound
import Mathlib.Algebra.BigOperators.Fin

namespace OneesanFormal.StripBound27

open OneesanFormal.GridFaceBoundary
open OneesanFormal.CheckerboardGridBound
open OneesanFormal.CornerPathCounting
open OneesanFormal.CheckerboardBound

abbrev Strip9x27 := Fin 9 × Fin 27 → F2

/-- No checkerboard entirely inside a 9x27 row strip. -/
def stripNoCheckerboard (f : Strip9x27) : Prop :=
  ∀ r : Fin 8, ∀ c : Fin 26,
    ¬ f2Checkerboard
      (f (⟨r.val, by omega⟩, ⟨c.val, by omega⟩))
      (f (⟨r.val, by omega⟩, ⟨c.val + 1, by omega⟩))
      (f (⟨r.val + 1, by omega⟩, ⟨c.val + 1, by omega⟩))
      (f (⟨r.val + 1, by omega⟩, ⟨c.val, by omega⟩))

noncomputable instance stripNoCheckerboardDecidable : DecidablePred stripNoCheckerboard :=
  Classical.decPred _

/-- Restrict a 27x27 face matrix to one of three consecutive 9-row strips. -/
def restrictStrip27 (f : FaceMatrix 27) (i : Fin 3) : Strip9x27 :=
  fun rc => f
    (⟨i.val * 9 + rc.1.val, by omega⟩,
     ⟨rc.2.val, by omega⟩)

theorem restrictStrip27_injective :
    Function.Injective (fun f : FaceMatrix 27 => fun i : Fin 3 => restrictStrip27 f i) := by
  intro f g h
  funext rc
  let r := rc.1.val
  by_cases h9 : r < 9
  · let i : Fin 3 := ⟨0, by omega⟩
    let lr : Fin 9 := ⟨r, h9⟩
    have hx := congrFun (congrFun h i) (lr, rc.2)
    simpa [restrictStrip27, i, lr, r] using hx
  · by_cases h18 : r < 18
    · let i : Fin 3 := ⟨1, by omega⟩
      let lr : Fin 9 := ⟨r - 9, by omega⟩
      have hx := congrFun (congrFun h i) (lr, rc.2)
      have hr : 9 + (r - 9) = r := by omega
      simpa [restrictStrip27, i, lr, r, hr] using hx
    · let i : Fin 3 := ⟨2, by omega⟩
      let lr : Fin 9 := ⟨r - 18, by omega⟩
      have hx := congrFun (congrFun h i) (lr, rc.2)
      have hr : 18 + (r - 18) = r := by omega
      simpa [restrictStrip27, i, lr, r, hr] using hx

theorem restrictStrip27_preserves
    (f : FaceMatrix 27) (hf : finiteNoCheckerboard 27 f) (i : Fin 3) :
    stripNoCheckerboard (restrictStrip27 f i) := by
  intro r c
  have h := hf
    ⟨i.val * 9 + r.val, by omega⟩
    ⟨c.val, by omega⟩
  simpa [restrictStrip27, Nat.add_assoc] using h

theorem card_valid27_le_three_strips :
    Fintype.card {f : FaceMatrix 27 // finiteNoCheckerboard 27 f} ≤
      (Fintype.card {s : Strip9x27 // stripNoCheckerboard s}) ^ 3 := by
  classical
  have h := card_global_valid_le_product_strips
    (Full := FaceMatrix 27) (I := Fin 3) (Strip := fun _ => Strip9x27)
    (finiteNoCheckerboard 27) (fun _ => stripNoCheckerboard)
    (fun f i => restrictStrip27 f i)
    restrictStrip27_injective
    (fun f hf i => restrictStrip27_preserves f hf i)
  rw [Fin.prod_univ_three] at h
  simpa [pow_succ, pow_two, mul_assoc] using h



/-- Concrete production n=27 strip upper bound: every corner-to-corner simple
path injects into a triple of independently valid 9x27 face strips. -/
theorem card_corner_paths_27_le_three_strips :
    Fintype.card ((OneesanFormal.GridGraph.gridGraph 27).Path
      (OneesanFormal.GridGraph.northwestCorner 27)
      (OneesanFormal.GridGraph.southeastCorner 27)) ≤
      (Fintype.card {s : Strip9x27 // stripNoCheckerboard s}) ^ 3 := by
  exact le_trans (card_corner_paths_le_checkerboard_free_faces 27 (by omega))
    card_valid27_le_three_strips

end OneesanFormal.StripBound27
