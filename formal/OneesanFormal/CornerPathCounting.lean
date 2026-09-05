import OneesanFormal.CornerPathBound
import OneesanFormal.FacePathInjective
import Mathlib.Combinatorics.SimpleGraph.Walk.Counting

namespace OneesanFormal.CornerPathCounting

open SimpleGraph
open OneesanFormal.GridGraph
open OneesanFormal.GridFaceBoundary
open OneesanFormal.CheckerboardGridBound
open OneesanFormal.PathGridEncoding
open OneesanFormal.OuterReference
open OneesanFormal.CornerPathBound
open OneesanFormal.FacePathInjective

abbrev FaceMatrix (n : Nat) := Fin n × Fin n → F2

/-- Finite `n×n` version of the no-checkerboard predicate. -/
def finiteNoCheckerboard (n : Nat) (f : FaceMatrix n) : Prop :=
  ∀ r : Fin (n - 1), ∀ c : Fin (n - 1),
    ¬ f2Checkerboard
      (f (⟨r.val, by omega⟩, ⟨c.val, by omega⟩))
      (f (⟨r.val, by omega⟩, ⟨c.val + 1, by omega⟩))
      (f (⟨r.val + 1, by omega⟩, ⟨c.val + 1, by omega⟩))
      (f (⟨r.val + 1, by omega⟩, ⟨c.val, by omega⟩))

noncomputable instance gridGraphAdjDecidable (n : Nat) : DecidableRel (gridGraph n).Adj :=
  Classical.decRel _

noncomputable instance finiteNoCheckerboardDecidable (n : Nat) :
    DecidablePred (finiteNoCheckerboard n) := Classical.decPred _

noncomputable def chosenCornerFace
    (n : Nat) (hn : 0 < n)
    (p : (gridGraph n).Path (northwestCorner n) (southeastCorner n)) :
    Nat → Nat → F2 :=
  Classical.choose (corner_simple_path_has_checkerboard_free_face_encoding n hn p.property)

theorem chosenCornerFace_spec
    (n : Nat) (hn : 0 < n)
    (p : (gridGraph n).Path (northwestCorner n) (southeastCorner n)) :
    RectBoundaryWitness n
      (edgeXor (pathHorizontal p.val) (outerHorizontal n))
      (edgeXor (pathVertical p.val) (outerVertical n))
      (chosenCornerFace n hn p) ∧
    noCheckerboardGrid n (chosenCornerFace n hn p) :=
  Classical.choose_spec (corner_simple_path_has_checkerboard_free_face_encoding n hn p.property)

noncomputable def cornerFaceEncoding
    (n : Nat) (hn : 0 < n) :
    (gridGraph n).Path (northwestCorner n) (southeastCorner n) → FaceMatrix n :=
  fun p rc => chosenCornerFace n hn p rc.1.val rc.2.val

theorem cornerFaceEncoding_valid
    (n : Nat) (hn : 0 < n)
    (p : (gridGraph n).Path (northwestCorner n) (southeastCorner n)) :
    finiteNoCheckerboard n (cornerFaceEncoding n hn p) := by
  intro r c
  have hnc := (chosenCornerFace_spec n hn p).2 r.val c.val (by omega) (by omega)
  simpa [cornerFaceEncoding] using hnc

theorem cornerFaceEncoding_injective
    (n : Nat) (hn : 0 < n) :
    Function.Injective (cornerFaceEncoding n hn) := by
  intro p q heq
  apply paths_eq_of_same_face_witness hn p q
    (outerHorizontal n) (outerVertical n)
    (chosenCornerFace n hn p) (chosenCornerFace n hn q)
  · exact (chosenCornerFace_spec n hn p).1
  · exact (chosenCornerFace_spec n hn q).1
  · intro r c hr hc
    have h := congrFun heq
      ((⟨r, hr⟩, ⟨c, hc⟩) : Fin n × Fin n)
    exact h

noncomputable def cornerValidFaceEncoding
    (n : Nat) (hn : 0 < n) :
    (gridGraph n).Path (northwestCorner n) (southeastCorner n) →
      {f : FaceMatrix n // finiteNoCheckerboard n f} :=
  fun p => ⟨cornerFaceEncoding n hn p, cornerFaceEncoding_valid n hn p⟩

theorem cornerValidFaceEncoding_injective
    (n : Nat) (hn : 0 < n) :
    Function.Injective (cornerValidFaceEncoding n hn) := by
  intro p q h
  apply cornerFaceEncoding_injective n hn
  exact congrArg Subtype.val h

theorem card_corner_paths_le_checkerboard_free_faces
    (n : Nat) (hn : 0 < n) :
    Fintype.card ((gridGraph n).Path (northwestCorner n) (southeastCorner n)) ≤
      Fintype.card {f : FaceMatrix n // finiteNoCheckerboard n f} := by
  classical
  exact Fintype.card_le_of_injective (cornerValidFaceEncoding n hn)
    (cornerValidFaceEncoding_injective n hn)


end OneesanFormal.CornerPathCounting
