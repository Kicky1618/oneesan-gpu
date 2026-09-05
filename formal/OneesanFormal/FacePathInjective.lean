import OneesanFormal.RectBoundaryInjective
import OneesanFormal.PathHVInjective

namespace OneesanFormal.FacePathInjective

open SimpleGraph
open OneesanFormal.GridGraph
open OneesanFormal.GridFaceBoundary
open OneesanFormal.PathGridEncoding
open OneesanFormal.RectBoundaryInjective
open OneesanFormal.PathHVInjective

theorem path_hv_eq_of_same_face_witness
    {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p q : (gridGraph n).Walk u v)
    (H₀ V₀ Fp Fq : Nat → Nat → F2)
    (hwp : RectBoundaryWitness n
      (edgeXor (pathHorizontal p) H₀)
      (edgeXor (pathVertical p) V₀) Fp)
    (hwq : RectBoundaryWitness n
      (edgeXor (pathHorizontal q) H₀)
      (edgeXor (pathVertical q) V₀) Fq)
    (hF : ∀ r c, r < n → c < n → Fp r c = Fq r c) :
    pathHorizontal p = pathHorizontal q ∧
    pathVertical p = pathVertical q := by
  obtain ⟨hHxor, hVxor⟩ := witness_real_edges_eq n hn
    (edgeXor (pathHorizontal p) H₀) (edgeXor (pathVertical p) V₀) Fp
    (edgeXor (pathHorizontal q) H₀) (edgeXor (pathVertical q) V₀) Fq
    hwp hwq hF
  constructor
  · funext r c
    by_cases hreal : r ≤ n ∧ 0 < c ∧ c ≤ n
    · have hx := hHxor r c hreal.1 hreal.2.1 hreal.2.2
      exact add_right_cancel hx
    · simp [pathHorizontal, hreal]
  · funext r c
    by_cases hreal : 0 < r ∧ r ≤ n ∧ c ≤ n
    · have hx := hVxor r c hreal.1 hreal.2.1 hreal.2.2
      exact add_right_cancel hx
    · simp [pathVertical, hreal]

theorem paths_eq_of_same_face_witness
    {n : Nat} (hn : 0 < n) {u v : GridVertex n}
    (p q : (gridGraph n).Path u v)
    (H₀ V₀ Fp Fq : Nat → Nat → F2)
    (hwp : RectBoundaryWitness n
      (edgeXor (pathHorizontal p.val) H₀)
      (edgeXor (pathVertical p.val) V₀) Fp)
    (hwq : RectBoundaryWitness n
      (edgeXor (pathHorizontal q.val) H₀)
      (edgeXor (pathVertical q.val) V₀) Fq)
    (hF : ∀ r c, r < n → c < n → Fp r c = Fq r c) : p = q := by
  apply path_hv_injective
  obtain ⟨hH, hV⟩ := path_hv_eq_of_same_face_witness hn p.val q.val H₀ V₀ Fp Fq hwp hwq hF
  exact Prod.ext hH hV


end OneesanFormal.FacePathInjective
