import OneesanFormal.GridFaceBoundary
import OneesanFormal.CheckerboardBound

namespace OneesanFormal.CheckerboardGridBound

open OneesanFormal.GridFaceBoundary

/-- Checkerboard relation directly on GF(2) face values. -/
def f2Checkerboard (nw ne se sw : F2) : Prop :=
  nw = se ∧ ne = sw ∧ nw ≠ ne

/-- A GF(2) edge bit contributes one to the ordinary graph degree iff it is
nonzero. -/
def bitDegree (x : F2) : Nat := if x = 0 then 0 else 1

/-- Ordinary degree of the four boundary edges around one interior vertex,
expressed from the surrounding face potential. -/
def faceLocalDegree (nw ne se sw : F2) : Nat :=
  bitDegree (nw + ne) + bitDegree (ne + se) +
  bitDegree (se + sw) + bitDegree (sw + nw)

/-- Alternating face values force all four GF(2) boundary edges to be present. -/
theorem f2_checkerboard_degree_four
    (nw ne se sw : F2) (h : f2Checkerboard nw ne se sw) :
    faceLocalDegree nw ne se sw = 4 := by
  rcases h with ⟨hse, hsw, hne⟩
  subst se
  subst sw
  have h01 : nw + ne ≠ 0 := by
    exact (not_congr CharTwo.add_eq_zero).2 hne
  have h10 : ne + nw ≠ 0 := by
    exact (not_congr CharTwo.add_eq_zero).2 (Ne.symm hne)
  simp [faceLocalDegree, bitDegree, h01, h10]

/-- Hence any face potential whose induced edge set has degree at most two at
an interior vertex cannot have a checkerboard there. -/
theorem no_checkerboard_of_face_local_degree_le_two
    (nw ne se sw : F2)
    (hdeg : faceLocalDegree nw ne se sw ≤ 2) :
    ¬ f2Checkerboard nw ne se sw := by
  intro hcb
  have hfour := f2_checkerboard_degree_four nw ne se sw hcb
  omega

/-- The four real edge bits around an interior vertex are exactly the four
pairwise face differences from a `RectBoundaryWitness`. -/
theorem witness_interior_degree_eq_faceLocalDegree
    (n : Nat) (H V F : Nat → Nat → F2)
    (hw : RectBoundaryWitness n H V F)
    (r c : Nat) (hr : r + 1 < n) (hc : c + 1 < n) :
    bitDegree (H (r + 1) (c + 1)) +
      bitDegree (H (r + 1) (c + 2)) +
      bitDegree (V (r + 1) (c + 1)) +
      bitDegree (V (r + 2) (c + 1)) =
    faceLocalDegree (F r c) (F r (c + 1))
      (F (r + 1) (c + 1)) (F (r + 1) c) := by
  have hleft := hw.horizontalInterior r c hr (by omega)
  have hright := hw.horizontalInterior r (c + 1) hr (by omega)
  have hup := hw.verticalInterior r c (by omega) hc
  have hdown := hw.verticalInterior (r + 1) c (by omega) hc
  rw [hleft, hright, hup, hdown]
  simp [faceLocalDegree, add_comm, add_left_comm]

/-- Main local obstruction in the exact rectangular indexing used by the
production checkerboard bound. -/
theorem witness_no_checkerboard_at_interior_vertex
    (n : Nat) (H V F : Nat → Nat → F2)
    (hw : RectBoundaryWitness n H V F)
    (r c : Nat) (hr : r + 1 < n) (hc : c + 1 < n)
    (hdeg :
      bitDegree (H (r + 1) (c + 1)) +
        bitDegree (H (r + 1) (c + 2)) +
        bitDegree (V (r + 1) (c + 1)) +
        bitDegree (V (r + 2) (c + 1)) ≤ 2) :
    ¬ f2Checkerboard (F r c) (F r (c + 1))
      (F (r + 1) (c + 1)) (F (r + 1) c) := by
  apply no_checkerboard_of_face_local_degree_le_two
  rw [← witness_interior_degree_eq_faceLocalDegree n H V F hw r c hr hc]
  exact hdeg


/-- No 2x2 checkerboard anywhere in the bounded-face matrix. -/
def noCheckerboardGrid (n : Nat) (F : Nat → Nat → F2) : Prop :=
  ∀ r c, r + 1 < n → c + 1 < n →
    ¬ f2Checkerboard (F r c) (F r (c + 1))
      (F (r + 1) (c + 1)) (F (r + 1) c)

/-- A rectangular boundary witness whose induced edge degree is at most two at
all interior vertices yields a globally checkerboard-free face assignment. -/
theorem witness_no_checkerboard_grid
    (n : Nat) (H V F : Nat → Nat → F2)
    (hw : RectBoundaryWitness n H V F)
    (hdeg : ∀ r c, r + 1 < n → c + 1 < n →
      bitDegree (H (r + 1) (c + 1)) +
        bitDegree (H (r + 1) (c + 2)) +
        bitDegree (V (r + 1) (c + 1)) +
        bitDegree (V (r + 2) (c + 1)) ≤ 2) :
    noCheckerboardGrid n F := by
  intro r c hr hc
  exact witness_no_checkerboard_at_interior_vertex n H V F hw r c hr hc
    (hdeg r c hr hc)

/-- If the reference edge set has no edges incident to an interior vertex, XOR
with that reference does not change the ordinary degree at that vertex.  This
is the local fact used for the fixed outer-boundary reference path `P0`. -/
theorem xor_interior_degree_eq_left_of_base_zero
    (H₁ V₁ H₀ V₀ : Nat → Nat → F2) (r c : Nat)
    (hhl : H₀ (r + 1) (c + 1) = 0)
    (hhr : H₀ (r + 1) (c + 2) = 0)
    (hvu : V₀ (r + 1) (c + 1) = 0)
    (hvd : V₀ (r + 2) (c + 1) = 0) :
    bitDegree (edgeXor H₁ H₀ (r + 1) (c + 1)) +
      bitDegree (edgeXor H₁ H₀ (r + 1) (c + 2)) +
      bitDegree (edgeXor V₁ V₀ (r + 1) (c + 1)) +
      bitDegree (edgeXor V₁ V₀ (r + 2) (c + 1)) =
    bitDegree (H₁ (r + 1) (c + 1)) +
      bitDegree (H₁ (r + 1) (c + 2)) +
      bitDegree (V₁ (r + 1) (c + 1)) +
      bitDegree (V₁ (r + 2) (c + 1)) := by
  simp [edgeXor, hhl, hhr, hvu, hvd]

/-- Concrete end-to-end local part of the checkerboard path bound.

Two edge assignments with the same terminal parity have a bounded-face XOR
representation.  If the fixed reference assignment is zero at every interior
vertex (as for the chosen outer-boundary path) and the candidate path has
ordinary degree at most two there, the recovered face assignment has no 2x2
checkerboard anywhere. -/
theorem outer_reference_xor_has_checkerboard_free_face_encoding
    (n : Nat) (hn : 0 < n)
    (H₁ V₁ H₀ V₀ : Nat → Nat → F2)
    (hHleft₁ : ∀ r, r ≤ n → H₁ r 0 = 0)
    (hHright₁ : ∀ r, r ≤ n → H₁ r (n + 1) = 0)
    (hVtop₁ : ∀ c, c ≤ n → V₁ 0 c = 0)
    (hVbottom₁ : ∀ c, c ≤ n → V₁ (n + 1) c = 0)
    (hHleft₀ : ∀ r, r ≤ n → H₀ r 0 = 0)
    (hHright₀ : ∀ r, r ≤ n → H₀ r (n + 1) = 0)
    (hVtop₀ : ∀ c, c ≤ n → V₀ 0 c = 0)
    (hVbottom₀ : ∀ c, c ≤ n → V₀ (n + 1) c = 0)
    (hsame : ∀ r c, r ≤ n → c ≤ n →
      vertexParity H₁ V₁ r c = vertexParity H₀ V₀ r c)
    (hbaseInterior : ∀ r c, r + 1 < n → c + 1 < n →
      H₀ (r + 1) (c + 1) = 0 ∧ H₀ (r + 1) (c + 2) = 0 ∧
      V₀ (r + 1) (c + 1) = 0 ∧ V₀ (r + 2) (c + 1) = 0)
    (hpathDegree : ∀ r c, r + 1 < n → c + 1 < n →
      bitDegree (H₁ (r + 1) (c + 1)) +
        bitDegree (H₁ (r + 1) (c + 2)) +
        bitDegree (V₁ (r + 1) (c + 1)) +
        bitDegree (V₁ (r + 2) (c + 1)) ≤ 2) :
    ∃ F : Nat → Nat → F2,
      RectBoundaryWitness n (edgeXor H₁ H₀) (edgeXor V₁ V₀) F ∧
      noCheckerboardGrid n F := by
  obtain ⟨F, hw⟩ := same_boundary_xor_has_bounded_face_boundary
    n hn H₁ V₁ H₀ V₀ hHleft₁ hHright₁ hVtop₁ hVbottom₁
    hHleft₀ hHright₀ hVtop₀ hVbottom₀ hsame
  refine ⟨F, hw, ?_⟩
  apply witness_no_checkerboard_grid n (edgeXor H₁ H₀) (edgeXor V₁ V₀) F hw
  intro r c hr hc
  rcases hbaseInterior r c hr hc with ⟨hhl, hhr, hvu, hvd⟩
  rw [xor_interior_degree_eq_left_of_base_zero H₁ V₁ H₀ V₀ r c hhl hhr hvu hvd]
  exact hpathDegree r c hr hc

end OneesanFormal.CheckerboardGridBound
