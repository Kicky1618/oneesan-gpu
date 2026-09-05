import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.CharP.Two
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic

namespace OneesanFormal.GridFaceBoundary

open Finset
abbrev F2 := ZMod 2

/-- In characteristic two, a sum of adjacent differences telescopes because
all interior terms occur twice. -/
theorem f2_telescope (a : Nat → F2) (r : Nat) :
    (∑ i ∈ Finset.range (r + 1), (a i + a (i + 1))) = a 0 + a (r + 1) := by
  induction r with
  | zero => simp
  | succ r ih =>
      rw [show Nat.succ r + 1 = (r + 1) + 1 by omega]
      rw [Finset.sum_range_succ, ih]
      calc
        a 0 + a (r + 1) + (a (r + 1) + a (r + 1 + 1))
            = a 0 + (a (r + 1) + a (r + 1)) + a (r + 1 + 1) := by abel
        _ = a 0 + 0 + a (r + 1 + 1) := by rw [CharTwo.add_self_eq_zero]
        _ = a 0 + a (Nat.succ r + 1) := by simp [Nat.succ_eq_add_one]

/-- Prefix integration of horizontal edge bits from the top boundary. Column
`c=0` and `c=n+1` are ghost columns; the actual face in grid column `c` is
represented by `facePrefix H r (c+1)`. -/
def facePrefix (H : Nat → Nat → F2) (r c : Nat) : F2 :=
  ∑ i ∈ Finset.range (r + 1), H i c

/-- Consecutive prefix rows differ by exactly the newly crossed horizontal
edge. -/
theorem facePrefix_succ (H : Nat → Nat → F2) (r c : Nat) :
    facePrefix H (r + 1) c + facePrefix H r c = H (r + 1) c := by
  unfold facePrefix
  rw [show r + 1 + 1 = (r + 1) + 1 by rfl]
  rw [Finset.sum_range_succ]
  calc
    (∑ i ∈ range (r + 1), H i c) + H (r + 1) c + (∑ i ∈ range (r + 1), H i c)
        = ((∑ i ∈ range (r + 1), H i c) + (∑ i ∈ range (r + 1), H i c)) + H (r + 1) c := by abel
    _ = 0 + H (r + 1) c := by rw [CharTwo.add_self_eq_zero]
    _ = H (r + 1) c := zero_add _

/-- Summing even-degree equations down a column reconstructs the vertical edge
at the bottom of the prefix from the two neighboring horizontal prefixes. -/
theorem vertical_from_vertex_parity
    (H V : Nat → Nat → F2) (r c : Nat)
    (hV0 : V 0 c = 0)
    (hpar : ∀ i ∈ Finset.range (r + 1),
      H i c + H i (c + 1) + V i c + V (i + 1) c = 0) :
    V (r + 1) c = facePrefix H r c + facePrefix H r (c + 1) := by
  have hs : (∑ i ∈ Finset.range (r + 1),
      (H i c + H i (c + 1) + V i c + V (i + 1) c)) = 0 := by
    exact Finset.sum_eq_zero hpar
  simp only [Finset.sum_add_distrib] at hs
  have ht := f2_telescope (fun i => V i c) r
  have hvsum :
      (∑ i ∈ Finset.range (r + 1), V i c) +
      (∑ i ∈ Finset.range (r + 1), V (i + 1) c) = V (r + 1) c := by
    rw [← Finset.sum_add_distrib, ht, hV0, zero_add]
  have hzero : facePrefix H r c + facePrefix H r (c + 1) + V (r + 1) c = 0 := by
    unfold facePrefix
    calc
      _ = (∑ i ∈ Finset.range (r + 1), H i c) +
          (∑ i ∈ Finset.range (r + 1), H i (c + 1)) +
          ((∑ i ∈ Finset.range (r + 1), V i c) +
           (∑ i ∈ Finset.range (r + 1), V (i + 1) c)) := by rw [hvsum]
      _ = 0 := by simpa only [add_assoc] using hs
  exact (CharTwo.add_eq_zero.mp hzero).symm

/-- Symmetric row-prefix version used to recover the bottom horizontal
boundary after all vertical edges have been reconstructed. -/
theorem horizontal_from_vertex_parity
    (H V : Nat → Nat → F2) (r c : Nat)
    (hH0 : H r 0 = 0)
    (hpar : ∀ j ∈ Finset.range (c + 1),
      H r j + H r (j + 1) + V r j + V (r + 1) j = 0) :
    H r (c + 1) =
      (∑ j ∈ Finset.range (c + 1), V r j) +
      (∑ j ∈ Finset.range (c + 1), V (r + 1) j) := by
  have hs : (∑ j ∈ Finset.range (c + 1),
      (H r j + H r (j + 1) + V r j + V (r + 1) j)) = 0 := by
    exact Finset.sum_eq_zero hpar
  simp only [Finset.sum_add_distrib] at hs
  have ht := f2_telescope (fun j => H r j) c
  have hhsum :
      (∑ j ∈ Finset.range (c + 1), H r j) +
      (∑ j ∈ Finset.range (c + 1), H r (j + 1)) = H r (c + 1) := by
    rw [← Finset.sum_add_distrib, ht, hH0, zero_add]
  have hzero :
      H r (c + 1) +
      ((∑ j ∈ Finset.range (c + 1), V r j) +
       (∑ j ∈ Finset.range (c + 1), V (r + 1) j)) = 0 := by
    calc
      _ = ((∑ j ∈ Finset.range (c + 1), H r j) +
           (∑ j ∈ Finset.range (c + 1), H r (j + 1))) +
          ((∑ j ∈ Finset.range (c + 1), V r j) +
           (∑ j ∈ Finset.range (c + 1), V (r + 1) j)) := by rw [hhsum]
      _ = 0 := by simpa only [add_assoc] using hs
  exact CharTwo.add_eq_zero.mp hzero

end OneesanFormal.GridFaceBoundary

namespace OneesanFormal.GridFaceBoundary

/-- The six families of edge equations saying that `F` is a bounded-face
potential whose GF(2) boundary is the edge assignment `(H,V)`. -/
structure RectBoundaryWitness
    (n : Nat) (H V F : Nat → Nat → F2) : Prop where
  top : ∀ c, c < n → H 0 (c + 1) = F 0 c
  horizontalInterior : ∀ r c, r + 1 < n → c < n →
    H (r + 1) (c + 1) = F r c + F (r + 1) c
  bottom : ∀ c, c < n → H n (c + 1) = F (n - 1) c
  left : ∀ r, r < n → V (r + 1) 0 = F r 0
  verticalInterior : ∀ r c, r < n → c + 1 < n →
    V (r + 1) (c + 1) = F r c + F r (c + 1)
  right : ∀ r, r < n → V (r + 1) n = F r (n - 1)

/-- Concrete rectangular-grid cycle-space theorem over GF(2).

`H r c` uses ghost horizontal columns `c=0,n+1`; real horizontal edges are
`c=1..n`. `V r c` uses ghost vertical rows `r=0,n+1`; real vertical edges are
`r=1..n`. If the ghost edges are zero and every grid vertex has even incident
parity, then the edge set is exactly the boundary of a subset of the `n^2`
bounded faces. -/
theorem even_grid_edges_have_bounded_face_boundary
    (n : Nat) (hn : 0 < n)
    (H V : Nat → Nat → F2)
    (hHleft : ∀ r, r ≤ n → H r 0 = 0)
    (hHright : ∀ r, r ≤ n → H r (n + 1) = 0)
    (hVtop : ∀ c, c ≤ n → V 0 c = 0)
    (hVbottom : ∀ c, c ≤ n → V (n + 1) c = 0)
    (hpar : ∀ r c, r ≤ n → c ≤ n →
      H r c + H r (c + 1) + V r c + V (r + 1) c = 0) :
    ∃ F : Nat → Nat → F2, RectBoundaryWitness n H V F := by
  let F : Nat → Nat → F2 := fun r c => facePrefix H r (c + 1)
  refine ⟨F, ?_⟩
  constructor
  · intro c hc
    simp [F, facePrefix]
  · intro r c hr hc
    have h := facePrefix_succ H r (c + 1)
    simpa [F, add_comm] using h.symm
  · intro c hc
    have hrow := horizontal_from_vertex_parity H V n c (hHleft n (le_refl n)) ?_
    · have hvbottomsum : (∑ j ∈ Finset.range (c + 1), V (n + 1) j) = 0 := by
        apply Finset.sum_eq_zero
        intro j hj
        apply hVbottom j
        have hjlt : j < c + 1 := Finset.mem_range.mp hj
        omega
      have hvformula : ∀ j ∈ Finset.range (c + 1),
          V n j = facePrefix H (n - 1) j + facePrefix H (n - 1) (j + 1) := by
        intro j hj
        have hjlt : j < c + 1 := Finset.mem_range.mp hj
        have hvert := vertical_from_vertex_parity H V (n - 1) j
          (hVtop j (by omega)) ?_
        · simpa [show n - 1 + 1 = n by omega] using hvert
        · intro i hi
          have hilt : i < (n - 1) + 1 := Finset.mem_range.mp hi
          exact hpar i j (by omega) (by omega)
      have hvsum : (∑ j ∈ Finset.range (c + 1), V n j) = facePrefix H (n - 1) (c + 1) := by
        calc
          (∑ j ∈ Finset.range (c + 1), V n j)
              = ∑ j ∈ Finset.range (c + 1),
                  (facePrefix H (n - 1) j + facePrefix H (n - 1) (j + 1)) := by
                    apply Finset.sum_congr rfl
                    intro j hj
                    exact hvformula j hj
          _ = facePrefix H (n - 1) 0 + facePrefix H (n - 1) (c + 1) :=
                f2_telescope (fun j => facePrefix H (n - 1) j) c
          _ = facePrefix H (n - 1) (c + 1) := by
                have hz : facePrefix H (n - 1) 0 = 0 := by
                  unfold facePrefix
                  apply Finset.sum_eq_zero
                  intro i hi
                  apply hHleft i
                  have hilt : i < (n - 1) + 1 := Finset.mem_range.mp hi
                  omega
                rw [hz, zero_add]
      rw [hvbottomsum, add_zero, hvsum] at hrow
      simpa [F] using hrow
    · intro j hj
      have hjlt : j < c + 1 := Finset.mem_range.mp hj
      exact hpar n j (le_refl n) (by omega)
  · intro r hr
    have hvert := vertical_from_vertex_parity H V r 0 (hVtop 0 (Nat.zero_le n)) ?_
    · have hz : facePrefix H r 0 = 0 := by
        unfold facePrefix
        apply Finset.sum_eq_zero
        intro i hi
        apply hHleft i
        have hilt : i < r + 1 := Finset.mem_range.mp hi
        omega
      rw [hz, zero_add] at hvert
      simpa [F] using hvert
    · intro i hi
      have hilt : i < r + 1 := Finset.mem_range.mp hi
      exact hpar i 0 (by omega) (Nat.zero_le n)
  · intro r c hr hc
    have hvert := vertical_from_vertex_parity H V r (c + 1)
      (hVtop (c + 1) (by omega)) ?_
    · simpa [F] using hvert
    · intro i hi
      have hilt : i < r + 1 := Finset.mem_range.mp hi
      exact hpar i (c + 1) (by omega) (by omega)
  · intro r hr
    have hvert := vertical_from_vertex_parity H V r n (hVtop n (le_refl n)) ?_
    · have hz : facePrefix H r (n + 1) = 0 := by
        unfold facePrefix
        apply Finset.sum_eq_zero
        intro i hi
        apply hHright i
        have hilt : i < r + 1 := Finset.mem_range.mp hi
        omega
      rw [hz, add_zero] at hvert
      have hnidx : (n - 1) + 1 = n := by omega
      simpa [F, hnidx] using hvert
    · intro i hi
      have hilt : i < r + 1 := Finset.mem_range.mp hi
      exact hpar i n (by omega) (le_refl n)

end OneesanFormal.GridFaceBoundary

namespace OneesanFormal.GridFaceBoundary

/-- GF(2) incidence parity at one grid vertex. -/
def vertexParity (H V : Nat → Nat → F2) (r c : Nat) : F2 :=
  H r c + H r (c + 1) + V r c + V (r + 1) c

/-- Symmetric difference/XOR of two edge assignments is pointwise addition in
GF(2). -/
def edgeXor (A B : Nat → Nat → F2) : Nat → Nat → F2 :=
  fun r c => A r c + B r c

/-- Two edge assignments with the same boundary parity have an even symmetric
difference. This is the T-join cancellation used for `P XOR P0`. -/
theorem xor_same_vertex_parity_is_even
    (H₁ V₁ H₀ V₀ : Nat → Nat → F2)
    (r c : Nat)
    (hsame : vertexParity H₁ V₁ r c = vertexParity H₀ V₀ r c) :
    vertexParity (edgeXor H₁ H₀) (edgeXor V₁ V₀) r c = 0 := by
  unfold vertexParity at hsame ⊢
  unfold edgeXor
  calc
    _ = (H₁ r c + H₁ r (c + 1) + V₁ r c + V₁ (r + 1) c) +
        (H₀ r c + H₀ r (c + 1) + V₀ r c + V₀ (r + 1) c) := by abel
    _ = 0 := by rw [hsame, CharTwo.add_self_eq_zero]

/-- Concrete bridge from two rectangular-grid T-joins with the same terminals
to a bounded-face representation of their XOR.

This is the exact topological step used by the checkerboard path bound: a
corner-to-corner path `P` and the fixed reference path `P0` have identical
vertex parity, so `P XOR P0` is the boundary of bounded faces. -/
theorem same_boundary_xor_has_bounded_face_boundary
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
      vertexParity H₁ V₁ r c = vertexParity H₀ V₀ r c) :
    ∃ F : Nat → Nat → F2,
      RectBoundaryWitness n (edgeXor H₁ H₀) (edgeXor V₁ V₀) F := by
  apply even_grid_edges_have_bounded_face_boundary n hn
      (edgeXor H₁ H₀) (edgeXor V₁ V₀)
  · intro r hr
    simp [edgeXor, hHleft₁ r hr, hHleft₀ r hr]
  · intro r hr
    simp [edgeXor, hHright₁ r hr, hHright₀ r hr]
  · intro c hc
    simp [edgeXor, hVtop₁ c hc, hVtop₀ c hc]
  · intro c hc
    simp [edgeXor, hVbottom₁ c hc, hVbottom₀ c hc]
  · intro r c hr hc
    exact xor_same_vertex_parity_is_even H₁ V₁ H₀ V₀ r c (hsame r c hr hc)

end OneesanFormal.GridFaceBoundary
