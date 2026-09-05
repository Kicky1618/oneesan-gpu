import OneesanFormal.GridFaceBoundary
import Mathlib.Tactic

namespace OneesanFormal.ProcessedStripParityBound

open OneesanFormal.GridFaceBoundary

/-- Binary edge choices in an `r`-row processed strip of width `w`.
`horiz` contains the `r(w-1)` horizontal edges; `vert` contains the `rw`
vertical edges leading from each processed row to the next vertex line. -/
structure StripBits (r w : Nat) where
  horiz : Fin r → Fin (w - 1) → F2
  vert : Fin r → Fin w → F2
  deriving Fintype

theorem StripBits.eq_of_horiz_vert {r w : Nat} {A B : StripBits r w}
    (hh : A.horiz = B.horiz) (hv : A.vert = B.vert) : A = B := by
  cases A
  cases B
  simp_all

/-- Vertical edge entering processed row `y` from the preceding row. -/
def previousVert {r w : Nat} (V : Fin r → Fin w → F2)
    (y : Fin r) (x : Fin w) : F2 :=
  if h : y.val = 0 then 0 else V ⟨y.val - 1, by omega⟩ x

/-- Parity at an upper processed vertex.  `hdeg` is deliberately abstract:
it can be the XOR of the one or two horizontal incident edges.  The only fact
needed for the counting bound is that it depends solely on `horiz`. -/
def upperParity {r w : Nat}
    (hdeg : (Fin r → Fin (w - 1) → F2) → Fin r → Fin w → F2)
    (A : StripBits r w) (y : Fin r) (x : Fin w) : F2 :=
  A.vert y x + previousVert A.vert y x + hdeg A.horiz y x

/-- For fixed horizontal bits and fixed upper-vertex parities, the vertical
bits are unique.  Algebraically the restricted incidence matrix contains a
unit lower-triangular `rw × rw` block formed by the vertical edges. -/
theorem vert_unique_of_horiz_parity {r w : Nat}
    (hdeg : (Fin r → Fin (w - 1) → F2) → Fin r → Fin w → F2)
    {A B : StripBits r w}
    (hh : A.horiz = B.horiz)
    (hp : upperParity hdeg A = upperParity hdeg B) :
    A.vert = B.vert := by
  funext y x
  have hrow : ∀ n : Nat, ∀ hn : n < r,
      A.vert ⟨n, hn⟩ x = B.vert ⟨n, hn⟩ x := by
    intro n
    induction n with
    | zero =>
        intro hn
        have h := congrFun (congrFun hp ⟨0, hn⟩) x
        simp [upperParity, previousVert, hh] at h
        exact h
    | succ n ih =>
        intro hn
        have hprev := ih (by omega)
        have h := congrFun (congrFun hp ⟨n + 1, hn⟩) x
        simp [upperParity, previousVert, hh, hprev] at h
        exact h
  exact hrow y.val y.isLt

/-- Projection to the horizontal choices is injective on every fixed-parity
fiber. -/
theorem horiz_injective_on_parity_fiber {r w : Nat}
    (hdeg : (Fin r → Fin (w - 1) → F2) → Fin r → Fin w → F2)
    (p : Fin r → Fin w → F2) :
    Function.Injective
      (fun A : {A : StripBits r w // upperParity hdeg A = p} => A.1.horiz) := by
  intro A B hab
  apply Subtype.ext
  exact StripBits.eq_of_horiz_vert hab
    (vert_unique_of_horiz_parity hdeg hab (A.property.trans B.property.symm))

/-- Hence a fixed upper-parity profile admits at most `2^(r(w-1))` processed
strip edge subsets.  This is the bound used by the deterministic Grid-FP vs.
structural row-8 CRT certificate. -/
theorem parity_fiber_card_le {r w : Nat}
    (hdeg : (Fin r → Fin (w - 1) → F2) → Fin r → Fin w → F2)
    (p : Fin r → Fin w → F2) :
    Fintype.card {A : StripBits r w // upperParity hdeg A = p} ≤
      2 ^ (r * (w - 1)) := by
  classical
  have hcard := Fintype.card_le_of_injective
    (fun A : {A : StripBits r w // upperParity hdeg A = p} => A.1.horiz)
    (horiz_injective_on_parity_fiber hdeg p)
  calc
    Fintype.card {A : StripBits r w // upperParity hdeg A = p} ≤
        (2 ^ (w - 1)) ^ r := by
      simpa [Fintype.card_fun] using hcard
    _ = 2 ^ ((w - 1) * r) := by rw [pow_mul]
    _ = 2 ^ (r * (w - 1)) := by rw [Nat.mul_comm]

/-- Row 8 / width 28 specialization: at most `2^216` edge subsets per fixed
upper-parity profile, versus the previous coarse `2^440` bound. -/
theorem row8_width28_parity_fiber_card_le
    (hdeg : (Fin 8 → Fin 27 → F2) → Fin 8 → Fin 28 → F2)
    (p : Fin 8 → Fin 28 → F2) :
    Fintype.card {A : StripBits 8 28 // upperParity hdeg A = p} ≤ 2 ^ 216 := by
  simpa using parity_fiber_card_le hdeg p

end OneesanFormal.ProcessedStripParityBound
