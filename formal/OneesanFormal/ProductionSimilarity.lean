import Mathlib.LinearAlgebra.Matrix.Invertible

open scoped Matrix

namespace OneesanFormal

/--
Suppose a Hankel pivot block at the target separator factors as `B = Sₜ C`,
and the one-symbol shifted block factors as `H = Sₛ Q C`.
If the production transition is formed as `H B⁻¹`, the suffix-coordinate
factor `C` cancels and the result is the quotient transition `Q` written in
the pivot-prefix coordinates `Sₛ,Sₜ`.
-/
theorem production_transition_similarity
    {R ι κ : Type*}
    [CommSemiring R]
    [Fintype ι] [DecidableEq ι]
    [Fintype κ] [DecidableEq κ]
    (Sₛ : Matrix ι ι R) (Sₜ C : Matrix κ κ R) (Q : Matrix ι κ R)
    [Invertible Sₜ] [Invertible C] :
    letI : Invertible (Sₜ * C) := invertibleMul Sₜ C
    (Sₛ * Q * C) * ⅟(Sₜ * C) = Sₛ * Q * ⅟Sₜ := by
  letI : Invertible (Sₜ * C) := invertibleMul Sₜ C
  rw [invOf_mul]
  simp [Matrix.mul_assoc]

/-- If the source prefix-coordinate matrix is also invertible, conjugating the
production transition back to quotient coordinates recovers `Q` exactly. -/
theorem production_transition_recovers_quotient
    {R ι κ : Type*}
    [CommSemiring R]
    [Fintype ι] [DecidableEq ι]
    [Fintype κ] [DecidableEq κ]
    (Sₛ : Matrix ι ι R) (Sₜ C : Matrix κ κ R) (Q : Matrix ι κ R)
    [Invertible Sₛ] [Invertible Sₜ] [Invertible C] :
    letI : Invertible (Sₜ * C) := invertibleMul Sₜ C
    ⅟Sₛ * ((Sₛ * Q * C) * ⅟(Sₜ * C)) * Sₜ = Q := by
  letI : Invertible (Sₜ * C) := invertibleMul Sₜ C
  rw [production_transition_similarity Sₛ Sₜ C Q]
  rw [Matrix.mul_assoc]
  rw [Matrix.invOf_mul_cancel_right]
  exact Matrix.invOf_mul_cancel_left Sₛ Q

end OneesanFormal
