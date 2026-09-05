import Mathlib

/-! Exact compression of a finite transfer by symmetry orbits.
The application must supply the row-moving, orbit-preserving bijections.
No assumption of equal orbit sizes or free group action is made. -/
namespace OneesanFormal.OrbitTransfer
open scoped BigOperators

variable {S Q : Type*} [Fintype S] [Fintype Q] [DecidableEq Q]

def counts (K : S → S → Nat) : Nat → S → Nat
  | 0, _ => 1
  | n + 1, x => ∑ y, K x y * counts K n y

def quotientKernel (K : S → S → Nat) (orbit : S → Q) (rep : Q → S)
    (q r : Q) : Nat :=
  ∑ y, if orbit y = r then K (rep q) y else 0

def orbitSize (orbit : S → Q) (q : Q) : Nat :=
  ∑ x, if orbit x = q then 1 else 0

theorem quotient_weighted_sum (K : S → S → Nat) (orbit : S → Q)
    (rep : Q → S) (v : Q → Nat) (q : Q) :
    (∑ r, quotientKernel K orbit rep q r * v r) =
      ∑ y, K (rep q) y * v (orbit y) := by
  classical
  simp only [quotientKernel, Finset.sum_mul, ite_mul, zero_mul]
  rw [Finset.sum_comm]
  simp

/-- Each source can be moved to its representative by a bijection preserving
orbit labels and transition weights. Group automorphisms supply such maps. -/
def SymmetricPartition (K : S → S → Nat) (orbit : S → Q) (rep : Q → S) : Prop :=
  ∀ x, ∃ e : S ≃ S,
    (∀ y, orbit (e y) = orbit y) ∧
    (∀ y, K x y = K (rep (orbit x)) (e y))

omit [Fintype S] [Fintype Q] [DecidableEq Q] in
/-- The four-map recipe used by complement/reflection compression. The
application supplies the two compatibility symmetries, invariant labels and
the fact that each chosen representative is one of the four transforms. -/
theorem symmetric_of_four (K : S → S → Nat) (orbit : S → Q) (rep : Q → S)
    (C R : S ≃ S)
    (hC : ∀ x y, K (C x) (C y) = K x y)
    (hR : ∀ x y, K (R x) (R y) = K x y)
    (oC : ∀ x, orbit (C x) = orbit x)
    (oR : ∀ x, orbit (R x) = orbit x)
    (hrep : ∀ x, rep (orbit x) = x ∨ rep (orbit x) = C x ∨
      rep (orbit x) = R x ∨ rep (orbit x) = C (R x)) :
    SymmetricPartition K orbit rep := by
  intro x
  rcases hrep x with hx | hx | hx | hx
  · refine ⟨Equiv.refl S, fun _ => rfl, ?_⟩
    intro y
    simp [hx]
  · refine ⟨C, oC, ?_⟩
    intro y
    rw [hx, hC]
  · refine ⟨R, oR, ?_⟩
    intro y
    rw [hx, hR]
  · refine ⟨R.trans C, ?_, ?_⟩
    · intro y
      change orbit (C (R y)) = orbit y
      rw [oC, oR]
    · intro y
      change K x y = K (rep (orbit x)) (C (R y))
      rw [hx, hC, hR]

theorem counts_exact (K : S → S → Nat) (orbit : S → Q) (rep : Q → S)
    (h : SymmetricPartition K orbit rep) (n : Nat) (x : S) :
    counts K n x = counts (quotientKernel K orbit rep) n (orbit x) := by
  classical
  induction n generalizing x with
  | zero => rfl
  | succ n ih =>
      rcases h x with ⟨e, heOrbit, heK⟩
      change (∑ y, K x y * counts K n y) =
        ∑ r, quotientKernel K orbit rep (orbit x) r *
          counts (quotientKernel K orbit rep) n r
      rw [quotient_weighted_sum]
      calc
        (∑ y, K x y * counts K n y) =
            ∑ y, K (rep (orbit x)) (e y) *
              counts (quotientKernel K orbit rep) n (orbit (e y)) := by
                apply Finset.sum_congr rfl
                intro y _
                rw [ih y, heK y, heOrbit y]
        _ = _ := e.sum_comp (fun y => K (rep (orbit x)) y *
          counts (quotientKernel K orbit rep) n (orbit y))

theorem weighted_total (orbit : S → Q) (v : Q → Nat) :
    (∑ q, orbitSize orbit q * v q) = ∑ x, v (orbit x) := by
  classical
  simp only [orbitSize, Finset.sum_mul, ite_mul, zero_mul, one_mul]
  rw [Finset.sum_comm]
  simp

theorem total_exact (K : S → S → Nat) (orbit : S → Q) (rep : Q → S)
    (h : SymmetricPartition K orbit rep) (n : Nat) :
    (∑ x, counts K n x) =
      ∑ q, orbitSize orbit q * counts (quotientKernel K orbit rep) n q := by
  classical
  simp_rw [counts_exact K orbit rep h n]
  exact (weighted_total orbit _).symm

end OneesanFormal.OrbitTransfer
