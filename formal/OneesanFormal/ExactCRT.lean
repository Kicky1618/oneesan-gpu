import Mathlib.Data.Nat.Basic

namespace OneesanFormal.ExactCRT

/-- Inside a rigorous bound `B < M`, a residue class modulo `M` has at most
one representative in `[0,B]`.  This is the mathematical uniqueness fact used
by the exact runner once the CRT modulus product exceeds the path bound. -/
theorem bounded_mod_unique {a x B M : Nat}
    (ha : a ≤ B) (hx : x ≤ B) (hBM : B < M)
    (hmod : x % M = a % M) : x = a := by
  have hxm : x < M := lt_of_le_of_lt hx hBM
  have ham : a < M := lt_of_le_of_lt ha hBM
  simpa [Nat.mod_eq_of_lt hxm, Nat.mod_eq_of_lt ham] using hmod

/-- If the true answer is bounded by `B`, a canonical CRT representative
`x < M` congruent to it cannot exceed `B` once `M > B`.  Therefore an observed
`x > B` is a sound corruption/inconsistency signal, not a valid alternative
representative. -/
theorem canonical_congruent_le_bound {a x B M : Nat}
    (ha : a ≤ B) (hxM : x < M) (hBM : B < M)
    (hmod : x % M = a % M) : x ≤ B := by
  have haM : a < M := lt_of_le_of_lt ha hBM
  have hxa : x = a := by
    simpa [Nat.mod_eq_of_lt hxM, Nat.mod_eq_of_lt haM] using hmod
  simpa [hxa] using ha

/-- Once both candidate exact answers satisfy the rigorous bound, agreement
modulo the accumulated CRT product forces ordinary integer equality. -/
theorem two_bounded_candidates_unique {x y B M : Nat}
    (hx : x ≤ B) (hy : y ≤ B) (hBM : B < M)
    (hmod : x % M = y % M) : x = y := by
  exact bounded_mod_unique hy hx hBM hmod

end OneesanFormal.ExactCRT
