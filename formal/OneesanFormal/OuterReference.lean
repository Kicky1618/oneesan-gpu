import OneesanFormal.GridFaceBoundary

namespace OneesanFormal.OuterReference

open OneesanFormal.GridFaceBoundary

/-- Horizontal edges of the fixed outer-boundary reference path
`(0,0) → (0,n) → (n,n)`, using the production ghost-column indexing. -/
def outerHorizontal (n r c : Nat) : F2 :=
  if r = 0 ∧ 0 < c ∧ c ≤ n then 1 else 0

/-- Vertical edges of the same outer-boundary reference path, using the
production ghost-row indexing. -/
def outerVertical (n r c : Nat) : F2 :=
  if c = n ∧ 0 < r ∧ r ≤ n then 1 else 0

/-- The terminal-parity vector of a corner-to-corner path. -/
def cornerParity (n r c : Nat) : F2 :=
  if (r = 0 ∧ c = 0) ∨ (r = n ∧ c = n) then 1 else 0

@[simp] theorem outerHorizontal_left_ghost (n r : Nat) :
    outerHorizontal n r 0 = 0 := by
  simp [outerHorizontal]

@[simp] theorem outerHorizontal_right_ghost (n r : Nat) :
    outerHorizontal n r (n + 1) = 0 := by
  simp [outerHorizontal]

@[simp] theorem outerVertical_top_ghost (n c : Nat) :
    outerVertical n 0 c = 0 := by
  simp [outerVertical]

@[simp] theorem outerVertical_bottom_ghost (n c : Nat) :
    outerVertical n (n + 1) c = 0 := by
  simp [outerVertical]

/-- The chosen reference path uses only the outer boundary, hence it has no
edge incident to a strictly interior grid vertex. -/
theorem outer_reference_interior_zero
    (n r c : Nat) (_hr : r + 1 < n) (hc : c + 1 < n) :
    outerHorizontal n (r + 1) (c + 1) = 0 ∧
      outerHorizontal n (r + 1) (c + 2) = 0 ∧
      outerVertical n (r + 1) (c + 1) = 0 ∧
      outerVertical n (r + 2) (c + 1) = 0 := by
  simp [outerHorizontal, outerVertical]
  omega

/-- The fixed outer-boundary reference has odd incidence exactly at the two
opposite corners and even incidence at every other grid vertex. -/
theorem outer_reference_vertexParity
    (n r c : Nat) (hn : 0 < n) (hr : r ≤ n) (hc : c ≤ n) :
    vertexParity (outerHorizontal n) (outerVertical n) r c =
      cornerParity n r c := by
  by_cases hr0 : r = 0 <;>
  by_cases hrn : r = n <;>
  by_cases hc0 : c = 0 <;>
  by_cases hcn : c = n <;>
    simp [vertexParity, outerHorizontal, outerVertical, cornerParity,
      hr0, hrn, hc0, hcn, CharTwo.add_self_eq_zero] <;>
    split_ifs <;> simp [CharTwo.add_self_eq_zero] <;> omega

end OneesanFormal.OuterReference
