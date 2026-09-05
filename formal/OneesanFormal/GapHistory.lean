import Mathlib.Data.Nat.Choose.Sum
import Mathlib.Tactic

namespace OneesanFormal

/-- Number of balanced length-`n` words over steps -1,0,+1: the central
trinomial coefficient.  The summand chooses the +1 steps and then the -1
steps; terms past floor(n/2) vanish automatically. -/
def centralTrinomial (n : Nat) : Nat :=
  ∑ k ∈ Finset.range (n + 1), Nat.choose n k * Nat.choose (n - k) k

/-- Convolution power of central trinomial coefficients.  Combinatorially,
`gapConv g m` counts `g` ordered balanced ternary histories whose total
length is `m`. -/
def gapConv : Nat → Nat → Nat
  | 0, 0 => 1
  | 0, _ + 1 => 0
  | g + 1, m => ∑ z ∈ Finset.range (m + 1), centralTrinomial z * gapConv g (m - z)

/-- Candidate dimension of the height-`h` separator block after `r` rows. -/
def gapTriangle (r h : Nat) : Nat :=
  if h ≤ r then gapConv (h + 1) (r - h) else 0

example : centralTrinomial 0 = 1 := by decide
example : centralTrinomial 1 = 1 := by decide
example : centralTrinomial 2 = 3 := by decide
example : centralTrinomial 6 = 141 := by decide

/-- The experimentally observed Hankel-rank triangle through row 8 is exactly
reproduced by the gap-history count.  This does *not* assert the rank theorem;
it machine-checks the independent combinatorial side of the conjecture. -/
theorem gapTriangle_rows_0_to_8 :
    ([[gapTriangle 0 0],
      [gapTriangle 1 0, gapTriangle 1 1],
      [gapTriangle 2 0, gapTriangle 2 1, gapTriangle 2 2],
      [gapTriangle 3 0, gapTriangle 3 1, gapTriangle 3 2, gapTriangle 3 3],
      [gapTriangle 4 0, gapTriangle 4 1, gapTriangle 4 2, gapTriangle 4 3, gapTriangle 4 4],
      [gapTriangle 5 0, gapTriangle 5 1, gapTriangle 5 2, gapTriangle 5 3, gapTriangle 5 4, gapTriangle 5 5],
      [gapTriangle 6 0, gapTriangle 6 1, gapTriangle 6 2, gapTriangle 6 3, gapTriangle 6 4, gapTriangle 6 5, gapTriangle 6 6],
      [gapTriangle 7 0, gapTriangle 7 1, gapTriangle 7 2, gapTriangle 7 3, gapTriangle 7 4, gapTriangle 7 5, gapTriangle 7 6, gapTriangle 7 7],
      [gapTriangle 8 0, gapTriangle 8 1, gapTriangle 8 2, gapTriangle 8 3, gapTriangle 8 4, gapTriangle 8 5, gapTriangle 8 6, gapTriangle 8 7, gapTriangle 8 8]] : List (List Nat)) =
    [[1], [1,1], [3,2,1], [7,7,3,1], [19,20,12,4,1],
     [51,61,40,18,5,1], [141,182,135,68,25,6,1],
     [393,547,441,251,105,33,7,1],
     [1107,1640,1428,888,420,152,42,8,1]] := by
  decide


/-- The row-8 gap-history block dimensions sum to the observed separator
state count 5686. -/
theorem gapTriangle_row8_total :
    gapTriangle 8 0 + gapTriangle 8 1 + gapTriangle 8 2 + gapTriangle 8 3 +
    gapTriangle 8 4 + gapTriangle 8 5 + gapTriangle 8 6 + gapTriangle 8 7 +
    gapTriangle 8 8 = 5686 := by
  decide

end OneesanFormal
