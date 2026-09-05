import OneesanFormal.GapHistory
import Mathlib.Data.Fintype.Sigma
import Mathlib.Tactic

namespace OneesanFormal

/-- A finite ranking code for one grand-Motzkin bridge of length `n`.
`k` chooses the common number of +1 and -1 steps; the two `Fin` factors
rank the choices of the + positions and then the - positions among the
remaining positions.  A later semantic layer can turn these ranks into
actual disjoint subsets of row ports. -/
def GrandBridgeCode (n : Nat) : Type :=
  Σ k : Fin (n + 1), Fin (Nat.choose n k) × Fin (Nat.choose (n - k) k)

noncomputable instance (n : Nat) : Fintype (GrandBridgeCode n) := by
  unfold GrandBridgeCode
  infer_instance

/-- `GrandBridgeCode` has exactly the central-trinomial cardinality. -/
theorem card_grandBridgeCode (n : Nat) :
    Fintype.card (GrandBridgeCode n) = centralTrinomial n := by
  calc
    Fintype.card (GrandBridgeCode n)
        = ∑ k : Fin (n + 1), Nat.choose n k * Nat.choose (n - k) k := by
            simp [GrandBridgeCode]
    _ = ∑ k ∈ Finset.range (n + 1), Nat.choose n k * Nat.choose (n - k) k := by
            simpa using (Fin.sum_univ_eq_sum_range
              (fun k : Nat => Nat.choose n k * Nat.choose (n - k) k) (n + 1))
    _ = centralTrinomial n := rfl

/-- `g` ordered grand-Motzkin bridge codes whose total length is `m`.
The first bridge length is selected explicitly; recursion stores the rest.
This is the finite interface suggested by the `h+1` planar gaps at a
separator of Motzkin height `h`. -/
def GapInterfaceCode : Nat → Nat → Type
  | 0, 0 => PUnit
  | 0, _ + 1 => Empty
  | g + 1, m => Σ z : Fin (m + 1), GrandBridgeCode z × GapInterfaceCode g (m - z)

noncomputable instance gapInterfaceCodeFintype : ∀ g m, Fintype (GapInterfaceCode g m)
  | 0, 0 => by
      change Fintype PUnit
      infer_instance
  | 0, _ + 1 => by
      change Fintype Empty
      infer_instance
  | g + 1, m => by
      change Fintype (Σ z : Fin (m + 1), GrandBridgeCode z × GapInterfaceCode g (m - z))
      letI : ∀ z : Fin (m + 1), Fintype (GapInterfaceCode g (m - z)) :=
        fun z => gapInterfaceCodeFintype g (m - z)
      infer_instance

/-- The explicit recursive interface type realizes `gapConv` exactly. -/
theorem card_gapInterfaceCode : ∀ g m,
    Fintype.card (GapInterfaceCode g m) = gapConv g m
  | 0, 0 => by
      change Fintype.card PUnit = 1
      simp
  | 0, m + 1 => by
      change Fintype.card Empty = 0
      simp
  | g + 1, m => by
      calc
        Fintype.card (GapInterfaceCode (g + 1) m)
            = ∑ z : Fin (m + 1), centralTrinomial z * gapConv g (m - z) := by
                simp [GapInterfaceCode, card_grandBridgeCode, card_gapInterfaceCode]
        _ = ∑ z ∈ Finset.range (m + 1), centralTrinomial z * gapConv g (m - z) := by
                simpa using (Fin.sum_univ_eq_sum_range
                  (fun z : Nat => centralTrinomial z * gapConv g (m - z)) (m + 1))
        _ = gapConv (g + 1) m := rfl

/-- Explicit finite interface candidate for an `r`-row strip at separator
height `h`.  The constant subtype predicate makes the type empty when `h>r`
and otherwise leaves the gap code unchanged. -/
def SeparatorInterfaceCode (r h : Nat) : Type :=
  {x : GapInterfaceCode (h + 1) (r - h) // h ≤ r}

noncomputable instance (r h : Nat) : Fintype (SeparatorInterfaceCode r h) := by
  unfold SeparatorInterfaceCode
  infer_instance

/-- The candidate separator interface has exactly the A111960/gap-triangle
size for every `r,h`, not merely for the experimentally checked rows. -/
theorem card_separatorInterfaceCode (r h : Nat) :
    Fintype.card (SeparatorInterfaceCode r h) = gapTriangle r h := by
  by_cases hh : h ≤ r
  · let e : SeparatorInterfaceCode r h ≃ GapInterfaceCode (h + 1) (r - h) :=
      { toFun := fun x => x.1
        invFun := fun x => ⟨x, hh⟩
        left_inv := by intro x; apply Subtype.ext; rfl
        right_inv := by intro x; rfl }
    rw [Fintype.card_congr e, card_gapInterfaceCode]
    simp [gapTriangle, hh]
  · let e : SeparatorInterfaceCode r h ≃ Empty :=
      { toFun := fun x => False.elim (hh x.2)
        invFun := fun x => nomatch x
        left_inv := by intro x; exact False.elim (hh x.2)
        right_inv := by intro x; nomatch x }
    rw [Fintype.card_congr e]
    simp [gapTriangle, hh]

/-- Row 8 therefore has 5686 explicitly enumerated interface codes in total. -/
theorem separatorInterfaceCode_row8_total :
    (∑ h ∈ Finset.range 9, Fintype.card (SeparatorInterfaceCode 8 h)) = 5686 := by
  simp [card_separatorInterfaceCode, gapTriangle]
  decide

end OneesanFormal
