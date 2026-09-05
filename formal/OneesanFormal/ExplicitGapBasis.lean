import OneesanFormal.GapInterface
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Fintype.Sigma
import Mathlib.Tactic

namespace OneesanFormal

/-- An explicit balanced ternary bridge of length `n`.

`up` is the set of `+1` positions and `down` is a set of the same cardinality
chosen from the complement of `up`; every remaining position is the zero step.
Thus disjointness and zero total displacement are structural, not predicates
checked afterwards. -/
def ExplicitGrandBridge (n : Nat) : Type :=
  Σ k : Fin (n + 1),
    Σ up : {s : Finset (Fin n) // s.card = k},
      {d : Finset {i : Fin n // i ∉ up.1} // d.card = k}

noncomputable instance (n : Nat) : Fintype (ExplicitGrandBridge n) := by
  unfold ExplicitGrandBridge
  infer_instance

/-- The actual ternary step encoded at one position. -/
inductive BridgeStep where
  | down | flat | up
  deriving DecidableEq, Repr

/-- Read an explicit bridge as a concrete `down/flat/up` word. -/
def ExplicitGrandBridge.step {n : Nat} (b : ExplicitGrandBridge n) (i : Fin n) : BridgeStep :=
  if hi : i ∈ b.2.1.1 then BridgeStep.up
  else if (⟨i, hi⟩ : {j : Fin n // j ∉ b.2.1.1}) ∈ b.2.2.1 then BridgeStep.down
  else BridgeStep.flat

@[simp] theorem explicitGrandBridge_up_card {n : Nat} (b : ExplicitGrandBridge n) :
    b.2.1.1.card = b.1 := b.2.1.2

@[simp] theorem explicitGrandBridge_down_card {n : Nat} (b : ExplicitGrandBridge n) :
    b.2.2.1.card = b.1 := b.2.2.2

/-- Number of available positions after choosing the up-steps. -/
theorem card_bridge_complement {n : Nat} (k : Fin (n + 1))
    (up : {s : Finset (Fin n) // s.card = k}) :
    Fintype.card {i : Fin n // i ∉ up.1} = n - k := by
  rw [Fintype.card_subtype_compl]
  simp [up.2]

/-- For a fixed up-set of size `k`, the down-steps can be chosen in exactly
`choose (n-k) k` ways from its complement. -/
theorem card_bridge_down_choices {n : Nat} (k : Fin (n + 1))
    (up : {s : Finset (Fin n) // s.card = k}) :
    Fintype.card {d : Finset {i : Fin n // i ∉ up.1} // d.card = k} =
      Nat.choose (n - k) k := by
  rw [Fintype.card_finset_len, card_bridge_complement]

/-- Concrete bridges have the central-trinomial cardinality. -/
theorem card_explicitGrandBridge (n : Nat) :
    Fintype.card (ExplicitGrandBridge n) = centralTrinomial n := by
  have hinner (k : Fin (n + 1)) :
      (∑ up : {s : Finset (Fin n) // s.card = k},
        Fintype.card {d : Finset {i : Fin n // i ∉ up.1} // d.card = k}) =
        Nat.choose n k * Nat.choose (n - k) k := by
    calc
      (∑ up : {s : Finset (Fin n) // s.card = k},
          Fintype.card {d : Finset {i : Fin n // i ∉ up.1} // d.card = k})
          = ∑ _up : {s : Finset (Fin n) // s.card = k},
              Nat.choose (n - k) k := by
                apply Fintype.sum_congr
                intro up
                exact card_bridge_down_choices k up
      _ = Nat.choose n k * Nat.choose (n - k) k := by
            simp [Fintype.card_finset_len]
  calc
    Fintype.card (ExplicitGrandBridge n)
        = ∑ k : Fin (n + 1), Nat.choose n k * Nat.choose (n - k) k := by
            rw [show Fintype.card (ExplicitGrandBridge n) =
              ∑ k : Fin (n + 1),
                ∑ up : {s : Finset (Fin n) // s.card = k},
                  Fintype.card {d : Finset {i : Fin n // i ∉ up.1} // d.card = k} by
                    simp [ExplicitGrandBridge]]
            exact Fintype.sum_congr _ _ hinner
    _ = ∑ k ∈ Finset.range (n + 1), Nat.choose n k * Nat.choose (n - k) k := by
            simpa using (Fin.sum_univ_eq_sum_range
              (fun k : Nat => Nat.choose n k * Nat.choose (n - k) k) (n + 1))
    _ = centralTrinomial n := rfl


/-- Down-step positions, viewed back in the original `Fin n` coordinate set. -/
def ExplicitGrandBridge.downPositions {n : Nat} (b : ExplicitGrandBridge n) : Finset (Fin n) :=
  b.2.2.1.map (Function.Embedding.subtype fun i : Fin n => i ∉ b.2.1.1)

@[simp] theorem explicitGrandBridge_downPositions_card {n : Nat} (b : ExplicitGrandBridge n) :
    b.downPositions.card = b.1 := by
  simp [ExplicitGrandBridge.downPositions, b.2.2.2]

/-- Up and down positions are disjoint by construction. -/
theorem explicitGrandBridge_up_down_disjoint {n : Nat} (b : ExplicitGrandBridge n) :
    Disjoint b.2.1.1 b.downPositions := by
  rw [Finset.disjoint_left]
  intro i hi hdown
  simp only [ExplicitGrandBridge.downPositions, Finset.mem_map] at hdown
  rcases hdown with ⟨j, hj, hji⟩
  have heq : (j : Fin n) = i := by simpa using hji
  subst i
  exact j.2 hi

/-- The explicit and ranking-code bridge types have the same finite size.
The equivalence is intentionally noncomputable; semantic code uses `step` and
`downPositions`, not this arbitrary finite-set bijection. -/
noncomputable def explicitGrandBridgeEquivCode (n : Nat) :
    ExplicitGrandBridge n ≃ GrandBridgeCode n :=
  Fintype.equivOfCardEq ((card_explicitGrandBridge n).trans (card_grandBridgeCode n).symm)

/-- Explicit gap tuples: `g` concrete grand-Motzkin bridges with total length `m`. -/
def ExplicitGapInterface : Nat → Nat → Type
  | 0, 0 => PUnit
  | 0, _ + 1 => Empty
  | g + 1, m => Σ z : Fin (m + 1), ExplicitGrandBridge z × ExplicitGapInterface g (m - z)

noncomputable instance explicitGapInterfaceFintype : ∀ g m, Fintype (ExplicitGapInterface g m)
  | 0, 0 => by change Fintype PUnit; infer_instance
  | 0, _ + 1 => by change Fintype Empty; infer_instance
  | g + 1, m => by
      change Fintype (Σ z : Fin (m + 1), ExplicitGrandBridge z × ExplicitGapInterface g (m - z))
      letI : ∀ z : Fin (m + 1), Fintype (ExplicitGapInterface g (m - z)) :=
        fun z => explicitGapInterfaceFintype g (m - z)
      infer_instance

/-- The concrete `U/D/N` gap basis realizes the same convolution exactly. -/
theorem card_explicitGapInterface : ∀ g m,
    Fintype.card (ExplicitGapInterface g m) = gapConv g m
  | 0, 0 => by change Fintype.card PUnit = 1; simp
  | 0, m + 1 => by change Fintype.card Empty = 0; simp
  | g + 1, m => by
      calc
        Fintype.card (ExplicitGapInterface (g + 1) m)
            = ∑ z : Fin (m + 1), centralTrinomial z * gapConv g (m - z) := by
                simp [ExplicitGapInterface, card_explicitGrandBridge, card_explicitGapInterface]
        _ = ∑ z ∈ Finset.range (m + 1), centralTrinomial z * gapConv g (m - z) := by
                simpa using (Fin.sum_univ_eq_sum_range
                  (fun z : Nat => centralTrinomial z * gapConv g (m - z)) (m + 1))
        _ = gapConv (g + 1) m := rfl

/-- Row-height block using explicit gap words rather than opaque ranking codes. -/
def ExplicitSeparatorBasis (r h : Nat) : Type :=
  {x : ExplicitGapInterface (h + 1) (r - h) // h ≤ r}

noncomputable instance (r h : Nat) : Fintype (ExplicitSeparatorBasis r h) := by
  unfold ExplicitSeparatorBasis
  infer_instance

/-- The explicit normal-form basis has exactly the gap-triangle/A111960 size. -/
theorem card_explicitSeparatorBasis (r h : Nat) :
    Fintype.card (ExplicitSeparatorBasis r h) = gapTriangle r h := by
  by_cases hh : h ≤ r
  · let e : ExplicitSeparatorBasis r h ≃ ExplicitGapInterface (h + 1) (r - h) :=
      { toFun := fun x => x.1
        invFun := fun x => ⟨x, hh⟩
        left_inv := by intro x; apply Subtype.ext; rfl
        right_inv := by intro x; rfl }
    rw [Fintype.card_congr e, card_explicitGapInterface]
    simp [gapTriangle, hh]
  · let e : ExplicitSeparatorBasis r h ≃ Empty :=
      { toFun := fun x => False.elim (hh x.2)
        invFun := fun x => nomatch x
        left_inv := by intro x; exact False.elim (hh x.2)
        right_inv := by intro x; nomatch x }
    rw [Fintype.card_congr e]
    simp [gapTriangle, hh]

/-- In particular the concrete row-8 normal forms total 5686. -/
theorem explicitSeparatorBasis_row8_total :
    (∑ h ∈ Finset.range 9, Fintype.card (ExplicitSeparatorBasis 8 h)) = 5686 := by
  simp [card_explicitSeparatorBasis, gapTriangle]
  decide

end OneesanFormal
