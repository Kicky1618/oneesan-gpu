import OneesanFormal.ProcessedStripCut
import OneesanFormal.ReverseMainCore
import Mathlib.Data.List.Count
import Mathlib.Tactic

namespace OneesanFormal.MateCutSemantics

open OneesanFormal.ReverseMain
open OneesanFormal.ProcessedStripCut

/-- Production frontier height after scanning a Mate-word prefix.  Grid-FP
starts the Motzkin scan at height one for the source component; `L` opens one
additional component and `R` closes one.  For a legal production prefix the
subtraction never truncates. -/
def openCount (xs : List V) : Nat := (1 + xs.count .L) - xs.count .R

@[simp] theorem openCount_nil : openCount [] = 1 := by simp [openCount]

@[simp] theorem openCount_cons_N (xs : List V) :
    openCount (.N :: xs) = openCount xs := by simp [openCount]

/-- Semantic realization of the unmatched arcs at one Mate-prefix cut.  `Arc`
indexes *arc occurrences*, not connected-component IDs: several occurrences
may belong to the same eventual path component.  Their physical subwalks are
pairwise edge-disjoint. -/
structure PrefixRealization {r w : Nat} (c : Fin (w - 1)) (xs : List V) where
  family : CrossingFamily (r := r) c
  card_eq_openCount : Fintype.card family.Component = openCount xs

/-- Every semantically realized Mate prefix has nesting height at most the
number of already processed physical rows. -/
theorem openCount_le_rows {r w : Nat} {c : Fin (w - 1)} {xs : List V}
    (R : PrefixRealization (r := r) c xs) : openCount xs ≤ r := by
  rw [← R.card_eq_openCount]
  exact R.family.card_le_rows

/-- Version for a complete frontier word: if every vertical cut admits an
edge-disjoint physical realization of the unmatched Mate arcs, every prefix
height at those cuts is at most `r`. -/
theorem every_realized_cut_le_rows {r w : Nat} (xs : List V)
    (realize : ∀ c : Fin (w - 1),
      PrefixRealization (r := r) c (xs.take (c.val + 1))) :
    ∀ c : Fin (w - 1), openCount (xs.take (c.val + 1)) ≤ r := by
  intro c
  exact openCount_le_rows (realize c)

/-- Once the complete frontier Mate word is balanced, physical realizations of
all proper vertical cuts bound *every* prefix stack height.  This is the form
consumed by the raw column transducer: reading `L/R/N` from left to right pushes,
pops, or leaves unchanged exactly this unmatched-arc count. -/
theorem all_prefix_openCount_le_rows {r w : Nat} (xs : List V)
    (hr : 0 < r)
    (hlen : xs.length = w)
    (hbalanced : openCount xs = 0)
    (realize : ∀ c : Fin (w - 1),
      PrefixRealization (r := r) c (xs.take (c.val + 1))) :
    ∀ k : Nat, k ≤ xs.length → openCount (xs.take k) ≤ r := by
  intro k hk
  by_cases hk0 : k = 0
  · subst k
    simp [openCount]
    omega
  by_cases hkfull : k = xs.length
  · subst k
    simpa [hbalanced] using (Nat.zero_le r)
  have hklt : k < xs.length := by omega
  have hcval : k - 1 < w - 1 := by omega
  let c : Fin (w - 1) := ⟨k - 1, hcval⟩
  have hc := openCount_le_rows (realize c)
  have htake : c.val + 1 = k := by
    dsimp [c]
    omega
  simpa [htake] using hc

/-- Stack-depth update performed by the column transducer when it consumes one
bottom Mate symbol.  The stack already contains the source marker at depth one. -/
def stackStep (h : Nat) : V → Option Nat
  | .N => some h
  | .L => some (h + 1)
  | .R => if h = 0 then none else some (h - 1)

/-- Consume a Mate word from an explicit initial stack depth. -/
def scanFrom : Nat → List V → Option Nat
  | h, [] => some h
  | h, x :: xs =>
      match stackStep h x with
      | none => none
      | some h' => scanFrom h' xs

/-- The production/raw convention starts with the source marker already open. -/
def scanStack (xs : List V) : Option Nat := scanFrom 1 xs

/-- A successful stack scan has exactly the arithmetic height encoded by symbol
counts.  This also proves that the `R` underflow guard is the only issue hidden
by natural-number subtraction. -/
theorem scanFrom_eq_count {h z : Nat} {xs : List V}
    (hs : scanFrom h xs = some z) :
    z = (h + xs.count .L) - xs.count .R := by
  induction xs generalizing h z with
  | nil =>
      simp [scanFrom] at hs ⊢
      exact hs.symm
  | cons x xs ih =>
      cases x with
      | N =>
          simp [scanFrom, stackStep] at hs ⊢
          exact ih hs
      | L =>
          simp [scanFrom, stackStep] at hs ⊢
          have hz := ih hs
          omega
      | R =>
          by_cases hh : h = 0
          · simp [scanFrom, stackStep, hh] at hs
          · have hhpos : 0 < h := Nat.pos_of_ne_zero hh
            simp [scanFrom, stackStep, hh] at hs ⊢
            have hz := ih hs
            omega

/-- In particular, the raw stack depth after any successful emitted prefix is
exactly the production frontier `openCount` for that prefix. -/
theorem scanStack_eq_openCount {xs : List V} {z : Nat}
    (hs : scanStack xs = some z) : z = openCount xs := by
  have h := scanFrom_eq_count (h := 1) hs
  simpa [scanStack, openCount, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h

/-- Combining physical cut realization with the raw stack parser gives the exact
cap theorem needed by the row-r column transducer. -/
theorem successful_prefix_stack_le_rows {r w : Nat} (xs : List V)
    (hr : 0 < r)
    (hlen : xs.length = w)
    (hbalanced : openCount xs = 0)
    (realize : ∀ c : Fin (w - 1),
      PrefixRealization (r := r) c (xs.take (c.val + 1)))
    (k z : Nat) (hk : k ≤ xs.length)
    (hscan : scanStack (xs.take k) = some z) : z ≤ r := by
  rw [scanStack_eq_openCount hscan]
  exact all_prefix_openCount_le_rows xs hr hlen hbalanced realize k hk


end OneesanFormal.MateCutSemantics
