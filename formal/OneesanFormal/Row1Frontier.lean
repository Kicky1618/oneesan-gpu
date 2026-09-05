import OneesanFormal.MateRowLipschitz
import Mathlib.Tactic

namespace OneesanFormal.Row1Frontier

open OneesanFormal.ReverseMain
open OneesanFormal.BoundedHeight
open OneesanFormal.MateRowLipschitz

/-- Integer occupancy height of the single horizontal strand entering a row-1
vertex. -/
def bitHeight : Bool → Int
  | false => 0
  | true => 1

/-- Mate symbol produced by the two adjacent 0/1 horizontal edge occupancies
at a row-1 vertex.  Equal bits continue/omit the strand, 0→1 opens an arc and
1→0 closes one. -/
def boundarySymbol : Bool → Bool → V
  | false, false => .N
  | false, true  => .L
  | true,  false => .R
  | true,  true  => .N

/-- The symbol update exactly transports the 0/1 cut height. -/
theorem boundarySymbol_delta (a b : Bool) :
    bitHeight a + symDelta (boundarySymbol a b) = bitHeight b := by
  cases a <;> cases b <;> decide

/-- Convert a list of selected row-1 horizontal edges into the production Mate
word. `prev` is the edge entering the first remaining vertex; when no internal
edge remains, the outside edge on the right is zero. -/
def wordFrom : Bool → List Bool → List V
  | prev, [] => [boundarySymbol prev false]
  | prev, next :: rest => boundarySymbol prev next :: wordFrom next rest

/-- Prefix bound relative to an arbitrary incoming 0/1 strand height. -/
def PrefixBoundFrom (h : Int) (xs : List V) : Prop :=
  ∀ k, h + prefixDelta xs k ≤ 1

/-- Every row-1 edge mask generates a word whose running height remains in
`{0,1}`; more precisely the running height after each vertex is the outgoing
horizontal edge occupancy. -/
theorem wordFrom_bound (prev : Bool) (edges : List Bool) :
    PrefixBoundFrom (bitHeight prev) (wordFrom prev edges) := by
  induction edges generalizing prev with
  | nil =>
      intro k
      cases k with
      | zero => cases prev <;> simp [PrefixBoundFrom, bitHeight, prefixDelta]
      | succ k =>
          cases k with
          | zero =>
              simpa [PrefixBoundFrom, wordFrom, prefixDelta] using
                show bitHeight prev + symDelta (boundarySymbol prev false) ≤ 1 by
                  rw [boundarySymbol_delta]
                  simp [bitHeight]
          | succ k =>
              cases prev <;> simp [PrefixBoundFrom, wordFrom, prefixDelta, boundarySymbol, bitHeight, symDelta]
  | cons next rest ih =>
      intro k
      cases k with
      | zero => cases prev <;> simp [PrefixBoundFrom, bitHeight, prefixDelta]
      | succ k =>
          have ht := ih next k
          have hd := boundarySymbol_delta prev next
          simp [PrefixBoundFrom, wordFrom, prefixDelta] at ht ⊢
          omega

/-- Production row 1 starts with the source strand present on the left. -/
def row1Word (edges : List Bool) : List V := wordFrom true edges

/-- Specialized row-1 initializer height theorem used by the row-8 cap proof. -/
theorem row1_frontier_bound (edges : List Bool) :
    FrontierBound 1 (row1Word edges) := by
  have h := wordFrom_bound true edges
  intro k
  simpa [FrontierBound, PrefixBoundFrom, row1Word, bitHeight] using h k

/-- `wordFrom` emits one vertex symbol more than there are internal horizontal
edge decisions. -/
theorem wordFrom_length (prev : Bool) (edges : List Bool) :
    (wordFrom prev edges).length = edges.length + 1 := by
  induction edges generalizing prev with
  | nil => simp [wordFrom]
  | cons a rest ih => simp [wordFrom, ih]

/-- A width `W` row has exactly `W-1` internal horizontal-edge decisions and
therefore produces a Mate word of width `W`. -/
theorem row1Word_length (edges : List Bool) :
    (row1Word edges).length = edges.length + 1 := by
  simpa [row1Word] using wordFrom_length true edges

end OneesanFormal.Row1Frontier
