import Mathlib.Data.Fintype.Card
import Mathlib.Tactic

namespace OneesanFormal.ProductionDecisionBound

/-- The only nondeterministic choice made by an ordinary Grid-FP cell is whether
its horizontal edge is excluded or included.  A blocked cell is forced and can
be represented by rejecting one of these bit words or ignoring its bit.
Thus a processed `rows × (width-1)` strip is driven by this fixed Boolean word. -/
abbrev DecisionWord (rows width : Nat) := Fin (rows * (width - 1)) → Bool

/-- There are exactly `2^(rows*(width-1))` horizontal include/exclude decision
words. -/
theorem card_decisionWord (rows width : Nat) :
    Fintype.card (DecisionWord rows width) = 2 ^ (rows * (width - 1)) := by
  simp [DecisionWord]

/-- Decision words that a deterministic execution maps to one target frontier
state.  The state type need not itself be finite. -/
def fiberFinset {State : Type*} [DecidableEq State] {rows width : Nat}
    (run : DecisionWord rows width → Option State) (target : State) :
    Finset (DecisionWord rows width) :=
  Finset.univ.filter fun d => run d = some target

/-- A coefficient of any deterministic binary-decision DP is bounded by the
number of its input decision words.  This certificate-safe bound requires no
path/planarity interpretation of the frontier state. -/
theorem fiber_card_le {State : Type*} [DecidableEq State] {rows width : Nat}
    (run : DecisionWord rows width → Option State) (target : State) :
    (fiberFinset run target).card ≤ 2 ^ (rows * (width - 1)) := by
  calc
    (fiberFinset run target).card ≤
        (Finset.univ : Finset (DecisionWord rows width)).card := Finset.card_le_univ _
    _ = Fintype.card (DecisionWord rows width) := by simp
    _ = 2 ^ (rows * (width - 1)) := card_decisionWord rows width

/-- Row-8 specialization used by the Grid-FP/structural integer-equality
certificate. -/
theorem row8_fiber_card_le {State : Type*} [DecidableEq State] {width : Nat}
    (run : DecisionWord 8 width → Option State) (target : State) :
    (fiberFinset run target).card ≤ 2 ^ (8 * (width - 1)) :=
  fiber_card_le run target

/-- Width 28 therefore needs only a 216-bit baseline coefficient bound. -/
theorem row8_width28_fiber_card_le {State : Type*} [DecidableEq State]
    (run : DecisionWord 8 28 → Option State) (target : State) :
    (fiberFinset run target).card ≤ 2 ^ 216 := by
  simpa using row8_fiber_card_le run target

end OneesanFormal.ProductionDecisionBound
