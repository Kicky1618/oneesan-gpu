import OneesanFormal.MateRowLipschitz
import OneesanFormal.Row1Frontier
import Mathlib.Tactic

namespace OneesanFormal.ProductionRowSemantics

open OneesanFormal.ReverseMain
open OneesanFormal.MateRowLipschitz
open OneesanFormal.Row1Frontier

/-- The eight successful included-edge semantic cases in
`src/common/gridfp_transition.hpp`.  `NN/RN/LN` remain in the main state for an
interior cell.  `NR/NL/RL/LL/RR` shrink to the blocked state and the next cell
performs the mandatory excluded-edge N insertion.  At the last cell the same
rewrite is returned directly to main. -/
inductive IncludeCase where
  | NN | NR | NL | RN | LN | RL | LL | RR
  deriving DecidableEq, Repr

/-- Interior-cell branch classification copied from the production case table.
This deliberately contains no packed-bit arithmetic; it is the semantic ABI
against which the C++ `include_horizontal` switch is audited. -/
def interiorBlocked : IncludeCase → Bool
  | .NN | .RN | .LN => false
  | .NR | .NL | .RL | .LL | .RR => true

@[simp] theorem interiorBlocked_NN : interiorBlocked .NN = false := rfl
@[simp] theorem interiorBlocked_RN : interiorBlocked .RN = false := rfl
@[simp] theorem interiorBlocked_LN : interiorBlocked .LN = false := rfl
@[simp] theorem interiorBlocked_NR : interiorBlocked .NR = true := rfl
@[simp] theorem interiorBlocked_NL : interiorBlocked .NL = true := rfl
@[simp] theorem interiorBlocked_RL : interiorBlocked .RL = true := rfl
@[simp] theorem interiorBlocked_LL : interiorBlocked .LL = true := rfl
@[simp] theorem interiorBlocked_RR : interiorBlocked .RR = true := rfl

/-- Word-level meaning of each successful C++ switch case.  `pre` is the
unchanged prefix before the currently processed pair.  The LL/RR cases also
carry the balanced middle region scanned by the production matching loop. -/
inductive CaseRewrite : IncludeCase → List V → List V → Nat → Prop where
  | nn (pre tail : List V) :
      CaseRewrite .NN (pre ++ (.N :: .N :: tail))
        (pre ++ (.L :: .R :: tail)) (pre.length + 1)
  | nr (pre tail : List V) :
      CaseRewrite .NR (pre ++ (.N :: .R :: tail))
        (pre ++ (.R :: .N :: tail)) (pre.length + 1)
  | nl (pre tail : List V) :
      CaseRewrite .NL (pre ++ (.N :: .L :: tail))
        (pre ++ (.L :: .N :: tail)) (pre.length + 1)
  | rn (pre tail : List V) :
      CaseRewrite .RN (pre ++ (.R :: .N :: tail))
        (pre ++ (.N :: .R :: tail)) (pre.length + 1)
  | ln (pre tail : List V) :
      CaseRewrite .LN (pre ++ (.L :: .N :: tail))
        (pre ++ (.N :: .L :: tail)) (pre.length + 1)
  | rl (pre tail : List V) :
      CaseRewrite .RL (pre ++ (.R :: .L :: tail))
        (pre ++ (.N :: .N :: tail)) (pre.length + 1)
  | ll (pre mid tail : List V) :
      CaseRewrite .LL
        (pre ++ (.L :: .L :: mid ++ (.R :: tail)))
        (pre ++ (.N :: .N :: mid ++ (.L :: tail)))
        (pre.length + 1)
  | rr (pre mid tail : List V) :
      CaseRewrite .RR
        (pre ++ (.L :: mid ++ (.R :: .R :: tail)))
        (pre ++ (.R :: mid ++ (.N :: .N :: tail)))
        (pre.length + mid.length + 2)

/-- The named production switch table is exactly contained in the already
proved height-safe `CellRewrite` relation. -/
theorem CaseRewrite.toCellRewrite {c : IncludeCase} {src dst : List V} {cut : Nat}
    (h : CaseRewrite c src dst cut) : CellRewrite src dst cut := by
  cases h with
  | nn pre tail => exact CellRewrite.nn pre tail
  | nr pre tail => exact CellRewrite.nr pre tail
  | nl pre tail => exact CellRewrite.nl pre tail
  | rn pre tail => exact CellRewrite.rn pre tail
  | ln pre tail => exact CellRewrite.ln pre tail
  | rl pre tail => exact CellRewrite.rl pre tail
  | ll pre mid tail => exact CellRewrite.ll pre mid tail
  | rr pre mid tail => exact CellRewrite.rr pre mid tail

/-- Every successful named include case obeys the one-cut height bound. -/
theorem CaseRewrite.bound {c : IncludeCase} {src dst : List V} {cut : Nat}
    (h : CaseRewrite c src dst cut) : CutStepBound src dst cut :=
  h.toCellRewrite.bound

/-- One semantic production action after composing a blocked included branch
with the mandatory N insertion at the immediately following cell.  A main
included branch consumes one cell; a blocked branch consumes two physical cell
positions because the second is forced-excluded. -/
inductive Action : Nat → List V → List V → Nat → Prop where
  | exclude (i : Nat) (s : List V) : Action i s s (i + 1)
  | main {i : Nat} {s t : List V} :
      CellRewrite s t i → Action i s t (i + 1)
  | blocked {i : Nat} {s t : List V} :
      CellRewrite s t i → Action i s t (i + 2)

/-- A complete left-to-right semantic execution of the remainder of one
production row.  The index is a monotone cut/cell cursor. -/
inductive Trace : Nat → List V → List V → Prop where
  | done (i : Nat) (s : List V) : Trace i s s
  | next {i j : Nat} {s t u : List V} :
      Action i s t j → Trace j t u → Trace i s u

/-- Every production semantic action embeds in the simpler `RowTrace` model.
For a blocked branch, the composed forced-N position is represented by one
`RowTrace.skip`; this is why the next executable production cell is `i+2`. -/
theorem Action.toRowTrace {i j : Nat} {s t u : List V}
    (a : Action i s t j) (rest : RowTrace j t u) : RowTrace i s u := by
  cases a with
  | exclude =>
      exact RowTrace.skip rest
  | main h =>
      exact RowTrace.step h rest
  | blocked h =>
      exact RowTrace.step h (RowTrace.skip rest)

/-- Hence every semantic production-row trace is a `RowTrace`; all height
bounds proved there apply without redoing the eight switch cases. -/
theorem Trace.toRowTrace {i : Nat} {s u : List V}
    (h : Trace i s u) : RowTrace i s u := by
  induction h with
  | done i s => exact RowTrace.done i s
  | next a hrest ih => exact a.toRowTrace ih

/-- One actual post-row1 production row raises every source-inclusive frontier
prefix height by at most one. -/
theorem Trace.frontierBound_succ {rows : Nat} {src dst : List V}
    (h : Trace 1 src dst) (hsrc : FrontierBound rows src) :
    FrontierBound (rows + 1) dst := by
  exact h.toRowTrace.frontierBound_succ hsrc

/-- Sequence of semantic production rows. -/
inductive Rows : Nat → List V → List V → Prop where
  | zero (s : List V) : Rows 0 s s
  | succ {n : Nat} {s t u : List V} :
      Trace 1 s t → Rows n t u → Rows (n + 1) s u

/-- Translate the production-level row sequence to the already proved abstract
row sequence. -/
theorem Rows.toRowsTrace {n : Nat} {s u : List V}
    (h : Rows n s u) : RowsTrace n s u := by
  induction h with
  | zero s => exact RowsTrace.zero s
  | succ hrow htail ih => exact RowsTrace.succ hrow.toRowTrace ih

/-- Row-8 cap theorem at the production semantic level.  Row 1 is generated by
its concrete horizontal-edge bit word; any seven subsequent production traces
therefore have source-inclusive Mate-prefix height at most eight. -/
theorem row8_frontier_bound (row1Edges : List Bool) {row8 : List V}
    (hrows : Rows 7 (row1Word row1Edges) row8) : FrontierBound 8 row8 := by
  exact row8_bound_of_row1 (row1_frontier_bound row1Edges) hrows.toRowsTrace

end OneesanFormal.ProductionRowSemantics
