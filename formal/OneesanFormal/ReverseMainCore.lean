namespace OneesanFormal.ReverseMain

/-- The nine legal local Mate-value pairs used by the p>1 Grid-FP update. -/
inductive Pair where
  | NN | NR | NL | RN | RR | RL | LN | LR | LL
  deriving DecidableEq, Repr

/-- Included-edge transitions that stay in the main state space for p>1.
All other valid included branches go to the blocked space (or are invalid). -/
def includeToMain : Pair → Option Pair
  | .NN => some .LR
  | .RN => some .NR
  | .LN => some .NL
  | _   => none

/-- The unique included-edge main predecessor of a target local pair, if any. -/
def includedMainPred : Pair → Option Pair
  | .LR => some .NN
  | .NR => some .RN
  | .NL => some .LN
  | _   => none

/-- Net frontier-height change of the two local symbols. -/
def pairDelta : Pair → Int
  | .NN => 0
  | .NR => -1
  | .NL => 1
  | .RN => -1
  | .RR => -2
  | .RL => 0
  | .LN => 1
  | .LR => 0
  | .LL => 2

/-- Every main→main included transition preserves the height on both sides of
its two-symbol window.  Hence the untouched factor half and the factor block
(intermediate height) can be reused verbatim by the inverse kernel. -/
theorem includeToMain_preserves_delta {s t : Pair}
    (h : includeToMain s = some t) : pairDelta s = pairDelta t := by
  cases s <;> cases t <;> simp [includeToMain, pairDelta] at h ⊢

/-- Soundness of the target-gather predecessor table. -/
theorem includedMainPred_sound {t s : Pair}
    (h : includedMainPred t = some s) : includeToMain s = some t := by
  cases t <;> cases s <;> simp [includedMainPred, includeToMain] at h ⊢

/-- Completeness of the target-gather predecessor table. -/
theorem includedMainPred_complete {s t : Pair}
    (h : includeToMain s = some t) : includedMainPred t = some s := by
  cases s <;> cases t <;> simp [includedMainPred, includeToMain] at h ⊢

/-- The contribution obtained by source-scattering the three main→main
included branches into a fixed target. -/
def scatterIncludedContribution (count : Pair → Nat) (t : Pair) : Nat :=
  (if t = .LR then count .NN else 0) +
  (if t = .NR then count .RN else 0) +
  (if t = .NL then count .LN else 0)

/-- The same contribution computed by one target thread from the inverse table. -/
def gatherIncludedContribution (count : Pair → Nat) (t : Pair) : Nat :=
  match includedMainPred t with
  | none   => 0
  | some s => count s

/-- Source-scatter and target-gather compute exactly the same main→main sum. -/
theorem scatter_eq_gather (count : Pair → Nat) (t : Pair) :
    scatterIncludedContribution count t = gatherIncludedContribution count t := by
  cases t <;> simp [scatterIncludedContribution, gatherIncludedContribution,
    includedMainPred]

/-- Therefore `includeToMain` is injective on its domain.  This is the fact
that lets one target thread gather at most one included main predecessor. -/
theorem includeToMain_injective {a b t : Pair}
    (ha : includeToMain a = some t) (hb : includeToMain b = some t) : a = b := by
  have hpa := includedMainPred_complete ha
  have hpb := includedMainPred_complete hb
  rw [hpa] at hpb
  exact Option.some.inj hpb

/-- Abstract zipper around position p.  The production packed-Mate operation
`blocked_exclude` inserts N at p, while `mshrink` removes that N. -/
inductive V where | N | R | L deriving DecidableEq, Repr

structure BlockedZipper where
  low  : List V
  high : List V
  deriving DecidableEq, Repr

structure MainZipper where
  low    : List V
  center : V
  high   : List V
  deriving DecidableEq, Repr

def blockedExclude (b : BlockedZipper) : MainZipper :=
  { low := b.low, center := .N, high := b.high }

def shrinkCenter (m : MainZipper) : BlockedZipper :=
  { low := m.low, high := m.high }

/-- `mshrink` is a left inverse of the forced-N blocked→main transition. -/
theorem shrink_blockedExclude (b : BlockedZipper) :
    shrinkCenter (blockedExclude b) = b := by
  rfl

/-- A main target has a blocked predecessor iff its inserted position is N,
and that predecessor is unique. -/
theorem blocked_predecessor_unique {m : MainZipper} (h : m.center = .N) :
    blockedExclude (shrinkCenter m) = m := by
  cases m
  simp_all [blockedExclude, shrinkCenter]


/-- p>1 included branches that land in the blocked state space. -/
inductive BlockBranch where | NR | NL | RL | LL | RR deriving DecidableEq, Repr

/-- Symbol exposed at compressed position p-1 in the blocked target. -/
def blockedTargetMarker : BlockBranch → V
  | .NR => .R
  | .NL => .L
  | .RL => .N
  | .LL => .N
  | .RR => .N

/-- NR/NL targets are disjoint from every RL/LL/RR target, so a source kernel
may plain-store NR/NL contributions while closure-like branches use atomics. -/
theorem nrnl_disjoint_closure
    (a b : BlockBranch)
    (ha : a = .NR ∨ a = .NL)
    (hb : b = .RL ∨ b = .LL ∨ b = .RR) :
    blockedTargetMarker a ≠ blockedTargetMarker b := by
  rcases ha with rfl | rfl <;> rcases hb with rfl | rfl | rfl <;> decide

end OneesanFormal.ReverseMain
