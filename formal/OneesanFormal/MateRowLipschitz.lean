import OneesanFormal.BoundedHeight
import Mathlib.Tactic

namespace OneesanFormal.MateRowLipschitz

open OneesanFormal.ReverseMain
open OneesanFormal.BoundedHeight

/-- Signed height contribution of the first `k` symbols.  Values beyond the
end of the word stay at the final height, which makes rewrite lemmas total in
`k`. -/
def prefixDelta : List V → Nat → Int
  | _, 0 => 0
  | [], _ + 1 => 0
  | x :: xs, k + 1 => symDelta x + prefixDelta xs k

/-- One effective cell rewrite may raise only its own internal cut, by at most
one.  `cut=1` means the cut between the first two symbols. -/
def CutStepBound (src dst : List V) (cut : Nat) : Prop :=
  ∀ k, prefixDelta dst k ≤ prefixDelta src k + (if k = cut then 1 else 0)

/-- Pointwise nonincrease is useful for the long LL/RR closure rewrites. -/
def PrefixNonIncrease (src dst : List V) : Prop :=
  ∀ k, prefixDelta dst k ≤ prefixDelta src k

/-- Same-prefix context shifts the distinguished cut by one. -/
theorem CutStepBound.prepend {src dst : List V} {cut : Nat}
    (h : CutStepBound src dst cut) (a : V) :
    CutStepBound (a :: src) (a :: dst) (cut + 1) := by
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      have hk := h k
      simp [prefixDelta] at hk ⊢
      by_cases hkc : k = cut <;> simp [hkc] at hk ⊢ <;> omega

/-- A common prefix of arbitrary length only translates the cut index. -/
theorem CutStepBound.prependList {src dst pre : List V} {cut : Nat}
    (h : CutStepBound src dst cut) :
    CutStepBound (pre ++ src) (pre ++ dst) (pre.length + cut) := by
  induction pre with
  | nil => simpa using h
  | cons a pre ih =>
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using ih.prepend a

/-- Generic two-symbol rewrite: if the first-symbol excursion grows by at most
one and the two-symbol net delta is unchanged, only the internal cut can grow. -/
theorem pairRewriteBound (a b c d : V) (tail : List V)
    (hfirst : symDelta c ≤ symDelta a + 1)
    (htotal : symDelta c + symDelta d = symDelta a + symDelta b) :
    CutStepBound (a :: b :: tail) (c :: d :: tail) 1 := by
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      cases k with
      | zero => simpa [prefixDelta] using hfirst
      | succ k =>
          simp [prefixDelta]
          omega

/-- The six short effective production rewrites. -/
theorem nn_lr (tail : List V) :
    CutStepBound (.N :: .N :: tail) (.L :: .R :: tail) 1 := by
  apply pairRewriteBound <;> decide

theorem nr_rn (tail : List V) :
    CutStepBound (.N :: .R :: tail) (.R :: .N :: tail) 1 := by
  apply pairRewriteBound <;> decide

theorem nl_ln (tail : List V) :
    CutStepBound (.N :: .L :: tail) (.L :: .N :: tail) 1 := by
  apply pairRewriteBound <;> decide

theorem rn_nr (tail : List V) :
    CutStepBound (.R :: .N :: tail) (.N :: .R :: tail) 1 := by
  apply pairRewriteBound <;> decide

theorem ln_nl (tail : List V) :
    CutStepBound (.L :: .N :: tail) (.N :: .L :: tail) 1 := by
  apply pairRewriteBound <;> decide

theorem rl_nn (tail : List V) :
    CutStepBound (.R :: .L :: tail) (.N :: .N :: tail) 1 := by
  apply pairRewriteBound <;> decide

/-- Prefix inequality with a constant additive allowance. -/
def PrefixLeOffset (dst src : List V) (off : Int) : Prop :=
  ∀ k, prefixDelta dst k ≤ prefixDelta src k + off

/-- A shared leading symbol preserves a constant prefix offset. -/
theorem PrefixLeOffset.prepend {dst src : List V} {off : Int}
    (h : PrefixLeOffset dst src off) (hoff : 0 ≤ off) (a : V) :
    PrefixLeOffset (a :: dst) (a :: src) off := by
  intro k
  cases k with
  | zero => simpa [prefixDelta] using hoff
  | succ k =>
      have hk := h k
      simp [prefixDelta] at hk ⊢
      omega

/-- Therefore an arbitrary common middle word preserves the same offset. -/
theorem PrefixLeOffset.prependList {dst src mid : List V} {off : Int}
    (h : PrefixLeOffset dst src off) (hoff : 0 ≤ off) :
    PrefixLeOffset (mid ++ dst) (mid ++ src) off := by
  induction mid with
  | nil => simpa using h
  | cons a mid ih => simpa using ih.prepend hoff a

/-- Replacing one future R by L can raise a prefix by at most two. -/
theorem l_vs_r_offset2 (tail : List V) :
    PrefixLeOffset (.L :: tail) (.R :: tail) 2 := by
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      simp [prefixDelta, symDelta]
      omega

/-- Replacing future RR by NN can likewise recover at most two units. -/
theorem nn_vs_rr_offset2 (tail : List V) :
    PrefixLeOffset (.N :: .N :: tail) (.R :: .R :: tail) 2 := by
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      cases k with
      | zero => simp [prefixDelta, symDelta]
      | succ k =>
          simp [prefixDelta, symDelta]
          omega

/-- Effective LL closure:
`LL mid R` becomes `NN mid L`.  Every prefix height weakly decreases; the
matching condition used to locate that R is not needed for this inequality. -/
theorem ll_closure_nonincrease (mid tail : List V) :
    PrefixNonIncrease
      (.L :: .L :: mid ++ (.R :: tail))
      (.N :: .N :: mid ++ (.L :: tail)) := by
  have hoff : PrefixLeOffset (mid ++ (.L :: tail)) (mid ++ (.R :: tail)) 2 :=
    (l_vs_r_offset2 tail).prependList (by omega)
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      cases k with
      | zero => simp [prefixDelta, symDelta]
      | succ k =>
          have h := hoff k
          simp [prefixDelta, symDelta] at h ⊢
          omega

/-- Effective RR closure:
`L mid RR` becomes `R mid NN`.  Again every prefix height weakly decreases. -/
theorem rr_closure_nonincrease (mid tail : List V) :
    PrefixNonIncrease
      (.L :: mid ++ (.R :: .R :: tail))
      (.R :: mid ++ (.N :: .N :: tail)) := by
  have hoff : PrefixLeOffset (mid ++ (.N :: .N :: tail))
      (mid ++ (.R :: .R :: tail)) 2 :=
    (nn_vs_rr_offset2 tail).prependList (by omega)
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      have h := hoff k
      simp [prefixDelta, symDelta] at h ⊢
      omega

/-- A nonincreasing rewrite satisfies the one-cut bound for any chosen cut. -/
theorem PrefixNonIncrease.toCutStepBound {src dst : List V}
    (h : PrefixNonIncrease src dst) (cut : Nat) : CutStepBound src dst cut := by
  intro k
  have hk := h k
  by_cases hkc : k = cut
  · subst k
    have hc := h cut
    simp
    omega
  · simpa [hkc] using hk


/-- Common prefix context preserves pointwise nonincrease. -/
theorem PrefixNonIncrease.prepend {src dst : List V}
    (h : PrefixNonIncrease src dst) (a : V) :
    PrefixNonIncrease (a :: src) (a :: dst) := by
  intro k
  cases k with
  | zero => simp [prefixDelta]
  | succ k =>
      have hk := h k
      simp [prefixDelta] at hk ⊢
      omega

theorem PrefixNonIncrease.prependList {src dst pre : List V}
    (h : PrefixNonIncrease src dst) :
    PrefixNonIncrease (pre ++ src) (pre ++ dst) := by
  induction pre with
  | nil => simpa using h
  | cons a pre ih => simpa using ih.prepend a

/-- Effective forward cell rewrites after composing a blocked branch with the
mandatory N insertion at the following cell.  The `cut` index is the unique
Mate-prefix cut geometrically associated with the horizontal cell update. -/
inductive CellRewrite : List V → List V → Nat → Prop where
  | nn (pre tail : List V) :
      CellRewrite (pre ++ (.N :: .N :: tail))
        (pre ++ (.L :: .R :: tail)) (pre.length + 1)
  | nr (pre tail : List V) :
      CellRewrite (pre ++ (.N :: .R :: tail))
        (pre ++ (.R :: .N :: tail)) (pre.length + 1)
  | nl (pre tail : List V) :
      CellRewrite (pre ++ (.N :: .L :: tail))
        (pre ++ (.L :: .N :: tail)) (pre.length + 1)
  | rn (pre tail : List V) :
      CellRewrite (pre ++ (.R :: .N :: tail))
        (pre ++ (.N :: .R :: tail)) (pre.length + 1)
  | ln (pre tail : List V) :
      CellRewrite (pre ++ (.L :: .N :: tail))
        (pre ++ (.N :: .L :: tail)) (pre.length + 1)
  | rl (pre tail : List V) :
      CellRewrite (pre ++ (.R :: .L :: tail))
        (pre ++ (.N :: .N :: tail)) (pre.length + 1)
  | ll (pre mid tail : List V) :
      CellRewrite
        (pre ++ (.L :: .L :: mid ++ (.R :: tail)))
        (pre ++ (.N :: .N :: mid ++ (.L :: tail)))
        (pre.length + 1)
  | rr (pre mid tail : List V) :
      CellRewrite
        (pre ++ (.L :: mid ++ (.R :: .R :: tail)))
        (pre ++ (.R :: mid ++ (.N :: .N :: tail)))
        (pre.length + mid.length + 2)

/-- Every effective production cell rewrite obeys the one-cut height bound. -/
theorem CellRewrite.bound {src dst : List V} {cut : Nat}
    (h : CellRewrite src dst cut) : CutStepBound src dst cut := by
  cases h with
  | nn pre tail => simpa [Nat.add_comm] using (nn_lr tail).prependList (pre := pre)
  | nr pre tail => simpa [Nat.add_comm] using (nr_rn tail).prependList (pre := pre)
  | nl pre tail => simpa [Nat.add_comm] using (nl_ln tail).prependList (pre := pre)
  | rn pre tail => simpa [Nat.add_comm] using (rn_nr tail).prependList (pre := pre)
  | ln pre tail => simpa [Nat.add_comm] using (ln_nl tail).prependList (pre := pre)
  | rl pre tail => simpa [Nat.add_comm] using (rl_nn tail).prependList (pre := pre)
  | ll pre mid tail =>
      exact ((ll_closure_nonincrease mid tail).prependList (pre := pre)).toCutStepBound _
  | rr pre mid tail =>
      exact ((rr_closure_nonincrease mid tail).prependList (pre := pre)).toCutStepBound _

/-- Abstract execution of one production row.  At cut `i` the implementation
may exclude the horizontal edge (`skip`) or perform exactly one effective
rewrite for that cut (`step`).  Either way execution continues with a strictly
larger cut index, so no cut can acquire the +1 allowance twice. -/
inductive RowTrace : Nat → List V → List V → Prop where
  | done (i : Nat) (s : List V) : RowTrace i s s
  | skip {i : Nat} {s u : List V} : RowTrace (i + 1) s u → RowTrace i s u
  | step {i : Nat} {s t u : List V} :
      CellRewrite s t i → RowTrace (i + 1) t u → RowTrace i s u

/-- Main row-Lipschitz theorem: once processing starts at cut `i`, every prefix
strictly before `i` can only decrease, while every prefix at or after `i` can
increase by at most one over the *entire remaining row*. -/
theorem RowTrace.prefix_le_one {i : Nat} {src dst : List V}
    (h : RowTrace i src dst) :
    ∀ k, prefixDelta dst k ≤ prefixDelta src k + (if i ≤ k then 1 else 0) := by
  induction h with
  | done i s =>
      intro k
      by_cases hik : i ≤ k <;> simp [hik]
  | @skip i s u h ih =>
      intro k
      have hk := ih k
      by_cases hik : i ≤ k
      · simp [hik]
        omega
      · have hi1 : ¬ i + 1 ≤ k := by omega
        simp [hik, hi1] at hk ⊢
        exact hk
  | @step i s t u hs hrest ih =>
      intro k
      have hcell := hs.bound k
      have htail := ih k
      by_cases hki : k = i
      · subst k
        have hi1 : ¬ i + 1 ≤ i := by omega
        simp [hi1] at htail
        simp
        omega
      · by_cases hik : i + 1 ≤ k
        · have hi : i ≤ k := by omega
          simp [hki, hik, hi] at hcell htail ⊢
          omega
        · have hlt : k < i := by omega
          have hi : ¬ i ≤ k := by omega
          simp [hki, hik, hi] at hcell htail ⊢
          omega

/-- Production row transfer starts at cut one, hence every proper/nonempty Mate
prefix grows by at most one in one physical row. -/
theorem RowTrace.from_first_cut {src dst : List V}
    (h : RowTrace 1 src dst) :
    ∀ k, prefixDelta dst k ≤ prefixDelta src k + 1 := by
  intro k
  have hk := h.prefix_le_one k
  by_cases h1 : 1 ≤ k <;> simp [h1] at hk ⊢ <;> omega


/-- Source-inclusive frontier height bound.  `prefixDelta` omits the permanent
source marker, hence the leading `1`. -/
def FrontierBound (rows : Nat) (xs : List V) : Prop :=
  ∀ k, (1 : Int) + prefixDelta xs k ≤ rows

/-- One production row raises a source-inclusive frontier bound by at most one. -/
theorem RowTrace.frontierBound_succ {rows : Nat} {src dst : List V}
    (hrow : RowTrace 1 src dst) (hsrc : FrontierBound rows src) :
    FrontierBound (rows + 1) dst := by
  intro k
  have ht := hrow.from_first_cut k
  have hs := hsrc k
  omega

/-- Sequence of ordinary post-row1 production row transfers. -/
inductive RowsTrace : Nat → List V → List V → Prop where
  | zero (s : List V) : RowsTrace 0 s s
  | succ {n : Nat} {s t u : List V} :
      RowTrace 1 s t → RowsTrace n t u → RowsTrace (n + 1) s u

/-- Iterating the row-Lipschitz theorem: `n` further physical rows add at most
`n` to every frontier prefix height. -/
theorem RowsTrace.frontierBound_add {n rows : Nat} {src dst : List V}
    (h : RowsTrace n src dst) (hsrc : FrontierBound rows src) :
    FrontierBound (rows + n) dst := by
  induction h generalizing rows with
  | zero s => simpa using hsrc
  | @succ n s t u hrow htail ih =>
      have ht : FrontierBound (rows + 1) t := hrow.frontierBound_succ hsrc
      have hu := ih ht
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hu

/-- Exact row-8 corollary used by the cap argument: if the specialized row-1
initializer produces height at most one, seven ordinary production row transfers
can never exceed height eight. -/
theorem row8_bound_of_row1 {row1 row8 : List V}
    (hrow1 : FrontierBound 1 row1)
    (hrows : RowsTrace 7 row1 row8) : FrontierBound 8 row8 := by
  have h := hrows.frontierBound_add hrow1
  simpa using h

end OneesanFormal.MateRowLipschitz
