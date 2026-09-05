namespace OneesanFormal.RankSplit

/-- The three frontier symbols.  The rank-splitting theorem below is more
    general than this type, but keeping the production alphabet here makes the
    intended application explicit. -/
inductive Symbol where
  | N | R | L
  deriving DecidableEq, Repr

/-- State reached after scanning a word from left to right.  In Grid-FP the
    state is the current Motzkin height. -/
def stateAfter {σ α : Type} (next : σ → α → σ) : List α → σ → σ
  | [], s => s
  | a :: as, s => stateAfter next as (next s a)

/-- Additive rank contribution of a scanned word.  In the production codec
    `weight s a` is the sum of the relevant `D_FULL_DP[pos][height]` terms;
    the absolute position can simply be included in `α`. -/
def scanRank {σ α : Type} (next : σ → α → σ) (weight : σ → α → Nat) :
    List α → σ → Nat
  | [], _ => 0
  | a :: as, s => weight s a + scanRank next weight as (next s a)

/-- Scanning a concatenation passes exactly the state reached by the prefix to
    the suffix. -/
theorem stateAfter_append {σ α : Type} (next : σ → α → σ)
    (xs ys : List α) (s : σ) :
    stateAfter next (xs ++ ys) s =
      stateAfter next ys (stateAfter next xs s) := by
  induction xs generalizing s with
  | nil => rfl
  | cons a as ih =>
      simp [stateAfter, ih]

/-- Canonical rank is additive across any cut of the scan.  This is the formal
    core of the row-6 Cartesian initialization:

      R(prefix, suffix) = R_prefix(prefix) + R_suffix(suffix),

    provided the suffix starts from the state/height produced by the prefix. -/
theorem scanRank_append {σ α : Type} (next : σ → α → σ)
    (weight : σ → α → Nat) (xs ys : List α) (s : σ) :
    scanRank next weight (xs ++ ys) s =
      scanRank next weight xs s +
        scanRank next weight ys (stateAfter next xs s) := by
  induction xs generalizing s with
  | nil => simp [scanRank, stateAfter]
  | cons a as ih =>
      simp [scanRank, stateAfter, ih, Nat.add_assoc]

/-- Fixed-length prefixes of an appended word are unique.  Thus once the cut
    position is fixed, a full frontier word has at most one `(prefix,suffix)`
    representation. -/
theorem append_injective_of_prefix_length {α : Type}
    {p₁ p₂ s₁ s₂ : List α} (hlen : p₁.length = p₂.length)
    (h : p₁ ++ s₁ = p₂ ++ s₂) : p₁ = p₂ ∧ s₁ = s₂ := by
  have hp : p₁ = p₂ := by
    calc
      p₁ = (p₁ ++ s₁).take p₁.length := by simp
      _ = (p₂ ++ s₂).take p₁.length := by rw [h]
      _ = (p₂ ++ s₂).take p₂.length := by rw [hlen]
      _ = p₂ := by simp
  subst p₂
  clear hlen
  refine ⟨rfl, ?_⟩
  induction p₁ with
  | nil => simpa using h
  | cons a p ih =>
      simp only [List.cons_append, List.cons.injEq] at h
      exact ih h.2

/-- Conversely every word has the canonical decomposition at a fixed cut. -/
theorem take_drop_decomposition {α : Type} (k : Nat) (w : List α) :
    w.take k ++ w.drop k = w := by
  exact List.take_append_drop k w

end OneesanFormal.RankSplit
