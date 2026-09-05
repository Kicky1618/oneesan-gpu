import OneesanFormal.ReverseMainCore
import Mathlib.Tactic

namespace OneesanFormal.BoundedHeight
open OneesanFormal.ReverseMain

/-- Height increment of one frontier symbol when scanning from high index to low. -/
def symDelta : V → Int
  | .N => 0
  | .R => -1
  | .L => 1

/-- The two symbols represented by a local Pair, in frontier scan order. -/
def pairWord : Pair → V × V
  | .NN => (.N,.N) | .NR => (.N,.R) | .NL => (.N,.L)
  | .RN => (.R,.N) | .RR => (.R,.R) | .RL => (.R,.L)
  | .LN => (.L,.N) | .LR => (.L,.R) | .LL => (.L,.L)

/-- Maximum height excursion, relative to the height before this two-symbol window. -/
def pairPeak (p : Pair) : Int :=
  let (a,b) := pairWord p
  max 0 (max (symDelta a) (symDelta a + symDelta b))

/-- Included main→main rewrites can raise the local frontier peak by at most one. -/
theorem includeToMain_peak_le_one {s t : Pair}
    (h : includeToMain s = some t) : pairPeak t ≤ pairPeak s + 1 := by
  cases s <;> cases t <;> simp [includeToMain, pairPeak, pairWord, symDelta] at h ⊢

/-- The only main→main included rewrite that strictly raises the local peak is NN→LR. -/
theorem includeToMain_peak_strict_iff {s t : Pair}
    (h : includeToMain s = some t) :
    pairPeak s < pairPeak t ↔ s = .NN ∧ t = .LR := by
  cases s <;> cases t <;> simp [includeToMain, pairPeak, pairWord, symDelta] at h ⊢

/-- RN→NR and LN→NL preserve the local peak; NN→LR raises it exactly by one. -/
theorem includeToMain_peak_delta {s t : Pair}
    (h : includeToMain s = some t) :
    pairPeak t - pairPeak s = (if s = .NN then 1 else 0) := by
  cases s <;> cases t <;> simp [includeToMain, pairPeak, pairWord, symDelta] at h ⊢

/-- Abstract maximum height of a whole frontier when a two-symbol window is
surrounded by an arbitrary prefix and suffix.  `prefixPeak` is the maximum
height already seen, `prefixDelta` is the height at the start of the window,
and `suffixPeak` is the suffix maximum relative to the end of the window. -/
def contextualPeak (prefixPeak prefixDelta suffixPeak : Int) (p : Pair) : Int :=
  max prefixPeak (prefixDelta + max (pairPeak p) (pairDelta p + suffixPeak))

/-- Elementary one-sided Lipschitz property of `max`: increasing one argument
by at most one increases the maximum by at most one. -/
theorem max_le_max_add_one {a x y : Int} (h : x ≤ y + 1) :
    max a x ≤ max a y + 1 := by
  have ha : a ≤ max a y := le_max_left _ _
  have hy : y ≤ max a y := le_max_right _ _
  omega

/-- The local `peak ≤ +1` fact is stable under *arbitrary* unchanged prefix
and suffix context.  Equality of `pairDelta` is crucial: it means the suffix
starts at exactly the same height after the rewrite. -/
theorem includeToMain_contextual_peak_le_one {s t : Pair}
    (h : includeToMain s = some t)
    (prefixPeak prefixDelta suffixPeak : Int) :
    contextualPeak prefixPeak prefixDelta suffixPeak t ≤
      contextualPeak prefixPeak prefixDelta suffixPeak s + 1 := by
  have hp := includeToMain_peak_le_one h
  have hd := includeToMain_preserves_delta h
  have hsuffix : pairDelta t + suffixPeak = pairDelta s + suffixPeak := by omega
  have hinner :
      max (pairPeak t) (pairDelta t + suffixPeak) ≤
        max (pairPeak s) (pairDelta s + suffixPeak) + 1 := by
    rw [hsuffix]
    simpa [max_comm] using
      (max_le_max_add_one (a := pairDelta s + suffixPeak) hp)
  unfold contextualPeak
  apply max_le_max_add_one
  omega

/-- Inserting the forced N used by blocked→main changes no height at all. -/
def wordHeight (xs : List V) : Int := xs.foldl (fun h x => h + symDelta x) 0

@[simp] theorem wordHeight_cons_N (xs : List V) :
    wordHeight (.N :: xs) = wordHeight xs := by
  simp [wordHeight, symDelta, List.foldl]

/-- The forced-N insertion is height-neutral at the algebraic level. -/
theorem inserted_N_delta_zero : symDelta V.N = 0 := by rfl

end OneesanFormal.BoundedHeight
