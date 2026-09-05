import OneesanFormal.StripBound27
import Mathlib.Data.Fin.Tuple.Basic

namespace OneesanFormal.StripDP
open OneesanFormal.GridFaceBoundary
open OneesanFormal.CheckerboardGridBound

abbrev Column (h : Nat) := Fin h → F2

def columnCompatible (x y : Column h) : Prop :=
  ∀ r : Fin (h - 1),
    ¬ f2Checkerboard
      (x ⟨r.val, by omega⟩)
      (y ⟨r.val, by omega⟩)
      (y ⟨r.val + 1, by omega⟩)
      (x ⟨r.val + 1, by omega⟩)

noncomputable instance columnCompatibleDecidable (h : Nat) :
    DecidableRel (columnCompatible (h := h)) := Classical.decRel _

/-- Snoc-recursive validity for a sequence of face columns. -/
def seqValid (h : Nat) : {w : Nat} → (Fin w → Column h) → Prop
  | 0, _ => True
  | 1, _ => True
  | k + 2, f =>
      seqValid h (Fin.init f) ∧
      columnCompatible (f ⟨k, by omega⟩) (f (Fin.last (k + 1)))

noncomputable instance seqValidDecidable (h w : Nat) :
    DecidablePred (seqValid h (w := w)) := Classical.decPred _

@[simp]theorem seqValid_zero (f : Fin 0 → Column h) : seqValid h f := by
  simp [seqValid]

@[simp]theorem seqValid_one (f : Fin 1 → Column h) : seqValid h f := by
  simp [seqValid]

@[simp]theorem seqValid_snoc_zero (f : Fin 0 → Column h) (y : Column h) :
    seqValid h (Fin.snoc f y) := by
  simp [seqValid]

theorem seqValid_snoc_succ
    (f : Fin (k + 1) → Column h) (y : Column h) :
    seqValid h (Fin.snoc f y) ↔
      seqValid h f ∧ columnCompatible (f (Fin.last k)) y := by
  let g : Fin (k + 2) → Column h := Fin.snoc f y
  change
    (seqValid h (Fin.init g) ∧
      columnCompatible (g ⟨k, by omega⟩) (g (Fin.last (k + 1)))) ↔ _
  have hginit : Fin.init g = f := by
    dsimp [g]
    exact Fin.init_snoc _ _
  have hglast : g (Fin.last (k + 1)) = y := by
    dsimp [g]
    exact Fin.snoc_last _ _
  have hidx : (⟨k, by omega⟩ : Fin (k + 2)) = (Fin.last k).castSucc := by
    apply Fin.ext
    rfl
  have hgprev : g ⟨k, by omega⟩ = f (Fin.last k) := by
    rw [hidx]
    dsimp [g]
    exact Fin.snoc_castSucc _ _ _
  rw [hginit, hglast, hgprev]



/-- Recursive type of valid column chains with `k` transitions and prescribed
last column `y`. -/
def EndChain (h : Nat) : (k : Nat) → Column h → Type
  | 0, _ => PUnit
  | k + 1, y => {z : (Σ x : Column h, EndChain h k x) // columnCompatible z.1 y}

@[instance_reducible] noncomputable def endChainFintype (h : Nat) :
    (k : Nat) → (y : Column h) → Fintype (EndChain h k y)
  | 0, _ => by
      change Fintype PUnit
      infer_instance
  | k + 1, y => by
      change Fintype {z : (Σ x : Column h, EndChain h k x) // columnCompatible z.1 y}
      letI (x : Column h) : Fintype (EndChain h k x) := endChainFintype h k x
      letI : DecidablePred (fun z : (Σ x : Column h, EndChain h k x) =>
          columnCompatible z.1 y) := Classical.decPred _
      exact Subtype.fintype _

noncomputable local instance endChainFintypeInst (h k : Nat) (y : Column h) :
    Fintype (EndChain h k y) := endChainFintype h k y

/-- Transfer-DP count of valid chains ending at `y`. -/
noncomputable def endCount (h : Nat) : (k : Nat) → Column h → Nat
  | 0, _ => 1
  | k + 1, y => ∑ x : Column h,
      if columnCompatible x y then endCount h k x else 0

/-- Reassociate a subtype of a sigma into a sigma of subtypes. -/
def sigmaSubtypeEquiv (h k : Nat) (y : Column h) :
    {z : (Σ x : Column h, EndChain h k x) // columnCompatible z.1 y} ≃
      (Σ x : Column h, {_c : EndChain h k x // columnCompatible x y}) where
  toFun z := ⟨z.1.1, ⟨z.1.2, z.2⟩⟩
  invFun z := ⟨⟨z.1, z.2.1⟩, z.2.2⟩
  left_inv _ := rfl
  right_inv _ := rfl

theorem endChain_card_eq_endCount (h : Nat) (k : Nat) (y : Column h) :
    Fintype.card (EndChain h k y) = endCount h k y := by
  induction k generalizing y with
  | zero =>
      change Fintype.card PUnit = 1
      simp
  | succ k ih =>
      have hc :
          Fintype.card (EndChain h (k + 1) y) =
            Fintype.card (Σ x : Column h,
              {_c : EndChain h k x // columnCompatible x y}) := by
        exact Fintype.card_congr (sigmaSubtypeEquiv h k y)
      rw [hc, Fintype.card_sigma]
      simp_rw [Fintype.card_subtype]
      simp only [endCount]
      apply Finset.sum_congr rfl
      intro x _
      by_cases hxy : columnCompatible x y
      · simp [hxy, ih x]
      · simp [hxy]


/-- Decode a recursive chain back to its full sequence of columns. -/
def endChainToSeq (h : Nat) : {k : Nat} → {y : Column h} →
    EndChain h k y → (Fin (k + 1) → Column h)
  | 0, y, _ => fun _ => y
  | _k + 1, y, z => Fin.snoc (endChainToSeq h z.1.2) y

/-- Encode a valid column sequence into the recursive chain ending at its last
column. -/
def seqToEndChain (h : Nat) : {k : Nat} →
    (f : Fin (k + 1) → Column h) → seqValid h f →
      EndChain h k (f (Fin.last k))
  | 0, _, _ => PUnit.unit
  | k + 1, f, hf => by
      change seqValid h (Fin.init f) ∧
        columnCompatible (f ⟨k, by omega⟩) (f (Fin.last (k + 1))) at hf
      have hc : columnCompatible
          ((Fin.init f) (Fin.last k)) (f (Fin.last (k + 1))) := by
        have hidx : (Fin.last k).castSucc = (⟨k, by omega⟩ : Fin (k + 2)) := by
          apply Fin.ext
          rfl
        change columnCompatible (f (Fin.last k).castSucc) (f (Fin.last (k + 1)))
        rw [hidx]
        exact hf.2
      exact ⟨⟨(Fin.init f) (Fin.last k), seqToEndChain h (Fin.init f) hf.1⟩, hc⟩

theorem endChainToSeq_seqToEndChain
    (h k : Nat) (f : Fin (k + 1) → Column h) (hf : seqValid h f) :
    endChainToSeq h (seqToEndChain h f hf) = f := by
  induction k with
  | zero =>
      funext i
      have hi : i = 0 := Fin.eq_zero i
      subst i
      rfl
  | succ k ih =>
      change seqValid h (Fin.init f) ∧
        columnCompatible (f ⟨k, by omega⟩) (f (Fin.last (k + 1))) at hf
      change Fin.snoc
        (endChainToSeq h (seqToEndChain h (Fin.init f) hf.1))
        (f (Fin.last (k + 1))) = f
      rw [ih (Fin.init f) hf.1]
      exact Fin.snoc_init_self _


abbrev ValidSeq (h w : Nat) := {f : Fin w → Column h // seqValid h f}

noncomputable def validSeqToEndSigma (h k : Nat) :
    ValidSeq h (k + 1) → (Σ y : Column h, EndChain h k y) :=
  fun f => ⟨f.val (Fin.last k), seqToEndChain h f.val f.property⟩

theorem validSeqToEndSigma_injective (h k : Nat) :
    Function.Injective (validSeqToEndSigma h k) := by
  intro f g hfg
  apply Subtype.ext
  have hseq := congrArg
    (fun z : (Σ y : Column h, EndChain h k y) => endChainToSeq h z.2) hfg
  simpa [validSeqToEndSigma,
    endChainToSeq_seqToEndChain h k f.val f.property,
    endChainToSeq_seqToEndChain h k g.val g.property] using hseq

theorem card_validSeq_le_dpSum (h k : Nat) :
    Fintype.card (ValidSeq h (k + 1)) ≤
      ∑ y : Column h, endCount h k y := by
  classical
  have hinj := validSeqToEndSigma_injective h k
  have hcard : Fintype.card (ValidSeq h (k + 1)) ≤
      Fintype.card (Σ y : Column h, EndChain h k y) :=
    Fintype.card_le_of_injective (validSeqToEndSigma h k) hinj
  rw [Fintype.card_sigma] at hcard
  have hsigma : (∑ y : Column h, Fintype.card (EndChain h k y)) =
      ∑ y : Column h, endCount h k y := by
    apply Finset.sum_congr rfl
    intro y _
    exact endChain_card_eq_endCount h k y
  rwa [hsigma] at hcard


/-- Pairwise compatibility of every consecutive column implies the snoc-recursive
`seqValid` predicate. -/
theorem seqValid_of_adjacent
    (h w : Nat) (f : Fin w → Column h)
    (hadj : ∀ i : Fin w, ∀ hi : i.val + 1 < w,
      columnCompatible (f i)
        (f ⟨i.val + 1, hi⟩)) :
    seqValid h f := by
  induction w with
  | zero => simp [seqValid]
  | succ w ih =>
      cases w with
      | zero => simp [seqValid]
      | succ k =>
          change seqValid h (Fin.init f) ∧
            columnCompatible (f ⟨k, by omega⟩) (f (Fin.last (k + 1)))
          constructor
          · apply ih (Fin.init f)
            intro i hi
            have hhi : i.castSucc.val + 1 < k + 2 := by
              simpa using Nat.lt_succ_of_lt hi
            have hh := hadj i.castSucc hhi
            have hnext :
                (⟨i.val + 1, by omega⟩ : Fin (k + 1)).castSucc =
                  (⟨i.val + 1, by omega⟩ : Fin (k + 2)) := by
              apply Fin.ext
              rfl
            simpa [Fin.init, hnext] using hh
          · let ik : Fin (k + 2) := ⟨k, by omega⟩
            have hik : ik.val + 1 < k + 2 := by
              dsimp [ik]
              omega
            have hh := hadj ik hik
            have hlast : (⟨k + 1, by omega⟩ : Fin (k + 2)) = Fin.last (k + 1) := by
              apply Fin.ext
              rfl
            dsimp [ik] at hh
            rwa [hlast] at hh


open OneesanFormal.StripBound27

/-- Curry a concrete 9x27 strip into a sequence of 27 nine-bit columns. -/
def stripColumns927 (s : Strip9x27) : Fin 27 → Column 9 :=
  fun c r => s (r, c)

theorem stripColumns927_injective : Function.Injective stripColumns927 := by
  intro s t h
  funext rc
  have hc := congrFun (congrFun h rc.2) rc.1
  exact hc

theorem stripColumns927_valid
    (s : Strip9x27) (hs : stripNoCheckerboard s) :
    seqValid 9 (stripColumns927 s) := by
  apply seqValid_of_adjacent 9 27 (stripColumns927 s)
  intro i hi r
  let ci : Fin 26 := ⟨i.val, by omega⟩
  have hcb := hs r ci
  dsimp [ci] at hcb
  have hcol : (⟨i.val, by omega⟩ : Fin 27) = i := by
    apply Fin.ext
    rfl
  have hnext : (⟨i.val + 1, by omega⟩ : Fin 27) =
      (⟨i.val + 1, hi⟩ : Fin 27) := by
    apply Fin.ext
    rfl
  simpa [stripColumns927, hcol, hnext] using hcb

noncomputable def validStripToValidSeq :
    {s : Strip9x27 // stripNoCheckerboard s} → ValidSeq 9 27 :=
  fun s => ⟨stripColumns927 s.val, stripColumns927_valid s.val s.property⟩

theorem validStripToValidSeq_injective :
    Function.Injective validStripToValidSeq := by
  intro s t h
  apply Subtype.ext
  apply stripColumns927_injective
  exact congrArg Subtype.val h

theorem card_validStrip927_le_dp :
    Fintype.card {s : Strip9x27 // stripNoCheckerboard s} ≤
      ∑ y : Column 9, endCount 9 26 y := by
  classical
  have h₁ : Fintype.card {s : Strip9x27 // stripNoCheckerboard s} ≤
      Fintype.card (ValidSeq 9 27) :=
    Fintype.card_le_of_injective validStripToValidSeq validStripToValidSeq_injective
  exact le_trans h₁ (card_validSeq_le_dpSum 9 26)

end OneesanFormal.StripDP
