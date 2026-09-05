import OneesanFormal.GridFaceBoundary

namespace OneesanFormal.RectBoundaryInjective

open OneesanFormal.GridFaceBoundary

theorem witness_real_edges_eq
    (n : Nat) (hn : 0 < n)
    (H₁ V₁ F₁ H₂ V₂ F₂ : Nat → Nat → F2)
    (h₁ : RectBoundaryWitness n H₁ V₁ F₁)
    (h₂ : RectBoundaryWitness n H₂ V₂ F₂)
    (hF : ∀ r c, r < n → c < n → F₁ r c = F₂ r c) :
    (∀ r c, r ≤ n → 0 < c → c ≤ n → H₁ r c = H₂ r c) ∧
    (∀ r c, 0 < r → r ≤ n → c ≤ n → V₁ r c = V₂ r c) := by
  constructor
  · intro r c hr hc0 hcn
    by_cases hr0 : r = 0
    · subst r
      have hc1 : c - 1 < n := by omega
      have hcidx : c - 1 + 1 = c := by omega
      rw [← hcidx, h₁.top (c - 1) hc1, h₂.top (c - 1) hc1]
      exact hF 0 (c - 1) hn hc1
    · by_cases hrn : r = n
      · subst r
        have hc1 : c - 1 < n := by omega
        have hcidx : c - 1 + 1 = c := by omega
        rw [← hcidx, h₁.bottom (c - 1) hc1, h₂.bottom (c - 1) hc1]
        exact hF (n - 1) (c - 1) (by omega) hc1
      · have hrpos : 0 < r := Nat.pos_of_ne_zero hr0
        have hrlt : r < n := by omega
        have hc1 : c - 1 < n := by omega
        have hridx : r - 1 + 1 = r := by omega
        have hcidx : c - 1 + 1 = c := by omega
        rw [← hridx, ← hcidx,
          h₁.horizontalInterior (r - 1) (c - 1) (by omega) hc1,
          h₂.horizontalInterior (r - 1) (c - 1) (by omega) hc1,
          hF (r - 1) (c - 1) (by omega) hc1,
          hF (r - 1 + 1) (c - 1) (by omega) hc1]
  · intro r c hr0 hrn hc
    by_cases hc0 : c = 0
    · subst c
      have hr1 : r - 1 < n := by omega
      have hridx : r - 1 + 1 = r := by omega
      rw [← hridx, h₁.left (r - 1) hr1, h₂.left (r - 1) hr1]
      exact hF (r - 1) 0 hr1 hn
    · by_cases hcn : c = n
      · subst c
        have hr1 : r - 1 < n := by omega
        have hridx : r - 1 + 1 = r := by omega
        rw [← hridx, h₁.right (r - 1) hr1, h₂.right (r - 1) hr1]
        exact hF (r - 1) (n - 1) hr1 (by omega)
      · have hcpos : 0 < c := Nat.pos_of_ne_zero hc0
        have hclt : c < n := by omega
        have hr1 : r - 1 < n := by omega
        have hridx : r - 1 + 1 = r := by omega
        have hcidx : c - 1 + 1 = c := by omega
        rw [← hridx, ← hcidx,
          h₁.verticalInterior (r - 1) (c - 1) hr1 (by omega),
          h₂.verticalInterior (r - 1) (c - 1) hr1 (by omega),
          hF (r - 1) (c - 1) hr1 (by omega),
          hF (r - 1) (c - 1 + 1) hr1 (by omega)]


end OneesanFormal.RectBoundaryInjective
