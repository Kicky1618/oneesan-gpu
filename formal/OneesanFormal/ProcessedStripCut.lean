import Mathlib.Combinatorics.SimpleGraph.Walk.Basic
import Mathlib.Tactic

namespace OneesanFormal.ProcessedStripCut

open SimpleGraph
open Sym2

/-- Vertices visible after exactly `r` physical rows have been processed.
There are `r+1` horizontal vertex lines but only `w` frontier columns. -/
abbrev Vertex (r w : Nat) := Fin (r + 1) × Fin w

/-- One positive-coordinate edge of the processed strip.  Horizontal edges
exist only on the `r` already processed row lines; the bottom frontier line
`row=r` has not had its horizontal edges processed yet. -/
def forwardStep {r w : Nat} (a b : Vertex r w) : Prop :=
  (a.1 = b.1 ∧ a.1.val < r ∧ a.2.val + 1 = b.2.val) ∨
  (a.2 = b.2 ∧ a.1.val + 1 = b.1.val)

/-- Undirected processed-strip graph. -/
def graph (r w : Nat) : SimpleGraph (Vertex r w) :=
  SimpleGraph.fromRel forwardStep

/-- Four-direction adjacency, with horizontal motion forbidden on the bottom
unprocessed frontier line. -/
theorem graph_adj_iff {r w : Nat} {a b : Vertex r w} :
    (graph r w).Adj a b ↔
      (a.1 = b.1 ∧ a.1.val < r ∧
        (a.2.val + 1 = b.2.val ∨ b.2.val + 1 = a.2.val)) ∨
      (a.2 = b.2 ∧
        (a.1.val + 1 = b.1.val ∨ b.1.val + 1 = a.1.val)) := by
  rw [graph, SimpleGraph.fromRel_adj]
  constructor
  · rintro ⟨_, h | h⟩
    · rcases h with h | h
      · exact Or.inl ⟨h.1, h.2.1, Or.inl h.2.2⟩
      · exact Or.inr ⟨h.1, Or.inl h.2⟩
    · rcases h with h | h
      · exact Or.inl ⟨h.1.symm, by simpa [h.1] using h.2.1, Or.inr h.2.2⟩
      · exact Or.inr ⟨h.1.symm, Or.inr h.2⟩
  · intro h
    refine ⟨?_, ?_⟩
    · intro hab
      subst b
      rcases h with ⟨_, _, hs⟩ | ⟨_, hs⟩ <;> omega
    · rcases h with h | h
      · rcases h.2.2 with hr | hl
        · exact Or.inl (Or.inl ⟨h.1, h.2.1, hr⟩)
        · exact Or.inr (Or.inl ⟨h.1.symm, by simpa [h.1] using h.2.1, hl⟩)
      · rcases h.2 with hd | hu
        · exact Or.inl (Or.inr ⟨h.1, hd⟩)
        · exact Or.inr (Or.inr ⟨h.1.symm, hu⟩)

/-- Physical horizontal edge in processed row `i` crossing the cut between
columns `c` and `c+1`. -/
def crossingEdge {r w : Nat} (c : Fin (w - 1)) (i : Fin r) :
    Sym2 (Vertex r w) :=
  s((⟨i, by omega⟩, ⟨c, by omega⟩),
    (⟨i, by omega⟩, ⟨c.val + 1, by omega⟩))

/-- Distinct processed rows give distinct physical crossing edges. -/
theorem crossingEdge_injective {r w : Nat} (c : Fin (w - 1)) :
    Function.Injective (crossingEdge (r := r) c) := by
  intro i j hij
  unfold crossingEdge at hij
  rw [Sym2.eq_iff] at hij
  rcases hij with hsame | hswap
  · apply Fin.ext
    exact congrArg (fun v : Vertex r w => v.1.val) hsame.1
  · have hc := congrArg (fun v : Vertex r w => v.2.val) hswap.1
    simp at hc

/-- A single processed-strip edge whose endpoints lie on opposite sides of a
vertical cut is exactly one of the `r` crossing edges. -/
theorem adj_crosses_cut_is_crossingEdge {r w : Nat}
    {a b : Vertex r w} (h : (graph r w).Adj a b)
    (c : Fin (w - 1)) (ha : a.2.val ≤ c.val) (hb : c.val < b.2.val) :
    ∃ i : Fin r, s(a,b) = crossingEdge c i := by
  rw [graph_adj_iff] at h
  rcases h with hh | hv
  · rcases hh with ⟨hrow, hproc, hright | hleft⟩
    · have hac : a.2.val = c.val := by omega
      have hbc : b.2.val = c.val + 1 := by omega
      let i : Fin r := ⟨a.1.val, hproc⟩
      refine ⟨i, ?_⟩
      unfold crossingEdge
      apply Sym2.eq_iff.mpr
      left
      constructor
      · apply Prod.ext
        · exact Fin.ext rfl
        · exact Fin.ext hac
      · apply Prod.ext
        · exact Fin.ext (congrArg Fin.val hrow.symm)
        · exact Fin.ext hbc
    · omega
  · rcases hv with ⟨hcol, _⟩
    have := congrArg Fin.val hcol
    omega

/-- Discrete intermediate-value theorem for a processed grid strip: every walk
from the left side of a vertical cut to the right side uses one of its `r`
physical crossing edges. -/
theorem walk_exists_crossingEdge {r w : Nat} {u v : Vertex r w}
    (p : (graph r w).Walk u v) (c : Fin (w - 1))
    (hu : u.2.val ≤ c.val) (hv : c.val < v.2.val) :
    ∃ i : Fin r, crossingEdge c i ∈ p.edges := by
  induction p with
  | nil => omega
  | @cons x y z hadj q ih =>
      by_cases hy : c.val < y.2.val
      · obtain ⟨i, hi⟩ := adj_crosses_cut_is_crossingEdge hadj c hu hy
        refine ⟨i, ?_⟩
        simp [Walk.edges_cons, ← hi]
      · have hyleft : y.2.val ≤ c.val := by omega
        obtain ⟨i, hi⟩ := ih hyleft hv
        exact ⟨i, by simp [Walk.edges_cons, hi]⟩

end OneesanFormal.ProcessedStripCut

namespace OneesanFormal.ProcessedStripCut

/-- A finite family of pairwise edge-disjoint path fragments, each of which
connects the left side of a fixed vertical cut to its right side. -/
structure CrossingFamily {r w : Nat} (c : Fin (w - 1)) where
  Component : Type*
  [componentFintype : Fintype Component]
  leftEnd  : Component → Vertex r w
  rightEnd : Component → Vertex r w
  path : (q : Component) → (graph r w).Walk (leftEnd q) (rightEnd q)
  left_side : ∀ q, (leftEnd q).2.val ≤ c.val
  right_side : ∀ q, c.val < (rightEnd q).2.val
  edge_disjoint : ∀ {a b : Component}, a ≠ b →
    Disjoint (path a).edgeSet (path b).edgeSet

attribute [instance] CrossingFamily.componentFintype

/-- Choose one physical crossing row from each component. -/
noncomputable def CrossingFamily.crossingRow {r w : Nat} {c : Fin (w - 1)}
    (F : CrossingFamily (r := r) c) (q : F.Component) : Fin r := by
  exact Classical.choose
    (walk_exists_crossingEdge (F.path q) c (F.left_side q) (F.right_side q))

/-- The chosen row really is traversed by that component. -/
theorem CrossingFamily.crossingRow_mem {r w : Nat} {c : Fin (w - 1)}
    (F : CrossingFamily (r := r) c) (q : F.Component) :
    crossingEdge (r := r) c (F.crossingRow q) ∈ (F.path q).edgeSet := by
  exact (Classical.choose_spec
    (walk_exists_crossingEdge (F.path q) c (F.left_side q) (F.right_side q)))

/-- Edge-disjoint components cannot choose the same crossing row. -/
theorem CrossingFamily.crossingRow_injective {r w : Nat} {c : Fin (w - 1)}
    (F : CrossingFamily (r := r) c) : Function.Injective F.crossingRow := by
  intro a b hab
  by_contra hne
  have hd := F.edge_disjoint hne
  have ha := F.crossingRow_mem a
  have hb := F.crossingRow_mem b
  have hs : crossingEdge (r := r) c (F.crossingRow a) = crossingEdge (r := r) c (F.crossingRow b) := by
    rw [hab]
  rw [hs] at ha
  exact Set.disjoint_left.1 hd ha hb

/-- At most `r` pairwise edge-disjoint path components can cross one vertical
cut of an `r`-row processed strip. -/
theorem CrossingFamily.card_le_rows {r w : Nat} {c : Fin (w - 1)}
    (F : CrossingFamily (r := r) c) : Fintype.card F.Component ≤ r := by
  have hcard := Fintype.card_le_of_injective F.crossingRow F.crossingRow_injective
  simpa using hcard

end OneesanFormal.ProcessedStripCut
