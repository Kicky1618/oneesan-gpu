import Mathlib.Combinatorics.SimpleGraph.Paths

namespace OneesanFormal.PathEdgeSet

open SimpleGraph

/-- A simple path with fixed start and end vertices is uniquely determined by
its undirected edge set.  The proof recovers the first neighbor from the edge
set and recurses on the tail; simplicity guarantees that the first edge does
not reappear in either tail. -/
theorem isPath_eq_of_edgeSet_eq
    {V : Type*} {G : SimpleGraph V} {u v : V}
    (p q : G.Walk u v) (hp : p.IsPath) (hq : q.IsPath)
    (he : p.edgeSet = q.edgeSet) : p = q := by
  induction p with
  | nil =>
      have hqn : q.Nil := Walk.isPath_iff_nil.mp hq
      cases q with
      | nil => rfl
      | cons _ _ => simp at hqn
  | @cons x y z h pt ih =>
      cases q with
      | nil =>
          have hfirst : s(x, y) ∈ (Walk.cons h pt).edgeSet := by
            simp [Walk.edgeSet]
          rw [he] at hfirst
          simp [Walk.edgeSet] at hfirst
      | @cons _ y' _ h' qt =>
          have hfirst : s(x, y) ∈ (Walk.cons h' qt).edges := by
            have hm : s(x, y) ∈ (Walk.cons h pt).edgeSet := by
              simp [Walk.edgeSet]
            rw [he] at hm
            simpa [Walk.edgeSet] using hm
          have hy : y = y' := by
            simpa using hq.eq_snd_of_mem_edges hfirst
          subst y'
          have hpt : pt.IsPath := hp.of_cons
          have hqt : qt.IsPath := hq.of_cons
          have hno_p : s(x, y) ∉ pt.edges :=
            (Walk.isTrail_cons h pt).mp hp.isTrail |>.2
          have hno_q : s(x, y) ∉ qt.edges :=
            (Walk.isTrail_cons h' qt).mp hq.isTrail |>.2
          have hetail : pt.edgeSet = qt.edgeSet := by
            ext e
            have hall :
                e ∈ (Walk.cons h pt).edgeSet ↔
                  e ∈ (Walk.cons h' qt).edgeSet := by
              rw [he]
            simp only [Walk.mem_edgeSet, Walk.edges_cons, List.mem_cons] at hall ⊢
            by_cases hef : e = s(x, y)
            · subst e
              simp [hno_p, hno_q]
            · simpa [hef] using hall
          have ht : pt = qt := ih qt hpt hqt hetail
          subst qt
          rfl

/-- Edge-set projection is injective on mathlib's subtype of simple paths with
fixed endpoints. -/
theorem path_edgeSet_injective
    {V : Type*} {G : SimpleGraph V} {u v : V} :
    Function.Injective (fun p : G.Path u v => (p : G.Walk u v).edgeSet) := by
  intro p q h
  apply Subtype.ext
  change p.val.edgeSet = q.val.edgeSet at h
  exact isPath_eq_of_edgeSet_eq p.val q.val p.property q.property h

end OneesanFormal.PathEdgeSet
