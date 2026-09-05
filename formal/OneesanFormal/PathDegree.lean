import Mathlib.Combinatorics.SimpleGraph.Connectivity.Subgraph

namespace OneesanFormal.PathDegree

open SimpleGraph

/-- Every vertex in the subgraph traced by a simple path has degree at most two.
Vertices outside the path support have degree zero, endpoints have degree one
for a nontrivial path, and internal vertices have degree two. -/
theorem isPath_neighborSet_ncard_le_two
    {V : Type*} {G : SimpleGraph V} {u v w : V}
    {p : G.Walk u v} (hp : p.IsPath) :
    (p.toSubgraph.neighborSet w).ncard ≤ 2 := by
  by_cases hw : w ∈ p.support
  · obtain ⟨i, hi, hile⟩ := Walk.mem_support_iff_exists_getVert.mp hw
    by_cases hnil : p.Nil
    · have hedges : p.edges = [] := Walk.edges_eq_nil.mpr hnil
      have hempty : p.toSubgraph.neighborSet w = ∅ := by
        ext x
        simp [Subgraph.mem_neighborSet, Walk.adj_toSubgraph_iff_mem_edges, hedges]
      rw [hempty]
      simp
    · by_cases hi0 : i = 0
      · subst i
        rw [p.getVert_zero] at hi
        subst w
        rw [hp.neighborSet_toSubgraph_startpoint hnil]
        simp
      · by_cases hil : i = p.length
        · rw [hil, p.getVert_length] at hi
          subst w
          rw [hp.neighborSet_toSubgraph_endpoint hnil]
          simp
        · have hilt : i < p.length := lt_of_le_of_ne hile hil
          rw [← hi, hp.ncard_neighborSet_toSubgraph_internal_eq_two hi0 hilt]
  · have hempty : p.toSubgraph.neighborSet w = ∅ := by
      ext x
      constructor
      · intro hx
        have hadj : p.toSubgraph.Adj w x := hx
        have hwvert : w ∈ p.toSubgraph.verts := hadj.fst_mem
        exact (hw (p.mem_verts_toSubgraph.mp hwvert)).elim
      · simp
    rw [hempty]
    simp


/-- Expected degree profile of a non-closed simple path: one at the two
endpoints, two at other support vertices, and zero off the path. -/
noncomputable def expectedPathDegree
    {V : Type*} {G : SimpleGraph V} {u v : V}
    (p : G.Walk u v) (w : V) : Nat := by
  classical
  exact if w = u ∨ w = v then 1 else if w ∈ p.support then 2 else 0

/-- Exact local degree classification for a non-closed simple path. -/
theorem isPath_neighborSet_ncard_eq
    {V : Type*} {G : SimpleGraph V} {u v w : V}
    {p : G.Walk u v} (hp : p.IsPath) (huv : u ≠ v) :
    (p.toSubgraph.neighborSet w).ncard = expectedPathDegree p w := by
  classical
  have hnil : ¬ p.Nil := Walk.not_nil_of_ne huv
  by_cases hwu : w = u
  · subst w
    rw [hp.neighborSet_toSubgraph_startpoint hnil]
    simp [expectedPathDegree]
  · by_cases hwv : w = v
    · subst w
      rw [hp.neighborSet_toSubgraph_endpoint hnil]
      simp [expectedPathDegree]
    · by_cases hw : w ∈ p.support
      · obtain ⟨i, hi, hile⟩ := Walk.mem_support_iff_exists_getVert.mp hw
        have hi0 : i ≠ 0 := by
          intro hiz
          subst i
          rw [p.getVert_zero] at hi
          exact hwu hi.symm
        have hil : i ≠ p.length := by
          intro hil
          rw [hil, p.getVert_length] at hi
          exact hwv hi.symm
        have hilt : i < p.length := lt_of_le_of_ne hile hil
        rw [← hi, hp.ncard_neighborSet_toSubgraph_internal_eq_two hi0 hilt]
        simp [expectedPathDegree, hi, hwu, hwv, hw]
      · have hempty : p.toSubgraph.neighborSet w = ∅ := by
          ext x
          constructor
          · intro hx
            have hadj : p.toSubgraph.Adj w x := hx
            have hwvert : w ∈ p.toSubgraph.verts := hadj.fst_mem
            exact (hw (p.mem_verts_toSubgraph.mp hwvert)).elim
          · simp
        rw [hempty]
        simp [expectedPathDegree, hwu, hwv, hw]

end OneesanFormal.PathDegree
