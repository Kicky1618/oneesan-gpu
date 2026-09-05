import Mathlib.Combinatorics.SimpleGraph.Acyclic
import Mathlib.Combinatorics.SimpleGraph.Finite
import Mathlib.Tactic

namespace OneesanFormal

open SimpleGraph

universe u
variable {V : Type u}

/-- Graph-theoretic XOR: keep exactly the edges that occur in one of `A`, `B`. -/
def xorGraph (A B : SimpleGraph V) : SimpleGraph V :=
  (A \ B) ⊔ (B \ A)

/-- A finite acyclic graph whose every vertex has even degree has no edges.

Choose an edge, pass to its connected component, use that an acyclic connected
component is a tree, and obtain a leaf.  Every neighbor of a vertex in the
component stays in that component, so the leaf also has degree one in the
original graph, contradicting even degree. -/
theorem SimpleGraph.IsAcyclic.eq_bot_of_forall_even_degree
    {G : SimpleGraph V} [Fintype V] [DecidableEq V] [DecidableRel G.Adj] (hacyc : G.IsAcyclic)
    (heven : ∀ v : V, Even (G.degree v)) :
    G = ⊥ := by
  classical
  by_contra hne
  obtain ⟨a, b, hab⟩ := SimpleGraph.ne_bot_iff_exists_adj.mp hne
  let C : G.ConnectedComponent := G.connectedComponentMk a
  have haC : a ∈ C.supp := by simp [C]
  have hbC : b ∈ C.supp := C.mem_supp_of_adj_mem_supp haC hab
  let aa : C := ⟨a, haC⟩
  let bb : C := ⟨b, hbC⟩
  have hab_ne : aa ≠ bb := by
    intro h
    have : a = b := congrArg Subtype.val h
    exact hab.ne this
  have hnontrivial : Nontrivial C := ⟨⟨aa, bb, hab_ne⟩⟩
  have htree : C.toSimpleGraph.IsTree := hacyc.isTree_connectedComponent C
  obtain ⟨u, hu⟩ := @SimpleGraph.IsTree.exists_vert_degree_one_of_nontrivial
    C C.toSimpleGraph inferInstance hnontrivial inferInstance htree
  have hu_unique : ∃! v : C, C.toSimpleGraph.Adj u v := by
    rw [← SimpleGraph.degree_eq_one_iff_existsUnique_adj]
    exact hu
  obtain ⟨v, huv, hvuniq⟩ := hu_unique
  have huG : G.degree u.val = 1 := by
    rw [SimpleGraph.degree_eq_one_iff_existsUnique_adj]
    refine ⟨v.val, ?_, ?_⟩
    · exact (C.toSimpleGraph_adj u.property v.property).mp huv
    · intro w huw
      have hwC : w ∈ C.supp := C.mem_supp_of_adj_mem_supp u.property huw
      let ww : C := ⟨w, hwC⟩
      have huww : C.toSimpleGraph.Adj u ww :=
        (C.toSimpleGraph_adj u.property hwC).mpr huw
      have hvw : ww = v := hvuniq ww huww
      exact congrArg Subtype.val hvw
  obtain ⟨k, hk⟩ := heven u.val
  rw [huG] at hk
  omega

end OneesanFormal
