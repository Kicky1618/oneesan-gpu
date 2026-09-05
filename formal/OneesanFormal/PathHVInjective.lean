import OneesanFormal.PathGridEncoding
import OneesanFormal.PathEdgeSet

namespace OneesanFormal.PathHVInjective

open SimpleGraph
open OneesanFormal.GridGraph
open OneesanFormal.GridFaceBoundary
open OneesanFormal.PathGridEncoding
open OneesanFormal.PathEdgeSet

@[simp] theorem pathEdgeBit_eq_one_iff {n : Nat} {u v : GridVertex n}
    (p : (gridGraph n).Walk u v) (a b : GridVertex n) :
    pathEdgeBit p a b = 1 ↔ p.toSubgraph.Adj a b := by
  classical
  simp [pathEdgeBit]

theorem edgeBit_eq_of_hv_eq_of_adj
    {n : Nat} {u v : GridVertex n}
    (p q : (gridGraph n).Walk u v)
    (hH : pathHorizontal p = pathHorizontal q)
    (hV : pathVertical p = pathVertical q)
    {a b : GridVertex n} (hab : (gridGraph n).Adj a b) :
    pathEdgeBit p a b = pathEdgeBit q a b := by
  rw [gridGraph_adj_iff] at hab
  rcases hab with ⟨hrow, hright | hleft⟩ | ⟨hcol, hdown | hup⟩
  · have haRow : a.1.val ≤ n := Nat.le_of_lt_succ a.1.isLt
    have hbCol : b.2.val ≤ n := Nat.le_of_lt_succ b.2.isLt
    have hbPos : 0 < b.2.val := by omega
    have hpH := pathHorizontal_of_valid p a.1.val b.2.val haRow hbPos hbCol
    have hqH := pathHorizontal_of_valid q a.1.val b.2.val haRow hbPos hbCol
    have hcoordA : ((⟨a.1.val, by omega⟩, ⟨b.2.val - 1, by omega⟩) : GridVertex n) = a := by
      apply Prod.ext
      · exact Fin.ext rfl
      · apply Fin.ext
        change b.2.val - 1 = a.2.val
        omega
    have hcoordB : ((⟨a.1.val, by omega⟩, ⟨b.2.val, by omega⟩) : GridVertex n) = b := by
      apply Prod.ext
      · exact hrow
      · exact Fin.ext rfl
    rw [hcoordA, hcoordB] at hpH hqH
    rw [pathEdgeBit_symm p a b, pathEdgeBit_symm q a b]
    rw [← hpH, ← hqH, hH]
  · have hbRow : b.1.val ≤ n := Nat.le_of_lt_succ b.1.isLt
    have haCol : a.2.val ≤ n := Nat.le_of_lt_succ a.2.isLt
    have haPos : 0 < a.2.val := by omega
    have hpH := pathHorizontal_of_valid p b.1.val a.2.val hbRow haPos haCol
    have hqH := pathHorizontal_of_valid q b.1.val a.2.val hbRow haPos haCol
    have hcoordA : ((⟨b.1.val, by omega⟩, ⟨a.2.val, by omega⟩) : GridVertex n) = a := by
      apply Prod.ext
      · exact hrow.symm
      · exact Fin.ext rfl
    have hcoordB : ((⟨b.1.val, by omega⟩, ⟨a.2.val - 1, by omega⟩) : GridVertex n) = b := by
      apply Prod.ext
      · exact Fin.ext rfl
      · apply Fin.ext
        change a.2.val - 1 = b.2.val
        omega
    rw [hcoordA, hcoordB] at hpH hqH
    rw [← hpH, ← hqH, hH]
  · have haCol : a.2.val ≤ n := Nat.le_of_lt_succ a.2.isLt
    have hbRow : b.1.val ≤ n := Nat.le_of_lt_succ b.1.isLt
    have hbPos : 0 < b.1.val := by omega
    have hpV := pathVertical_of_valid p b.1.val a.2.val hbPos hbRow haCol
    have hqV := pathVertical_of_valid q b.1.val a.2.val hbPos hbRow haCol
    have hcoordA : ((⟨b.1.val - 1, by omega⟩, ⟨a.2.val, by omega⟩) : GridVertex n) = a := by
      apply Prod.ext
      · apply Fin.ext
        change b.1.val - 1 = a.1.val
        omega
      · exact Fin.ext rfl
    have hcoordB : ((⟨b.1.val, by omega⟩, ⟨a.2.val, by omega⟩) : GridVertex n) = b := by
      apply Prod.ext
      · exact Fin.ext rfl
      · exact hcol
    rw [hcoordA, hcoordB] at hpV hqV
    rw [pathEdgeBit_symm p a b, pathEdgeBit_symm q a b]
    rw [← hpV, ← hqV, hV]
  · have hbCol : b.2.val ≤ n := Nat.le_of_lt_succ b.2.isLt
    have haRow : a.1.val ≤ n := Nat.le_of_lt_succ a.1.isLt
    have haPos : 0 < a.1.val := by omega
    have hpV := pathVertical_of_valid p a.1.val b.2.val haPos haRow hbCol
    have hqV := pathVertical_of_valid q a.1.val b.2.val haPos haRow hbCol
    have hcoordA : ((⟨a.1.val, by omega⟩, ⟨b.2.val, by omega⟩) : GridVertex n) = a := by
      apply Prod.ext
      · exact Fin.ext rfl
      · exact hcol.symm
    have hcoordB : ((⟨a.1.val - 1, by omega⟩, ⟨b.2.val, by omega⟩) : GridVertex n) = b := by
      apply Prod.ext
      · apply Fin.ext
        change a.1.val - 1 = b.1.val
        omega
      · exact Fin.ext rfl
    rw [hcoordA, hcoordB] at hpV hqV
    rw [← hpV, ← hqV, hV]


theorem path_hv_injective
    {n : Nat} {u v : GridVertex n} :
    Function.Injective (fun p : (gridGraph n).Path u v =>
      (pathHorizontal p.val, pathVertical p.val)) := by
  intro p q hpq
  have hH : pathHorizontal p.val = pathHorizontal q.val := congrArg Prod.fst hpq
  have hV : pathVertical p.val = pathVertical q.val := congrArg Prod.snd hpq
  apply Subtype.ext
  apply isPath_eq_of_edgeSet_eq p.val q.val p.property q.property
  ext e
  induction e using Sym2.inductionOn with
  | hf a b =>
      constructor
      · intro he
        have hpAdj : p.val.toSubgraph.Adj a b := by
          rw [Walk.adj_toSubgraph_iff_mem_edges]
          exact Walk.mem_edgeSet.mp he
        have hgAdj : (gridGraph n).Adj a b := p.val.toSubgraph.adj_sub hpAdj
        have hbit := edgeBit_eq_of_hv_eq_of_adj p.val q.val hH hV hgAdj
        have hpOne : pathEdgeBit p.val a b = 1 :=
          (pathEdgeBit_eq_one_iff p.val a b).2 hpAdj
        have hqOne : pathEdgeBit q.val a b = 1 := by rw [← hbit]; exact hpOne
        have hqAdj : q.val.toSubgraph.Adj a b :=
          (pathEdgeBit_eq_one_iff q.val a b).1 hqOne
        exact Walk.mem_edgeSet.mpr (Walk.adj_toSubgraph_iff_mem_edges.mp hqAdj)
      · intro he
        have hqAdj : q.val.toSubgraph.Adj a b := by
          rw [Walk.adj_toSubgraph_iff_mem_edges]
          exact Walk.mem_edgeSet.mp he
        have hgAdj : (gridGraph n).Adj a b := q.val.toSubgraph.adj_sub hqAdj
        have hbit := edgeBit_eq_of_hv_eq_of_adj p.val q.val hH hV hgAdj
        have hqOne : pathEdgeBit q.val a b = 1 :=
          (pathEdgeBit_eq_one_iff q.val a b).2 hqAdj
        have hpOne : pathEdgeBit p.val a b = 1 := by rw [hbit]; exact hqOne
        have hpAdj : p.val.toSubgraph.Adj a b :=
          (pathEdgeBit_eq_one_iff p.val a b).1 hpOne
        exact Walk.mem_edgeSet.mpr (Walk.adj_toSubgraph_iff_mem_edges.mp hpAdj)


end OneesanFormal.PathHVInjective
