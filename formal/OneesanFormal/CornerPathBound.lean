import OneesanFormal.PathGridEncoding
import OneesanFormal.OuterReference
import OneesanFormal.PathGridParity

namespace OneesanFormal.CornerPathBound

open OneesanFormal.GridGraph
open OneesanFormal.GridFaceBoundary
open OneesanFormal.CheckerboardGridBound
open OneesanFormal.PathGridEncoding
open OneesanFormal.OuterReference
open OneesanFormal.PathGridParity

/-- End-to-end checkerboard-free face encoding for a concrete mathlib simple
corner-to-corner grid path, reduced to the single remaining boundary-parity
bridge for the H/V coordinate encoding. -/
theorem corner_path_has_checkerboard_free_face_encoding
    (n : Nat) (hn : 0 < n)
    {p : (gridGraph n).Walk (northwestCorner n) (southeastCorner n)}
    (hp : p.IsPath)
    (hpathParity : ∀ r c, r ≤ n → c ≤ n →
      vertexParity (pathHorizontal p) (pathVertical p) r c =
        cornerParity n r c) :
    ∃ F : Nat → Nat → F2,
      RectBoundaryWitness n
        (edgeXor (pathHorizontal p) (outerHorizontal n))
        (edgeXor (pathVertical p) (outerVertical n)) F ∧
      noCheckerboardGrid n F := by
  apply isPath_with_reference_has_checkerboard_free_face_encoding
    hn hp (outerHorizontal n) (outerVertical n)
  · intro r _
    exact outerHorizontal_left_ghost n r
  · intro r _
    exact outerHorizontal_right_ghost n r
  · intro c _
    exact outerVertical_top_ghost n c
  · intro c _
    exact outerVertical_bottom_ghost n c
  · intro r c hr hc
    rw [hpathParity r c hr hc]
    exact (outer_reference_vertexParity n r c hn hr hc).symm
  · intro r c hr hc
    exact outer_reference_interior_zero n r c hr hc


/-- Fully concrete per-path checkerboard bound theorem. Every mathlib simple
NW-to-SE path in the rectangular grid has a bounded-face XOR encoding relative
to the fixed outer-boundary reference, and that face matrix contains no 2x2
checkerboard. No parity/topology hypothesis remains. -/
theorem corner_simple_path_has_checkerboard_free_face_encoding
    (n : Nat) (hn : 0 < n)
    {p : (gridGraph n).Walk (northwestCorner n) (southeastCorner n)}
    (hp : p.IsPath) :
    ∃ F : Nat → Nat → F2,
      RectBoundaryWitness n
        (edgeXor (pathHorizontal p) (outerHorizontal n))
        (edgeXor (pathVertical p) (outerVertical n)) F ∧
      noCheckerboardGrid n F := by
  exact corner_path_has_checkerboard_free_face_encoding n hn hp
    (corner_path_vertexParity hn hp)

end OneesanFormal.CornerPathBound
