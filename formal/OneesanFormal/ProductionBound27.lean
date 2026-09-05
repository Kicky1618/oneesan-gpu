import OneesanFormal.StripBound27
import OneesanFormal.StripDP
import OneesanFormal.BitColumnDP
import OneesanFormal.RawDP
import OneesanFormal.FastDP9
import OneesanFormal.CompressedDP9
import OneesanFormal.CompressedTableDP9

namespace OneesanFormal.ProductionBound27

open OneesanFormal.GridGraph
open OneesanFormal.StripBound27
open OneesanFormal.StripDP
open OneesanFormal.BitColumnDP
open OneesanFormal.RawDP
open OneesanFormal.FastDP9
open OneesanFormal.CompressedDP9
open OneesanFormal.CompressedTableDP9

/-- The exact transfer-DP count for one 9x27 checkerboard-free row strip. -/
def stripCount9x27 : Nat :=
  3165928478117342768922265826341920493835329849417470440184018662

/-- The production n=27 checkerboard-strip upper bound: three independent
9x27 strips. -/
def pathBound27 : Nat :=
  31732427633797389964407887052573851640105323179333844527763421102211579188310190597412900756550874123129342585261840138964010190312364508107168927006232277067783279927977876779029754107293528

/-- The semantic 9x27 strip count is bounded by the fully executable raw-state
DP total. All implementation refinements are equalities; only the earlier
strip-to-column map contributes an inequality. -/
theorem card_valid_strip927_le_rawTotal :
    Fintype.card {s : Strip9x27 // stripNoCheckerboard s} ≤ rawTotal 9 26 := by
  have h := card_validStrip927_le_dp
  rw [← fastTotal_eq_endCount_sum 9 26] at h
  rw [← rawTotal_eq_fastTotal 9 26] at h
  exact h

/-- Fully formal n=27 production-shaped upper bound before evaluating the
executable DP constant. -/
theorem card_corner_paths_27_le_rawTotal_cube :
    Fintype.card ((gridGraph 27).Path
      (northwestCorner 27) (southeastCorner 27)) ≤ (rawTotal 9 26) ^ 3 := by
  let a : Nat := Fintype.card {s : Strip9x27 // stripNoCheckerboard s}
  let b : Nat := rawTotal 9 26
  have hab : a ≤ b := by exact card_valid_strip927_le_rawTotal
  have hp : a ^ 3 ≤ b ^ 3 := Nat.pow_le_pow_left hab 3
  exact le_trans card_corner_paths_27_le_three_strips hp

/-- The raw executable DP evaluates to the exact 9x27 strip count. The native
constant computation is performed in `CompressedTableDP9`; all arrows here are
proved refinement equalities. -/
theorem rawTotal_9_26_value : rawTotal 9 26 = stripCount9x27 := by
  calc
    rawTotal 9 26 = fast9Total 26 := (fast9Total_eq_rawTotal 26).symm
    _ = compressedTotal 26 := (compressedTotal_eq_fast9Total 26).symm
    _ = tableTotal := tableTotal_eq_compressedTotal.symm
    _ = stripCount9x27 := by simpa [stripCount9x27] using tableTotal_value

/-- The cube of the formally verified strip count is the exact 633-bit
production path bound. -/
theorem rawTotal_cube_value : (rawTotal 9 26) ^ 3 = pathBound27 := by
  rw [rawTotal_9_26_value]
  norm_num [stripCount9x27, pathBound27]

/-- End-to-end numerical n=27 theorem: the number of simple NW-to-SE grid
paths is at most the exact 633-bit checkerboard-strip bound used to size CRT. -/
theorem card_corner_paths_27_le_pathBound27 :
    Fintype.card ((gridGraph 27).Path
      (northwestCorner 27) (southeastCorner 27)) ≤ pathBound27 := by
  calc
    Fintype.card ((gridGraph 27).Path
      (northwestCorner 27) (southeastCorner 27)) ≤ (rawTotal 9 26) ^ 3 :=
        card_corner_paths_27_le_rawTotal_cube
    _ = pathBound27 := rawTotal_cube_value

end OneesanFormal.ProductionBound27
