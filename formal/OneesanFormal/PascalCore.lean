import OneesanFormal.ExplicitGapBasis
import Mathlib.Tactic

namespace OneesanFormal

/-- Number of flat-free balanced `U/D` bridges of length `n`.
It is zero for odd `n`, and the central binomial coefficient for even `n`. -/
def binaryBridge (n : Nat) : Nat :=
  if n % 2 = 0 then Nat.choose n (n / 2) else 0

/-- Convolution of `g` flat-free bridge gaps with total active gap length `m`.
This is the A111959 core before choosing where the flat `N` rows are inserted. -/
def coreGapConv : Nat → Nat → Nat
  | 0, 0 => 1
  | 0, _ + 1 => 0
  | g + 1, m => ∑ z ∈ Finset.range (m + 1), binaryBridge z * coreGapConv g (m - z)

/-- Flat-free row/interface triangle.  `j` is the number of active row slots
(`T/U/D`) after deleting all `N` slots, and `h` is the separator height. -/
def coreTriangle (j h : Nat) : Nat :=
  if h ≤ j then coreGapConv (h + 1) (j - h) else 0

/-- Pascal lift of the flat-free core: choose the `j` active slots among `r`
physical row slots, then use a `coreTriangle j h` interface on them. -/
def pascalCoreTriangle (r h : Nat) : Nat :=
  ∑ j ∈ Finset.range (r + 1), Nat.choose r j * coreTriangle j h

/-- For row 8, deleting the flat slots and then reinserting them by a Pascal
choice gives exactly the previously proved A111960/gap-triangle dimensions. -/
theorem pascalCoreTriangle_row8 (h : Fin 9) :
    pascalCoreTriangle 8 h = gapTriangle 8 h := by
  fin_cases h <;> native_decide

/-- The nine row-8 block dimensions reconstructed from the core are exactly
`[1107,1640,1428,888,420,152,42,8,1]`. -/
theorem pascalCoreTriangle_row8_values :
    (List.ofFn (fun h : Fin 9 => pascalCoreTriangle 8 h)) =
      [1107, 1640, 1428, 888, 420, 152, 42, 8, 1] := by
  native_decide

/-- Only 529 distinct flat-free core coordinates occur for `j ≤ 8`; the 5686
row-8 coordinates are copies of these cores over different active-row masks. -/
theorem row8_core_coordinate_total :
    (∑ j ∈ Finset.range 9, ∑ h ∈ Finset.range (j + 1), coreTriangle j h) = 529 := by
  native_decide

/-- Pascal lifting the 529 core families over row subsets reconstructs all 5686
row-8 interface coordinates. -/
theorem row8_pascal_lift_total :
    (∑ h ∈ Finset.range 9, pascalCoreTriangle 8 h) = 5686 := by
  native_decide

end OneesanFormal
