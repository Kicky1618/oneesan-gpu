import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Pi

namespace OneesanFormal.CheckerboardBound

/-- One boundary edge adjacent to two face bits is present exactly when the
bits differ.  This is the local form of `boundary(F)` over GF(2). -/
def boundaryEdge (a b : Bool) : Nat := if a = b then 0 else 1

/-- Degree contributed at one interior grid vertex by the boundary of four
surrounding face bits, listed clockwise as NW, NE, SE, SW. -/
def localBoundaryDegree (nw ne se sw : Bool) : Nat :=
  boundaryEdge nw ne + boundaryEdge ne se +
  boundaryEdge se sw + boundaryEdge sw nw

/-- The two alternating 2x2 face patterns. -/
def checkerboard (nw ne se sw : Bool) : Prop :=
  nw = se ∧ ne = sw ∧ nw ≠ ne

/-- A checkerboard 2x2 face pattern puts all four incident edges into the
boundary, hence creates degree four at the interior vertex. -/
theorem checkerboard_boundary_degree_four
    (nw ne se sw : Bool) (h : checkerboard nw ne se sw) :
    localBoundaryDegree nw ne se sw = 4 := by
  cases nw <;> cases ne <;> cases se <;> cases sw <;>
    simp [checkerboard, localBoundaryDegree, boundaryEdge] at h ⊢

/-- Therefore a boundary with local degree at most two, as required for a
simple path at an interior vertex, cannot contain a checkerboard pattern. -/
theorem checkerboard_impossible_for_simple_local_degree
    (nw ne se sw : Bool)
    (hdeg : localBoundaryDegree nw ne se sw ≤ 2) :
    ¬ checkerboard nw ne se sw := by
  intro hcb
  have hfour := checkerboard_boundary_degree_four nw ne se sw hcb
  omega

/-- XOR with a fixed edge set is a deterministic map from a face-boundary edge
set to a path edge set. -/
def xorWithBase {Edge : Type*} (base delta : Edge → Bool) : Edge → Bool :=
  fun e => Bool.xor (base e) (delta e)

/-- Abstract form of the face encoding used by the rigorous path bound.

For each path, it is enough to choose *some* face assignment whose boundary,
XORed with the fixed reference path, equals the path edge set.  Uniqueness of
the face representation is not needed for the counting injection. -/
theorem chosen_face_encoding_injective
    {Path Edge Face : Type*}
    (pathEdges : Path → Edge → Bool)
    (pathEdges_injective : Function.Injective pathEdges)
    (base : Edge → Bool)
    (boundary : (Face → Bool) → Edge → Bool)
    (encode : Path → Face → Bool)
    (represents : ∀ p, pathEdges p = xorWithBase base (boundary (encode p))) :
    Function.Injective encode := by
  intro p q heq
  apply pathEdges_injective
  rw [represents p, represents q, heq]

/-- If every path is encoded into the subtype of face assignments satisfying a
validity predicate, injectivity immediately gives the desired counting upper
bound.  For the production proof, `valid` is "no checkerboard 2x2". -/
theorem card_paths_le_valid_face_assignments
    {Path Face : Type*}
    [Fintype Path] [Fintype Face] [Fintype (Face → Bool)]
    (valid : (Face → Bool) → Prop)
    [DecidablePred valid]
    (encode : Path → {f : Face → Bool // valid f})
    (hinj : Function.Injective encode) :
    Fintype.card Path ≤ Fintype.card {f : Face → Bool // valid f} := by
  exact Fintype.card_le_of_injective encode hinj

/-- Complete abstract counting step used by the checkerboard upper bound.

Once a concrete rectangular-grid development provides, for every simple path,
a face assignment `encode p` whose boundary represents the path relative to
`base`, and proves that assignment has no checkerboard, the cardinality bound
follows with no uniqueness assumption on face representations. -/
theorem card_bound_of_valid_face_encoding
    {Path Edge Face : Type*}
    [Fintype Path] [Fintype Face] [Fintype (Face → Bool)]
    (valid : (Face → Bool) → Prop)
    [DecidablePred valid]
    (pathEdges : Path → Edge → Bool)
    (pathEdges_injective : Function.Injective pathEdges)
    (base : Edge → Bool)
    (boundary : (Face → Bool) → Edge → Bool)
    (encode : Path → Face → Bool)
    (valid_encode : ∀ p, valid (encode p))
    (represents : ∀ p, pathEdges p = xorWithBase base (boundary (encode p))) :
    Fintype.card Path ≤ Fintype.card {f : Face → Bool // valid f} := by
  let subEncode : Path → {f : Face → Bool // valid f} :=
    fun p => ⟨encode p, valid_encode p⟩
  have hraw : Function.Injective encode :=
    chosen_face_encoding_injective pathEdges pathEdges_injective base boundary encode represents
  have hsub : Function.Injective subEncode := by
    intro p q h
    apply hraw
    exact congrArg Subtype.val h
  exact card_paths_le_valid_face_assignments valid subEncode hsub


/-- Dropping constraints between row strips can only enlarge the admissible set.

`restrict x i` is the assignment seen by strip `i`.  If the collection of
strip restrictions determines the full assignment and every globally valid
assignment is valid inside every strip, then the number of globally valid
assignments is at most the product of the per-strip valid counts.  This is the
abstract counting step used by the production checkerboard-strip bound. -/
theorem card_global_valid_le_product_strips
    {Full I : Type*} {Strip : I → Type*}
    [Fintype Full] [Fintype I] [DecidableEq I] [∀ i, Fintype (Strip i)]
    (globalValid : Full → Prop) [DecidablePred globalValid]
    (stripValid : ∀ i, Strip i → Prop) [∀ i, DecidablePred (stripValid i)]
    [∀ i, Fintype {y : Strip i // stripValid i y}]
    (restrict : Full → ∀ i, Strip i)
    (restrict_injective : Function.Injective restrict)
    (preserves : ∀ x, globalValid x → ∀ i, stripValid i (restrict x i)) :
    Fintype.card {x : Full // globalValid x} ≤
      ∏ i, Fintype.card {y : Strip i // stripValid i y} := by
  let subRestrict : {x : Full // globalValid x} →
      ∀ i, {y : Strip i // stripValid i y} :=
    fun x i => ⟨restrict x i, preserves x.1 x.2 i⟩
  have hinj : Function.Injective subRestrict := by
    intro x y hxy
    apply Subtype.ext
    apply restrict_injective
    funext i
    exact congrArg Subtype.val (congrFun hxy i)
  have hcard := Fintype.card_le_of_injective subRestrict hinj
  simpa only [Fintype.card_pi] using hcard

end OneesanFormal.CheckerboardBound
