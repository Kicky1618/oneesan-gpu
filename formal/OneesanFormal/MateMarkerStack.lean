import OneesanFormal.MateCutSemantics
import Mathlib.Tactic

namespace OneesanFormal.MateMarkerStack

open OneesanFormal.ReverseMain
open OneesanFormal.MateCutSemantics

/-- A live stack occurrence remembers where it was opened. `none` is the
source marker already present before the first frontier symbol; `some i` is the
`L` consumed at frontier position `i`.  These are occurrence identities, not
raw connected-component labels, so several markers may later belong to one
physical component without becoming the same stack occurrence. -/
abbrev Marker := Option Nat

/-- One marker-stack update at frontier position `i`. -/
def markerStep (i : Nat) (s : List Marker) : V → Option (List Marker)
  | .N => some s
  | .L => some (some i :: s)
  | .R => match s with
          | [] => none
          | _ :: rest => some rest

/-- Scan a suffix beginning at absolute frontier position `i`. -/
def scanMarkersFrom : Nat → List Marker → List V → Option (List Marker)
  | _, s, [] => some s
  | i, s, x :: xs =>
      match markerStep i s x with
      | none => none
      | some s' => scanMarkersFrom (i + 1) s' xs

/-- Production/raw convention: one source occurrence is live initially. -/
def scanMarkers (xs : List V) : Option (List Marker) :=
  scanMarkersFrom 0 [none] xs

/-- Forgetting occurrence identities from the marker parser gives exactly the
numeric stack parser used in `MateCutSemantics`. -/
theorem scanMarkersFrom_length {i h : Nat} {s out : List Marker} {xs : List V}
    (hlen : s.length = h)
    (hs : scanMarkersFrom i s xs = some out) :
    scanFrom h xs = some out.length := by
  induction xs generalizing i h s out with
  | nil =>
      simp [scanMarkersFrom, scanFrom] at hs ⊢
      subst out
      simpa [hlen]
  | cons x xs ih =>
      cases x with
      | N =>
          simp [scanMarkersFrom, markerStep, scanFrom, stackStep] at hs ⊢
          exact ih hlen hs
      | L =>
          simp [scanMarkersFrom, markerStep, scanFrom, stackStep] at hs ⊢
          exact ih (by simp [hlen]) hs
      | R =>
          cases s with
          | nil =>
              simp at hlen
              subst h
              simp [scanMarkersFrom, markerStep, scanFrom, stackStep] at hs
          | cons a rest =>
              have hh : h = rest.length + 1 := by simpa using hlen.symm
              simp [scanMarkersFrom, markerStep, scanFrom, stackStep, hh] at hs ⊢
              exact ih (h := rest.length) rfl hs

/-- Therefore the number of live *occurrences* after a successful emitted
prefix is exactly the source-inclusive production frontier height. -/
theorem scanMarkers_length_eq_openCount {xs : List V} {out : List Marker}
    (hs : scanMarkers xs = some out) : out.length = openCount xs := by
  have hscan : scanStack xs = some out.length := by
    unfold scanMarkers at hs
    unfold scanStack
    exact scanMarkersFrom_length (i := 0) (h := 1) (s := [none]) (by simp) hs
  exact scanStack_eq_openCount hscan

/-- Every non-source marker still live after scanning a prefix was opened at a
position strictly before the current prefix length. -/
theorem scanMarkersFrom_marker_lt {i : Nat} {s out : List Marker} {xs : List V}
    (hbase : ∀ m ∈ s, ∀ q, m = some q → q < i)
    (hs : scanMarkersFrom i s xs = some out) :
    ∀ m ∈ out, ∀ q, m = some q → q < i + xs.length := by
  induction xs generalizing i s out with
  | nil =>
      simp [scanMarkersFrom] at hs
      subst out
      simpa using hbase
  | cons x xs ih =>
      cases x with
      | N =>
          simp [scanMarkersFrom, markerStep] at hs
          have hbase' : ∀ m ∈ s, ∀ q, m = some q → q < i + 1 := by
            intro m hm q hmq
            exact Nat.lt_succ_of_lt (hbase m hm q hmq)
          have h := ih (i := i + 1) hbase' hs
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h
      | L =>
          simp [scanMarkersFrom, markerStep] at hs
          have hbase' : ∀ m ∈ (some i :: s), ∀ q, m = some q → q < i + 1 := by
            intro m hm q hmq
            simp only [List.mem_cons] at hm
            rcases hm with rfl | hm
            · have hiq : i = q := Option.some.inj hmq
              omega
            · exact Nat.lt_succ_of_lt (hbase m hm q hmq)
          have h := ih (i := i + 1) hbase' hs
          simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h
      | R =>
          cases s with
          | nil => simp [scanMarkersFrom, markerStep] at hs
          | cons a rest =>
              simp [scanMarkersFrom, markerStep] at hs
              have hbase' : ∀ m ∈ rest, ∀ q, m = some q → q < i + 1 := by
                intro m hm q hmq
                exact Nat.lt_succ_of_lt (hbase m (by simp [hm]) q hmq)
              have h := ih (i := i + 1) hbase' hs
              simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h

/-- In a successful scan from the canonical source stack, every live non-source
marker refers to an `L` occurrence inside the consumed prefix. -/
theorem scanMarkers_marker_lt {xs : List V} {out : List Marker}
    (hs : scanMarkers xs = some out) :
    ∀ m ∈ out, ∀ q, m = some q → q < xs.length := by
  unfold scanMarkers at hs
  have h := scanMarkersFrom_marker_lt (i := 0) (s := [none])
    (out := out) (xs := xs) (by simp) hs
  simpa using h


/-- Physical realization of the *occurrences* currently stored in the marker
stack.  The component type is `Fin stack.length`, so repeated raw component IDs
cannot accidentally identify two live opening occurrences. -/
structure StackRealization {r w : Nat} (c : Fin (w - 1)) (stack : List Marker) where
  leftEnd  : Fin stack.length → OneesanFormal.ProcessedStripCut.Vertex r w
  rightEnd : Fin stack.length → OneesanFormal.ProcessedStripCut.Vertex r w
  path : (q : Fin stack.length) →
    (OneesanFormal.ProcessedStripCut.graph r w).Walk (leftEnd q) (rightEnd q)
  left_side : ∀ q, (leftEnd q).2.val ≤ c.val
  right_side : ∀ q, c.val < (rightEnd q).2.val
  edge_disjoint : ∀ {a b : Fin stack.length}, a ≠ b →
    Disjoint (path a).edgeSet (path b).edgeSet

/-- A marker-occurrence realization is exactly a crossing family whose
components are the live stack slots. -/
def StackRealization.toCrossingFamily {r w : Nat} {c : Fin (w - 1)}
    {stack : List Marker} (R : StackRealization (r := r) c stack) :
    OneesanFormal.ProcessedStripCut.CrossingFamily (r := r) c where
  Component := Fin stack.length
  leftEnd := R.leftEnd
  rightEnd := R.rightEnd
  path := R.path
  left_side := R.left_side
  right_side := R.right_side
  edge_disjoint := R.edge_disjoint

/-- Hence the number of live marker occurrences is bounded by the number of
processed physical rows. -/
theorem StackRealization.length_le_rows {r w : Nat} {c : Fin (w - 1)}
    {stack : List Marker} (R : StackRealization (r := r) c stack) :
    stack.length ≤ r := by
  have h := R.toCrossingFamily.card_le_rows
  change Fintype.card (Fin stack.length) ≤ r at h
  simpa using h

/-- End-to-end cap statement for the occurrence parser: whenever a consumed
Mate prefix succeeds and its live opening occurrences have edge-disjoint
physical cut realizations, the actual parser stack cannot exceed `r`. -/
theorem scanMarkers_length_le_rows {r w : Nat} {c : Fin (w - 1)}
    {xs : List V} {stack : List Marker}
    (hscan : scanMarkers xs = some stack)
    (R : StackRealization (r := r) c stack) :
    stack.length ≤ r := by
  exact R.length_le_rows

end OneesanFormal.MateMarkerStack
