namespace OneesanFormal.MmapResume

universe u

/-- Durable phases of one transition-closed mmap group.

The production protocol advances monotonically through these phases:
`beforeJournal → journalDurable → scattered → committed`.
A crash may happen between any two durable steps. -/
inductive Phase where
  | beforeJournal
  | journalDurable
  | scattered
  | committed
  deriving DecidableEq, Repr

/-- Logical state after crash recovery.

Before the durable commit bit, recovery must expose the old group state.  In
`scattered`, this is implemented by restoring the durable undo journal.  Once
the commit bit is durable, the new mmap state is authoritative. -/
def recover {α : Type u} (old new : α) : Phase → α
  | .beforeJournal  => old
  | .journalDurable => old
  | .scattered      => old
  | .committed      => new

/-- All pre-commit crash points recover to the old state. -/
theorem recover_precommit {α : Type u} (old new : α) :
    recover old new .beforeJournal = old ∧
    recover old new .journalDurable = old ∧
    recover old new .scattered = old := by
  simp [recover]

/-- A durable commit makes the new state authoritative. -/
theorem recover_committed {α : Type u} (old new : α) :
    recover old new .committed = new := by
  rfl

/-- Resume behavior for one group.

A committed group is skipped.  Every earlier phase is first logically restored
to `old` and then the deterministic transition is recomputed. -/
def resume {α : Type u} (step : α → α) (old new : α) : Phase → α
  | .beforeJournal  => step old
  | .journalDurable => step old
  | .scattered      => step old
  | .committed      => new

/-- The central crash-safety theorem: if the uninterrupted group transition
maps `old` to `new`, then restart produces `new` for *every* crash phase. -/
theorem resume_eq_uninterrupted {α : Type u}
    (step : α → α) (old new : α) (phase : Phase)
    (hstep : step old = new) :
    resume step old new phase = new := by
  cases phase <;> simp [resume, hstep]

/-- A transition-closed window decomposes into independent groups.  The
production bitmap stores one durable `Phase` summary per group; disjointness of
group ranges means recovery can be reasoned about pointwise. -/
def resumeAll {ι : Type u} {α : Type u}
    (step : ι → α → α) (old new : ι → α) (phase : ι → Phase) : ι → α :=
  fun g => resume (step g) (old g) (new g) (phase g)

/-- Arbitrary out-of-order group completion is safe.  Some groups may already
be committed while others are journaled or scattered; after restart and
recomputation the whole window equals the uninterrupted result pointwise. -/
theorem resumeAll_eq_uninterrupted {ι : Type u} {α : Type u}
    (step : ι → α → α) (old new : ι → α) (phase : ι → Phase)
    (hstep : ∀ g, step g (old g) = new g) :
    resumeAll step old new phase = new := by
  funext g
  exact resume_eq_uninterrupted (step g) (old g) (new g) (phase g) (hstep g)

/-- Re-running recovery is idempotent at the logical-state level.  This
matches the implementation rule that a surviving undo journal can be applied
again before its group is recomputed. -/
theorem recover_idempotent {α : Type u} (old new : α) (phase : Phase) :
    recover old (recover old new phase) phase = recover old new phase := by
  cases phase <;> rfl


/-- Abstract membership in one fixed-occupancy partition.  `ι` indexes the
positions selected by `fixed_pos`; both a frontier state and a group id induce
one Boolean occupancy key on those positions. -/
def inGroup {ι : Type u} (stateKey groupKey : ι → Bool) : Prop :=
  ∀ i, stateKey i = groupKey i

/-- Two distinct occupancy keys cannot own the same frontier state.  This is
the mathematical reason concurrent transition-closed groups have disjoint
authoritative ranges. -/
theorem group_partition_disjoint {ι : Type u}
    {g h : ι → Bool} (hne : g ≠ h) :
    ¬ ∃ stateKey, inGroup stateKey g ∧ inGroup stateKey h := by
  intro hex
  rcases hex with ⟨stateKey, hg, hh⟩
  apply hne
  funext i
  rw [← hg i, hh i]

/-- Every frontier state belongs to the partition selected by its own fixed
occupancy key, so the family of groups covers the entire state space. -/
theorem group_partition_covers {ι : Type u} (stateKey : ι → Bool) :
    inGroup stateKey stateKey := by
  intro i
  rfl

end OneesanFormal.MmapResume
