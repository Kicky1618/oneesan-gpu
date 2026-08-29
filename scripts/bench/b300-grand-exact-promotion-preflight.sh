#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh"
CHECKPOINT_TEST="$ONEESAN_ROOT/scripts/bench/b300-exact-checkpoint-compat-preflight.py"
[[ -f "$PROMOTE" && -f "$CHECKPOINT_TEST" ]] || { echo 'promotion preflight dependency missing' >&2; exit 2; }
bash -n "$PROMOTE"
python3 -m py_compile "$CHECKPOINT_TEST" "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py"
python3 "$CHECKPOINT_TEST" | grep -Fq 'b300-exact-checkpoint-compat-preflight OK'

for s in \
  'B300_GRAND_SELECTED_VALIDATED' \
  'B300_GRAND_SELECTED_BINARY_SHA256' \
  'B300_GRAND_SELECTED_PROFILE_SHA256' \
  'B300_GRAND_SELECTED_RACE_RESULT_SHA256' \
  'checkpoint fingerprint mismatch' \
  'race winner contract mismatch' \
  'ALLOW_HEAD_DRIFT' \
  'ALLOW_DIRTY_FIRSTPASS' \
  'ALLOW_WORKTREE_DIRTY' \
  'DRY_RUN' \
  'solve_b300_exact_batch.py' \
  'B300 GRAND EXACT PROMOTION VALIDATED'; do
  grep -Fq "$s" "$PROMOTE" || { echo "promotion marker missing: $s" >&2; exit 3; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/fake-batch"
PROFILE="$TMP/profile.env"
RESULT="$TMP/race.tsv"
WORK="$TMP/work"
META="$TMP/firstpass.meta"
ENVF="$TMP/selected.env"
mkdir -p "$WORK"
cat >"$BIN" <<'SH'
#!/usr/bin/env bash
exit 99
SH
chmod +x "$BIN"
cat >"$PROFILE" <<'ENV'
BUCKET_THREADS=256
WARP_GX=32
WARP_GY=8
ORBIT_GY=128
LOW_GX=16
LOW_GY=8
ORBITCTA_FLAT=0
ORBITCTA_FLAT_BLOCKS_PER_SM=0
ENV
printf 'backend\tprofile\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
printf 'synthetic_forced\tt256\t%s\tok\t7\t1.250000\t90\t95\t4\n' "$BIN" >>"$RESULT"
printf 'schema=2\nexit_code=0\npromotion_contract=1\n' >"$META"

BIN_SHA="$(sha256sum "$BIN" | awk '{print $1}')"
PROFILE_SHA="$(sha256sum "$PROFILE" | awk '{print $1}')"
RESULT_SHA="$(sha256sum "$RESULT" | awk '{print $1}')"
HEAD_SHA="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
cat >"$WORK/checkpoint.json" <<EOF
{
  "n": 27,
  "solver_fingerprint": {
    "schema": 3,
    "binary_sha256": "$BIN_SHA",
    "profile_sha256": "$PROFILE_SHA"
  },
  "residues": {
    "4294967291": {"residue": 7, "wall_s": 1.25}
  }
}
EOF
cat >"$ENVF" <<EOF
B300_GRAND_SELECTED_SCHEMA=1
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_N=27
B300_GRAND_SELECTED_HEAD_SHA=$(printf '%q' "$HEAD_SHA")
B300_GRAND_SELECTED_HEAD_DIRTY=0
B300_GRAND_SELECTED_PROFILE_FILE=$(printf '%q' "$PROFILE")
B300_GRAND_SELECTED_PROFILE_SHA256=$PROFILE_SHA
B300_GRAND_SELECTED_BACKEND=synthetic_forced
B300_GRAND_SELECTED_PROFILE=t256
B300_GRAND_SELECTED_BINARY=$(printf '%q' "$BIN")
B300_GRAND_SELECTED_BINARY_SHA256=$BIN_SHA
B300_GRAND_SELECTED_RESIDUE=7
B300_GRAND_SELECTED_WALL_S=1.250000
B300_GRAND_SELECTED_SMOKE_PRIME=4294967291
B300_GRAND_SELECTED_RUNTIME_KIND=forced
B300_GRAND_SELECTED_THREADS=256
B300_GRAND_SELECTED_TARGET_MIB=65536
B300_GRAND_SELECTED_MAX_WINDOW=14
B300_GRAND_SELECTED_WORK_DIR=$(printf '%q' "$WORK")
B300_GRAND_SELECTED_CHECKPOINT=$(printf '%q' "$WORK/checkpoint.json")
B300_GRAND_SELECTED_RACE_RESULT=$(printf '%q' "$RESULT")
B300_GRAND_SELECTED_RACE_RESULT_SHA256=$RESULT_SHA
B300_GRAND_SELECTED_FIRSTPASS_META=$(printf '%q' "$META")
EOF

OUT="$TMP/promote.out"
ERR="$TMP/promote.err"
DRY_RUN=1 SELECTED_ENV="$ENVF" bash "$PROMOTE" 27 >"$OUT" 2>"$ERR"
grep -Fq 'B300 GRAND EXACT PROMOTION VALIDATED command=' "$ERR"
grep -Fq -- '--binary' "$ERR"
grep -Fq "$BIN" "$ERR"
grep -Fq -- '--work-dir' "$ERR"
grep -Fq "$WORK" "$ERR"
[[ -s "$WORK/promotion.meta" ]] || { echo 'promotion meta was not written' >&2; exit 4; }
grep -Fq 'selected_binary_sha256=' "$WORK/promotion.meta"

# Tampering with any immutable first-pass artifact must block promotion before
# the exact solver can be launched.
printf 'tamper\n' >>"$RESULT"
set +e
DRY_RUN=1 SELECTED_ENV="$ENVF" bash "$PROMOTE" 27 >"$TMP/tamper.out" 2>"$TMP/tamper.err"
rc=$?
set -e
((rc!=0)) || { echo 'tampered race TSV unexpectedly accepted' >&2; exit 4; }
grep -Fq 'single-pass TSV fingerprint mismatch' "$TMP/tamper.err"

echo 'b300-grand-exact-promotion-preflight OK bash_syntax=1 checkpoint_schema2_schema3=1 synthetic_dry_run=1 selected_binary=1 runtime_reconstruction=forced tamper_rejected=1 gpu_work=0'
