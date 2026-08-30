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
  'unsupported grand selection schema' \
  'grand summary fingerprint mismatch' \
  'schema-3 grand summary Stage-I/J/K single-race proof missing' \
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
ENV1="$TMP/selected-schema1.env"
ENV3="$TMP/selected-schema3.env"
SUMMARY="$TMP/grand-summary.env"
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
printf 'schema=4\nexit_code=0\npromotion_contract=3\n' >"$META"

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

write_common(){
  local schema="$1" out="$2"
  cat >"$out" <<EOF
B300_GRAND_SELECTED_SCHEMA=$schema
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
}

# Backward-compatible schema-1 contract remains accepted.
write_common 1 "$ENV1"
DRY_RUN=1 SELECTED_ENV="$ENV1" bash "$PROMOTE" 27 >"$TMP/schema1.out" 2>"$TMP/schema1.err"
grep -Fq 'B300 GRAND EXACT PROMOTION VALIDATED schema=1 command=' "$TMP/schema1.err"
grep -Fq -- '--binary' "$TMP/schema1.err"
grep -Fq "$BIN" "$TMP/schema1.err"

# Current hardened schema-3 contract binds the selected candidate to the exact
# grand summary and to the one-complete-prime Stage-I/J/K selection proof.
cat >"$SUMMARY" <<'EOF'
B300_GRAND_PREPARED=1
B300_GRAND_STAGEI_NAMESPACE_ISOLATED=1
B300_GRAND_STAGEJ_INTEGRATED=1
B300_GRAND_STAGEK_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEI_OK=1
B300_GRAND_STAGEJ_OK=1
B300_GRAND_STAGEK_OK=1
EOF
SUMMARY_SHA="$(sha256sum "$SUMMARY" | awk '{print $1}')"
write_common 3 "$ENV3"
cat >>"$ENV3" <<EOF
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$SUMMARY")
B300_GRAND_SELECTED_GRAND_SUMMARY_SHA256=$SUMMARY_SHA
B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1
B300_GRAND_SELECTED_STAGEI_ACCEPTED=1
B300_GRAND_SELECTED_STAGEJ_ACCEPTED=1
B300_GRAND_SELECTED_STAGEK_ACCEPTED=1
EOF
DRY_RUN=1 SELECTED_ENV="$ENV3" bash "$PROMOTE" 27 >"$TMP/schema3.out" 2>"$TMP/schema3.err"
grep -Fq 'B300 GRAND EXACT PROMOTION VALIDATED schema=3 command=' "$TMP/schema3.err"
grep -Fq -- '--work-dir' "$TMP/schema3.err"
grep -Fq "$WORK" "$TMP/schema3.err"
[[ -s "$WORK/promotion.meta" ]] || { echo 'promotion meta was not written' >&2; exit 4; }
grep -Fq 'selection_schema=3' "$WORK/promotion.meta"
grep -Fq "selected_grand_summary_sha256=$SUMMARY_SHA" "$WORK/promotion.meta"

# Tampering with the complete-prime result must block promotion.
printf 'tamper\n' >>"$RESULT"
set +e
DRY_RUN=1 SELECTED_ENV="$ENV3" bash "$PROMOTE" 27 >"$TMP/tamper-race.out" 2>"$TMP/tamper-race.err"
rc=$?
set -e
((rc!=0)) || { echo 'tampered race TSV unexpectedly accepted' >&2; exit 4; }
grep -Fq 'single-pass TSV fingerprint mismatch' "$TMP/tamper-race.err"
# Restore the race file and its contract for the independent summary tamper test.
sed -i '$d' "$RESULT"
[[ "$(sha256sum "$RESULT" | awk '{print $1}')" == "$RESULT_SHA" ]] || exit 4
printf 'B300_GRAND_STAGEK_OK=0\n' >>"$SUMMARY"
set +e
DRY_RUN=1 SELECTED_ENV="$ENV3" bash "$PROMOTE" 27 >"$TMP/tamper-summary.out" 2>"$TMP/tamper-summary.err"
rc=$?
set -e
((rc!=0)) || { echo 'tampered grand summary unexpectedly accepted' >&2; exit 4; }
grep -Fq 'grand summary fingerprint mismatch' "$TMP/tamper-summary.err"

echo 'b300-grand-exact-promotion-preflight OK selection_schema1=1 selection_schema3=1 checkpoint_schema3=1 grand_summary_fingerprint=1 single_complete_prime_proof=1 synthetic_dry_run=1 race_tamper_rejected=1 summary_tamper_rejected=1 gpu_work=0'