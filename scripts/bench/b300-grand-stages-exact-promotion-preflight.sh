#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stages.sh"
[[ -f "$PROMOTE" ]] || exit 2; bash -n "$PROMOTE"
for s in 'B300_GRAND_SELECTED_STAGES_ACCEPTED' 'Stage S accepted without Stage-R acceptance' 'accepted Stage S retained exact Stage-R zero-hint tuple' 'Stage-S pair hint active on non-cg low policy' 'Stage-S high-state tuple drift from accepted Stage R' 'B300_GRAND_STAGES_INTEGRATED' 'Stage-S selected low L2 tuple differs from grand summary' 'sha256sum -c "$B300_GRAND_STAGES_MANIFEST"' 'b300x8-grand-promote-exact-stager.sh' 'PINNED_BASE_PROMOTER'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-S exact marker missing: $s" >&2; exit 3; }; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stages-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake.sh"; cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf called >"${FAKE_CALLED:?}"
SH
chmod +x "$fake"
run_ok(){ FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$2.called" ]] || exit 4; }
run_bad(){ set +e; FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$2.out" 2>"$tmp/$2.err"; rc=$?; set -e; ((rc!=0)) || exit 4; grep -Fq "$3" "$tmp/$2.err" || { cat "$tmp/$2.err" >&2; exit 4; }; [[ ! -e "$tmp/$2.called" ]] || exit 4; }
legacy="$tmp/legacy.env"; printf 'B300_GRAND_SELECTED_SCHEMA=3\nB300_GRAND_SELECTED_VALIDATED=1\n' >"$legacy"; run_ok "$legacy" legacy
payload="$tmp/payload"; echo s >"$payload"; manifest="$tmp/s.sha"; sha256sum "$payload" >"$manifest"
summary="$tmp/grand.env"; cat >"$summary" <<EOF
B300_GRAND_STAGES_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGES_OK=1
B300_GRAND_STAGES_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_STAGES_LOW_PAIR_POLICY=cg
B300_GRAND_STAGES_LOW_BLOCK_POLICY=default
B300_GRAND_STAGES_HIGH_PAIR_POLICY=cg
B300_GRAND_STAGES_HIGH_BLOCK_POLICY=cg
B300_GRAND_STAGES_HIGH_PAIR_L2_BYTES=256
B300_GRAND_STAGES_HIGH_BLOCK_L2_BYTES=64
B300_GRAND_STAGES_PAIR_L2_BYTES=128
B300_GRAND_STAGES_BLOCK_L2_BYTES=0
B300_GRAND_STAGES_MANIFEST=$(printf '%q' "$manifest")
EOF
valid="$tmp/valid.env"; cat >"$valid" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGER_ACCEPTED=1
B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_SELECTED_STAGER_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES=256
B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGES_ENABLED=1
B300_GRAND_SELECTED_STAGES_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGES_ACCEPTED=1
B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGES_HIGH_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES=256
B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=128
B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES=0
B300_GRAND_SELECTED_STAGES_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGES_SEARCH_PAIR_L2='0 64 128 256'
B300_GRAND_SELECTED_STAGES_SEARCH_BLOCK_L2='0'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
BASE_PROMOTER=/bin/false
EOF
run_ok "$valid" valid
no_r="$tmp/no-r.env"; cp "$valid" "$no_r"; sed -i 's/B300_GRAND_SELECTED_STAGER_ACCEPTED=1/B300_GRAND_SELECTED_STAGER_ACCEPTED=0/' "$no_r"; run_bad "$no_r" no-r 'Stage S accepted without Stage-R acceptance'
unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"; sed -i 's/B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=128/B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=0/' "$unchanged"; run_bad "$unchanged" unchanged 'accepted Stage S retained exact Stage-R zero-hint tuple'
noncg="$tmp/noncg.env"; cp "$valid" "$noncg"; sed -i 's/B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY=cg/B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY=default/' "$noncg"; sed -i 's/B300_GRAND_SELECTED_STAGER_PAIR_POLICY=cg/B300_GRAND_SELECTED_STAGER_PAIR_POLICY=default/' "$noncg"; run_bad "$noncg" noncg 'Stage-S pair hint active on non-cg low policy'
high="$tmp/high.env"; cp "$valid" "$high"; sed -i 's/B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES=256/B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES=128/' "$high"; run_bad "$high" high 'Stage-S high-state tuple drift from accepted Stage R'
mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"; sed -i 's/B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=128/B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=64/' "$mismatch"; run_bad "$mismatch" mismatch 'Stage-S selected low L2 tuple differs from grand summary'
echo corrupt >>"$payload"; run_bad "$valid" manifest 'Stage-S promotion manifest failed before exact continuation'
[[ "$(find "$tmp" -name '*.called'|wc -l)" == 2 ]] || exit 4
echo 'b300-grand-stages-exact-promotion-preflight OK legacy_schema3=1 stages_requires_stager=1 low_tuple_change_required=1 noncg_zero=1 high_tuple_preserved=1 summary_match=1 manifest_gate=1 control_paths_pinned=1 rejection_before_promoter=1 gpu_work=0'
