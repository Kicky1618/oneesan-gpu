#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagep.sh"; [[ -f "$PROMOTE" ]] || exit 2; bash -n "$PROMOTE"
for s in 'B300_GRAND_SELECTED_STAGEP_ACCEPTED' 'Stage P accepted without Stage-N acceptance' 'accepted Stage P retained inherited mate L2 baseline' 'Stage P ignored accepted Stage O upstream' 'B300_GRAND_STAGEP_INTEGRATED' 'Stage-P mate L2 differs from grand summary' 'sha256sum -c "$B300_GRAND_STAGEP_MANIFEST"' 'b300x8-grand-promote-exact-stageo.sh'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-P exact marker missing: $s" >&2; exit 3; }; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagep-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake.sh"; cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf called >"${FAKE_CALLED:?}"
SH
chmod +x "$fake"
run_ok(){ FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$2.called" ]] || exit 4; }
run_bad(){ set +e; FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$2.out" 2>"$tmp/$2.err"; rc=$?; set -e; ((rc!=0)) || exit 4; grep -Fq "$3" "$tmp/$2.err" || { cat "$tmp/$2.err" >&2; exit 4; }; [[ ! -e "$tmp/$2.called" ]] || exit 4; }
legacy="$tmp/legacy.env"; printf 'B300_GRAND_SELECTED_SCHEMA=3\nB300_GRAND_SELECTED_VALIDATED=1\n' >"$legacy"; run_ok "$legacy" legacy
payload="$tmp/payload"; echo p >"$payload"; manifest="$tmp/p.sha"; sha256sum "$payload" >"$manifest"
summary="$tmp/grand.env"; cat >"$summary" <<EOF
B300_GRAND_STAGEP_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEP_OK=1
B300_GRAND_STAGEP_COUNT_UPSTREAM=stagen
B300_GRAND_STAGEP_BASE_MATE_L2_BYTES=0
B300_GRAND_STAGEP_MATE_L2_BYTES=128
B300_GRAND_STAGEP_MANIFEST=$(printf '%q' "$manifest")
EOF
valid="$tmp/valid.env"; cat >"$valid" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGEN_ACCEPTED=1
B300_GRAND_SELECTED_STAGEM_POLICY=cg
B300_GRAND_SELECTED_STAGEO_ACCEPTED=0
B300_GRAND_SELECTED_STAGEP_ENABLED=1
B300_GRAND_SELECTED_STAGEP_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEP_ACCEPTED=1
B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM=stagen
B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES=0
B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=128
B300_GRAND_SELECTED_STAGEP_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGEP_SEARCH_MATE_L2='0 64 128 256'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
run_ok "$valid" valid
no_n="$tmp/no-n.env"; cp "$valid" "$no_n"; sed -i 's/B300_GRAND_SELECTED_STAGEN_ACCEPTED=1/B300_GRAND_SELECTED_STAGEN_ACCEPTED=0/' "$no_n"; run_bad "$no_n" no-n 'Stage P accepted without Stage-N acceptance'
unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"; sed -i 's/B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=128/B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=0/' "$unchanged"; run_bad "$unchanged" unchanged 'accepted Stage P retained inherited mate L2 baseline'
oaccept="$tmp/oaccept.env"; cp "$valid" "$oaccept"; sed -i 's/B300_GRAND_SELECTED_STAGEO_ACCEPTED=0/B300_GRAND_SELECTED_STAGEO_ACCEPTED=1/' "$oaccept"; run_bad "$oaccept" oaccept 'Stage P ignored accepted Stage O upstream'
mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"; sed -i 's/B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=128/B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=256/' "$mismatch"; run_bad "$mismatch" mismatch 'Stage-P mate L2 differs from grand summary'
echo corrupt >>"$payload"; run_bad "$valid" manifest 'Stage-P promotion manifest failed before exact continuation'
[[ "$(find "$tmp" -name '*.called'|wc -l)" == 2 ]] || exit 4
echo 'b300-grand-stagep-exact-promotion-preflight OK legacy_schema3=1 stagep_requires_stagen=1 mate_l2_change_required=1 maximal_o_upstream=1 summary_match=1 manifest_gate=1 rejection_before_promoter=1 gpu_work=0'
