#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stageq.sh"; [[ -f "$PROMOTE" ]] || exit 2; bash -n "$PROMOTE"
for s in 'B300_GRAND_SELECTED_STAGEQ_ACCEPTED' 'Stage Q accepted without Stage-N acceptance' 'Stage Q ignored accepted Stage P upstream' 'Stage Q ignored accepted Stage O upstream' 'accepted Stage Q retained exact upstream Count L2 tuple' 'B300_GRAND_STAGEQ_INTEGRATED' 'Stage-Q selected Count L2 differs from grand summary' 'sha256sum -c "$B300_GRAND_STAGEQ_MANIFEST"' 'b300x8-grand-promote-exact-stagep.sh'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-Q exact marker missing: $s" >&2; exit 3; }; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageq-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake.sh"; cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf called >"${FAKE_CALLED:?}"
SH
chmod +x "$fake"
run_ok(){ FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$2.called" ]] || exit 4; }
run_bad(){ set +e; FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$2.out" 2>"$tmp/$2.err"; rc=$?; set -e; ((rc!=0)) || exit 4; grep -Fq "$3" "$tmp/$2.err" || { cat "$tmp/$2.err" >&2; exit 4; }; [[ ! -e "$tmp/$2.called" ]] || exit 4; }
legacy="$tmp/legacy.env"; printf 'B300_GRAND_SELECTED_SCHEMA=3\nB300_GRAND_SELECTED_VALIDATED=1\n' >"$legacy"; run_ok "$legacy" legacy
payload="$tmp/payload"; echo q >"$payload"; manifest="$tmp/q.sha"; sha256sum "$payload" >"$manifest"
summary="$tmp/grand.env"; cat >"$summary" <<EOF
B300_GRAND_STAGEQ_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEQ_OK=1
B300_GRAND_STAGEQ_UPSTREAM_KIND=stagep
B300_GRAND_STAGEQ_UPSTREAM_PAIR_L2_BYTES=128
B300_GRAND_STAGEQ_UPSTREAM_BLOCK_L2_BYTES=64
B300_GRAND_STAGEQ_PAIR_L2_BYTES=256
B300_GRAND_STAGEQ_BLOCK_L2_BYTES=64
B300_GRAND_STAGEQ_MANIFEST=$(printf '%q' "$manifest")
EOF
valid="$tmp/valid.env"; cat >"$valid" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGEN_ACCEPTED=1
B300_GRAND_SELECTED_STAGEO_ACCEPTED=1
B300_GRAND_SELECTED_STAGEP_ACCEPTED=1
B300_GRAND_SELECTED_STAGEQ_ENABLED=1
B300_GRAND_SELECTED_STAGEQ_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEQ_ACCEPTED=1
B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagep
B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES=128
B300_GRAND_SELECTED_STAGEQ_UPSTREAM_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=256
B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGEQ_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGEQ_SEARCH_PAIR_L2='0 64 128 256'
B300_GRAND_SELECTED_STAGEQ_SEARCH_BLOCK_L2='0 64 128 256'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
run_ok "$valid" valid
no_n="$tmp/no-n.env"; cp "$valid" "$no_n"; sed -i 's/B300_GRAND_SELECTED_STAGEN_ACCEPTED=1/B300_GRAND_SELECTED_STAGEN_ACCEPTED=0/' "$no_n"; run_bad "$no_n" no-n 'Stage Q accepted without Stage-N acceptance'
wrong_up="$tmp/wrong-up.env"; cp "$valid" "$wrong_up"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagep/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stageo/' "$wrong_up"; run_bad "$wrong_up" wrong-up 'Stage Q ignored accepted Stage P upstream'
# Explicit O fallback: once P is not accepted, Q must use O rather than N.
wrong_o="$tmp/wrong-o.env"; cp "$valid" "$wrong_o"; sed -i 's/B300_GRAND_SELECTED_STAGEP_ACCEPTED=1/B300_GRAND_SELECTED_STAGEP_ACCEPTED=0/' "$wrong_o"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagep/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagen/' "$wrong_o"; run_bad "$wrong_o" wrong-o 'Stage Q ignored accepted Stage O upstream'
unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=256/B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=128/' "$unchanged"; run_bad "$unchanged" unchanged 'accepted Stage Q retained exact upstream Count L2 tuple'
mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES=64/B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES=128/' "$mismatch"; run_bad "$mismatch" mismatch 'Stage-Q selected Count L2 differs from grand summary'
echo corrupt >>"$payload"; run_bad "$valid" manifest 'Stage-Q promotion manifest failed before exact continuation'
[[ "$(find "$tmp" -name '*.called'|wc -l)" == 2 ]] || exit 4
echo 'b300-grand-stageq-exact-promotion-preflight OK legacy_schema3=1 stageq_requires_stagen=1 maximal_upstream_P=1 maximal_upstream_O=1 fallback_N_contract=1 tuple_change_required=1 summary_match=1 manifest_gate=1 rejection_before_promoter=1 gpu_work=0'
