#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stager.sh"; [[ -f "$PROMOTE" ]] || exit 2; bash -n "$PROMOTE"
for s in 'B300_GRAND_SELECTED_STAGER_ACCEPTED' 'Stage R accepted without Stage-N acceptance' 'Stage R ignored maximal accepted upstream' 'accepted Stage R retained exact upstream ILP2 tuple' 'Stage-R did not preserve accepted Stage-Q high L2 tuple' 'B300_GRAND_STAGER_INTEGRATED' 'Stage-R selected tuple differs from grand summary' 'sha256sum -c "$B300_GRAND_STAGER_MANIFEST"' 'b300x8-grand-promote-exact-stageq.sh'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-R exact marker missing: $s" >&2; exit 3; }; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stager-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake.sh"; cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf called >"${FAKE_CALLED:?}"
SH
chmod +x "$fake"
run_ok(){ FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$2.called" ]] || exit 4; }
run_bad(){ set +e; FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$2.out" 2>"$tmp/$2.err"; rc=$?; set -e; ((rc!=0)) || exit 4; grep -Fq "$3" "$tmp/$2.err" || { cat "$tmp/$2.err" >&2; exit 4; }; [[ ! -e "$tmp/$2.called" ]] || exit 4; }
legacy="$tmp/legacy.env"; printf 'B300_GRAND_SELECTED_SCHEMA=3\nB300_GRAND_SELECTED_VALIDATED=1\n' >"$legacy"; run_ok "$legacy" legacy
payload="$tmp/payload"; echo r >"$payload"; manifest="$tmp/r.sha"; sha256sum "$payload" >"$manifest"
summary="$tmp/grand.env"; cat >"$summary" <<EOF
B300_GRAND_STAGER_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGER_OK=1
B300_GRAND_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_STAGER_PAIR_POLICY=cs
B300_GRAND_STAGER_BLOCK_POLICY=default
B300_GRAND_STAGER_HIGH_PAIR_POLICY=cg
B300_GRAND_STAGER_HIGH_BLOCK_POLICY=cg
B300_GRAND_STAGER_HIGH_PAIR_L2_BYTES=256
B300_GRAND_STAGER_HIGH_BLOCK_L2_BYTES=64
B300_GRAND_STAGER_MANIFEST=$(printf '%q' "$manifest")
EOF
valid="$tmp/valid.env"; cat >"$valid" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGEN_ACCEPTED=1
B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGEO_ACCEPTED=1
B300_GRAND_SELECTED_STAGEP_ACCEPTED=1
B300_GRAND_SELECTED_STAGEQ_ACCEPTED=1
B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=256
B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGER_ENABLED=1
B300_GRAND_SELECTED_STAGER_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGER_ACCEPTED=1
B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_SELECTED_STAGER_UPSTREAM_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGER_UPSTREAM_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES=256
B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGER_PAIR_POLICY=cs
B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGER_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGER_SEARCH_PAIR_POLICIES='default cg cs'
B300_GRAND_SELECTED_STAGER_SEARCH_BLOCK_POLICIES='default cg cs'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
run_ok "$valid" valid
no_n="$tmp/no-n.env"; cp "$valid" "$no_n"; sed -i 's/B300_GRAND_SELECTED_STAGEN_ACCEPTED=1/B300_GRAND_SELECTED_STAGEN_ACCEPTED=0/' "$no_n"; run_bad "$no_n" no-n 'Stage R accepted without Stage-N acceptance'
# Q accepted -> any lower upstream is stale.
wrong_q="$tmp/wrong-q.env"; cp "$valid" "$wrong_q"; sed -i 's/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stagep/' "$wrong_q"; run_bad "$wrong_q" wrong-q 'Stage R ignored maximal accepted upstream'
# Q rejected, P accepted -> O/N are stale.
wrong_p="$tmp/wrong-p.env"; cp "$valid" "$wrong_p"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_ACCEPTED=1/B300_GRAND_SELECTED_STAGEQ_ACCEPTED=0/' "$wrong_p"; sed -i 's/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageo/' "$wrong_p"; run_bad "$wrong_p" wrong-p 'Stage R ignored maximal accepted upstream'
# Q/P rejected, O accepted -> N is stale.
wrong_o="$tmp/wrong-o.env"; cp "$valid" "$wrong_o"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_ACCEPTED=1/B300_GRAND_SELECTED_STAGEQ_ACCEPTED=0/' "$wrong_o"; sed -i 's/B300_GRAND_SELECTED_STAGEP_ACCEPTED=1/B300_GRAND_SELECTED_STAGEP_ACCEPTED=0/' "$wrong_o"; sed -i 's/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stagen/' "$wrong_o"; run_bad "$wrong_o" wrong-o 'Stage R ignored maximal accepted upstream'
# Q/P/O rejected -> only N is legal. Claiming O must fail.
wrong_n="$tmp/wrong-n.env"; cp "$valid" "$wrong_n"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_ACCEPTED=1/B300_GRAND_SELECTED_STAGEQ_ACCEPTED=0/' "$wrong_n"; sed -i 's/B300_GRAND_SELECTED_STAGEP_ACCEPTED=1/B300_GRAND_SELECTED_STAGEP_ACCEPTED=0/' "$wrong_n"; sed -i 's/B300_GRAND_SELECTED_STAGEO_ACCEPTED=1/B300_GRAND_SELECTED_STAGEO_ACCEPTED=0/' "$wrong_n"; sed -i 's/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq/B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageo/' "$wrong_n"; run_bad "$wrong_n" wrong-n 'Stage R ignored maximal accepted upstream'
unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"; sed -i 's/B300_GRAND_SELECTED_STAGER_PAIR_POLICY=cs/B300_GRAND_SELECTED_STAGER_PAIR_POLICY=cg/' "$unchanged"; sed -i 's/B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=default/B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=cg/' "$unchanged"; run_bad "$unchanged" unchanged 'accepted Stage R retained exact upstream ILP2 tuple'
badl2="$tmp/badl2.env"; cp "$valid" "$badl2"; sed -i 's/B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES=256/B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES=128/' "$badl2"; run_bad "$badl2" badl2 'Stage-R did not preserve accepted Stage-Q high L2 tuple'
mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"; sed -i 's/B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=default/B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=cs/' "$mismatch"; run_bad "$mismatch" mismatch 'Stage-R selected tuple differs from grand summary'
echo corrupt >>"$payload"; run_bad "$valid" manifest 'Stage-R promotion manifest failed before exact continuation'
[[ "$(find "$tmp" -name '*.called'|wc -l)" == 2 ]] || exit 4
echo 'b300-grand-stager-exact-promotion-preflight OK legacy_schema3=1 stager_requires_stagen=1 maximal_upstream_Q=1 maximal_upstream_P=1 maximal_upstream_O=1 fallback_N=1 low_tuple_change_required=1 stageq_high_tuple_preserved=1 summary_match=1 manifest_gate=1 rejection_before_promoter=1 gpu_work=0'
