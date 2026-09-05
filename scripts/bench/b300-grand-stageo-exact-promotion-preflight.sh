#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stageo.sh"
[[ -f "$PROMOTE" ]] || exit 2; bash -n "$PROMOTE"
for s in 'B300_GRAND_SELECTED_STAGEO_ACCEPTED' 'Stage O accepted without Stage-N acceptance' 'accepted Stage O retained inherited L2 baseline' 'Stage O changed L2 on non-CG pair axis' 'B300_GRAND_STAGEO_INTEGRATED' 'Stage-O pair L2 differs from grand summary' 'sha256sum -c "$B300_GRAND_STAGEO_MANIFEST"' 'b300x8-grand-promote-exact-stagen.sh'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-O exact marker missing: $s" >&2; exit 3; }; done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageo-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake-promoter.sh"
cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'called selected=%s n=%s\n' "${SELECTED_ENV:?}" "${1:-}" >"${FAKE_CALLED:?}"
SH
chmod +x "$fake"
run_ok(){ local envf="$1" tag="$2"; FAKE_CALLED="$tmp/$tag.called" SELECTED_ENV="$envf" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$tag.called" ]] || exit 4; }
run_bad(){ local envf="$1" tag="$2" needle="$3"; set +e; FAKE_CALLED="$tmp/$tag.called" SELECTED_ENV="$envf" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$tag.out" 2>"$tmp/$tag.err"; local rc=$?; set -e; ((rc!=0)) || { echo "$tag unexpectedly accepted" >&2; exit 4; }; grep -Fq "$needle" "$tmp/$tag.err" || { cat "$tmp/$tag.err" >&2; exit 4; }; [[ ! -e "$tmp/$tag.called" ]] || exit 4; }
legacy="$tmp/legacy.env"; printf 'B300_GRAND_SELECTED_SCHEMA=3\nB300_GRAND_SELECTED_VALIDATED=1\n' >"$legacy"; run_ok "$legacy" legacy
payload="$tmp/payload"; printf 'stage-o-manifest-payload\n' >"$payload"; manifest="$tmp/stageo.sha256"; sha256sum "$payload" >"$manifest"
summary="$tmp/grand.env"
cat >"$summary" <<EOF
B300_GRAND_STAGEO_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEO_OK=1
B300_GRAND_STAGEO_PAIR_L2_BYTES=64
B300_GRAND_STAGEO_BLOCK_L2_BYTES=0
B300_GRAND_STAGEO_BASE_PAIR_L2_BYTES=128
B300_GRAND_STAGEO_BASE_BLOCK_L2_BYTES=0
B300_GRAND_STAGEO_MANIFEST=$(printf '%q' "$manifest")
EOF
valid="$tmp/valid.env"
cat >"$valid" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGEN_ACCEPTED=1
B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY=cs
B300_GRAND_SELECTED_STAGEO_ENABLED=1
B300_GRAND_SELECTED_STAGEO_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEO_ACCEPTED=1
B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES=64
B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES=0
B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES=128
B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES=0
B300_GRAND_SELECTED_STAGEO_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGEO_SEARCH_PAIR_L2='0 64 128 256'
B300_GRAND_SELECTED_STAGEO_SEARCH_BLOCK_L2='0'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
run_ok "$valid" valid
no_n="$tmp/no-n.env"; cp "$valid" "$no_n"; sed -i 's/B300_GRAND_SELECTED_STAGEN_ACCEPTED=1/B300_GRAND_SELECTED_STAGEN_ACCEPTED=0/' "$no_n"; run_bad "$no_n" no-n 'Stage O accepted without Stage-N acceptance'
unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"; sed -i 's/B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES=64/B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES=128/' "$unchanged"; run_bad "$unchanged" unchanged 'accepted Stage O retained inherited L2 baseline'
noncg="$tmp/noncg.env"; cp "$valid" "$noncg"; sed -i 's/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cg/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cs/' "$noncg"; run_bad "$noncg" noncg 'Stage O changed L2 on non-CG pair axis'
mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"; sed -i 's/B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES=64/B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES=256/' "$mismatch"; run_bad "$mismatch" mismatch 'Stage-O pair L2 differs from grand summary'
printf 'corrupt\n' >>"$payload"
run_bad "$valid" manifest 'Stage-O promotion manifest failed before exact continuation'
[[ "$(find "$tmp" -name '*.called' | wc -l)" == 2 ]] || { echo 'only legacy and valid Stage O should reach base promoter' >&2; exit 4; }
echo 'b300-grand-stageo-exact-promotion-preflight OK legacy_schema3=1 stageo_requires_stagen=1 changed_l2_required=1 noncg_axis_guard=1 summary_l2_match=1 manifest_gate=1 rejection_before_promoter=1 gpu_work=0'
