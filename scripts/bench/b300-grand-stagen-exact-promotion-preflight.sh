#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagen.sh"
[[ -f "$PROMOTE" ]] || { echo "missing Stage-N exact promoter=$PROMOTE" >&2; exit 2; }
bash -n "$PROMOTE"
for s in \
  'B300_GRAND_SELECTED_STAGEN_ENABLED' \
  'B300_GRAND_SELECTED_STAGEN_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND' \
  'B300_GRAND_SELECTED_STAGEN_PAIR_POLICY' \
  'B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY' \
  'B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY' \
  'accepted Stage N retained inherited pair/block baseline' \
  'accepted Stage-N speedup below threshold' \
  'Stage N uses Stage M without Stage-M acceptance' \
  'Stage N uses Stage L without Stage-L acceptance' \
  'Stage-N search policy set omits inherited baseline' \
  'B300_GRAND_STAGEN_INTEGRATED' \
  'B300_GRAND_COMPLETE_PRIME_RACES' \
  'Stage-N pair policy differs from grand summary' \
  'Stage-N block policy differs from grand summary' \
  'b300x8-grand-promote-exact-stagem.sh'; do
  grep -Fq "$s" "$PROMOTE" || { echo "Stage-N exact marker missing: $s" >&2; exit 3; }
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagen-exact.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
FAKE="$tmp/fake-promoter.sh"
cat >"$FAKE" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'fake_promoter_called=1 selected=%s n=%s\n' "${SELECTED_ENV:-}" "${1:-}" >"${FAKE_CALLED:?}"
SH
chmod +x "$FAKE"

run_ok(){ local envf="$1" tag="$2"; FAKE_CALLED="$tmp/$tag.called" SELECTED_ENV="$envf" BASE_PROMOTER="$FAKE" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$tag.called" ]] || { echo "$tag selection was not delegated" >&2; exit 4; }; }
run_bad(){ local envf="$1" tag="$2" needle="$3"; set +e; FAKE_CALLED="$tmp/$tag.called" SELECTED_ENV="$envf" BASE_PROMOTER="$FAKE" bash "$PROMOTE" 27 >"$tmp/$tag.out" 2>"$tmp/$tag.err"; local rc=$?; set -e; ((rc!=0)) || { echo "$tag unexpectedly accepted" >&2; exit 4; }; grep -Fq "$needle" "$tmp/$tag.err" || { cat "$tmp/$tag.err" >&2; exit 4; }; [[ ! -e "$tmp/$tag.called" ]] || { echo "$tag reached base promoter" >&2; exit 4; }; }

legacy="$tmp/legacy.env"
cat >"$legacy" <<'EOF'
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
EOF
run_ok "$legacy" legacy

summary="$tmp/grand.env"
cat >"$summary" <<'EOF'
B300_GRAND_STAGEN_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEN_OK=1
B300_GRAND_STAGEN_UPSTREAM_KIND=stagem
B300_GRAND_STAGEN_PAIR_POLICY=cs
B300_GRAND_STAGEN_BLOCK_POLICY=cg
B300_GRAND_STAGEN_BASE_COUNT_POLICY=cg
EOF
valid="$tmp/valid.env"
cat >"$valid" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGEL_ACCEPTED=1
B300_GRAND_SELECTED_STAGEM_ACCEPTED=1
B300_GRAND_SELECTED_STAGEN_ENABLED=1
B300_GRAND_SELECTED_STAGEN_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEN_ACCEPTED=1
B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND=stagem
B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cs
B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY=cg
B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP=1.007
B300_GRAND_SELECTED_STAGEN_SEARCH_PAIR_POLICIES='default cg cs'
B300_GRAND_SELECTED_STAGEN_SEARCH_BLOCK_POLICIES='default cg cs'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
run_ok "$valid" valid

unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"
sed -i 's/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cs/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cg/' "$unchanged"
run_bad "$unchanged" unchanged 'accepted Stage N retained inherited pair/block baseline'

mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"
sed -i 's/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cs/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=default/' "$mismatch"
run_bad "$mismatch" mismatch 'Stage-N pair policy differs from grand summary'

block_mismatch="$tmp/block-mismatch.env"; cp "$valid" "$block_mismatch"
sed -i 's/B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY=cg/B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY=default/' "$block_mismatch"
run_bad "$block_mismatch" block-mismatch 'Stage-N block policy differs from grand summary'

slow="$tmp/slow.env"; cp "$valid" "$slow"
sed -i 's/B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP=1.007/B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP=1.001/' "$slow"
run_bad "$slow" slow 'accepted Stage-N speedup below threshold'

missing_base="$tmp/missing-base.env"; cp "$valid" "$missing_base"
sed -i "s/B300_GRAND_SELECTED_STAGEN_SEARCH_PAIR_POLICIES='default cg cs'/B300_GRAND_SELECTED_STAGEN_SEARCH_PAIR_POLICIES='default cs'/" "$missing_base"
run_bad "$missing_base" missing-base 'Stage-N search policy set omits inherited baseline'

m_not_accepted="$tmp/m-not-accepted.env"; cp "$valid" "$m_not_accepted"
sed -i 's/B300_GRAND_SELECTED_STAGEM_ACCEPTED=1/B300_GRAND_SELECTED_STAGEM_ACCEPTED=0/' "$m_not_accepted"
run_bad "$m_not_accepted" m-not-accepted 'Stage N uses Stage M without Stage-M acceptance'

summary_l="$tmp/grand-l.env"
sed 's/B300_GRAND_STAGEN_UPSTREAM_KIND=stagem/B300_GRAND_STAGEN_UPSTREAM_KIND=stagel/' "$summary" >"$summary_l"
valid_l="$tmp/valid-l.env"
sed -e "s|B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=.*|B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary_l")|" \
    -e 's/B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND=stagem/B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND=stagel/' \
    -e 's/B300_GRAND_SELECTED_STAGEM_ACCEPTED=1/B300_GRAND_SELECTED_STAGEM_ACCEPTED=0/' "$valid" >"$valid_l"
run_ok "$valid_l" valid-l

l_not_accepted="$tmp/l-not-accepted.env"; cp "$valid_l" "$l_not_accepted"
sed -i 's/B300_GRAND_SELECTED_STAGEL_ACCEPTED=1/B300_GRAND_SELECTED_STAGEL_ACCEPTED=0/' "$l_not_accepted"
run_bad "$l_not_accepted" l-not-accepted 'Stage N uses Stage L without Stage-L acceptance'

[[ "$(find "$tmp" -name '*.called' | wc -l)" == 3 ]] || { echo 'only legacy + valid M + valid L should reach the base promoter' >&2; exit 4; }
echo 'b300-grand-stagen-exact-promotion-preflight OK legacy_schema3=1 stagen_to_stagem=1 stagen_to_stagel=1 inherited_baseline_change_required=1 threshold_gate=1 search_baseline_gate=1 grand_summary_pair_block_match=1 single_complete_prime=1 delegation=1 gpu_work=0'
