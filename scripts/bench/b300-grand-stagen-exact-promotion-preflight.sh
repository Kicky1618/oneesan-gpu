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
  'Stage N uses Stage M without Stage-M acceptance' \
  'B300_GRAND_STAGEN_INTEGRATED' \
  'B300_GRAND_COMPLETE_PRIME_RACES' \
  'Stage-N pair policy differs from grand summary' \
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

legacy="$tmp/legacy.env"
cat >"$legacy" <<'EOF'
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
EOF
FAKE_CALLED="$tmp/legacy.called" SELECTED_ENV="$legacy" BASE_PROMOTER="$FAKE" bash "$PROMOTE" 27 >/dev/null
[[ -s "$tmp/legacy.called" ]] || { echo 'legacy selection was not delegated' >&2; exit 4; }

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
FAKE_CALLED="$tmp/valid.called" SELECTED_ENV="$valid" BASE_PROMOTER="$FAKE" bash "$PROMOTE" 27 >/dev/null
[[ -s "$tmp/valid.called" ]] || { echo 'valid Stage-N selection was not delegated' >&2; exit 4; }

unchanged="$tmp/unchanged.env"; cp "$valid" "$unchanged"
sed -i 's/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cs/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cg/' "$unchanged"
set +e
FAKE_CALLED="$tmp/unchanged.called" SELECTED_ENV="$unchanged" BASE_PROMOTER="$FAKE" bash "$PROMOTE" 27 >"$tmp/unchanged.out" 2>"$tmp/unchanged.err"
rc=$?
set -e
((rc!=0)) || { echo 'unchanged Stage-N policy unexpectedly accepted' >&2; exit 4; }
grep -Fq 'accepted Stage N retained inherited pair/block baseline' "$tmp/unchanged.err"
[[ ! -e "$tmp/unchanged.called" ]] || { echo 'invalid unchanged Stage N reached base promoter' >&2; exit 4; }

mismatch="$tmp/mismatch.env"; cp "$valid" "$mismatch"
sed -i 's/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=cs/B300_GRAND_SELECTED_STAGEN_PAIR_POLICY=default/' "$mismatch"
set +e
FAKE_CALLED="$tmp/mismatch.called" SELECTED_ENV="$mismatch" BASE_PROMOTER="$FAKE" bash "$PROMOTE" 27 >"$tmp/mismatch.out" 2>"$tmp/mismatch.err"
rc=$?
set -e
((rc!=0)) || { echo 'Stage-N summary mismatch unexpectedly accepted' >&2; exit 4; }
grep -Fq 'Stage-N pair policy differs from grand summary' "$tmp/mismatch.err"
[[ ! -e "$tmp/mismatch.called" ]] || { echo 'summary mismatch reached base promoter' >&2; exit 4; }

echo 'b300-grand-stagen-exact-promotion-preflight OK legacy_schema3=1 stagen_to_stagem=1 inherited_baseline_change_required=1 grand_summary_match=1 single_complete_prime=1 delegation=1 gpu_work=0'
