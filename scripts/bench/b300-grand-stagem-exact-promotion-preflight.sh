#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagem.sh"
[[ -f "$WRAP" ]] || exit 2
bash -n "$WRAP"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagem-exact.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake-promoter.sh"; calls="$tmp/calls"
cat >"$fake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${SELECTED_ENV:?}" >>"${FAKE_CALLS:?}"
[[ "${1:-}" == 27 ]] || exit 9
exit 0
EOF
chmod +x "$fake"

run_ok(){ local envf="$1"; FAKE_CALLS="$calls" SELECTED_ENV="$envf" BASE_PROMOTER="$fake" bash "$WRAP" 27 >/dev/null 2>"$tmp/ok.err"; }
run_bad(){ local envf="$1"; set +e; FAKE_CALLS="$calls" SELECTED_ENV="$envf" BASE_PROMOTER="$fake" bash "$WRAP" 27 >/dev/null 2>"$tmp/bad.err"; local rc=$?; set -e; ((rc!=0)) || { echo "expected rejection: $envf" >&2; exit 3; }; }

# Old schema-3 selections did not carry Stage L/M keys and remain compatible.
old="$tmp/old.env"
cat >"$old" <<'EOF'
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
EOF
run_ok "$old"

grand="$tmp/grand.env"
cat >"$grand" <<'EOF'
B300_GRAND_PREPARED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEL_INTEGRATED=1
B300_GRAND_STAGEL_OK=1
B300_GRAND_STAGEL_PROFILE=pp
B300_GRAND_STAGEL_SELF_GUARD=predicated
B300_GRAND_STAGEL_MATE_GUARD=predicated
B300_GRAND_STAGEM_INTEGRATED=1
B300_GRAND_STAGEM_OK=1
B300_GRAND_STAGEM_POLICY=cg
EOF
new="$tmp/new.env"
cat >"$new" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$grand")
B300_GRAND_SELECTED_STAGEL_ENABLED=1
B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEL_ACCEPTED=1
B300_GRAND_SELECTED_STAGEL_PROFILE=pp
B300_GRAND_SELECTED_STAGEL_SELF_GUARD=predicated
B300_GRAND_SELECTED_STAGEL_MATE_GUARD=predicated
B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES='bb pb bp pp'
B300_GRAND_SELECTED_STAGEM_ENABLED=1
B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEM_ACCEPTED=1
B300_GRAND_SELECTED_STAGEM_POLICY=cg
B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP=1.004
B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES='default cg cs'
EOF
run_ok "$new"

# Stage M can only exist downstream of an accepted Stage L candidate.
bad_chain="$tmp/bad-chain.env"; cp "$new" "$bad_chain"; sed -i 's/B300_GRAND_SELECTED_STAGEL_ACCEPTED=1/B300_GRAND_SELECTED_STAGEL_ACCEPTED=0/' "$bad_chain"
run_bad "$bad_chain"
grep -Fq 'Stage-M accepted without Stage-L acceptance' "$tmp/bad.err" || { cat "$tmp/bad.err" >&2; exit 3; }

# Selected policy and immutable grand-summary policy must agree.
bad_policy="$tmp/bad-policy.env"; cp "$new" "$bad_policy"; sed -i 's/B300_GRAND_SELECTED_STAGEM_POLICY=cg/B300_GRAND_SELECTED_STAGEM_POLICY=cs/' "$bad_policy"
run_bad "$bad_policy"
grep -Fq 'Stage-M policy differs from grand summary' "$tmp/bad.err" || { cat "$tmp/bad.err" >&2; exit 3; }

# An accepted Stage-L refinement cannot claim the bb control profile.
bad_l="$tmp/bad-l.env"; cp "$new" "$bad_l"; sed -i 's/B300_GRAND_SELECTED_STAGEL_PROFILE=pp/B300_GRAND_SELECTED_STAGEL_PROFILE=bb/' "$bad_l"
run_bad "$bad_l"
grep -Fq 'accepted Stage L cannot retain bb control profile' "$tmp/bad.err" || { cat "$tmp/bad.err" >&2; exit 3; }

[[ "$(wc -l <"$calls")" == 2 ]] || { echo 'fake promoter should run only for old+valid contracts' >&2; exit 3; }
echo 'b300-grand-stagem-exact-promotion-preflight OK legacy_schema3=1 stage_l_semantics=1 stage_m_requires_l=1 summary_policy_binding=1 rejection_before_promoter=1 gpu_work=0'
