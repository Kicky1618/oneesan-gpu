#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stages.sh"; [[ -f "$PROMOTE" ]] || exit 2; bash -n "$PROMOTE"
for s in 'readonly PINNED_SELECTED_ENV PINNED_BASE_PROMOTER' 'env -u BASE_PROMOTER SELECTED_ENV="$PINNED_SELECTED_ENV" "$PINNED_BASE_PROMOTER"'; do grep -Fq "$s" "$PROMOTE" || { echo "Stage-S control-path marker missing: $s" >&2; exit 3; }; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stages-control.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake.sh"; evil="$tmp/evil.sh"
cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf called >"${FAKE_CALLED:?}"
SH
cat >"$evil" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf evil >"${EVIL_CALLED:?}"
SH
chmod +x "$fake" "$evil"
base_selected(){
  local out="$1" summary="$2"
  cat >"$out" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGES_ENABLED=1
B300_GRAND_SELECTED_STAGES_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGES_ACCEPTED=0
B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND=
B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY=default
B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGES_HIGH_PAIR_POLICY=default
B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES=0
B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES=0
B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=0
B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES=0
B300_GRAND_SELECTED_STAGES_STAGED_SPEEDUP=1.0
B300_GRAND_SELECTED_STAGES_SEARCH_PAIR_L2='0 64'
B300_GRAND_SELECTED_STAGES_SEARCH_BLOCK_L2='0 64'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
}
run_redirect_bad(){
  local selected="$1" tag="$2"
  set +e
  FAKE_CALLED="$tmp/$tag.fake" EVIL_CALLED="$tmp/$tag.evil" SELECTED_ENV="$selected" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$tag.out" 2>"$tmp/$tag.err"
  rc=$?; set -e
  ((rc!=0)) || { echo "Stage-S control-path redirect unexpectedly succeeded tag=$tag" >&2; exit 4; }
  [[ ! -e "$tmp/$tag.fake" && ! -e "$tmp/$tag.evil" ]] || { echo "Stage-S control-path redirect reached promoter tag=$tag" >&2; exit 4; }
  grep -Fq 'readonly variable' "$tmp/$tag.err" || { cat "$tmp/$tag.err" >&2; echo "expected readonly rejection tag=$tag" >&2; exit 4; }
}
# Selected artifact cannot overwrite the pinned base promoter.
summary1="$tmp/summary1.env"; printf 'B300_GRAND_STAGES_INTEGRATED=1\nB300_GRAND_COMPLETE_PRIME_RACES=1\nB300_GRAND_STAGES_OK=0\n' >"$summary1"
selected1="$tmp/selected1.env"; base_selected "$selected1" "$summary1"; printf 'PINNED_BASE_PROMOTER=%q\n' "$evil" >>"$selected1"
run_redirect_bad "$selected1" selected
# Grand summary cannot overwrite the pinned base promoter either.
summary2="$tmp/summary2.env"; cat >"$summary2" <<EOF
PINNED_BASE_PROMOTER=$(printf '%q' "$evil")
B300_GRAND_STAGES_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGES_OK=0
EOF
selected2="$tmp/selected2.env"; base_selected "$selected2" "$summary2"
run_redirect_bad "$selected2" summary
echo 'b300-grand-stages-exact-controlpath-preflight OK selected_redirect_rejected=1 summary_redirect_rejected=1 pinned_readonly=1 base_env_unset_on_delegate=1 rejection_before_promoter=1 gpu_work=0'
