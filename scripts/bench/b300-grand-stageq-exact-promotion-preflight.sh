#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stageq.sh"
[[ -f "$PROMOTE" ]] || exit 2
bash -n "$PROMOTE"
for s in \
  'B300_GRAND_SELECTED_STAGEQ_ACCEPTED' \
  'Stage Q accepted without Stage-N acceptance' \
  'Stage Q ignored accepted Stage P upstream' \
  'Stage Q ignored accepted Stage O upstream' \
  'Stage Q upstream must fall back to Stage N' \
  'accepted Stage Q retained exact upstream Count L2 tuple' \
  'B300_GRAND_STAGEQ_INTEGRATED' \
  'Stage-Q upstream Count L2 differs from grand summary' \
  'Stage-Q selected Count L2 differs from grand summary' \
  'sha256sum -c "$B300_GRAND_STAGEQ_MANIFEST"' \
  'b300x8-grand-promote-exact-stagep.sh'; do
  grep -Fq "$s" "$PROMOTE" || { echo "Stage-Q exact marker missing: $s" >&2; exit 3; }
done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageq-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
fake="$tmp/fake.sh"
cat >"$fake" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf called >"${FAKE_CALLED:?}"
SH
chmod +x "$fake"
run_ok(){ FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >/dev/null; [[ -s "$tmp/$2.called" ]] || exit 4; }
run_bad(){ set +e; FAKE_CALLED="$tmp/$2.called" SELECTED_ENV="$1" BASE_PROMOTER="$fake" bash "$PROMOTE" 27 >"$tmp/$2.out" 2>"$tmp/$2.err"; rc=$?; set -e; ((rc!=0)) || { echo "expected rejection case=$2" >&2; exit 4; }; grep -Fq "$3" "$tmp/$2.err" || { cat "$tmp/$2.err" >&2; exit 4; }; [[ ! -e "$tmp/$2.called" ]] || { echo "promoter called after rejected case=$2" >&2; exit 4; }; }

legacy="$tmp/legacy.env"
printf 'B300_GRAND_SELECTED_SCHEMA=3\nB300_GRAND_SELECTED_VALIDATED=1\n' >"$legacy"
run_ok "$legacy" legacy
payload="$tmp/payload"; printf 'q\n' >"$payload"
manifest="$tmp/q.sha"; sha256sum "$payload" >"$manifest"

make_summary(){
  local out="$1" upstream="$2" up_pair="$3" up_block="$4" pair="$5" block="$6"
  cat >"$out" <<EOF
B300_GRAND_STAGEQ_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGEQ_OK=1
B300_GRAND_STAGEQ_UPSTREAM_KIND=$upstream
B300_GRAND_STAGEQ_UPSTREAM_PAIR_L2_BYTES=$up_pair
B300_GRAND_STAGEQ_UPSTREAM_BLOCK_L2_BYTES=$up_block
B300_GRAND_STAGEQ_PAIR_L2_BYTES=$pair
B300_GRAND_STAGEQ_BLOCK_L2_BYTES=$block
B300_GRAND_STAGEQ_MANIFEST=$(printf '%q' "$manifest")
EOF
}
make_selected(){
  local out="$1" n="$2" o="$3" p="$4" upstream="$5" up_pair="$6" up_block="$7" pair="$8" block="$9" summary="${10}"
  cat >"$out" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_VALIDATED=1
B300_GRAND_SELECTED_STAGEN_ACCEPTED=$n
B300_GRAND_SELECTED_STAGEO_ACCEPTED=$o
B300_GRAND_SELECTED_STAGEP_ACCEPTED=$p
B300_GRAND_SELECTED_STAGEQ_ENABLED=1
B300_GRAND_SELECTED_STAGEQ_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGEQ_ACCEPTED=1
B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=$upstream
B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES=$up_pair
B300_GRAND_SELECTED_STAGEQ_UPSTREAM_BLOCK_L2_BYTES=$up_block
B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=$pair
B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES=$block
B300_GRAND_SELECTED_STAGEQ_STAGED_SPEEDUP=1.006
B300_GRAND_SELECTED_STAGEQ_SEARCH_PAIR_L2='0 64 128 256'
B300_GRAND_SELECTED_STAGEQ_SEARCH_BLOCK_L2='0 64 128 256'
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$(printf '%q' "$summary")
EOF
}

p_summary="$tmp/grand-p.env"; make_summary "$p_summary" stagep 128 64 256 64
p_valid="$tmp/p-valid.env"; make_selected "$p_valid" 1 1 1 stagep 128 64 256 64 "$p_summary"
run_ok "$p_valid" p-valid
no_n="$tmp/no-n.env"; cp "$p_valid" "$no_n"; sed -i 's/B300_GRAND_SELECTED_STAGEN_ACCEPTED=1/B300_GRAND_SELECTED_STAGEN_ACCEPTED=0/' "$no_n"
run_bad "$no_n" no-n 'Stage Q accepted without Stage-N acceptance'
wrong_p="$tmp/wrong-p.env"; cp "$p_valid" "$wrong_p"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagep/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stageo/' "$wrong_p"
run_bad "$wrong_p" wrong-p 'Stage Q ignored accepted Stage P upstream'

o_summary="$tmp/grand-o.env"; make_summary "$o_summary" stageo 64 256 64 128
o_valid="$tmp/o-valid.env"; make_selected "$o_valid" 1 1 0 stageo 64 256 64 128 "$o_summary"
run_ok "$o_valid" o-valid
wrong_o="$tmp/wrong-o.env"; cp "$o_valid" "$wrong_o"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stageo/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagen/' "$wrong_o"
run_bad "$wrong_o" wrong-o 'Stage Q ignored accepted Stage O upstream'

n_summary="$tmp/grand-n.env"; make_summary "$n_summary" stagen 128 128 256 128
n_valid="$tmp/n-valid.env"; make_selected "$n_valid" 1 0 0 stagen 128 128 256 128 "$n_summary"
run_ok "$n_valid" n-valid
wrong_n="$tmp/wrong-n.env"; cp "$n_valid" "$wrong_n"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stagen/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND=stageo/' "$wrong_n"
run_bad "$wrong_n" wrong-n 'Stage Q upstream must fall back to Stage N'

unchanged="$tmp/unchanged.env"; make_selected "$unchanged" 1 1 1 stagep 128 64 128 64 "$p_summary"
run_bad "$unchanged" unchanged 'accepted Stage Q retained exact upstream Count L2 tuple'
mismatch_up="$tmp/mismatch-up.env"; cp "$p_valid" "$mismatch_up"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES=128/B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES=64/' "$mismatch_up"
run_bad "$mismatch_up" mismatch-up 'Stage-Q upstream Count L2 differs from grand summary'
mismatch_sel="$tmp/mismatch-sel.env"; cp "$p_valid" "$mismatch_sel"; sed -i 's/B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=256/B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES=0/' "$mismatch_sel"
run_bad "$mismatch_sel" mismatch-sel 'Stage-Q selected Count L2 differs from grand summary'

echo corrupt >>"$payload"
run_bad "$p_valid" manifest 'Stage-Q promotion manifest failed before exact continuation'
[[ "$(find "$tmp" -name '*.called' | wc -l)" == 4 ]] || { find "$tmp" -name '*.called' >&2; exit 4; }
echo 'b300-grand-stageq-exact-promotion-preflight OK legacy_schema3=1 stageq_requires_stagen=1 maximal_upstream_P=1 maximal_upstream_O=1 maximal_upstream_N=1 tuple_change_required=1 summary_match=1 manifest_gate=1 rejection_before_promoter=1 gpu_work=0'
