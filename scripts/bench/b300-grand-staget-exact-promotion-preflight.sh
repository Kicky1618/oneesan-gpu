#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
PROM="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-staget.sh"
[[ -s "$PROM" ]] || exit 2; bash -n "$PROM"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-staget-exact.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
cat >"$tmp/base.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
echo BASE_CALLED
SH
chmod +x "$tmp/base.sh"
echo payload >"$tmp/payload"; sha256sum "$tmp/payload" >"$tmp/manifest"
manifest="$tmp/manifest"
make_summary(){ local f="$1" up="$2" lp="$3" lb="$4" lpl2="$5" lbl2="$6" hp="$7" hb="$8" hpl2="$9"; shift 9; local hbl2="$1" hm="$2" hml2="$3" pol="$4" man="$5"; cat >"$f" <<EOF
B300_GRAND_STAGET_INTEGRATED=1
B300_GRAND_COMPLETE_PRIME_RACES=1
B300_GRAND_STAGET_OK=1
B300_GRAND_STAGET_UPSTREAM_KIND=$up
B300_GRAND_STAGET_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_STAGET_LOW_PAIR_POLICY=$lp
B300_GRAND_STAGET_LOW_BLOCK_POLICY=$lb
B300_GRAND_STAGET_LOW_PAIR_L2_BYTES=$lpl2
B300_GRAND_STAGET_LOW_BLOCK_L2_BYTES=$lbl2
B300_GRAND_STAGET_HIGH_PAIR_POLICY=$hp
B300_GRAND_STAGET_HIGH_BLOCK_POLICY=$hb
B300_GRAND_STAGET_HIGH_PAIR_L2_BYTES=$hpl2
B300_GRAND_STAGET_HIGH_BLOCK_L2_BYTES=$hbl2
B300_GRAND_STAGET_HIGH_MATE_POLICY=$hm
B300_GRAND_STAGET_HIGH_MATE_L2_BYTES=$hml2
B300_GRAND_STAGET_POLICY=$pol
B300_GRAND_STAGET_MANIFEST=$man
EOF
}
make_selected(){ local f="$1" summary="$2" s_ok="$3" up="$4" lp="$5" lb="$6" lpl2="$7" lbl2="$8" hp="$9"; shift 9; local hb="$1" hpl2="$2" hbl2="$3" hm="$4" hml2="$5" pol="$6"; cat >"$f" <<EOF
B300_GRAND_SELECTED_SCHEMA=3
B300_GRAND_SELECTED_STAGET_ENABLED=1
B300_GRAND_SELECTED_STAGET_MIN_SPEEDUP=1.002
B300_GRAND_SELECTED_STAGET_ACCEPTED=1
B300_GRAND_SELECTED_STAGET_UPSTREAM_KIND=$up
B300_GRAND_SELECTED_STAGET_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_SELECTED_STAGET_LOW_PAIR_POLICY=$lp
B300_GRAND_SELECTED_STAGET_LOW_BLOCK_POLICY=$lb
B300_GRAND_SELECTED_STAGET_LOW_PAIR_L2_BYTES=$lpl2
B300_GRAND_SELECTED_STAGET_LOW_BLOCK_L2_BYTES=$lbl2
B300_GRAND_SELECTED_STAGET_HIGH_PAIR_POLICY=$hp
B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_POLICY=$hb
B300_GRAND_SELECTED_STAGET_HIGH_PAIR_L2_BYTES=$hpl2
B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_L2_BYTES=$hbl2
B300_GRAND_SELECTED_STAGET_HIGH_MATE_POLICY=$hm
B300_GRAND_SELECTED_STAGET_HIGH_MATE_L2_BYTES=$hml2
B300_GRAND_SELECTED_STAGET_POLICY=$pol
B300_GRAND_SELECTED_STAGET_STAGED_SPEEDUP=1.010
B300_GRAND_SELECTED_STAGET_SEARCH_POLICIES='default cg cs'
B300_GRAND_SELECTED_STAGER_ACCEPTED=1
B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND=stageq
B300_GRAND_SELECTED_STAGER_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGER_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY=cg
B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES=256
B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES=64
B300_GRAND_SELECTED_STAGES_ACCEPTED=$s_ok
B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY=cg
B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY=default
B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES=128
B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES=0
B300_GRAND_SELECTED_STAGEN_MATE_LOAD_POLICY=cg
B300_GRAND_SELECTED_STAGEP_ACCEPTED=1
B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES=128
B300_GRAND_SELECTED_GRAND_SUMMARY_ENV=$summary
EOF
}
run_ok(){ local sel="$1"; SELECTED_ENV="$sel" BASE_PROMOTER="$tmp/base.sh" bash "$PROM" 27 >"$tmp/ok.out" 2>"$tmp/ok.err"; grep -Fq 'BASE_CALLED' "$tmp/ok.out"; grep -Fq 'Stage-T exact provenance OK has_t=1' "$tmp/ok.err"; }
run_fail(){ local sel="$1" needle="$2"; set +e; SELECTED_ENV="$sel" BASE_PROMOTER="$tmp/base.sh" bash "$PROM" 27 >"$tmp/fail.out" 2>"$tmp/fail.err"; rc=$?; set -e; ((rc!=0)) || { echo "expected Stage-T exact failure: $needle" >&2; exit 3; }; ! grep -Fq 'BASE_CALLED' "$tmp/fail.out" || { echo 'bad Stage-T contract reached base promoter' >&2; exit 3; }; grep -Fq "$needle" "$tmp/fail.err" || { cat "$tmp/fail.err" >&2; exit 3; }; }
# Valid T -> exact R immediate upstream.
make_summary "$tmp/sr.sum" stager cg default 0 0 cg cg 256 64 cg 128 cs "$manifest"
make_selected "$tmp/sr.env" "$tmp/sr.sum" 0 stager cg default 0 0 cg cg 256 64 cg 128 cs
run_ok "$tmp/sr.env"
# Valid T -> exact S immediate upstream, preserving S low Count tuple.
make_summary "$tmp/ss.sum" stages cg default 128 0 cg cg 256 64 cg 128 cs "$manifest"
make_selected "$tmp/ss.env" "$tmp/ss.sum" 1 stages cg default 128 0 cg cg 256 64 cg 128 cs
run_ok "$tmp/ss.env"
# S rejected but T claims S upstream.
make_selected "$tmp/bad-up.env" "$tmp/ss.sum" 0 stages cg default 128 0 cg cg 256 64 cg 128 cs
run_fail "$tmp/bad-up.env" 'Stage T ignored maximal immediate upstream'
# High mate provenance must preserve accepted P's 128B hint.
make_selected "$tmp/bad-mate.env" "$tmp/sr.sum" 0 stager cg default 0 0 cg cg 256 64 cg 64 cs
run_fail "$tmp/bad-mate.env" 'Stage-T high mate provenance drift'
# Summary cannot shadow selected policy.
make_summary "$tmp/bad-pol.sum" stager cg default 0 0 cg cg 256 64 cg 128 cg "$manifest"
make_selected "$tmp/bad-pol.env" "$tmp/bad-pol.sum" 0 stager cg default 0 0 cg cg 256 64 cg 128 cs
run_fail "$tmp/bad-pol.env" 'Stage-T selected policy differs from grand summary'
# Corrupted promotion manifest must fail before exact continuation.
echo bad >>"$tmp/payload"
make_summary "$tmp/bad-man.sum" stager cg default 0 0 cg cg 256 64 cg 128 cs "$manifest"
make_selected "$tmp/bad-man.env" "$tmp/bad-man.sum" 0 stager cg default 0 0 cg cg 256 64 cg 128 cs
run_fail "$tmp/bad-man.env" 'Stage-T promotion manifest failed before exact continuation'
echo 'b300-grand-staget-exact-promotion-preflight OK valid_t_to_r=1 valid_t_to_s=1 maximal_upstream=1 high_mate_lock=1 summary_lock=1 manifest_lock=1 base_not_called_on_invalid=1 gpu_work=0'
