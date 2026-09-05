#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"
NGPU="${NGPU:-8}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-sync}"; PM_ACCUM="${PM_ACCUM:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_bucket_closure_key4_ab_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}}"
R0="${R0:-${PREFIX}_scalar.tsv}"; R1="${R1:-${PREFIX}_key4.tsv}"; OUT="${OUT:-${PREFIX}_compare.tsv}"
L0="${L0:-${PREFIX}_scalar_logs}"; L1="${L1:-${PREFIX}_key4_logs}"

run_mode(){
  local mode="$1" result="$2" logdir="$3"
  echo "=== closure key A/B ternary_key4=$mode ===" >&2
  N="$N" MOD="$MOD" EXPECT="$EXPECT" NGPU="$NGPU" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$mode" RESULT="$result" LOGDIR="$logdir" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-bucket-closure-ab.sh" >/dev/null
}

run_mode 0 "$R0" "$L0"
run_mode 1 "$R1" "$L1"

awk -F '\t' -v OFS='\t' '
function ratio(a,b){return (a=="NA"||b=="NA"||a==""||b==""||b+0==0)?"NA":sprintf("%.6f",(a+0)/(b+0))}
function delta(a,b){return (a=="NA"||b=="NA"||a==""||b=="")?"NA":sprintf("%.6f",(a+0)-(b+0))}
NR==FNR {
  if(FNR==1) next
  seen[$1]=1; wall0[$1]=$4; fh0[$1]=$5; fl0[$1]=$6; rl0[$1]=$7; rh0[$1]=$8; ts0[$1]=$9
  next
}
FNR==1 {
  print "backend","scalar_wall_s","key4_wall_s","key4_speedup","wall_saved_s","forward_high_speedup","forward_low_speedup","reverse_low_speedup","reverse_high_speedup","transpose_ratio"
  next
}
{
  b=$1
  if(!seen[b]) {print "missing scalar backend " b > "/dev/stderr"; bad=1; next}
  print b,wall0[b],$4,ratio(wall0[b],$4),delta(wall0[b],$4),ratio(fh0[b],$5),ratio(fl0[b],$6),ratio(rl0[b],$7),ratio(rh0[b],$8),ratio(ts0[b],$9)
  got[b]=1
}
END {
  for(b in seen) if(!got[b]) {print "missing key4 backend " b > "/dev/stderr"; bad=1}
  if(bad) exit 3
}
' "$R0" "$R1" >"$OUT"

cat "$OUT"
echo "closure-key4-ab OK n=$N transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM scalar=$R0 key4=$R1 compare=$OUT" >&2
