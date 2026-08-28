#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}";CAL_RUNS="${CAL_RUNS:-1}";CAL_ROWS="${CAL_ROWS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}";B300_STAGED_META_MAX_MIB="${B300_STAGED_META_MAX_MIB:-512}";B300_STAGED_INTERVAL_MAX_MIB="${B300_STAGED_INTERVAL_MAX_MIB:-256}"
CAL_MIN_WALL_SPEEDUP="${CAL_MIN_WALL_SPEEDUP:-1.005}";CAL_MIN_ACTIVE_SPEEDUP="${CAL_MIN_ACTIVE_SPEEDUP:-1.005}";CAL_MIN_TOTAL_SPEEDUP="${CAL_MIN_TOTAL_SPEEDUP:-0.980}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_static_lpt_staged_intervals_ab}";CAL_RESULT="${CAL_RESULT:-${PREFIX}_cal.tsv}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"
((RUNS>=1&&CAL_RUNS>=1&&CAL_ROWS>=1&&CAL_ROWS<=28))||exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 static LPT staged interval A/B";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh" >"$LOGDIR/meta-plan-proof.out" 2>"$LOGDIR/meta-plan-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh" >"$LOGDIR/local-meta-proof.out" 2>"$LOGDIR/local-meta-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-interval-staging-proof.sh" >"$LOGDIR/interval-stage-proof.out" 2>"$LOGDIR/interval-stage-proof.err"
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"

BIN_BASE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_static_lpt_interval_base"
BIN_STAGE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_static_lpt_interval_stage"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_BASE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_STAGE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-staged-intervals.sh" >"$LOGDIR/stage.build.out" 2>"$LOGDIR/stage.build.err"
grep -Fq 'staged_intervals=1 per_group_interval_h2d=0' "$LOGDIR/stage.build.out"
grep -Fq 'expected_default_descriptors=8453518' "$LOGDIR/stage.build.out"

printf 'mode\trun\trows\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\n' >"$CAL_RESULT"
printf 'mode\trun\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\tinterval_descriptors\tinterval_max_mib_per_gpu\tinterval_total_h2d_gib\tcombined_stage_max_mib_per_gpu\n' >"$RESULT"
run_one(){
 local mode="$1" run="$2" rows="$3" dest="$4" bin;[[ "$mode" == base ]]&&bin="$BIN_BASE"||bin="$BIN_STAGE"
 local out="$LOGDIR/${dest}_${mode}_${run}.out" err="$LOGDIR/${dest}_${mode}_${run}.err"
 B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" B300_STAGED_INTERVAL_MAX_MIB="$B300_STAGED_INTERVAL_MAX_MIB" GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
 grep -Fq "B300 row limit: rows=${rows}/28 calibration=$((rows<28?1:0))" "$err"||{ tail -n120 "$err" >&2;exit 3; }
 local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n120 "$err" >&2;exit 4; }
 field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
 local residue="$(field residue)" wall="$(field wall_s)" prep="$(field prepare_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" ints="$(field max_intervals)";[[ -n "$residue"&&-n "$wall"&&-n "$prep"&&-n "$active"&&-n "$sum"&&-n "$ints" ]]||exit 5
 local total;total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
 if [[ "$dest" == cal ]];then printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$rows" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" >>"$CAL_RESULT";return;fi
 local desc='-' maxm='-' h2d='-' combined='-'
 if [[ "$mode" == stage ]];then
  local il;il="$(grep '^static LPT staged intervals: ' "$err"|tail -n1||true)";[[ -n "$il" ]]||{ echo "missing staged interval runtime line" >&2;exit 6; }
  ifield(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$il"; }
  desc="$(ifield descriptors)";maxm="$(ifield max_mib_per_gpu)";h2d="$(ifield total_h2d_gib)";combined="$(ifield combined_stage_max_mib_per_gpu)"
  [[ "$desc" == 8453518 ]]||{ echo "unexpected staged interval count $desc" >&2;exit 7; }
  grep -Fq 'copy_mode=H2D_once_local_then_zero_interval_copy_per_group scheduler=static_lpt' <<<"$il"||exit 8
 fi
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" "$desc" "$maxm" "$h2d" "$combined" >>"$RESULT"
}
for((r=1;r<=CAL_RUNS;++r));do if((r&1));then order=(base stage);else order=(stage base);fi;for m in "${order[@]}";do run_one "$m" "$r" "$CAL_ROWS" cal;done;done
cat "$CAL_RESULT"
read -r CAL_WALL CAL_ACTIVE CAL_TOTAL < <(python3 - "$CAL_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(m,k):return statistics.median(float(x[k]) for x in r if x['mode']==m)
res={m:{x['residue'] for x in r if x['mode']==m} for m in ('base','stage')}
if any(len(v)!=1 for v in res.values()) or res['base']!=res['stage']:raise SystemExit(f'cal residue mismatch {res}')
print(f"{med('base','wall_s')/med('stage','wall_s'):.9f} {med('base','active_max_s')/med('stage','active_max_s'):.9f} {med('base','total_s')/med('stage','total_s'):.9f}")
PY
)
echo "b300_interval_stage_cal wall_speedup=${CAL_WALL}x active_max_speedup=${CAL_ACTIVE}x total_speedup=${CAL_TOTAL}x" >&2
set +e
python3 - "$CAL_WALL" "$CAL_MIN_WALL_SPEEDUP" "$CAL_ACTIVE" "$CAL_MIN_ACTIVE_SPEEDUP" "$CAL_TOTAL" "$CAL_MIN_TOTAL_SPEEDUP" <<'PY'
import sys
w,wm,a,am,t,tm=map(float,sys.argv[1:]);raise SystemExit(0 if w>=wm and a>=am and t>=tm else 9)
PY
gate_rc=$?;set -e
case "$gate_rc" in 0);;9)echo "b300-vmm-static-lpt-staged-intervals-production-ab SKIP_FULL cal_wall=${CAL_WALL}x cal_active=${CAL_ACTIVE}x cal_total=${CAL_TOTAL}x";exit 0;;*)exit "$gate_rc";;esac
for((r=1;r<=RUNS;++r));do if((r&1));then order=(base stage);else order=(stage base);fi;for m in "${order[@]}";do run_one "$m" "$r" 28 full;done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$CAL_WALL" "$CAL_ACTIVE" "$CAL_TOTAL" <<'PY'
import csv,statistics,sys
src,dst,cw,ca,ct=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for m in ('base','stage'):
 rs=[x for x in rows if x['mode']==m];rr={x['residue'] for x in rs}
 if len(rr)!=1:raise SystemExit(f'unstable residue {m}: {rr}')
 res[m]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in rs);d=dict(mode=m,runs=len(rs),residue=res[m],median_wall_s=f'{med("wall_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',median_total_s=f'{med("total_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in rs));
 if m=='stage':
  for k in ('interval_descriptors','interval_max_mib_per_gpu','interval_total_h2d_gib','combined_stage_max_mib_per_gpu'):d[k]=rs[0][k]
 out.append(d)
if len(set(res.values()))!=1:raise SystemExit(f'full residue mismatch {res}')
keys=[]
for x in out:
 for k in x:
  if k not in keys:keys.append(k)
with open(dst,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};b=q['base'];s=q['stage'];bw=float(b['median_wall_s']);sw=float(s['median_wall_s']);bt=float(b['median_total_s']);st=float(s['median_total_s']);ba=float(b['median_active_max_s']);sa=float(s['median_active_max_s'])
print(f'b300_interval_stage_cal_wall_speedup={float(cw):.6f}x')
print(f'b300_interval_stage_cal_active_max_speedup={float(ca):.6f}x')
print(f'b300_interval_stage_cal_total_speedup={float(ct):.6f}x')
print(f'b300_interval_stage_full_wall_speedup={bw/sw:.6f}x')
print(f'b300_interval_stage_full_total_speedup={bt/st:.6f}x')
print(f'b300_interval_stage_full_active_max_speedup={ba/sa:.6f}x')
print(f'b300_interval_stage_full_total_delta_pct={(st/bt-1)*100:.4f}%')
print(f'b300_interval_stage_best={"stage" if st<bt else "base"}')
print('interval_h2d_old_full_gib=5.290623843670 interval_h2d_staged_gib=0.188950851560 interval_h2d_reduction=28x')
print(f'residue={res["base"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-static-lpt-staged-intervals-production-ab OK cal_rows=$CAL_ROWS cal_runs=$CAL_RUNS runs=$RUNS result=$RESULT" >&2
