#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}";CAL_RUNS="${CAL_RUNS:-1}";CAL_ROWS="${CAL_ROWS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}";B300_STAGED_META_MAX_MIB="${B300_STAGED_META_MAX_MIB:-512}"
CAL_MIN_WALL_SPEEDUP="${CAL_MIN_WALL_SPEEDUP:-0.995}";CAL_MIN_ACTIVE_SPEEDUP="${CAL_MIN_ACTIVE_SPEEDUP:-0.995}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_static_lpt_meta_source_ab}";CAL_RESULT="${CAL_RESULT:-${PREFIX}_cal.tsv}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"
((RUNS>=1&&CAL_RUNS>=1&&CAL_ROWS>=1&&CAL_ROWS<=28))||exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 static LPT metadata-source A/B";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh" >"$LOGDIR/plan-proof.out" 2>"$LOGDIR/plan-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh" >"$LOGDIR/local-meta-proof.out" 2>"$LOGDIR/local-meta-proof.err"
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metaptr-ptx-proof.sh" >"$LOGDIR/ptr-ptx.out" 2>"$LOGDIR/ptr-ptx.err"
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metakernelarg-ptx-proof.sh" >"$LOGDIR/arg-ptx.out" 2>"$LOGDIR/arg-ptx.err"

BIN_CONST="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_meta_const_select"
BIN_PTR="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_meta_ptr_select"
BIN_ARG="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_meta_arg_select"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_CONST" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh" >"$LOGDIR/const.build.out" 2>"$LOGDIR/const.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_PTR" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-metaptr.sh" >"$LOGDIR/ptr.build.out" 2>"$LOGDIR/ptr.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_ARG" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-metakernelarg.sh" >"$LOGDIR/arg.build.out" 2>"$LOGDIR/arg.build.err"
grep -Fq 'per_group_meta_copy=D2D_sync' "$LOGDIR/const.build.out"
grep -Fq 'group_meta_constant_payload_bytes_per_group=8' "$LOGDIR/ptr.build.out"
grep -Fq 'per_group_meta_copy_bytes=0 per_group_meta_symbol_copy_calls=0' "$LOGDIR/arg.build.out"

printf 'mode\trun\trows\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\n' >"$CAL_RESULT"
printf 'mode\trun\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\n' >"$RESULT"
run_one(){
 local mode="$1" run="$2" rows="$3" dest="$4" bin
 case "$mode" in const)bin="$BIN_CONST";;ptr)bin="$BIN_PTR";;arg)bin="$BIN_ARG";;*)return 2;;esac
 local out="$LOGDIR/${dest}_${mode}_${run}.out" err="$LOGDIR/${dest}_${mode}_${run}.err"
 B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
 grep -Fq "B300 row limit: rows=${rows}/28 calibration=$((rows<28?1:0))" "$err"||{ tail -n120 "$err" >&2;exit 3; }
 local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n120 "$err" >&2;exit 4; }
 field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
 local residue="$(field residue)" wall="$(field wall_s)" prep="$(field prepare_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" ints="$(field max_intervals)";[[ -n "$residue"&&-n "$wall"&&-n "$prep"&&-n "$active"&&-n "$sum"&&-n "$ints" ]]||exit 5
 local total;total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
 local meta;meta="$(grep '^static LPT staged group meta: ' "$err"|tail -n1||true)";[[ -n "$meta" ]]||exit 6
 case "$mode" in
  const) grep -Fq 'copy_mode=H2D_once_local_then_D2D_per_group scheduler=static_lpt' <<<"$meta"||exit 7;;
  ptr) grep -Fq 'copy_mode=H2D_once_local_then_pointer_H2D_per_group scheduler=static_lpt meta_access=staged_global' <<<"$meta"||exit 8;;
  arg) grep -Fq 'copy_mode=H2D_once_local_then_no_meta_copy_per_group scheduler=static_lpt meta_access=staged_global_kernelarg' <<<"$meta"||exit 9;;
 esac
 if [[ "$dest" == cal ]];then printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$rows" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" >>"$CAL_RESULT";else printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" >>"$RESULT";fi
}
for((r=1;r<=CAL_RUNS;++r));do case $(((r-1)%3)) in 0)order=(const ptr arg);;1)order=(ptr arg const);;*)order=(arg const ptr);;esac;for m in "${order[@]}";do run_one "$m" "$r" "$CAL_ROWS" cal;done;done
cat "$CAL_RESULT"
read -r WINNER PTR_W PTR_A ARG_W ARG_A < <(python3 - "$CAL_RESULT" "$CAL_MIN_WALL_SPEEDUP" "$CAL_MIN_ACTIVE_SPEEDUP" <<'PY'
import csv,statistics,sys
src,wmin,amin=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]);r=list(csv.DictReader(open(src),delimiter='\t'))
def med(m,k):return statistics.median(float(x[k]) for x in r if x['mode']==m)
res={m:{x['residue'] for x in r if x['mode']==m} for m in ('const','ptr','arg')}
if any(len(v)!=1 for v in res.values()) or len({next(iter(v)) for v in res.values()})!=1:raise SystemExit(f'cal residue mismatch {res}')
cw,ca=med('const','wall_s'),med('const','active_max_s')
vals={}
for m in ('ptr','arg'):
 vals[m]=(cw/med(m,'wall_s'),ca/med(m,'active_max_s'),med(m,'wall_s'))
elig=[m for m,(w,a,_) in vals.items() if w>=wmin and a>=amin]
winner=min(elig,key=lambda m:vals[m][2]) if elig else 'none'
print(winner,f'{vals["ptr"][0]:.9f}',f'{vals["ptr"][1]:.9f}',f'{vals["arg"][0]:.9f}',f'{vals["arg"][1]:.9f}')
PY
)
echo "b300_meta_source_cal winner=$WINNER ptr_wall=${PTR_W}x ptr_active=${PTR_A}x arg_wall=${ARG_W}x arg_active=${ARG_A}x" >&2
if [[ "$WINNER" == none ]];then echo "b300-vmm-static-lpt-meta-source-production-ab SKIP_FULL no candidate passed calibration thresholds";exit 0;fi
for((r=1;r<=RUNS;++r));do if((r&1));then order=(const "$WINNER");else order=("$WINNER" const);fi;for m in "${order[@]}";do run_one "$m" "$r" 28 full;done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$WINNER" "$PTR_W" "$PTR_A" "$ARG_W" "$ARG_A" <<'PY'
import csv,statistics,sys
src,dst,winner,pw,pa,aw,aa=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for m in ('const',winner):
 rs=[x for x in rows if x['mode']==m];rr={x['residue'] for x in rs}
 if len(rr)!=1:raise SystemExit(f'unstable residue {m}: {rr}')
 res[m]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in rs);out.append(dict(mode=m,runs=len(rs),residue=res[m],median_wall_s=f'{med("wall_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',median_total_s=f'{med("total_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in rs)))
if len(set(res.values()))!=1:raise SystemExit(f'full residue mismatch {res}')
with open(dst,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};c=q['const'];n=q[winner];cw=float(c['median_wall_s']);nw=float(n['median_wall_s']);ct=float(c['median_total_s']);nt=float(n['median_total_s']);ca=float(c['median_active_max_s']);na=float(n['median_active_max_s'])
print(f'b300_meta_source_selected={winner}')
print(f'b300_meta_source_cal_ptr_wall_speedup={float(pw):.6f}x')
print(f'b300_meta_source_cal_ptr_active_speedup={float(pa):.6f}x')
print(f'b300_meta_source_cal_arg_wall_speedup={float(aw):.6f}x')
print(f'b300_meta_source_cal_arg_active_speedup={float(aa):.6f}x')
print(f'b300_meta_source_full_wall_speedup={cw/nw:.6f}x')
print(f'b300_meta_source_full_total_speedup={ct/nt:.6f}x')
print(f'b300_meta_source_full_active_max_speedup={ca/na:.6f}x')
print(f'b300_meta_source_full_total_delta_pct={(nt/ct-1)*100:.4f}%')
print('meta_const_payload_bytes=13936 meta_ptr_payload_bytes=8 meta_arg_payload_bytes=0')
print(f'residue={res["const"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-static-lpt-meta-source-production-ab OK winner=$WINNER cal_rows=$CAL_ROWS cal_runs=$CAL_RUNS runs=$RUNS result=$RESULT" >&2
