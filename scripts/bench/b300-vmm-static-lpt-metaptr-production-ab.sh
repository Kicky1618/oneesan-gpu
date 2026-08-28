#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}";CAL_RUNS="${CAL_RUNS:-1}";CAL_ROWS="${CAL_ROWS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}";B300_STAGED_META_MAX_MIB="${B300_STAGED_META_MAX_MIB:-512}"
CAL_MIN_WALL_SPEEDUP="${CAL_MIN_WALL_SPEEDUP:-0.995}";CAL_MIN_ACTIVE_SPEEDUP="${CAL_MIN_ACTIVE_SPEEDUP:-0.995}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_static_lpt_metaptr_ab}";CAL_RESULT="${CAL_RESULT:-${PREFIX}_cal.tsv}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"
((RUNS>=1&&CAL_RUNS>=1&&CAL_ROWS>=1&&CAL_ROWS<=28))||exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 static LPT metadata-pointer production A/B";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh" >"$LOGDIR/plan-proof.out" 2>"$LOGDIR/plan-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh" >"$LOGDIR/local-meta-proof.out" 2>"$LOGDIR/local-meta-proof.err"
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metaptr-ptx-proof.sh" >"$LOGDIR/metaptr-ptx.out" 2>"$LOGDIR/metaptr-ptx.err"

BIN_CONST="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_static_lpt_metaconst_ab"
BIN_PTR="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_static_lpt_metaptr_ab"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_CONST" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh" >"$LOGDIR/const.build.out" 2>"$LOGDIR/const.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_PTR" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-metaptr.sh" >"$LOGDIR/ptr.build.out" 2>"$LOGDIR/ptr.build.err"
grep -Fq 'per_group_meta_copy=D2D_sync' "$LOGDIR/const.build.out"
grep -Fq 'group_meta_constant_payload_bytes_per_group=8' "$LOGDIR/ptr.build.out"
grep -Fq 'group_meta_payload_reduction=1742x' "$LOGDIR/ptr.build.out"

printf 'phase\tmode\trun\trows\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\n' >"$CAL_RESULT"
printf 'mode\trun\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\tcopy_mode\tmeta_access\n' >"$RESULT"
run_one(){
 local phase="$1" mode="$2" run="$3" rows="$4" dest="$5" bin;[[ "$mode" == const ]]&&bin="$BIN_CONST"||bin="$BIN_PTR"
 local out="$LOGDIR/${phase}_${mode}_${run}.out" err="$LOGDIR/${phase}_${mode}_${run}.err"
 B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
 grep -Fq "B300 row limit: rows=${rows}/28 calibration=$(( rows<28 ? 1 : 0 ))" "$err"||{ tail -n120 "$err" >&2;exit 3; }
 local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n120 "$err" >&2;exit 4; }
 field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
 local residue="$(field residue)" wall="$(field wall_s)" prep="$(field prepare_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" ints="$(field max_intervals)";[[ -n "$residue"&&-n "$wall"&&-n "$prep"&&-n "$active"&&-n "$sum"&&-n "$ints" ]]||exit 5
 local total;total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
 local meta;meta="$(grep '^static LPT staged group meta: ' "$err"|tail -n1||true)";[[ -n "$meta" ]]||{ echo "missing static LPT metadata line" >&2;exit 6; }
 if [[ "$mode" == const ]];then grep -Fq 'copy_mode=H2D_once_local_then_D2D_per_group scheduler=static_lpt' <<<"$meta"||exit 7;copy='full_D2D_13936B';access='constant';else grep -Fq 'copy_mode=H2D_once_local_then_pointer_H2D_per_group scheduler=static_lpt meta_access=staged_global' <<<"$meta"||exit 8;copy='pointer_H2D_8B';access='staged_global';fi
 if [[ "$dest" == cal ]];then printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$mode" "$run" "$rows" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" >>"$CAL_RESULT";else printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" "$copy" "$access" >>"$RESULT";fi
}
for((r=1;r<=CAL_RUNS;++r));do if((r&1));then order=(const ptr);else order=(ptr const);fi;for m in "${order[@]}";do run_one calibration "$m" "$r" "$CAL_ROWS" cal;done;done
cat "$CAL_RESULT"
read -r CAL_WALL CAL_ACTIVE CAL_TOTAL < <(python3 - "$CAL_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(m,k):return statistics.median(float(x[k]) for x in r if x['mode']==m)
res={m:{x['residue'] for x in r if x['mode']==m} for m in ('const','ptr')}
if any(len(x)!=1 for x in res.values()) or res['const']!=res['ptr']:raise SystemExit(f'cal residue mismatch {res}')
print(f"{med('const','wall_s')/med('ptr','wall_s'):.9f} {med('const','active_max_s')/med('ptr','active_max_s'):.9f} {med('const','total_s')/med('ptr','total_s'):.9f}")
PY
)
echo "b300_metaptr_calibration wall_speedup=${CAL_WALL}x active_max_speedup=${CAL_ACTIVE}x total_speedup=${CAL_TOTAL}x" >&2
set +e
python3 - "$CAL_WALL" "$CAL_MIN_WALL_SPEEDUP" "$CAL_ACTIVE" "$CAL_MIN_ACTIVE_SPEEDUP" <<'PY'
import sys
w,wm,a,am=map(float,sys.argv[1:]);raise SystemExit(0 if w>=wm and a>=am else 9)
PY
gate_rc=$?;set -e
case "$gate_rc" in 0);;9)echo "b300-vmm-static-lpt-metaptr-production-ab SKIP_FULL cal_wall=${CAL_WALL}x wall_threshold=${CAL_MIN_WALL_SPEEDUP}x cal_active=${CAL_ACTIVE}x active_threshold=${CAL_MIN_ACTIVE_SPEEDUP}x";exit 0;;*)exit "$gate_rc";;esac
for((r=1;r<=RUNS;++r));do if((r&1));then order=(const ptr);else order=(ptr const);fi;for m in "${order[@]}";do run_one full "$m" "$r" 28 full;done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$CAL_WALL" "$CAL_ACTIVE" "$CAL_TOTAL" <<'PY'
import csv,statistics,sys
src,dst,cw,ca,ct=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for m in ('const','ptr'):
 rs=[x for x in rows if x['mode']==m];rr={x['residue'] for x in rs}
 if len(rr)!=1:raise SystemExit(f'unstable residue {m}: {rr}')
 res[m]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in rs);out.append(dict(mode=m,runs=len(rs),residue=res[m],median_wall_s=f'{med("wall_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',median_total_s=f'{med("total_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in rs),copy_mode=rs[0]['copy_mode'],meta_access=rs[0]['meta_access']))
if len(set(res.values()))!=1:raise SystemExit(f'full residue mismatch {res}')
with open(dst,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};c=q['const'];p=q['ptr'];cwf=float(c['median_wall_s']);pw=float(p['median_wall_s']);ctf=float(c['median_total_s']);pt=float(p['median_total_s']);ca2=float(c['median_active_max_s']);pa=float(p['median_active_max_s'])
print(f'b300_metaptr_cal_wall_speedup={float(cw):.6f}x')
print(f'b300_metaptr_cal_active_max_speedup={float(ca):.6f}x')
print(f'b300_metaptr_cal_total_speedup={float(ct):.6f}x')
print(f'b300_metaptr_full_wall_speedup={cwf/pw:.6f}x')
print(f'b300_metaptr_full_total_speedup={ctf/pt:.6f}x')
print(f'b300_metaptr_full_active_max_speedup={ca2/pa:.6f}x')
print(f'b300_metaptr_full_total_delta_pct={(pt/ctf-1)*100:.4f}%')
print(f'b300_metaptr_best={"ptr" if pt<ctf else "const"}')
print('constant_payload_old_bytes=13936 constant_payload_ptr_bytes=8 payload_reduction=1742x')
print(f'residue={res["const"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-static-lpt-metaptr-production-ab OK cal_rows=$CAL_ROWS cal_runs=$CAL_RUNS runs=$RUNS result=$RESULT" >&2
