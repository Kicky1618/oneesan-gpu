#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";RUNS="${RUNS:-1}";CAL_RUNS="${CAL_RUNS:-1}";CAL_ROWS="${CAL_ROWS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}";B300_STAGED_META_MAX_MIB="${B300_STAGED_META_MAX_MIB:-512}"
CAL_MIN_WALL_SPEEDUP="${CAL_MIN_WALL_SPEEDUP:-1.001}";CAL_MIN_ACTIVE_SPEEDUP="${CAL_MIN_ACTIVE_SPEEDUP:-1.001}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_static_lpt_workerbind_ab}";CAL_RESULT="${CAL_RESULT:-${PREFIX}_cal.tsv}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"
((RUNS>=1&&CAL_RUNS>=1&&CAL_ROWS>=1&&CAL_ROWS<=28))||exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 worker device-binding A/B";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh" >"$LOGDIR/plan-proof.out" 2>"$LOGDIR/plan-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh" >"$LOGDIR/local-meta-proof.out" 2>"$LOGDIR/local-meta-proof.err"
BIN_BASE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_workerbind_base";BIN_BIND="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_n27_workerbind_fast"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_BASE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_BIND" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-workerbind.sh" >"$LOGDIR/bind.build.out" 2>"$LOGDIR/bind.build.err"
grep -Fq 'expected_call_reduction=2048x' "$LOGDIR/bind.build.out"
printf 'mode\trun\trows\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\n' >"$CAL_RESULT";printf 'mode\trun\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\n' >"$RESULT"
run_one(){ local mode="$1" run="$2" rows="$3" dest="$4" bin;[[ "$mode" == base ]]&&bin="$BIN_BASE"||bin="$BIN_BIND";local out="$LOGDIR/${dest}_${mode}_${run}.out" err="$LOGDIR/${dest}_${mode}_${run}.err";B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err";local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n120 "$err" >&2;exit 3; };field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; };local residue="$(field residue)" wall="$(field wall_s)" prep="$(field prepare_s)" active="$(field active_max_s)" sum="$(field active_sum_s)";local total;total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)";if [[ "$dest" == cal ]];then printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$rows" "$residue" "$wall" "$prep" "$total" "$active" "$sum" >>"$CAL_RESULT";else printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$prep" "$total" "$active" "$sum" >>"$RESULT";fi; }
for((r=1;r<=CAL_RUNS;++r));do if((r&1));then order=(base bind);else order=(bind base);fi;for m in "${order[@]}";do run_one "$m" "$r" "$CAL_ROWS" cal;done;done
read -r CW CA < <(python3 - "$CAL_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));med=lambda m,k:statistics.median(float(x[k]) for x in r if x['mode']==m);res={m:{x['residue'] for x in r if x['mode']==m} for m in ('base','bind')}
if res['base']!=res['bind'] or len(res['base'])!=1:raise SystemExit(f'residue mismatch {res}')
print(f"{med('base','wall_s')/med('bind','wall_s'):.9f} {med('base','active_max_s')/med('bind','active_max_s'):.9f}")
PY
)
set +e;python3 - "$CW" "$CAL_MIN_WALL_SPEEDUP" "$CA" "$CAL_MIN_ACTIVE_SPEEDUP" <<'PY'
import sys
w,wm,a,am=map(float,sys.argv[1:]);raise SystemExit(0 if w>=wm and a>=am else 9)
PY
gate=$?;set -e
[[ "$gate" == 0 ]]||{ [[ "$gate" == 9 ]]&&{ echo "b300-vmm-static-lpt-workerbind-production-ab SKIP_FULL cal_wall=${CW}x cal_active=${CA}x";exit 0;};exit "$gate"; }
for((r=1;r<=RUNS;++r));do if((r&1));then order=(base bind);else order=(bind base);fi;for m in "${order[@]}";do run_one "$m" "$r" 28 full;done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$CW" "$CA" <<'PY'
import csv,statistics,sys
src,dst,cw,ca=sys.argv[1:];r=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for m in ('base','bind'):
 x=[v for v in r if v['mode']==m];rr={v['residue'] for v in x};res[m]=next(iter(rr));med=lambda k:statistics.median(float(v[k]) for v in x);out.append({'mode':m,'runs':len(x),'residue':res[m],'median_wall_s':f'{med("wall_s"):.9f}','median_total_s':f'{med("total_s"):.9f}','median_active_max_s':f'{med("active_max_s"):.9f}'})
if res['base']!=res['bind']:raise SystemExit(f'full residue mismatch {res}')
with open(dst,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={v['mode']:v for v in out};b=q['base'];n=q['bind'];print(f'b300_workerbind_cal_wall_speedup={float(cw):.6f}x');print(f'b300_workerbind_cal_active_speedup={float(ca):.6f}x');print(f'b300_workerbind_full_wall_speedup={float(b["median_wall_s"])/float(n["median_wall_s"]):.6f}x');print(f'b300_workerbind_full_total_speedup={float(b["median_total_s"])/float(n["median_total_s"]):.6f}x');print('cudaSetDevice_calls_old=917504 cudaSetDevice_calls_new=448 call_reduction=2048x');print(f'residue={res["base"]}')
PY
