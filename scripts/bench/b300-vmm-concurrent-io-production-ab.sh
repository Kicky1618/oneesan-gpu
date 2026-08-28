#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
MIN_IO_SPEEDUP="${MIN_IO_SPEEDUP:-1.01}";MICRO_REPEATS="${MICRO_REPEATS:-9}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_concurrent_io_production_ab}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";MICRO_RESULT="${MICRO_RESULT:-${PREFIX}_micro.tsv}"
((RUNS>=1))||exit 2;require_nvcc_version_at_least "$NVCC" 13 0 "B300 concurrent VMM I/O production A/B";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" REPEATS="$MICRO_REPEATS" RESULT="$MICRO_RESULT" PREFIX="${PREFIX}_micro" LOGDIR="$LOGDIR/micro" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-concurrent-io-ab.sh" >"$LOGDIR/micro.out" 2>"$LOGDIR/micro.err"
IO_SPEEDUP="$(python3 - "$MICRO_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));print(f"{statistics.median(float(x['speedup']) for x in r):.9f}")
PY
)"
echo "b300_concurrent_io_micro_median_speedup=${IO_SPEEDUP}x" >&2
set +e
python3 - "$IO_SPEEDUP" "$MIN_IO_SPEEDUP" <<'PY'
import sys
s,t=map(float,sys.argv[1:]);raise SystemExit(0 if s>=t else 9)
PY
gate_rc=$?;set -e
case "$gate_rc" in 0);;9)echo "b300-vmm-concurrent-io-production-ab SKIP_FULL micro_speedup=${IO_SPEEDUP}x threshold=${MIN_IO_SPEEDUP}x";exit 0;;*)exit "$gate_rc";;esac
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
BIN_BASE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_basearg_ioab"
BIN_CONC="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_basearg_concurrent_io_ab"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_BASE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_CONC" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg-concurrent-io.sh" >"$LOGDIR/conc.build.out" 2>"$LOGDIR/conc.build.err"
grep -Fq 'concurrent_authoritative_io=1' "$LOGDIR/conc.build.out"
printf 'mode\trun\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\n' >"$RESULT"
run_one(){
 local mode="$1" run="$2" bin;[[ "$mode" == base ]]&&bin="$BIN_BASE"||bin="$BIN_CONC";local out="$LOGDIR/${mode}_run${run}.out" err="$LOGDIR/${mode}_run${run}.err"
 GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
 local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n120 "$err" >&2;exit 3; }
 field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
 local residue="$(field residue)" wall="$(field wall_s)" prep="$(field prepare_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" ints="$(field max_intervals)";[[ -n "$residue"&&-n "$wall"&&-n "$prep"&&-n "$active"&&-n "$sum"&&-n "$ints" ]]||exit 4
 local total;total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)";printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" >>"$RESULT"
}
for((r=1;r<=RUNS;++r));do if((r&1));then order=(base concurrent);else order=(concurrent base);fi;for m in "${order[@]}";do run_one "$m" "$r";done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$IO_SPEEDUP" <<'PY'
import csv,statistics,sys
src,dst,micro=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for mode in ('base','concurrent'):
 r=[x for x in rows if x['mode']==mode];rr={x['residue'] for x in r}
 if len(rr)!=1:raise SystemExit(f'unstable residue {mode}: {rr}')
 res[mode]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in r);out.append(dict(mode=mode,runs=len(r),residue=res[mode],median_wall_s=f'{med("wall_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',median_total_s=f'{med("total_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in r)))
if len(set(res.values()))!=1:raise SystemExit(f'residue mismatch {res}')
with open(dst,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};b=float(q['base']['median_wall_s']);c=float(q['concurrent']['median_wall_s']);bt=float(q['base']['median_total_s']);ct=float(q['concurrent']['median_total_s'])
print(f'b300_concurrent_io_micro_speedup={float(micro):.6f}x')
print(f'b300_concurrent_io_full_wall_speedup={b/c:.6f}x')
print(f'b300_concurrent_io_full_total_speedup={bt/ct:.6f}x')
print(f'b300_concurrent_io_full_total_delta_pct={(ct/bt-1)*100:.4f}%')
print(f'b300_concurrent_io_best={"concurrent" if ct<bt else "base"}')
print(f'residue={res["base"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-concurrent-io-production-ab OK runs=$RUNS micro_gate=${IO_SPEEDUP}x threshold=${MIN_IO_SPEEDUP}x result=$RESULT" >&2
