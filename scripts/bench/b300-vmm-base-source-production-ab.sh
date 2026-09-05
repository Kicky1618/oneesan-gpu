#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
MIN_ARG_SPEEDUP="${MIN_ARG_SPEEDUP:-1.002}"
MICRO_ITERS="${MICRO_ITERS:-4096}";MICRO_REPEATS="${MICRO_REPEATS:-9}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_base_source_production_ab}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";MICRO_RESULT="${MICRO_RESULT:-${PREFIX}_micro.tsv}"
(( RUNS >= 1 )) || exit 2
command -v "$NVCC" >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= 8 )) || { echo "need 8 visible GPUs" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

GPUS=8 ALL_SRC_GPUS=1 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" ITERS="$MICRO_ITERS" REPEATS="$MICRO_REPEATS" \
  RESULT="$MICRO_RESULT" PREFIX="${PREFIX}_micro" LOGDIR="$LOGDIR/micro" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-contiguous-shards-ab.sh" \
  >"$LOGDIR/micro.out" 2>"$LOGDIR/micro.err"
ARG_SPEEDUP="$(python3 - "$MICRO_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
print(f"{statistics.median(float(x['arg_vs_symbol_speedup']) for x in r):.9f}")
PY
)"
echo "b300_vmm_arg_micro_median_speedup_vs_symbol=${ARG_SPEEDUP}x" >&2
if python3 - "$ARG_SPEEDUP" "$MIN_ARG_SPEEDUP" <<'PY'
import sys
s,t=map(float,sys.argv[1:])
raise SystemExit(0 if s>=t else 9)
PY
then
  :
else
  rc=$?
  if (( rc == 9 )); then
    echo "b300-vmm-base-source-production-ab SKIP_FULL micro_arg_speedup=${ARG_SPEEDUP}x threshold=${MIN_ARG_SPEEDUP}x"
    exit 0
  fi
  exit "$rc"
fi

GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-production-ptx-proof.sh" >"$LOGDIR/basearg_ptx.out" 2>"$LOGDIR/basearg_ptx.err"
BIN_SYMBOL="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_symbol_ab"
BIN_ARG="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_basearg_ab"
N=27 ARCH="$ARCH" OUT="$BIN_SYMBOL" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm.sh" >"$LOGDIR/symbol.build.out" 2>"$LOGDIR/symbol.build.err"
N=27 ARCH="$ARCH" OUT="$BIN_ARG" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg.sh" >"$LOGDIR/arg.build.out" 2>"$LOGDIR/arg.build.err"
grep -Fq 'vmm_base_source=kernel_param' "$LOGDIR/arg.build.out"
grep -Fq 'vmm_base_symbols=0' "$LOGDIR/arg.build.out"
printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\tmax_intervals\n' >"$RESULT"
run_one(){
  local mode="$1" run="$2" bin
  case "$mode" in symbol) bin="$BIN_SYMBOL" ;; arg) bin="$BIN_ARG" ;; *) return 2 ;; esac
  local out="$LOGDIR/${mode}_run${run}.out" err="$LOGDIR/${mode}_run${run}.err"
  GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local line
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 100 "$err" >&2; exit 3; }
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$line"; }
  local residue="$(field residue)" wall="$(field wall_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" prep="$(field prepare_s)" ints="$(field max_intervals)"
  [[ -n "$residue" && -n "$wall" && -n "$active" && -n "$sum" && -n "$prep" && -n "$ints" ]] || exit 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$active" "$sum" "$prep" "$ints" >>"$RESULT"
}
for ((r=1;r<=RUNS;++r)); do
  if (( r & 1 )); then order=(symbol arg); else order=(arg symbol); fi
  for m in "${order[@]}"; do run_one "$m" "$r"; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$ARG_SPEEDUP" <<'PY'
import csv,statistics,sys
src,dst,micro=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for mode in ('symbol','arg'):
    r=[x for x in rows if x['mode']==mode];rr={x['residue'] for x in r}
    if len(rr)!=1: raise SystemExit(f'unstable residue {mode}: {rr}')
    res[mode]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in r)
    out.append(dict(mode=mode,runs=len(r),residue=res[mode],median_wall_s=f'{med("wall_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in r)))
if len(set(res.values()))!=1: raise SystemExit(f'residue mismatch {res}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};s=float(q['symbol']['median_wall_s']);a=float(q['arg']['median_wall_s'])
print(f'b300_vmm_arg_micro_speedup_vs_symbol={float(micro):.6f}x')
print(f'b300_vmm_arg_full_wall_speedup_vs_symbol={s/a:.6f}x')
print(f'b300_vmm_arg_full_wall_delta_pct_vs_symbol={(a/s-1)*100:.4f}%')
print(f'b300_vmm_base_source_best={"arg" if a<s else "symbol"}')
print(f'residue={res["symbol"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-base-source-production-ab OK runs=$RUNS micro_gate=${ARG_SPEEDUP}x threshold=${MIN_ARG_SPEEDUP}x result=$RESULT" >&2
