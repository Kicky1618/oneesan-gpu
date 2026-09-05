#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}"
MIN_COPY_SPEEDUP="${MIN_COPY_SPEEDUP:-1.02}";MICRO_ITERS="${MICRO_ITERS:-4096}";MICRO_REPEATS="${MICRO_REPEATS:-9}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_packedmeta_production_ab}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";MICRO_RESULT="${MICRO_RESULT:-${PREFIX}_micro.tsv}"
(( RUNS>=1 )) || exit 2;require_nvcc_version_at_least "$NVCC" 13 0 "B300 packed metadata production A/B";command -v nvidia-smi >/dev/null || exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

NVCC="$NVCC" GPUS=8 ARCH="$ARCH" ITERS="$MICRO_ITERS" REPEATS="$MICRO_REPEATS" RESULT="$MICRO_RESULT" PREFIX="${PREFIX}_micro" LOGDIR="$LOGDIR/micro" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-group-meta-symbol-copy-ab.sh" >"$LOGDIR/micro.out" 2>"$LOGDIR/micro.err"
COPY_SPEEDUP="$(python3 - "$MICRO_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
print(f"{statistics.median(float(x['packed_speedup_vs_unpacked']) for x in r):.9f}")
PY
)"
echo "b300_group_meta_copy_median_speedup=${COPY_SPEEDUP}x" >&2
set +e
python3 - "$COPY_SPEEDUP" "$MIN_COPY_SPEEDUP" <<'PY'
import sys
s,t=map(float,sys.argv[1:])
raise SystemExit(0 if s>=t else 9)
PY
gate_rc=$?
set -e
case "$gate_rc" in
  0) ;;
  9) echo "b300-vmm-packedmeta-production-ab SKIP_FULL micro_copy_speedup=${COPY_SPEEDUP}x threshold=${MIN_COPY_SPEEDUP}x"; exit 0 ;;
  *) exit "$gate_rc" ;;
esac

NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-packedmeta-ptx-proof.sh" >"$LOGDIR/packedmeta_ptx.out" 2>"$LOGDIR/packedmeta_ptx.err"
BIN_BASE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_basearg_metaab"
BIN_PACK="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_basearg_packedmeta_ab"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_BASE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_PACK" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg-packedmeta.sh" >"$LOGDIR/pack.build.out" 2>"$LOGDIR/pack.build.err"
grep -Fq 'group_meta_symbol_copies_per_group=1' "$LOGDIR/pack.build.out"

printf 'mode\trun\tresidue\twall_s\tactive_max_s\tactive_sum_s\tprepare_s\tmax_intervals\n' >"$RESULT"
run_one(){
  local mode="$1" run="$2" bin;[[ "$mode" == base ]]&&bin="$BIN_BASE"||bin="$BIN_PACK"
  local out="$LOGDIR/${mode}_run${run}.out" err="$LOGDIR/${mode}_run${run}.err"
  GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n100 "$err" >&2;exit 3; }
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
  local residue="$(field residue)" wall="$(field wall_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" prep="$(field prepare_s)" ints="$(field max_intervals)"
  [[ -n "$residue"&&-n "$wall"&&-n "$active"&&-n "$sum"&&-n "$prep"&&-n "$ints" ]]||exit 4
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$active" "$sum" "$prep" "$ints" >>"$RESULT"
}
for((r=1;r<=RUNS;++r));do if((r&1));then order=(base pack);else order=(pack base);fi;for m in "${order[@]}";do run_one "$m" "$r";done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$COPY_SPEEDUP" <<'PY'
import csv,statistics,sys
src,dst,micro=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for mode in ('base','pack'):
 r=[x for x in rows if x['mode']==mode];rr={x['residue'] for x in r}
 if len(rr)!=1: raise SystemExit(f'unstable residue {mode}: {rr}')
 res[mode]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in r)
 out.append(dict(mode=mode,runs=len(r),residue=res[mode],median_wall_s=f'{med("wall_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in r)))
if len(set(res.values()))!=1: raise SystemExit(f'residue mismatch {res}')
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};b=float(q['base']['median_wall_s']);p=float(q['pack']['median_wall_s']);ba=float(q['base']['median_active_max_s']);pa=float(q['pack']['median_active_max_s'])
print(f'b300_packedmeta_micro_copy_speedup={float(micro):.6f}x')
print(f'b300_packedmeta_full_wall_speedup={b/p:.6f}x')
print(f'b300_packedmeta_full_wall_delta_pct={(p/b-1)*100:.4f}%')
print(f'b300_packedmeta_full_active_max_speedup={ba/pa:.6f}x')
print(f'b300_packedmeta_best={"pack" if p<b else "base"}')
print(f'residue={res["base"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-packedmeta-production-ab OK runs=$RUNS micro_gate=${COPY_SPEEDUP}x threshold=${MIN_COPY_SPEEDUP}x result=$RESULT" >&2
