#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}";RUNS="${RUNS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}";GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}";B300_STAGED_META_MAX_MIB="${B300_STAGED_META_MAX_MIB:-512}"
MIN_STAGED_SPEEDUP="${MIN_STAGED_SPEEDUP:-1.02}";MIN_WORST_STAGED_SPEEDUP="${MIN_WORST_STAGED_SPEEDUP:-0.995}";MICRO_ITERS="${MICRO_ITERS:-4096}";MICRO_REPEATS="${MICRO_REPEATS:-9}";STAGED_SLOTS="${STAGED_SLOTS:-1024}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_stagedmeta_production_ab}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";MICRO_RESULT="${MICRO_RESULT:-${PREFIX}_micro.tsv}"
(( RUNS>=1 ))||exit 2;require_nvcc_version_at_least "$NVCC" 13 0 "B300 staged metadata production A/B";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=8))||{ echo "need 8 visible GPUs" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

NVCC="$NVCC" GPUS=8 ARCH="$ARCH" ITERS="$MICRO_ITERS" REPEATS="$MICRO_REPEATS" STAGED_SLOTS="$STAGED_SLOTS" RESULT="$MICRO_RESULT" PREFIX="${PREFIX}_micro" LOGDIR="$LOGDIR/micro" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-group-meta-symbol-copy-ab.sh" >"$LOGDIR/micro.out" 2>"$LOGDIR/micro.err"
read -r STAGED_SPEEDUP STAGED_WORST < <(python3 - "$MICRO_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
x=[float(v['staged_speedup_vs_packed']) for v in r]
print(f"{statistics.median(x):.9f} {min(x):.9f}")
PY
)
echo "b300_group_meta_staged_median_speedup_vs_packed=${STAGED_SPEEDUP}x worst_gpu=${STAGED_WORST}x" >&2
set +e
python3 - "$STAGED_SPEEDUP" "$MIN_STAGED_SPEEDUP" "$STAGED_WORST" "$MIN_WORST_STAGED_SPEEDUP" <<'PY'
import sys
med,med_min,worst,worst_min=map(float,sys.argv[1:]);raise SystemExit(0 if med>=med_min and worst>=worst_min else 9)
PY
gate_rc=$?;set -e
case "$gate_rc" in
  0);;
  9) echo "b300-vmm-stagedmeta-production-ab SKIP_FULL median=${STAGED_SPEEDUP}x median_threshold=${MIN_STAGED_SPEEDUP}x worst_gpu=${STAGED_WORST}x worst_threshold=${MIN_WORST_STAGED_SPEEDUP}x";exit 0;;
  *)exit "$gate_rc";;
esac

NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-packedmeta-ptx-proof.sh" >"$LOGDIR/packedmeta_ptx.out" 2>"$LOGDIR/packedmeta_ptx.err"
BIN_PACK="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_packedmeta_stageab"
BIN_STAGE="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_stagedmeta_ab"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_PACK" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg-packedmeta.sh" >"$LOGDIR/pack.build.out" 2>"$LOGDIR/pack.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" OUT="$BIN_STAGE" bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg-stagedmeta.sh" >"$LOGDIR/stage.build.out" 2>"$LOGDIR/stage.build.err"
grep -Fq 'per_group_meta_copy=D2D_sync' "$LOGDIR/stage.build.out"

printf 'mode\trun\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\tstaged_groups\tstaged_mib_per_gpu\tstaged_cap_mib\tstaged_total_h2d_gib\n' >"$RESULT"
run_one(){
  local mode="$1" run="$2" bin;[[ "$mode" == pack ]]&&bin="$BIN_PACK"||bin="$BIN_STAGE"
  local out="$LOGDIR/${mode}_run${run}.out" err="$LOGDIR/${mode}_run${run}.err"
  B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ tail -n120 "$err" >&2;exit 3; }
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
  local residue="$(field residue)" wall="$(field wall_s)" prep="$(field prepare_s)" active="$(field active_max_s)" sum="$(field active_sum_s)" ints="$(field max_intervals)"
  [[ -n "$residue"&&-n "$wall"&&-n "$prep"&&-n "$active"&&-n "$sum"&&-n "$ints" ]]||exit 4
  local total;total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  local groups='-' mib='-' cap='-' h2d='-'
  if [[ "$mode" == stage ]];then
    local meta;meta="$(grep '^staged group meta: ' "$err"|tail -n1||true)";[[ -n "$meta" ]]||{ echo "missing staged metadata runtime line" >&2;exit 5; }
    mfield(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$meta"; }
    groups="$(mfield groups)";mib="$(mfield mib_per_gpu)";cap="$(mfield cap_mib)";h2d="$(mfield total_h2d_gib)";[[ -n "$groups"&&-n "$mib"&&-n "$cap"&&-n "$h2d" ]]||exit 6
    [[ "$cap" == "$B300_STAGED_META_MAX_MIB" ]]||{ echo "staged metadata cap mismatch runtime=$cap requested=$B300_STAGED_META_MAX_MIB" >&2;exit 7; }
    if [[ "$TARGET_MIB" == 16384 && "$MAX_WINDOW" == 14 ]];then [[ "$groups" == 16384 ]]||{ echo "unexpected default staged group count $groups" >&2;exit 8; };fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$run" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" "$groups" "$mib" "$cap" "$h2d" >>"$RESULT"
}
for((r=1;r<=RUNS;++r));do if((r&1));then order=(pack stage);else order=(stage pack);fi;for m in "${order[@]}";do run_one "$m" "$r";done;done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$STAGED_SPEEDUP" "$STAGED_WORST" <<'PY'
import csv,statistics,sys
src,dst,micro,worst=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for mode in ('pack','stage'):
 r=[x for x in rows if x['mode']==mode];rr={x['residue'] for x in r}
 if len(rr)!=1:raise SystemExit(f'unstable residue {mode}: {rr}')
 res[mode]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in r)
 d=dict(mode=mode,runs=len(r),residue=res[mode],median_wall_s=f'{med("wall_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',median_total_s=f'{med("total_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in r))
 if mode=='stage':
  for k in ('staged_groups','staged_mib_per_gpu','staged_cap_mib','staged_total_h2d_gib'):
   vals={x[k] for x in r};
   if len(vals)!=1:raise SystemExit(f'unstable {k}: {vals}')
   d[k]=next(iter(vals))
 out.append(d)
if len(set(res.values()))!=1:raise SystemExit(f'residue mismatch {res}')
keys=[]
for x in out:
 for k in x:
  if k not in keys:keys.append(k)
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys,delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};p=float(q['pack']['median_wall_s']);s=float(q['stage']['median_wall_s']);pt=float(q['pack']['median_total_s']);st=float(q['stage']['median_total_s'])
print(f'b300_stagedmeta_micro_median_speedup_vs_packed={float(micro):.6f}x')
print(f'b300_stagedmeta_micro_worst_gpu_speedup_vs_packed={float(worst):.6f}x')
print(f'b300_stagedmeta_full_wall_speedup={p/s:.6f}x')
print(f'b300_stagedmeta_full_total_speedup={pt/st:.6f}x')
print(f'b300_stagedmeta_full_total_delta_pct={(st/pt-1)*100:.4f}%')
print(f'b300_stagedmeta_best_total={"stage" if st<pt else "pack"}')
print(f'b300_stagedmeta_groups={q["stage"].get("staged_groups","-")}')
print(f'b300_stagedmeta_mib_per_gpu={q["stage"].get("staged_mib_per_gpu","-")}')
print(f'b300_stagedmeta_cap_mib={q["stage"].get("staged_cap_mib","-")}')
print(f'b300_stagedmeta_total_h2d_gib={q["stage"].get("staged_total_h2d_gib","-")}')
print(f'residue={res["pack"]}')
print(f'summary={dst}')
PY
echo "b300-vmm-stagedmeta-production-ab OK runs=$RUNS median_gate=${STAGED_SPEEDUP}x median_threshold=${MIN_STAGED_SPEEDUP}x worst_gate=${STAGED_WORST}x worst_threshold=${MIN_WORST_STAGED_SPEEDUP}x stage_cap_mib=$B300_STAGED_META_MAX_MIB result=$RESULT" >&2
