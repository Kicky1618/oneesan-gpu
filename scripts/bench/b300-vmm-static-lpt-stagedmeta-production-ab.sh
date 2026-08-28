#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";PTX_ARCH="${PTX_ARCH:-sm_103}"
RUNS="${RUNS:-1}";CAL_RUNS="${CAL_RUNS:-1}";CAL_ROWS="${CAL_ROWS:-1}"
MOD="${MOD:-4294967291}";TARGET_MIB="${TARGET_MIB:-16384}";MAX_WINDOW="${MAX_WINDOW:-14}"
GRIDFP_VRAM_RESERVE_MIB="${GRIDFP_VRAM_RESERVE_MIB:-8192}";B300_STAGED_META_MAX_MIB="${B300_STAGED_META_MAX_MIB:-512}"
CAL_MIN_WALL_SPEEDUP="${CAL_MIN_WALL_SPEEDUP:-0.995}";CAL_MIN_ACTIVE_SPEEDUP="${CAL_MIN_ACTIVE_SPEEDUP:-0.990}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_static_lpt_stagedmeta_ab}"
CAL_RESULT="${CAL_RESULT:-${PREFIX}_cal.tsv}";RESULT="${RESULT:-${PREFIX}.tsv}";SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"

(( RUNS>=1 && CAL_RUNS>=1 && CAL_ROWS>=1 && CAL_ROWS<=28 )) || { echo "invalid RUNS/CAL_RUNS/CAL_ROWS" >&2; exit 2; }
require_nvcc_version_at_least "$NVCC" 13 0 "B300 static LPT staged metadata production A/B"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)";(( visible>=8 )) || { echo "need 8 visible GPUs; got $visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh" >"$LOGDIR/plan-proof.out" 2>"$LOGDIR/plan-proof.err"
NVCC="$NVCC" GPUS=8 ARCH="$ARCH" PTX_ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-preflight.sh" >"$LOGDIR/preflight.out" 2>"$LOGDIR/preflight.err"
NVCC="$NVCC" ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-packedmeta-ptx-proof.sh" >"$LOGDIR/packedmeta-ptx.out" 2>"$LOGDIR/packedmeta-ptx.err"

BIN_DYNAMIC="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_staged_dynamic_lptab"
BIN_STATIC="$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n27_vmm_staged_static_lptab"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_DYNAMIC" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-basearg-stagedmeta.sh" >"$LOGDIR/dynamic.build.out" 2>"$LOGDIR/dynamic.build.err"
NVCC="$NVCC" N=27 ARCH="$ARCH" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" OUT="$BIN_STATIC" \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-vmm-static-lpt-stagedmeta.sh" >"$LOGDIR/static.build.out" 2>"$LOGDIR/static.build.err"
grep -Fq 'row_limit_env=B300_ROW_LIMIT default_rows=28' "$LOGDIR/dynamic.build.out"
grep -Fq 'scheduler=static_lpt' "$LOGDIR/static.build.out"
grep -Fq 'metadata_replication=0' "$LOGDIR/static.build.out"
grep -Fq 'expected_default_max_mib_per_gpu=27.271911621' "$LOGDIR/static.build.out"

printf 'phase\tmode\trun\trows\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\n' >"$CAL_RESULT"
printf 'phase\tmode\trun\trows\tresidue\twall_s\tprepare_s\ttotal_s\tactive_max_s\tactive_sum_s\tmax_intervals\tmeta_groups\tmeta_group_min\tmeta_group_max\tmeta_mib_per_gpu\tmeta_total_h2d_gib\tmeta_work_spread_pct\n' >"$RESULT"

run_one(){
  local phase="$1" mode="$2" run="$3" rows="$4" dest="$5" bin
  [[ "$mode" == dynamic ]] && bin="$BIN_DYNAMIC" || bin="$BIN_STATIC"
  local out="$LOGDIR/${phase}_${mode}_${run}.out" err="$LOGDIR/${phase}_${mode}_${run}.err"
  echo "=== B300 static-LPT A/B phase=$phase mode=$mode rows=$rows run=$run ===" >&2
  B300_ROW_LIMIT="$rows" B300_STAGED_META_MAX_MIB="$B300_STAGED_META_MAX_MIB" GRIDFP_VRAM_RESERVE_MIB="$GRIDFP_VRAM_RESERVE_MIB" \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  grep -Fq "B300 row limit: rows=${rows}/28 calibration=$(( rows<28 ? 1 : 0 ))" "$err" || { echo "missing row-limit proof in $err" >&2; tail -n120 "$err" >&2; exit 3; }
  local line;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN-vmm ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n120 "$err" >&2 || true; cat "$out" >&2; exit 4; }
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$line"; }
  local residue wall prep active sum ints total
  residue="$(field residue)";wall="$(field wall_s)";prep="$(field prepare_s)";active="$(field active_max_s)";sum="$(field active_sum_s)";ints="$(field max_intervals)"
  [[ -n "$residue" && -n "$wall" && -n "$prep" && -n "$active" && -n "$sum" && -n "$ints" ]] || { echo "$line" >&2; exit 5; }
  total="$(python3 - "$wall" "$prep" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  if [[ "$dest" == cal ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$mode" "$run" "$rows" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" >>"$CAL_RESULT"
    return
  fi
  local groups='-' gmin='-' gmax='-' mib='-' h2d='-' spread='-'
  if [[ "$mode" == dynamic ]]; then
    local meta;meta="$(grep '^staged group meta: ' "$err" | tail -n1 || true)";[[ -n "$meta" ]] || { echo "missing dynamic staged metadata line" >&2; exit 6; }
    mfield(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$meta"; }
    groups="$(mfield groups)";mib="$(mfield mib_per_gpu)";h2d="$(mfield total_h2d_gib)"
  else
    local meta;meta="$(grep '^static LPT staged group meta: ' "$err" | tail -n1 || true)";[[ -n "$meta" ]] || { echo "missing static LPT metadata line" >&2; exit 7; }
    mfield(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$meta"; }
    groups="$(mfield groups)";gmin="$(mfield group_min)";gmax="$(mfield group_max)";mib="$(mfield max_mib_per_gpu)";h2d="$(mfield total_h2d_gib)";spread="$(mfield work_spread_pct)"
    [[ "$TARGET_MIB" != 16384 || "$MAX_WINDOW" != 14 || ( "$groups" == 16384 && "$gmin" == 2044 && "$gmax" == 2052 ) ]] || { echo "unexpected static default assignment: $meta" >&2; exit 8; }
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$mode" "$run" "$rows" "$residue" "$wall" "$prep" "$total" "$active" "$sum" "$ints" "$groups" "$gmin" "$gmax" "$mib" "$h2d" "$spread" >>"$RESULT"
}

for((r=1;r<=CAL_RUNS;++r));do
  if((r&1));then order=(dynamic static);else order=(static dynamic);fi
  for mode in "${order[@]}";do run_one calibration "$mode" "$r" "$CAL_ROWS" cal;done
done
cat "$CAL_RESULT"
read -r CAL_WALL_SPEEDUP CAL_ACTIVE_SPEEDUP CAL_TOTAL_SPEEDUP < <(python3 - "$CAL_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(mode,key): return statistics.median(float(x[key]) for x in r if x['mode']==mode)
res={m:{x['residue'] for x in r if x['mode']==m} for m in ('dynamic','static')}
if any(len(v)!=1 for v in res.values()) or res['dynamic']!=res['static']: raise SystemExit(f'calibration residue mismatch {res}')
dw,sw=med('dynamic','wall_s'),med('static','wall_s');da,sa=med('dynamic','active_max_s'),med('static','active_max_s');dt,st=med('dynamic','total_s'),med('static','total_s')
print(f'{dw/sw:.9f} {da/sa:.9f} {dt/st:.9f}')
PY
)
echo "b300_static_lpt_calibration wall_speedup=${CAL_WALL_SPEEDUP}x active_max_speedup=${CAL_ACTIVE_SPEEDUP}x total_speedup=${CAL_TOTAL_SPEEDUP}x" >&2
set +e
python3 - "$CAL_WALL_SPEEDUP" "$CAL_MIN_WALL_SPEEDUP" "$CAL_ACTIVE_SPEEDUP" "$CAL_MIN_ACTIVE_SPEEDUP" <<'PY'
import sys
wall,wall_min,active,active_min=map(float,sys.argv[1:]);raise SystemExit(0 if wall>=wall_min and active>=active_min else 9)
PY
gate_rc=$?;set -e
case "$gate_rc" in
  0);;
  9) echo "b300-vmm-static-lpt-stagedmeta-production-ab SKIP_FULL cal_wall=${CAL_WALL_SPEEDUP}x wall_threshold=${CAL_MIN_WALL_SPEEDUP}x cal_active=${CAL_ACTIVE_SPEEDUP}x active_threshold=${CAL_MIN_ACTIVE_SPEEDUP}x";exit 0;;
  *)exit "$gate_rc";;
esac

for((r=1;r<=RUNS;++r));do
  if((r&1));then order=(dynamic static);else order=(static dynamic);fi
  for mode in "${order[@]}";do run_one full "$mode" "$r" 28 full;done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$CAL_WALL_SPEEDUP" "$CAL_ACTIVE_SPEEDUP" "$CAL_TOTAL_SPEEDUP" <<'PY'
import csv,statistics,sys
src,dst,calw,cala,calt=sys.argv[1:];rows=list(csv.DictReader(open(src),delimiter='\t'));out=[];res={}
for mode in ('dynamic','static'):
    rs=[x for x in rows if x['mode']==mode];rr={x['residue'] for x in rs}
    if len(rr)!=1: raise SystemExit(f'unstable residue {mode}: {rr}')
    res[mode]=next(iter(rr));med=lambda k:statistics.median(float(x[k]) for x in rs)
    d=dict(mode=mode,runs=len(rs),residue=res[mode],median_wall_s=f'{med("wall_s"):.9f}',median_prepare_s=f'{med("prepare_s"):.9f}',median_total_s=f'{med("total_s"):.9f}',median_active_max_s=f'{med("active_max_s"):.9f}',median_active_sum_s=f'{med("active_sum_s"):.9f}',max_intervals=max(int(x['max_intervals']) for x in rs))
    for k in ('meta_groups','meta_group_min','meta_group_max','meta_mib_per_gpu','meta_total_h2d_gib','meta_work_spread_pct'):
        vals={x[k] for x in rs};d[k]=next(iter(vals)) if len(vals)==1 else 'varies'
    out.append(d)
if len(set(res.values()))!=1: raise SystemExit(f'full residue mismatch {res}')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t');w.writeheader();w.writerows(out)
q={x['mode']:x for x in out};d=q['dynamic'];s=q['static'];dw=float(d['median_wall_s']);sw=float(s['median_wall_s']);dt=float(d['median_total_s']);st=float(s['median_total_s']);da=float(d['median_active_max_s']);sa=float(s['median_active_max_s'])
print(f'b300_static_lpt_cal_wall_speedup={float(calw):.6f}x')
print(f'b300_static_lpt_cal_active_max_speedup={float(cala):.6f}x')
print(f'b300_static_lpt_cal_total_speedup={float(calt):.6f}x')
print(f'b300_static_lpt_full_wall_speedup={dw/sw:.6f}x')
print(f'b300_static_lpt_full_total_speedup={dt/st:.6f}x')
print(f'b300_static_lpt_full_active_max_speedup={da/sa:.6f}x')
print(f'b300_static_lpt_full_total_delta_pct={(st/dt-1)*100:.4f}%')
print(f'b300_static_lpt_best={"static" if st<dt else "dynamic"}')
print(f'b300_dynamic_meta_mib_per_gpu={d["meta_mib_per_gpu"]}')
print(f'b300_static_meta_max_mib_per_gpu={s["meta_mib_per_gpu"]}')
print(f'b300_dynamic_meta_total_h2d_gib={d["meta_total_h2d_gib"]}')
print(f'b300_static_meta_total_h2d_gib={s["meta_total_h2d_gib"]}')
print(f'b300_static_meta_work_spread_pct={s["meta_work_spread_pct"]}')
print(f'residue={res["dynamic"]}')
print(f'summary={dst}')
PY

echo "b300-vmm-static-lpt-stagedmeta-production-ab OK cal_rows=$CAL_ROWS cal_runs=$CAL_RUNS runs=$RUNS result=$RESULT" >&2
