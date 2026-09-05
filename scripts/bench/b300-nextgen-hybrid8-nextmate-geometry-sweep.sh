#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextself_staged_winner.env}"
ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"
MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"
MATE_EVICT="${MATE_EVICT:-default}"
SELF_EVICT="${SELF_EVICT:-default}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_nextmate_geometry_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh"
SELF_BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh"

mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"
[[ -s "$INPUT_ENV" ]] || { echo "missing Stage-F INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
[[ "$ROWS" =~ ^[1-9][0-9]*$ ]] && ((ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be >=1' >&2; exit 2; }
for ev in SELF_EVICT MATE_EVICT; do case "${!ev}" in default|normal|last) ;; *) echo "$ev must be default,normal,last" >&2; exit 2;; esac; done

mate_widths=()
for w in $MATE_WIDTH_LIST; do
  case "$w" in 1|2|4|8) ;; *) echo "bad MATE_WIDTH_LIST entry=$w" >&2; exit 2;; esac
  seen=0; for old in "${mate_widths[@]}"; do [[ "$old" == "$w" ]] && seen=1; done
  ((seen)) || mate_widths+=("$w")
done
((${#mate_widths[@]})) || { echo 'MATE_WIDTH_LIST must not be empty' >&2; exit 2; }
mate_distances=()
for d in $MATE_DISTANCE_LIST; do
  case "$d" in 1|2|4) ;; *) echo "bad MATE_DISTANCE_LIST entry=$d" >&2; exit 2;; esac
  seen=0; for old in "${mate_distances[@]}"; do [[ "$old" == "$d" ]] && seen=1; done
  ((seen)) || mate_distances+=("$d")
done
((${#mate_distances[@]})) || { echo 'MATE_DISTANCE_LIST must not be empty' >&2; exit 2; }
threads=()
for th in $THREADS_LIST; do
  [[ "$th" =~ ^[0-9]+$ ]] && ((th>=32&&th<=1024&&th%32==0)) || { echo "bad THREADS_LIST entry=$th" >&2; exit 2; }
  seen=0; for old in "${threads[@]}"; do [[ "$old" == "$th" ]] && seen=1; done
  ((seen)) || threads+=("$th")
done
((${#threads[@]})) || { echo 'THREADS_LIST must not be empty' >&2; exit 2; }

# shellcheck disable=SC1090
source "$INPUT_ENV"
for k in \
  B300_HYBRID8_NEXTSELF_STAGED_VALIDATED B300_HYBRID8_NEXTSELF_FINAL_ENABLED \
  B300_HYBRID8_NEXTSELF_FINAL_WIDTH B300_HYBRID8_NEXTSELF_FINAL_DISTANCE \
  B300_HYBRID8_NEXTSELF_THRESHOLD B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK \
  B300_HYBRID8_NEXTSELF_RANDOM_CG B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES \
  B300_HYBRID8_NEXTSELF_PREFETCH_L2 B300_HYBRID8_NEXTSELF_DUALMASK \
  B300_HYBRID8_NEXTSELF_CLOSURE_BATCH B300_HYBRID8_NEXTSELF_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "Stage-F env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_NEXTSELF_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_NEXTSELF_FINAL_ENABLED" == 1 ]] || {
  echo 'Stage-F geometry is not promotable' >&2; exit 4;
}
SELF_W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"
SELF_D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"
T="$B300_HYBRID8_NEXTSELF_THRESHOLD"
H="$B300_HYBRID8_NEXTSELF_HIGH_DROP_CHUNK"
CG="$B300_HYBRID8_NEXTSELF_RANDOM_CG"
CGL2="$B300_HYBRID8_NEXTSELF_RANDOM_CG_L2_FETCH_BYTES"
PRE="$B300_HYBRID8_NEXTSELF_PREFETCH_L2"
DUAL="$B300_HYBRID8_NEXTSELF_DUALMASK"
BATCH="$B300_HYBRID8_NEXTSELF_CLOSURE_BATCH"
CAP="$B300_HYBRID8_NEXTSELF_MAXRREGCOUNT"
case "$SELF_W" in 1|2|4|8) ;; *) exit 3;; esac
case "$SELF_D" in 1|2|4) ;; *) exit 3;; esac

command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

COMMON=(
  N=27 ARCH="$ARCH" HIGH_DROP_CHUNK="$H" HYBRID_THRESHOLD="$T"
  RANDOM_CG="$CG" RANDOM_CG_L2_FETCH_BYTES="$CGL2" PREFETCH_L2="$PRE"
  DUALMASK="$DUAL" CLOSURE_BATCH="$BATCH" MAXRREGCOUNT="$CAP" PTXAS_VERBOSE=1
)
CONTROL_BIN="$ONEESAN_BUILD_DIR/b300_stagei_self_w${SELF_W}_d${SELF_D}_sev${SELF_EVICT}_t${T}_n27"
CONTROL_ERR="$LOGDIR/control.build.err"
CONTROL_OUT="$LOGDIR/control.build.out"
env "${COMMON[@]}" OUT="$CONTROL_BIN" NEXTSELF_WIDTH="$SELF_W" NEXTSELF_DISTANCE="$SELF_D" NEXTSELF_EVICT="$SELF_EVICT" \
  BUILD_ERR="$CONTROL_ERR" bash "$SELF_BUILDER" >"$CONTROL_OUT" 2>"$LOGDIR/control.driver.err"
[[ -x "$CONTROL_BIN" ]] || { echo 'Stage-I control binary missing' >&2; exit 3; }
grep -Fq "recurrence_hybrid_ilp8_nextself_width=$SELF_W recurrence_hybrid_ilp8_nextself_distance=$SELF_D" "$CONTROL_OUT" || exit 3

BINS="$LOGDIR/binaries.tsv"
printf 'candidate\tmate_width\tmate_distance\tbinary\tbuild_err\n' >"$BINS"
for mw in "${mate_widths[@]}"; do
  for md in "${mate_distances[@]}"; do
    name="mate_w${mw}_d${md}"
    bin="$ONEESAN_BUILD_DIR/b300_stagei_${name}_selfw${SELF_W}_selfd${SELF_D}_sev${SELF_EVICT}_mev${MATE_EVICT}_t${T}_n27"
    err="$LOGDIR/${name}.build.err"
    out="$LOGDIR/${name}.build.out"
    env "${COMMON[@]}" OUT="$bin" SELF_WIDTH="$SELF_W" SELF_DISTANCE="$SELF_D" MATE_WIDTH="$mw" MATE_DISTANCE="$md" \
      SELF_EVICT="$SELF_EVICT" MATE_EVICT="$MATE_EVICT" BUILD_ERR="$err" bash "$BUILDER" >"$out" 2>"$LOGDIR/${name}.driver.err"
    [[ -x "$bin" ]] || { echo "Stage-I candidate missing: $name" >&2; exit 3; }
    grep -Fq "self_geometry width=$SELF_W distance=$SELF_D evict=$SELF_EVICT" "$out" || exit 3
    grep -Fq "mate_geometry width=$mw distance=$md evict=$MATE_EVICT" "$out" || exit 3
    grep -Fq 'geometry_decoupled=1' "$out" || exit 3
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$mw" "$md" "$bin" "$err" >>"$BINS"
  done
done

printf 'candidate\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$CONTROL_ERR" --label control --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
python3 "$PARSER" "$CONTROL_ERR" --label control --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
while IFS=$'\t' read -r name mw md bin err; do
  [[ "$name" == candidate ]] && continue
  python3 "$PARSER" "$err" --label "$name" --contains main_pull_kernel_ilp2 >>"$RESOURCE" || true
  python3 "$PARSER" "$err" --label "$name" --contains main_pull_kernel_ilp8_hybrid >>"$RESOURCE" || true
done <"$BINS"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'candidate\tmate_width\tmate_distance\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tforward_high_s\treverse_high_s\n' >"$RESULT"
run_one(){
  local name="$1" mw="$2" md="$3" bin="$4" th="$5" r="$6"
  local so="$LOGDIR/${name}_t${th}_r${r}.out" se="$LOGDIR/${name}_t${th}_r${r}.err" line
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { tail -n 80 "$se" >&2 || true; return 4; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$mw" "$md" "$th" "$r" "$(field residue "$line")" "$(field wall_s "$line")" \
    "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$(field forward_high_s "$line")" "$(field reverse_high_s "$line")" >>"$RESULT"
}
for th in "${threads[@]}"; do
  for ((r=1;r<=REPEATS;++r)); do
    echo "=== Stage I control threads=$th repeat=$r rows=$ROWS self=w${SELF_W}d${SELF_D} ===" >&2
    run_one control 0 0 "$CONTROL_BIN" "$th" "$r"
  done
  while IFS=$'\t' read -r name mw md bin err; do
    [[ "$name" == candidate ]] && continue
    for ((r=1;r<=REPEATS;++r)); do
      echo "=== Stage I $name threads=$th repeat=$r rows=$ROWS self=w${SELF_W}d${SELF_D} mate=w${mw}d${md} ===" >&2
      run_one "$name" "$mw" "$md" "$bin" "$th" "$r"
    done
  done <"$BINS"
done

python3 - "$RESULT" "$RESOURCE" "$BINS" "$WINNER_ENV" "$CONTROL_BIN" "$ROWS" "$SELF_W" "$SELF_D" "$SELF_EVICT" "$MATE_EVICT" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,bins_path,winner,control_bin,rows_arg,self_w,self_d,self_evict,mate_evict=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
rr=list(csv.DictReader(open(resource),delimiter='\t'))
bins={r['candidate']:r for r in csv.DictReader(open(bins_path),delimiter='\t')}
if not rows: raise SystemExit('no Stage-I rows')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL Stage-I residue mismatch '+repr({(r['candidate'],r['threads'],r['repeat']):r['residue'] for r in rows}))
resources={name:[] for name in ['control',*bins]}
for r in rr:
    try: resources[r['candidate']].append((int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes'])))
    except (ValueError,KeyError): pass
agg=[]
for name,t in {(r['candidate'],int(r['threads'])) for r in rows}:
    rs=[r for r in rows if r['candidate']==name and int(r['threads'])==t]
    wall=statistics.median(float(r['wall_s']) for r in rs)
    hs=[]
    for r in rs:
        try: hs.append(float(r['forward_high_s'])+float(r['reverse_high_s']))
        except ValueError: pass
    high=statistics.median(hs) if hs else math.nan
    rv=resources[name]
    regs=max((x[0] for x in rv),default=-1)
    ss=max((x[1] for x in rv),default=-1); sl=max((x[2] for x in rv),default=-1)
    clean=len(rv)>=2 and ss==0 and sl==0
    agg.append((wall,name,t,high,regs,ss,sl,clean))
def rank(x): return (x[0],x[3] if not math.isnan(x[3]) else math.inf,x[4],x[2])
for x in sorted(agg,key=rank):
    print(f'STAGE_I candidate={x[1]} threads={x[2]} wall_s={x[0]:.9f} high_s={x[3]:.9f} regs={x[4]} spill_store={x[5]} spill_load={x[6]} spill_free={int(x[7])}',file=sys.stderr)
control=min((x for x in agg if x[1]=='control' and x[7]),default=None,key=rank)
tests=[x for x in agg if x[1]!='control' and x[7]]
if control is None or not tests: raise SystemExit('Stage-I needs spill-free control and mate candidate')
best=min(tests,key=rank)
meta=bins[best[1]]
speed=control[0]/best[0]
enabled=int(rank(best)<rank(control))
q=lambda x:shlex.quote(str(x))
vals={
 'B300_STAGEI_ROWS':rows_arg,
 'B300_STAGEI_RESIDUE':next(iter(res)),
 'B300_STAGEI_SELF_WIDTH':int(self_w),
 'B300_STAGEI_SELF_DISTANCE':int(self_d),
 'B300_STAGEI_SELF_EVICT':self_evict,
 'B300_STAGEI_CONTROL_BIN':control_bin,
 'B300_STAGEI_CONTROL_THREADS':control[2],
 'B300_STAGEI_CONTROL_WALL_S':f'{control[0]:.9f}',
 'B300_STAGEI_CONTROL_HIGH_S':f'{control[3]:.9f}',
 'B300_STAGEI_CONTROL_SPILL_FREE':1,
 'B300_STAGEI_MATE_WIDTH':int(meta['mate_width']),
 'B300_STAGEI_MATE_DISTANCE':int(meta['mate_distance']),
 'B300_STAGEI_MATE_EVICT':mate_evict,
 'B300_STAGEI_BIN':meta['binary'],
 'B300_STAGEI_THREADS':best[2],
 'B300_STAGEI_WALL_S':f'{best[0]:.9f}',
 'B300_STAGEI_HIGH_S':f'{best[3]:.9f}',
 'B300_STAGEI_SPILL_FREE':1,
 'B300_STAGEI_SPEEDUP':f'{speed:.9f}',
 'B300_STAGEI_BEST_ENABLED':enabled,
}
with open(winner,'w') as f:
    for k,v in vals.items(): f.write(k+'='+q(v)+'\n')
print('b300_stagei_exact_match=1')
print(f'b300_stagei_self_geometry=w{self_w}d{self_d}')
print(f'b300_stagei_mate_geometry=w{meta["mate_width"]}d{meta["mate_distance"]}')
print(f'b300_stagei_speedup={speed:.9f}x')
print(f'b300_stagei_best_enabled={enabled}')
PY

cat "$RESULT"
cat "$RESOURCE"
echo "b300-nextgen-hybrid8-nextmate-geometry-sweep OK rows=$ROWS self=w${SELF_W}d${SELF_D} mate_widths=${mate_widths[*]} mate_distances=${mate_distances[*]} winner_env=$WINNER_ENV" >&2
