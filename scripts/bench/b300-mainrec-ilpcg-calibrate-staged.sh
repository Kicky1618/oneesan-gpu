#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"
SEARCH_ROWS="${SEARCH_ROWS:-1}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
TRANSFORM_MIN_SPEEDUP="${TRANSFORM_MIN_SPEEDUP:-1.01}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_staged}"
SEARCH_LOG="${SEARCH_LOG:-${PREFIX}_search.log}"
RESULT="${RESULT:-${PREFIX}.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"

[[ "$SEARCH_ROWS" =~ ^[0-9]+$ ]] && ((SEARCH_ROWS>=1&&SEARCH_ROWS<=28)) || { echo 'SEARCH_ROWS must be 1..28' >&2; exit 2; }
[[ "$SEARCH_REPEATS" =~ ^[0-9]+$ ]] && ((SEARCH_REPEATS>=1)) || { echo 'SEARCH_REPEATS must be >=1' >&2; exit 2; }
[[ "$VALIDATE_REPEATS" =~ ^[0-9]+$ ]] && ((VALIDATE_REPEATS>=1)) || { echo 'VALIDATE_REPEATS must be >=1' >&2; exit 2; }
python3 - "$TRANSFORM_MIN_SPEEDUP" "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('TRANSFORM_MIN_SPEEDUP must be >=1')
if float(sys.argv[2]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY
for rows in $VALIDATE_ROWS; do
  [[ "$rows" =~ ^[0-9]+$ ]] && ((rows>=1&&rows<=28)) || { echo "bad VALIDATE_ROWS entry: $rows" >&2; exit 2; }
done
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

getv(){ local key="$1" file="$2"; sed -nE "s/^${key}=([^[:space:]]+).*/\\1/p" "$file" | tail -n1; }

echo "=== stage 1: broad ILP/CG search rows=$SEARCH_ROWS ===" >&2
ARCH="$ARCH" MOD="$MOD" ROWS="$SEARCH_ROWS" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" \
  THREADS_LIST="$THREADS_LIST" HIGHDROP_LIST="$HIGHDROP_LIST" REPEATS="$SEARCH_REPEATS" \
  TRANSFORM_MIN_SPEEDUP="$TRANSFORM_MIN_SPEEDUP" PREFIX="${PREFIX}_search" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilpcg-calibrate.sh" | tee "$SEARCH_LOG"

[[ "$(getv b300_mainrec_ilpcg_calibrate_exact_gates "$SEARCH_LOG")" == 1 ]] || { echo 'stage-1 exact gate missing' >&2; exit 3; }
SEARCH_RESIDUE="$(getv b300_mainrec_ilpcg_calibrate_residue "$SEARCH_LOG")"
BASE_HIGH="$(getv b300_mainrec_ilpcg_calibrate_global_base_high_drop "$SEARCH_LOG")"
BASE_THREADS="$(getv b300_mainrec_ilpcg_calibrate_global_base_threads "$SEARCH_LOG")"
BASE_WALL="$(getv b300_mainrec_ilpcg_calibrate_global_base_wall_s "$SEARCH_LOG")"
FINAL_HIGH="$(getv b300_mainrec_ilpcg_calibrate_final_high_drop_chunk "$SEARCH_LOG")"
FINAL_MODE="$(getv b300_mainrec_ilpcg_calibrate_final_mode "$SEARCH_LOG")"
FINAL_ILP="$(getv b300_mainrec_ilpcg_calibrate_final_ilp "$SEARCH_LOG")"
FINAL_CG="$(getv b300_mainrec_ilpcg_calibrate_final_random_cg "$SEARCH_LOG")"
FINAL_THREADS="$(getv b300_mainrec_ilpcg_calibrate_final_threads "$SEARCH_LOG")"
ADOPT="$(getv b300_mainrec_ilpcg_calibrate_adopt_transform "$SEARCH_LOG")"
[[ "$BASE_HIGH" == 0 || "$BASE_HIGH" == 1 ]] || exit 3
[[ "$FINAL_HIGH" == 0 || "$FINAL_HIGH" == 1 ]] || exit 3
[[ "$BASE_THREADS" =~ ^[0-9]+$ && "$FINAL_THREADS" =~ ^[0-9]+$ ]] || exit 3
[[ "$FINAL_MODE" =~ ^ilp(2|4|8)(cg)?$ ]] || { echo "bad stage-1 mode=$FINAL_MODE" >&2; exit 3; }
[[ "$ADOPT" == 0 || "$ADOPT" == 1 ]] || exit 3

BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_ilpcg_ilp2_hd${BASE_HIGH}_n27"
FINAL_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_ilpcg_${FINAL_MODE}_hd${FINAL_HIGH}_n27"
[[ -x "$BASE_BIN" ]] || { echo "missing stage baseline binary=$BASE_BIN" >&2; exit 3; }
[[ -x "$FINAL_BIN" ]] || { echo "missing stage winner binary=$FINAL_BIN" >&2; exit 3; }

printf 'stage_rows\tvariant\trepeat\thigh_drop\tmode\tthreads\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"

sample_mem(){
  local pid="$1" out="$2"
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

run_one(){
  local rows="$1" variant="$2" high="$3" mode="$4" threads="$5" bin="$6" rep="$7"
  local stem="$LOGDIR/r${rows}_${variant}_r${rep}" so="${stem}.out" se="${stem}.err" util="${stem}.mem"
  set +e
  B300_ROW_LIMIT="$rows" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  local pid=$!
  sample_mem "$pid" "$util" & local spid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$spid" || true
  ((rc==0)) || { echo "stage rows=$rows $variant failed rc=$rc" >&2; tail -n 100 "$se" >&2 || true; return "$rc"; }
  local line residue wall mc_avg mc_max mc_n
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "stage rows=$rows $variant missing result line" >&2; return 4; }
  residue="$(sed -nE 's/(^|.*[[:space:]])residue=([^[:space:]]+).*/\2/p' <<<"$line" | tail -n1)"
  wall="$(sed -nE 's/(^|.*[[:space:]])wall_s=([^[:space:]]+).*/\2/p' <<<"$line" | tail -n1)"
  [[ -n "$residue" && -n "$wall" ]] || return 4
  read -r mc_avg mc_max mc_n < <(awk '{s+=$1;if($1>m)m=$1;n++}END{if(n)printf "%.6f %.6f %d\n",s/n,m,n;else print "nan nan 0"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rows" "$variant" "$rep" "$high" "$mode" "$threads" "$residue" "$wall" "$mc_avg" "$mc_max" "$mc_n" >>"$RESULT"
}

if [[ "$ADOPT" == 1 ]]; then
  for rows in $VALIDATE_ROWS; do
    echo "=== stage validate rows=$rows baseline vs $FINAL_MODE ===" >&2
    for ((r=1;r<=VALIDATE_REPEATS;++r)); do
      run_one "$rows" baseline "$BASE_HIGH" ilp2 "$BASE_THREADS" "$BASE_BIN" "$r"
      run_one "$rows" candidate "$FINAL_HIGH" "$FINAL_MODE" "$FINAL_THREADS" "$FINAL_BIN" "$r"
    done
  done
fi

python3 - "$RESULT" "$WINNER_ENV" "$TRANSFORM_MIN_SPEEDUP" "$ADOPT" "$SEARCH_RESIDUE" \
  "$BASE_HIGH" "$BASE_THREADS" "$BASE_WALL" "$FINAL_HIGH" "$FINAL_MODE" "$FINAL_ILP" "$FINAL_CG" "$FINAL_THREADS" <<'PY'
import csv,statistics,sys
(result,wenv,minsp_s,adopt_s,search_res,base_high,base_threads,base_wall,
 final_high,final_mode,final_ilp,final_cg,final_threads)=sys.argv[1:]
minsp=float(minsp_s); adopt=adopt_s=='1'
rows=list(csv.DictReader(open(result),delimiter='\t'))
validated=adopt
stage_speedups=[]
if adopt:
    stages=sorted({int(r['stage_rows']) for r in rows})
    if not stages: raise SystemExit('transformed winner requested but no validation stages ran')
    for stage in stages:
        sr=[r for r in rows if int(r['stage_rows'])==stage]
        b=[r for r in sr if r['variant']=='baseline']; c=[r for r in sr if r['variant']=='candidate']
        if not b or not c: raise SystemExit(f'missing baseline/candidate rows at stage {stage}')
        residues={r['residue'] for r in sr}
        if len(residues)!=1:
            raise SystemExit(f'FATAL staged residue mismatch rows={stage}: '+repr({r['variant']:r['residue'] for r in sr}))
        bw=statistics.median(float(r['wall_s']) for r in b)
        cw=statistics.median(float(r['wall_s']) for r in c)
        speed=bw/cw
        stage_speedups.append((stage,speed,bw,cw))
        if speed < minsp: validated=False
        bmc=[float(r['mc_avg_pct']) for r in b if r['mc_avg_pct']!='nan']
        cmc=[float(r['mc_avg_pct']) for r in c if r['mc_avg_pct']!='nan']
        bm=statistics.median(bmc) if bmc else float('nan')
        cm=statistics.median(cmc) if cmc else float('nan')
        print(f'STAGED rows={stage} baseline_wall={bw:.9f} candidate_wall={cw:.9f} speedup={speed:.6f}x baseline_mc={bm:.3f} candidate_mc={cm:.3f} mc_delta={cm-bm:.3f}pp exact=1',file=sys.stderr)

if validated:
    high,mode,ilp,cg,threads=final_high,final_mode,final_ilp,final_cg,final_threads
else:
    high,mode,ilp,cg,threads=base_high,'ilp2','2','0',base_threads
with open(wenv,'w') as f:
    f.write(f'B300_MAINREC_STAGED_VALIDATED={int(validated)}\n')
    f.write(f'B300_MAINREC_HIGH_DROP_CHUNK={high}\n')
    f.write(f'B300_MAINREC_MODE={mode}\n')
    f.write(f'B300_MAINREC_ILP={ilp}\n')
    f.write(f'B300_MAINREC_RANDOM_CG={cg}\n')
    f.write(f'B300_MAINREC_THREADS={threads}\n')
    f.write(f'B300_MAINREC_SEARCH_RESIDUE={search_res}\n')
    f.write(f'B300_MAINREC_TRANSFORM_MIN_SPEEDUP={minsp:.9f}\n')
    if stage_speedups:
        f.write('B300_MAINREC_STAGE_SPEEDUPS='+','.join(f'{s}:{v:.9f}' for s,v,_,_ in stage_speedups)+'\n')
print(f'b300_mainrec_ilpcg_staged_validated={int(validated)}')
print(f'b300_mainrec_ilpcg_staged_final_high_drop_chunk={high}')
print(f'b300_mainrec_ilpcg_staged_final_mode={mode}')
print(f'b300_mainrec_ilpcg_staged_final_ilp={ilp}')
print(f'b300_mainrec_ilpcg_staged_final_random_cg={cg}')
print(f'b300_mainrec_ilpcg_staged_final_threads={threads}')
print(f'b300_mainrec_ilpcg_staged_winner_env={wenv}')
PY

cat "$RESULT" >&2
cat "$WINNER_ENV"
echo "b300-mainrec-ilpcg-calibrate-staged OK result=$RESULT winner=$WINNER_ENV" >&2
