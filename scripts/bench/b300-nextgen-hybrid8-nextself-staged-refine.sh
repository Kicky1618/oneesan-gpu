#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

INPUT_ENV="${INPUT_ENV:-$ONEESAN_ROOT/work/b300_nextgen_hybrid8_staged_winner.env}"
OUTPUT_ENV="${OUTPUT_ENV:-${INPUT_ENV%.env}_nextself.env}"
PREFIX="${PREFIX:-${OUTPUT_ENV%.env}}"
ARCH="${ARCH:-native}"
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SEARCH_REPEATS="${SEARCH_REPEATS:-1}"
VALIDATE_REPEATS="${VALIDATE_REPEATS:-1}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.005}"
VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
SUMMARY="${SUMMARY:-${PREFIX}.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$OUTPUT_ENV")" "$(dirname "$SUMMARY")"

[[ -s "$INPUT_ENV" ]] || { echo "missing INPUT_ENV=$INPUT_ENV" >&2; exit 2; }
[[ "$SEARCH_REPEATS" =~ ^[1-9][0-9]*$ && "$VALIDATE_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'repeat counts must be >=1' >&2; exit 2; }
python3 - "$MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < 1.0: raise SystemExit('MIN_SPEEDUP must be >=1')
PY
# shellcheck disable=SC1090
source "$INPUT_ENV"

for k in \
  B300_HYBRID8_STAGED_VALIDATED B300_HYBRID8_FINAL_ENABLED B300_HYBRID8_FINAL_THRESHOLD \
  B300_HYBRID8_FINAL_BIN B300_HYBRID8_FINAL_THREADS B300_HYBRID8_FINAL_SPILL_FREE \
  B300_HYBRID8_FINAL_STAGE_ROWS B300_HYBRID8_FINAL_STAGE_RESIDUE B300_HYBRID8_CORE_ROWS B300_HYBRID8_RESIDUE \
  B300_HYBRID8_HIGH_DROP_CHUNK B300_HYBRID8_RANDOM_CG B300_HYBRID8_RANDOM_CG_L2_FETCH_BYTES \
  B300_HYBRID8_PREFETCH_L2 B300_HYBRID8_DUALMASK B300_HYBRID8_CLOSURE_BATCH B300_HYBRID8_MAXRREGCOUNT; do
  [[ -n "${!k+x}" ]] || { echo "input hybrid env missing $k" >&2; exit 3; }
done
[[ "$B300_HYBRID8_STAGED_VALIDATED" == 1 && "$B300_HYBRID8_FINAL_ENABLED" == 1 && "$B300_HYBRID8_FINAL_SPILL_FREE" == 1 ]] || {
  echo 'hybrid8 next-self refine skipped: input hybrid is not staged-valid/spill-free' >&2
  cp "$INPUT_ENV" "$OUTPUT_ENV"
  cat >>"$OUTPUT_ENV" <<EOF
B300_HYBRID8_NS_STAGED_VALIDATED=0
B300_HYBRID8_NS_FINAL_BIN=$(printf '%q' "$B300_HYBRID8_FINAL_BIN")
B300_HYBRID8_NS_CONTROL_BIN=$(printf '%q' "$B300_HYBRID8_FINAL_BIN")
B300_HYBRID8_NS_FINAL_THREADS=$(printf '%q' "$B300_HYBRID8_FINAL_THREADS")
B300_HYBRID8_NS_REASON=input_not_promotable
EOF
  exit 0
}
[[ -x "$B300_HYBRID8_FINAL_BIN" ]] || { echo 'input hybrid final binary missing' >&2; exit 3; }
THREADS="$B300_HYBRID8_FINAL_THREADS"
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32 && THREADS<=1024 && THREADS%32==0)) || { echo 'bad input hybrid threads' >&2; exit 3; }
THRESHOLD="$B300_HYBRID8_FINAL_THRESHOLD"
[[ "$THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'bad input hybrid threshold' >&2; exit 3; }
SEARCH_ROWS="$B300_HYBRID8_CORE_ROWS"
[[ "$SEARCH_ROWS" =~ ^[1-9][0-9]*$ ]] && ((SEARCH_ROWS<=28)) || { echo 'bad input core rows' >&2; exit 3; }
for rows in $VALIDATE_ROWS; do [[ "$rows" =~ ^[1-9][0-9]*$ ]] && ((rows<=28)) || { echo "bad validation rows=$rows" >&2; exit 2; }; done

CONTROL_BIN="$B300_HYBRID8_FINAL_BIN"
NS_BIN="${NS_BIN:-$ONEESAN_BUILD_DIR/b300_nextgen_hybrid8_t${THRESHOLD}_nextself_h${B300_HYBRID8_HIGH_DROP_CHUNK}_cg${B300_HYBRID8_RANDOM_CG}_l2${B300_HYBRID8_RANDOM_CG_L2_FETCH_BYTES}_pre${B300_HYBRID8_PREFETCH_L2}_d${B300_HYBRID8_DUALMASK}_b${B300_HYBRID8_CLOSURE_BATCH}_r${B300_HYBRID8_MAXRREGCOUNT}_t${THREADS}_n27}"
BUILD_OUT="${PREFIX}.build.out"
BUILD_ERR="${PREFIX}.build.err"

echo "=== build hybrid8+next-self threshold=$THRESHOLD threads=$THREADS ===" >&2
N=27 ARCH="$ARCH" OUT="$NS_BIN" HIGH_DROP_CHUNK="$B300_HYBRID8_HIGH_DROP_CHUNK" \
  RECURRENCE_ILP=2 RECURRENCE_HYBRID_ILP8=1 RECURRENCE_HYBRID_ILP8_MIN_STATES="$THRESHOLD" RECURRENCE_HYBRID_ILP8_NEXTSELF=1 \
  RANDOM_CG="$B300_HYBRID8_RANDOM_CG" RANDOM_CG_L2_FETCH_BYTES="$B300_HYBRID8_RANDOM_CG_L2_FETCH_BYTES" PREFETCH_L2="$B300_HYBRID8_PREFETCH_L2" \
  DUALMASK="$B300_HYBRID8_DUALMASK" CLOSURE_BATCH="$B300_HYBRID8_CLOSURE_BATCH" MAXRREGCOUNT="$B300_HYBRID8_MAXRREGCOUNT" \
  BUILD_ERR="$BUILD_ERR" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$BUILD_OUT" 2>"${PREFIX}.build.driver.err"
[[ -x "$NS_BIN" ]] || { echo 'hybrid8+next-self binary missing' >&2; exit 4; }
grep -Fq "recurrence_hybrid_ilp8=1 recurrence_hybrid_ilp8_min_states=$THRESHOLD recurrence_hybrid_ilp8_nextself=1" "$BUILD_OUT" || { echo 'hybrid8+next-self build marker missing' >&2; exit 4; }

python3 "$PARSER" "$BUILD_ERR" --header --label hybrid8_nextself --contains main_pull_kernel_ilp2 --contains main_pull_kernel_ilp8_hybrid >"$RESOURCE"
read -r REGS_MAX SPILL_STORE_MAX SPILL_LOAD_MAX RESOURCE_ROWS < <(python3 - "$RESOURCE" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if len(rows)<2: raise SystemExit(f'expected ILP2+ILP8 ptxas rows, got {len(rows)}')
def vals(k):
 out=[]
 for r in rows:
  try: out.append(int(r[k]))
  except (ValueError,TypeError): raise SystemExit(f'unknown ptxas {k}: {r!r}')
 return out
regs=vals('registers'); ss=vals('spill_store_bytes'); sl=vals('spill_load_bytes')
print(max(regs),max(ss),max(sl),len(rows))
PY
)
[[ "$SPILL_STORE_MAX" == 0 && "$SPILL_LOAD_MAX" == 0 ]] || { echo "hybrid8+next-self spills store=$SPILL_STORE_MAX load=$SPILL_LOAD_MAX" >&2; exit 4; }

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
printf 'rows\tprofile\trepeat\tresidue\twall_s\n' >"$SUMMARY"
run_one(){
  local rows="$1" profile="$2" bin="$3" rep="$4"
  local so="$LOGDIR/${profile}_rows${rows}_r${rep}.out" se="$LOGDIR/${profile}_rows${rows}_r${rep}.err"
  set +e
  B300_ROW_LIMIT="$rows" GRIDFP_THREADS="$THREADS" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"
  local rc=$?
  set -e
  ((rc==0)) || { echo "$profile rows=$rows repeat=$rep failed rc=$rc" >&2; tail -n 100 "$se" >&2 || true; exit "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$profile rows=$rows missing backend line" >&2; exit 5; }
  local residue="$(field residue "$line")" wall="$(field wall_s "$line")"
  [[ "$residue" =~ ^[0-9]+$ && -n "$wall" ]] || { echo "$profile rows=$rows malformed result" >&2; exit 5; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$rows" "$profile" "$rep" "$residue" "$wall" >>"$SUMMARY"
}

stage_rows=("$SEARCH_ROWS")
for rows in $VALIDATE_ROWS; do
  seen=0; for x in "${stage_rows[@]}"; do [[ "$x" == "$rows" ]] && seen=1; done
  ((seen)) || stage_rows+=("$rows")
done
for rows in "${stage_rows[@]}"; do
  reps="$VALIDATE_REPEATS"; [[ "$rows" == "$SEARCH_ROWS" ]] && reps="$SEARCH_REPEATS"
  for ((r=1;r<=reps;++r)); do run_one "$rows" control "$CONTROL_BIN" "$r"; done
  for ((r=1;r<=reps;++r)); do run_one "$rows" nextself "$NS_BIN" "$r"; done
done

read -r VALID MIN_OBS_SPEED FINAL_SPEED FINAL_ROWS FINAL_RES < <(python3 - "$SUMMARY" "$SEARCH_ROWS" "$B300_HYBRID8_RESIDUE" "$B300_HYBRID8_FINAL_STAGE_ROWS" "$B300_HYBRID8_FINAL_STAGE_RESIDUE" "$MIN_SPEEDUP" <<'PY'
import csv,statistics,sys
summary,core_rows,core_res,final_rows,final_res,min_speed=sys.argv[1:]
core_rows=int(core_rows); final_rows=int(final_rows); min_speed=float(min_speed)
rows=list(csv.DictReader(open(summary),delimiter='\t'))
if not rows: raise SystemExit('no hybrid8 next-self rows')
by={}
for r in rows: by.setdefault((int(r['rows']),r['profile']),[]).append(r)
all_rows=sorted({k[0] for k in by})
speeds=[]; last=None
for n in all_rows:
 c=by.get((n,'control'),[]); p=by.get((n,'nextself'),[])
 if not c or not p: raise SystemExit(f'missing control/nextself rows={n}')
 cres={r['residue'] for r in c}; pres={r['residue'] for r in p}
 if len(cres)!=1 or len(pres)!=1 or cres!=pres:
  raise SystemExit(f'FATAL hybrid8-nextself residue mismatch rows={n} control={cres} nextself={pres}')
 residue=next(iter(cres))
 if n==core_rows and residue!=core_res:
  raise SystemExit(f'FATAL core-row residue mismatch got={residue} expected={core_res}')
 if n==final_rows and residue!=final_res:
  raise SystemExit(f'FATAL final-stage residue mismatch got={residue} expected={final_res}')
 cw=statistics.median(float(r['wall_s']) for r in c)
 pw=statistics.median(float(r['wall_s']) for r in p)
 speed=cw/pw
 speeds.append(speed); last=(n,residue,speed,cw,pw)
 print(f'HYBRID8_NEXTSELF_STAGE rows={n} control_wall_s={cw:.9f} nextself_wall_s={pw:.9f} speedup={speed:.6f}x residue={residue} exact=1',file=sys.stderr)
valid=int(all(x>=min_speed for x in speeds))
assert last is not None
print(valid,f'{min(speeds):.9f}',f'{last[2]:.9f}',last[0],last[1])
PY
)

cp "$INPUT_ENV" "$OUTPUT_ENV"
cat >>"$OUTPUT_ENV" <<EOF
B300_HYBRID8_NS_STAGED_VALIDATED=$VALID
B300_HYBRID8_NS_FINAL_BIN=$(printf '%q' "$([[ "$VALID" == 1 ]] && printf '%s' "$NS_BIN" || printf '%s' "$CONTROL_BIN")")
B300_HYBRID8_NS_CONTROL_BIN=$(printf '%q' "$CONTROL_BIN")
B300_HYBRID8_NS_FINAL_THREADS=$THREADS
B300_HYBRID8_NS_THRESHOLD=$THRESHOLD
B300_HYBRID8_NS_MIN_OBSERVED_SPEEDUP=$MIN_OBS_SPEED
B300_HYBRID8_NS_FINAL_STAGE_SPEEDUP=$FINAL_SPEED
B300_HYBRID8_NS_FINAL_STAGE_ROWS=$FINAL_ROWS
B300_HYBRID8_NS_FINAL_STAGE_RESIDUE=$FINAL_RES
B300_HYBRID8_NS_SPILL_FREE=1
B300_HYBRID8_NS_REGISTERS_MAX=$REGS_MAX
B300_HYBRID8_NS_RESOURCE_ROWS=$RESOURCE_ROWS
B300_HYBRID8_NS_BUILD_ERR=$(printf '%q' "$BUILD_ERR")
B300_HYBRID8_NS_SUMMARY=$(printf '%q' "$SUMMARY")
B300_HYBRID8_NS_INPUT_ENV=$(printf '%q' "$INPUT_ENV")
EOF

cat "$SUMMARY" >&2
cat "$OUTPUT_ENV"
echo "b300-nextgen-hybrid8-nextself-staged-refine OK validated=$VALID threshold=$THRESHOLD min_speedup=$MIN_OBS_SPEED final_stage_speedup=$FINAL_SPEED final_rows=$FINAL_ROWS residue=$FINAL_RES spill_free=1 output_env=$OUTPUT_ENV" >&2
