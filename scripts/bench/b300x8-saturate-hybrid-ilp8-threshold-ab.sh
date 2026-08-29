#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608}"
REPEATS="${REPEATS:-1}"
REBUILD="${REBUILD:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_hybrid_ilp8_threshold_ab_n${N}_rows${ROWS}}"
RESULT="${RESULT:-$LOGDIR/results.tsv}"
mkdir -p "$LOGDIR"

[[ "$N" == 27 ]] || { echo 'hybrid ILP8 threshold A/B currently targets n=27' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= 28 )) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && (( THREADS >= 32 && THREADS <= 1024 && THREADS % 32 == 0 )) || { echo 'GRIDFP_THREADS must be warp multiple 32..1024' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 1 )) || { echo 'REPEATS must be >=1' >&2; exit 2; }
for x in RANDOM_CG WARP_SCAN REBUILD; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

read -r -a threshold_list <<<"$THRESHOLDS"
(( ${#threshold_list[@]} > 0 )) || { echo 'ILP8_THRESHOLDS is empty' >&2; exit 2; }
declare -A seen=()
for t in "${threshold_list[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "invalid threshold=$t" >&2; exit 2; }
  [[ -z "${seen[$t]+x}" ]] || { echo "duplicate threshold=$t" >&2; exit 2; }
  seen[$t]=1
done
[[ -n "${seen[0]+x}" ]] || { echo 'ILP8_THRESHOLDS must include 0 (always-ILP8 baseline)' >&2; exit 2; }

field(){
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

printf 'mode\tthreshold\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\n' >"$RESULT"

run_one(){
  local mode="$1" threshold="$2" rep="$3"; shift 3
  local tag="${mode}_t${threshold}_r${rep}"
  local log="$LOGDIR/${tag}.log" dmon="$LOGDIR/${tag}.dmon"
  echo "=== hybrid ILP8 threshold A/B mode=$mode threshold=$threshold repeat=$rep ===" >&2
  : >"$dmon"
  nvidia-smi dmon -s u -d 1 >"$dmon" 2>&1 &
  local mpid=$!
  set +e
  "$@" >"$log" 2>&1
  local rc=$?
  set -e
  kill "$mpid" 2>/dev/null || true
  wait "$mpid" 2>/dev/null || true
  (( rc == 0 )) || { tail -n 120 "$log" >&2 || true; echo "$tag failed rc=$rc" >&2; return "$rc"; }

  local line residue wall active_max active_sum mem_avg mem_max
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing backend result line" >&2; return 4; }
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  active_max="$(field active_max_s "$line")"
  active_sum="$(field active_sum_s "$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$tag missing residue/wall_s" >&2; return 4; }
  read -r mem_avg mem_max < <(awk '
    $1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {s+=$3;n++;if($3>m)m=$3}
    END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}
  ' "$dmon")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$threshold" "$rep" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mem_avg" "$mem_max" >>"$RESULT"
}

for (( rep=1; rep<=REPEATS; ++rep )); do
  if (( rep % 2 == 1 )); then
    run_one ilp4 NA "$rep" \
      env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
      "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"
    for t in "${threshold_list[@]}"; do
      run_one hybrid "$t" "$rep" \
        env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" ILP8_MIN_STATES="$t" REBUILD="$REBUILD" \
        "$ONEESAN_ROOT/scripts/run/b300x8-saturate-hybrid-ilp8.sh" "$N" "$MOD"
    done
  else
    for (( j=${#threshold_list[@]}-1; j>=0; --j )); do
      t="${threshold_list[$j]}"
      run_one hybrid "$t" "$rep" \
        env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" ILP8_MIN_STATES="$t" REBUILD="$REBUILD" \
        "$ONEESAN_ROOT/scripts/run/b300x8-saturate-hybrid-ilp8.sh" "$N" "$MOD"
    done
    run_one ilp4 NA "$rep" \
      env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
      "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"
  fi
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows: raise SystemExit('no hybrid threshold rows')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL hybrid ILP4/ILP8 residue mismatch '+repr({(r['mode'],r['threshold'],r['repeat']):r['residue'] for r in rows}))
by={}
for r in rows:
    key=(r['mode'],r['threshold'])
    by.setdefault(key,[]).append(float(r['wall_s']))
med={k:statistics.median(v) for k,v in by.items()}
base=med.get(('ilp4','NA'))
if base is None: raise SystemExit('missing pure ILP4 baseline')
best_key=min(med,key=med.get); best=med[best_key]
for key,wall in sorted(med.items(),key=lambda kv:kv[1]):
    mode,t=key
    speed=base/wall
    print('HYBRID_ILP8_CANDIDATE',f'mode={mode}',f'threshold={t}',f'median_wall_s={wall:.9f}',f'speedup_vs_ilp4={speed:.6f}x',file=sys.stderr)
print('HYBRID_ILP8_SELECTED',
      f'mode={best_key[0]}',f'threshold={best_key[1]}',
      f'median_wall_s={best:.9f}',f'speedup_vs_ilp4={base/best:.6f}x',
      f'residue={next(iter(res))}',f'exact_gate=1',file=sys.stderr)
PY

cat "$RESULT"
echo "hybrid ILP8 threshold A/B OK result=$RESULT rows=$ROWS repeats=$REPEATS" >&2
