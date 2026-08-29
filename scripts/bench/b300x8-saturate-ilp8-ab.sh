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
ILP8_REGS="${ILP8_REGS:-0 96 128 160}"
REBUILD="${REBUILD:-0}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_saturate_ilp8_ab_n${N}}"
SUMMARY="${SUMMARY:-$LOGDIR/summary.tsv}"
mkdir -p "$LOGDIR"

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

printf 'profile\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tregs_max\tspill_store_max_bytes\tspill_load_max_bytes\n' >"$SUMMARY"
cat "$SUMMARY"

run_one(){
  local name="$1";shift
  local log="$LOGDIR/${name}.log" dmon="$LOGDIR/${name}.dmon"
  echo "=== $name ===" >&2
  : >"$dmon"
  nvidia-smi dmon -s u -d 1 >"$dmon" 2>&1 &
  local mpid=$!
  set +e
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$mpid" 2>/dev/null || true
  wait "$mpid" 2>/dev/null || true
  (( rc==0 )) || return "$rc"

  local line wall active_max active_sum mem_avg mem_max regs_max spill_store spill_load row
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  wall="$(sed -nE 's/.* wall_s=([^ ]+).*/\1/p' <<<"$line")"
  active_max="$(sed -nE 's/.* active_max_s=([^ ]+).*/\1/p' <<<"$line")"
  active_sum="$(sed -nE 's/.* active_sum_s=([^ ]+).*/\1/p' <<<"$line")"
  read -r mem_avg mem_max < <(awk '
    $1 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {s+=$3;n++;if($3>m)m=$3}
    END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}
  ' "$dmon")
  read -r regs_max spill_store spill_load < <(python3 - "$log" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='replace').read()
regs=[int(x) for x in re.findall(r'Used\s+(\d+)\s+registers',text)]
spill=[(int(a),int(b)) for a,b in re.findall(r'(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads',text)]
print(max(regs) if regs else 'nan', max((x for x,_ in spill),default='nan'), max((y for _,y in spill),default='nan'))
PY
)
  row="$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$name" "${wall:-nan}" "${active_max:-nan}" "${active_sum:-nan}" "$mem_avg" "$mem_max" "$regs_max" "$spill_store" "$spill_load")"
  printf '%s\n' "$row" | tee -a "$SUMMARY"
}

run_one ilp4 \
  env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
      RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" REBUILD="$REBUILD" \
  "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" "$N" "$MOD"

for r in $ILP8_REGS; do
  run_one "ilp8_r${r}" \
    env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
        RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" MAXRREGCOUNT="$r" REBUILD="$REBUILD" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD"
done

python3 - "$SUMMARY" <<'PY'
import csv,math,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ilp8=[r for r in rows if r['profile'].startswith('ilp8_') and r['wall_s'] not in ('','nan')]
if not ilp8:
    raise SystemExit('no successful ILP8 candidate')
def num(r,k,default=math.inf):
    try:return float(r[k])
    except (ValueError,TypeError):return default
def spill_free(r):
    return num(r,'spill_store_max_bytes',1)==0 and num(r,'spill_load_max_bytes',1)==0
clean=[r for r in ilp8 if spill_free(r)]
pool=clean or ilp8
best=min(pool,key=lambda r:(num(r,'wall_s'),-num(r,'mc_avg_pct',-math.inf)))
base=next((r for r in rows if r['profile']=='ilp4' and r['wall_s'] not in ('','nan')),None)
if base:
    print(f"ILP8_SELECTED profile={best['profile']} wall_s={best['wall_s']} speedup={num(base,'wall_s')/num(best,'wall_s'):.6f}x mc_avg_pct={best['mc_avg_pct']} regs_max={best['regs_max']} spill_store_max={best['spill_store_max_bytes']} spill_load_max={best['spill_load_max_bytes']} spill_free_pool={int(bool(clean))}",file=sys.stderr)
else:
    print(f"ILP8_SELECTED profile={best['profile']} wall_s={best['wall_s']} mc_avg_pct={best['mc_avg_pct']} regs_max={best['regs_max']} spill_store_max={best['spill_store_max_bytes']} spill_load_max={best['spill_load_max_bytes']} spill_free_pool={int(bool(clean))}",file=sys.stderr)
PY

echo "summary=$SUMMARY logs=$LOGDIR" >&2
