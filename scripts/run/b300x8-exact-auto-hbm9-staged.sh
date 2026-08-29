#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo "staged HBM selector currently targets n=27" >&2; exit 2; }
ARCH="${ARCH:-native}"
PRIME="${SMOKE_PRIME:-4294967291}"
MAX_WINDOW="${MAX_WINDOW:-14}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"
FORCED_THREADS="${GRIDFP_THREADS:-256}"
FORCED_ROWS="${FORCED_PRESELECT_ROWS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm_staged_n27}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PRESELECT="${PRESELECT:-${PREFIX}_forced_preselect.tsv}"
REBUILD="${REBUILD:-1}"
mkdir -p "$LOGDIR" "$(dirname "$PRESELECT")"
[[ "$REBUILD" == 0 || "$REBUILD" == 1 ]] || { echo "REBUILD must be 0 or 1" >&2; exit 2; }
[[ "$FORCED_ROWS" =~ ^[0-9]+$ ]] && ((FORCED_ROWS>=1&&FORCED_ROWS<28)) || { echo "FORCED_PRESELECT_ROWS must be 1..27" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo "need 8 visible GPUs" >&2; exit 2; }

# mode | ILP | high-drop-chunk | low-main-recurrence | high-main-recurrence | unified-main-recurrence
SPECS=(
  'forced_ilp1|1|0|0|0|0'
  'forced|2|0|0|0|0'
  'forced_ilp3|3|0|0|0|0'
  'forced_ilp4|4|0|0|0|0'
  'forced_high|2|1|0|0|0'
  'forced_lowrec|2|0|1|0|0'
  'forced_highrec|2|0|0|1|0'
  'forced_highrec_high|2|1|0|1|0'
  'forced_mainrec|2|0|0|0|1'
  'forced_mainrec_high|2|1|0|0|1'
)
declare -A BINS BUILD_OK BUILD_ERR

build_metadata_ok(){
  local bout="$1" ilp="$2" high="$3" lowrec="$4" highrec="$5" mainrec="$6"
  [[ -f "$bout" ]] || return 1
  grep -Fq "main_pull_ilp=$ilp" "$bout" || return 1
  grep -Fq "high_drop_chunk=$high" "$bout" || return 1
  grep -Fq "low_main_recurrence=$lowrec" "$bout" || return 1
  grep -Fq "high_main_recurrence=$highrec" "$bout" || return 1
  grep -Fq "main_recurrence=$mainrec" "$bout" || return 1
  grep -Fq 'batch_row_limit_env=B300_ROW_LIMIT' "$bout" || return 1
  if [[ "$highrec" == 1 ]]; then
    grep -Fq 'high_p_lo=14 high_symbol_range=13..27 high_trit_positions=15 high_min_fixed=7 high_fixed_lt7_fallback=raw_mate_rank' "$bout" || return 1
  fi
  if [[ "$mainrec" == 1 ]]; then
    grep -Fq 'high_delta_bits=35 high_p_lo=14 high_symbol_range=13..27 high_trit_positions=15 high_min_fixed=7 high_fixed_lt7_fallback=raw_mate_rank' "$bout" || return 1
  fi
  return 0
}

build_candidate(){
  local mode="$1" ilp="$2" high="$3" lowrec="$4" highrec="$5" mainrec="$6"
  local bin="$ONEESAN_BUILD_DIR/b300_forced_pre_${mode}_n27" bout="$LOGDIR/${mode}.pre.build.out" berr="$LOGDIR/${mode}.pre.build.err"
  BINS[$mode]="$bin";BUILD_ERR[$mode]="$berr"
  if [[ "$REBUILD" == 0 && -x "$bin" ]] && build_metadata_ok "$bout" "$ilp" "$high" "$lowrec" "$highrec" "$mainrec"; then
    BUILD_OK[$mode]=1;return 0
  fi
  if [[ "$REBUILD" == 0 && -x "$bin" ]]; then echo "stale or unverified binary: rebuilding $mode" >&2;fi
  echo "=== staged build $mode ilp=$ilp high=$high lowrec=$lowrec highrec=$highrec mainrec=$mainrec ===" >&2
  set +e
  N=27 ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP="$ilp" HIGH_DROP_CHUNK="$high" LOW_MAIN_RECURRENCE="$lowrec" HIGH_MAIN_RECURRENCE="$highrec" MAIN_RECURRENCE="$mainrec" \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
    LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$bout" 2>"$berr"
  local rc=$?
  set -e
  if ((rc)); then echo "warning: exclude $mode build rc=$rc" >&2;BUILD_OK[$mode]=0;return 0;fi
  if ! build_metadata_ok "$bout" "$ilp" "$high" "$lowrec" "$highrec" "$mainrec"; then
    echo "warning: exclude $mode build metadata mismatch" >&2;BUILD_OK[$mode]=0;return 0
  fi
  BUILD_OK[$mode]=1
}

for spec in "${SPECS[@]}";do IFS='|' read -r mode ilp high lowrec highrec mainrec<<<"$spec";build_candidate "$mode" "$ilp" "$high" "$lowrec" "$highrec" "$mainrec";done

field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1;}
row_smoke(){
  local mode="$1" bin="${BINS[$1]}" so="$LOGDIR/${1}.pre.out" se="$LOGDIR/${1}.pre.err" dm="$LOGDIR/${1}.pre.dmon"
  if [[ "${BUILD_OK[$mode]:-0}" != 1 ]];then printf '%s\t%s\tfailed:build\tNA\tNA\tNA\tNA\n' "$mode" "$bin" >>"$PRESELECT";return 0;fi
  echo "=== forced preselect $mode rows=$FORCED_ROWS ===" >&2
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e
  B300_ROW_LIMIT="$FORCED_ROWS" GRIDFP_THREADS="$FORCED_THREADS" "$bin" 27 "$FORCED_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  local rc=$?
  set -e
  kill "$dp" 2>/dev/null||true;wait "$dp" 2>/dev/null||true
  if ((rc));then printf '%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\n' "$mode" "$bin" "$rc" >>"$PRESELECT";return 0;fi
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so"|tail -n1||true)"
  if [[ -z "$line" ]];then printf '%s\t%s\tfailed:no_result\tNA\tNA\tNA\tNA\n' "$mode" "$bin" >>"$PRESELECT";return 0;fi
  grep -Fq " rows=$FORCED_ROWS calibration=1 "<<<"$line"||{ printf '%s\t%s\tfailed:no_row_metadata\tNA\tNA\tNA\tNA\n' "$mode" "$bin" >>"$PRESELECT";return 0;}
  local avg mx;read -r avg mx < <(awk 'BEGIN{n=0;s=0;m=0} !/^#/ && NF>=3 {x=$3+0;s+=x;if(x>m)m=x;n++} END{if(n)printf "%.3f %.3f\n",s/n,m;else print "NA NA"}' "$dm")
  printf '%s\t%s\tok\t%s\t%s\t%s\t%s\n' "$mode" "$bin" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" >>"$PRESELECT"
}
printf 'backend\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\n' >"$PRESELECT"
for spec in "${SPECS[@]}";do IFS='|' read -r mode _<<<"$spec";row_smoke "$mode";done

selection="$(python3 - "$PRESELECT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));ok=[r for r in rows if r['status']=='ok']
if not ok:raise SystemExit('no successful forced preselection candidate')
res={r['residue'] for r in ok}
if len(res)!=1:raise SystemExit('FATAL forced partial-row residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])):print('FORCED_PRESELECT',r['backend'],'wall_s='+r['wall_s'],'mc_avg='+r['mc_avg_pct'],'mc_max='+r['mc_max_pct'],file=sys.stderr)
b=min(ok,key=lambda x:float(x['wall_s']));print('\t'.join([b['backend'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST_FORCED BEST_FORCED_BIN BEST_PARTIAL_RES BEST_PARTIAL_WALL<<<"$selection"
echo "FORCED PRESELECTED backend=$BEST_FORCED partial_wall_s=$BEST_PARTIAL_WALL partial_residue=$BEST_PARTIAL_RES rows=$FORCED_ROWS" >&2;cat "$PRESELECT" >&2

export FORCED_OVERRIDE_BIN="$BEST_FORCED_BIN"
export FORCED_OVERRIDE_LABEL="$BEST_FORCED"
export FORCED_OVERRIDE_BUILD_ERR="${BUILD_ERR[$BEST_FORCED]}"
export REBUILD_BUCKETS="${REBUILD_BUCKETS:-$REBUILD}"
export PREFIX="${FINAL_PREFIX:-$ONEESAN_ROOT/work/b300_exact_final5_n27}"
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-final5.sh" 27 "$@"
