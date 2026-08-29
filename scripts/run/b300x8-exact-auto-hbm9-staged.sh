#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo "staged HBM9 selector currently targets n=27" >&2; exit 2; }
ARCH="${ARCH:-native}"
PRIME="${SMOKE_PRIME:-4294967291}"
MAX_WINDOW="${MAX_WINDOW:-14}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"
FORCED_THREADS="${GRIDFP_THREADS:-256}"
FORCED_ROWS="${FORCED_PRESELECT_ROWS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm9_staged_n27}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PRESELECT="${PRESELECT:-${PREFIX}_forced_preselect.tsv}"
REBUILD="${REBUILD:-1}"
mkdir -p "$LOGDIR" "$(dirname "$PRESELECT")"
[[ "$REBUILD" == 0 || "$REBUILD" == 1 ]] || { echo "REBUILD must be 0 or 1" >&2; exit 2; }
[[ "$FORCED_ROWS" =~ ^[0-9]+$ ]] && ((FORCED_ROWS>=1&&FORCED_ROWS<28)) || { echo "FORCED_PRESELECT_ROWS must be 1..27" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo "need 8 visible GPUs" >&2; exit 2; }

FORCED_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_n27"
FORCED_ILP3_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_ilp3_n27"
FORCED_ILP4_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_ilp4_n27"
FORCED_HIGH_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_high_n27"
FORCED_LOWREC_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_lowrec_n27"

build_forced(){
  local mode="$1" bin="$2" ilp="$3" high="$4" rec="$5"
  [[ "$REBUILD" == 1 || ! -x "$bin" ]] || return 0
  echo "=== staged build $mode ilp=$ilp high=$high lowrec=$rec ===" >&2
  N=27 ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP="$ilp" HIGH_DROP_CHUNK="$high" LOW_MAIN_RECURRENCE="$rec" \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
    LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$LOGDIR/${mode}.pre.build.out" 2>"$LOGDIR/${mode}.pre.build.err"
  grep -Fq "main_pull_ilp=$ilp" "$LOGDIR/${mode}.pre.build.out"
  grep -Fq "high_drop_chunk=$high" "$LOGDIR/${mode}.pre.build.out"
  grep -Fq "low_main_recurrence=$rec" "$LOGDIR/${mode}.pre.build.out"
  grep -Fq 'batch_row_limit_env=B300_ROW_LIMIT' "$LOGDIR/${mode}.pre.build.out"
}

build_forced forced "$FORCED_BIN" 2 0 0
build_forced forced_ilp3 "$FORCED_ILP3_BIN" 3 0 0
build_forced forced_ilp4 "$FORCED_ILP4_BIN" 4 0 0
build_forced forced_high "$FORCED_HIGH_BIN" 2 1 0
build_forced forced_lowrec "$FORCED_LOWREC_BIN" 2 0 1

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
row_smoke(){
  local mode="$1" bin="$2" so="$LOGDIR/${mode}.pre.out" se="$LOGDIR/${mode}.pre.err" dm="$LOGDIR/${mode}.pre.dmon"
  echo "=== forced preselect $mode rows=$FORCED_ROWS ===" >&2
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e
  B300_ROW_LIMIT="$FORCED_ROWS" GRIDFP_THREADS="$FORCED_THREADS" \
    "$bin" 27 "$FORCED_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  local rc=$?
  set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  if ((rc)); then printf '%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\n' "$mode" "$bin" "$rc" >>"$PRESELECT"; return 0; fi
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  if [[ -z "$line" ]]; then printf '%s\t%s\tfailed:no_result\tNA\tNA\tNA\tNA\n' "$mode" "$bin" >>"$PRESELECT"; return 0; fi
  grep -Fq " rows=$FORCED_ROWS calibration=1 " <<<"$line" || { printf '%s\t%s\tfailed:no_row_metadata\tNA\tNA\tNA\tNA\n' "$mode" "$bin" >>"$PRESELECT"; return 0; }
  local avg mx
  read -r avg mx < <(awk 'BEGIN{n=0;s=0;m=0} !/^#/ && NF>=3 {x=$3+0;s+=x;if(x>m)m=x;n++} END{if(n)printf "%.3f %.3f\n",s/n,m;else print "NA NA"}' "$dm")
  printf '%s\t%s\tok\t%s\t%s\t%s\t%s\n' "$mode" "$bin" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" >>"$PRESELECT"
}

printf 'backend\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\n' >"$PRESELECT"
row_smoke forced "$FORCED_BIN"
row_smoke forced_ilp3 "$FORCED_ILP3_BIN"
row_smoke forced_ilp4 "$FORCED_ILP4_BIN"
row_smoke forced_high "$FORCED_HIGH_BIN"
row_smoke forced_lowrec "$FORCED_LOWREC_BIN"

selection="$(python3 - "$PRESELECT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r['status']=='ok']
if not ok:raise SystemExit('no successful forced preselection candidate')
res={r['residue'] for r in ok}
if len(res)!=1:raise SystemExit('FATAL forced partial-row residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])):
 print('FORCED_PRESELECT',r['backend'],'wall_s='+r['wall_s'],'mc_avg='+r['mc_avg_pct'],'mc_max='+r['mc_max_pct'],file=sys.stderr)
b=min(ok,key=lambda x:float(x['wall_s']))
print('\t'.join([b['backend'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST_FORCED BEST_FORCED_BIN BEST_PARTIAL_RES BEST_PARTIAL_WALL <<<"$selection"
echo "FORCED PRESELECTED backend=$BEST_FORCED partial_wall_s=$BEST_PARTIAL_WALL partial_residue=$BEST_PARTIAL_RES rows=$FORCED_ROWS" >&2
cat "$PRESELECT" >&2

# The selected forced binary is already built. Let HBM9 build any missing bucket
# candidates, full-smoke only this one forced backend plus the four bucket paths,
# exact-compare their complete residues, and continue CRT on the fastest backend.
export REBUILD=0
export CANDIDATES="$BEST_FORCED warp_dense warp_sparse orbit_dense orbit_sparse"
export PREFIX="${FINAL_PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm9_final_n27}"
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm9.sh" 27 "$@"
