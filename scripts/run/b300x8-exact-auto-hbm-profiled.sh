#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'profiled HBM selector currently targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile file: $PROFILE_FILE" >&2; echo 'run: bash scripts/bench/b300-hbm-profile-auto21.sh' >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"

# Backward-compatible defaults for profiles generated before compact prectx and
# flat persistent orbit scheduling were added. psm=0 means occupancy-derived
# forward/reverse pool sizes, i.e. leave the runtime override unset.
WARP_PRECTX_COMPACT="${WARP_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"

PRIME="${SMOKE_PRIME:-4294967291}"; ARCH="${ARCH:-native}"; MAX_WINDOW="${MAX_WINDOW:-14}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
THREADS="${BUCKET_THREADS:-256}"; FORCED_THREADS="${GRIDFP_THREADS:-256}"
WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
PM_ACCUM="${PM_ACCUM:-1}"; REBUILD="${REBUILD:-1}"; SELECT_ONLY="${SELECT_ONLY:-0}"
CANDIDATES="${CANDIDATES:-forced warp_tuned orbit_tuned}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm_profiled_n27}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; PTXAS="${PTXAS:-${PREFIX}_ptxas.tsv}"; WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

for x in PM_ACCUM REBUILD SELECT_ONLY WARP_PRECTX_COMPACT ORBIT_PRECTX_COMPACT ORBITCTA_FLAT; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'ORBITCTA_FLAT_BLOCKS_PER_SM must be non-negative integer (0=occupancy)' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

needvar(){ local n="$1"; [[ -n "${!n+x}" ]] || { echo "profile missing $n" >&2; exit 2; }; }
for n in WARP_PROFILE WARP_COL_ILP WARP_SPARSE64 WARP_SORTED WARP_CPASYNC_PAIR WARP_CPASYNC_LOCAL_PAIR WARP_CPASYNC_OVERLAP_LOCAL_PAIR WARP_QUAD_MLP WARP_PRECTX_FORWARD WARP_PRECTX_REVERSE ORBIT_PROFILE ORBIT_COL_ILP ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_CPASYNC_LOCAL_PAIR ORBIT_CPASYNC_OVERLAP_LOCAL_PAIR ORBIT_QUAD_MLP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE; do needvar "$n"; done
for n in WARP_SPARSE64 WARP_SORTED WARP_CPASYNC_PAIR WARP_CPASYNC_LOCAL_PAIR WARP_CPASYNC_OVERLAP_LOCAL_PAIR WARP_QUAD_MLP WARP_PRECTX_FORWARD WARP_PRECTX_REVERSE WARP_PRECTX_COMPACT ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_CPASYNC_LOCAL_PAIR ORBIT_CPASYNC_OVERLAP_LOCAL_PAIR ORBIT_QUAD_MLP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT; do
  v="${!n}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$n must be 0 or 1, got $v" >&2; exit 2; }
done
[[ "$WARP_PRECTX_COMPACT" == 0 || "$WARP_PRECTX_FORWARD" == 1 || "$WARP_PRECTX_REVERSE" == 1 ]] || { echo 'warp compact prectx requires prectx' >&2; exit 2; }
[[ "$ORBIT_PRECTX_COMPACT" == 0 || "$ORBIT_PRECTX_FORWARD" == 1 || "$ORBIT_PRECTX_REVERSE" == 1 ]] || { echo 'orbit compact prectx requires prectx' >&2; exit 2; }
case "$WARP_COL_ILP" in 2|4) ;; *) echo "invalid WARP_COL_ILP=$WARP_COL_ILP" >&2; exit 2;; esac
case "$ORBIT_COL_ILP" in 2|4) ;; *) echo "invalid ORBIT_COL_ILP=$ORBIT_COL_ILP" >&2; exit 2;; esac
[[ "$WARP_PROFILE" =~ ^[A-Za-z0-9_.-]+$ && "$ORBIT_PROFILE" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo 'unsafe profile name' >&2; exit 2; }
[[ "$ORBIT_SORTED" == 0 && "$ORBIT_CPASYNC_LOCAL_PAIR" == 0 && "$ORBIT_CPASYNC_OVERLAP_LOCAL_PAIR" == 0 && "$ORBIT_QUAD_MLP" == 0 ]] || { echo 'profile requests orbit features not wired by this profiled path' >&2; exit 2; }

has(){ [[ " $CANDIDATES " == *" $1 "* ]]; }
FORCED_BIN="$ONEESAN_BUILD_DIR/b300_profiled_forced_n27"
WARP_BIN="$ONEESAN_BUILD_DIR/b300_profiled_warp_${WARP_PROFILE}_n27"
ORBIT_BIN="$ONEESAN_BUILD_DIR/b300_profiled_orbit_${ORBIT_PROFILE}_flat${ORBITCTA_FLAT}_psm${ORBITCTA_FLAT_BLOCKS_PER_SM}_n27"

if (( WARP_CPASYNC_PAIR || ORBIT_CPASYNC_PAIR )); then
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { cat "$LOGDIR/cpasync-peer.out" >&2; cat "$LOGDIR/cpasync-peer.err" >&2; exit 4; }
fi
if (( WARP_PRECTX_COMPACT || ORBIT_PRECTX_COMPACT )); then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
fi

if has forced && [[ "$REBUILD" == 1 || ! -x "$FORCED_BIN" ]]; then
  N=27 ARCH="$ARCH" OUT="$FORCED_BIN" MAIN_PULL_ILP=2 HIGH_DROP_CHUNK=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$LOGDIR/forced.build.out" 2>"$LOGDIR/forced.build.err"
fi
if has warp_tuned && [[ "$REBUILD" == 1 || ! -x "$WARP_BIN" ]]; then
  N=27 ARCH="$ARCH" OUT="$WARP_BIN" COL_ILP="$WARP_COL_ILP" PM_ACCUM="$PM_ACCUM" DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 \
    DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$WARP_SPARSE64" SORTED="$WARP_SORTED" QUAD_MLP="$WARP_QUAD_MLP" \
    CPASYNC_PAIR="$WARP_CPASYNC_PAIR" CPASYNC_LOCAL_PAIR="$WARP_CPASYNC_LOCAL_PAIR" CPASYNC_OVERLAP_LOCAL_PAIR="$WARP_CPASYNC_OVERLAP_LOCAL_PAIR" \
    PRECTX_FORWARD="$WARP_PRECTX_FORWARD" PRECTX_REVERSE="$WARP_PRECTX_REVERSE" PRECTX_COMPACT="$WARP_PRECTX_COMPACT" FORCE7=0 PREFETCH_NEXT=0 PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$LOGDIR/warp_tuned.build.out" 2>"$LOGDIR/warp_tuned.build.err"
fi
if has orbit_tuned && [[ "$REBUILD" == 1 || ! -x "$ORBIT_BIN" ]]; then
  N=27 ARCH="$ARCH" OUT="$ORBIT_BIN" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" DIRECTGATHER_SORT_RANKS=0 ORBITCTA_COL_ILP="$ORBIT_COL_ILP" ORBITCTA_FLAT="$ORBITCTA_FLAT" ORBITCTA_FLAT_CHUNK=1 \
    PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" \
    RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/orbit_tuned.build.out" 2>"$LOGDIR/orbit_tuned.build.err"
fi

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$PTXAS"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
for mode in forced warp_tuned orbit_tuned; do
  has "$mode" || continue; [[ -f "$LOGDIR/$mode.build.err" ]] || continue
  if [[ "$mode" == forced ]]; then python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains main_pull_kernel --contains block_pull_kernel >>"$PTXAS" || true
  else python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$PTXAS" || true; fi
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample_summary(){ python3 - "$1" <<'PY'
import sys
v=[]
try:
 for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
  s=line.strip()
  if not s or s.startswith('#'):continue
  a=s.split()
  if len(a)>=3:
   try:v.append(float(a[2]))
   except ValueError:pass
except FileNotFoundError:pass
if len(v)>16:v=v[8:]
print('NA\tNA\t0' if not v else f'{sum(v)/len(v):.3f}\t{max(v):.3f}\t{len(v)}')
PY
}

plan_bucket(){
  local mode="$1" bin="$2" pe="$LOGDIR/$mode.plan.err"
  set +e; "$bin" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 --plan-only >"$LOGDIR/$mode.plan.out" 2>"$pe"; local rc=$?; set -e
  ((rc==0)) || return "$rc"
  grep 'backend=gridfp-b300-bucket-snake-onepass-graph-batch-v0.1-plan' "$pe" | tail -n1
}
WARP_PLAN=''; ORBIT_PLAN=''
has warp_tuned && WARP_PLAN="$(plan_bucket warp_tuned "$WARP_BIN")" || true
has orbit_tuned && ORBIT_PLAN="$(plan_bucket orbit_tuned "$ORBIT_BIN")" || true

printf 'backend\tprofile\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\textra_metadata_mib\tmax_device_need_gib\n' >"$RESULT"
smoke(){
  local mode="$1" profile="$2" bin="$3" target="$4" family="$5" plan="$6"
  local so="$LOGDIR/$mode.smoke.out" se="$LOGDIR/$mode.smoke.err" dm="$LOGDIR/$mode.smoke.dmon"
  echo "=== profiled smoke $mode profile=$profile ===" >&2
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e
  if [[ "$family" == forced ]]; then
    GRIDFP_THREADS="$FORCED_THREADS" "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  elif [[ "$family" == warp ]]; then
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  elif [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
      BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  else
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" \
      BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  fi
  local rc=$?; set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  if ((rc)); then printf '%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\tNA\tNA\n' "$mode" "$profile" "$bin" "$rc" >>"$RESULT"; return 0; fi
  local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { printf '%s\t%s\t%s\tfailed:no_residue\tNA\tNA\tNA\tNA\t0\tNA\tNA\n' "$mode" "$profile" "$bin" >>"$RESULT"; return 0; }
  local avg mx ns extra=0 need=0; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
  [[ -n "$plan" ]] && { extra="$(field extra_metadata_mib "$plan")"; need="$(field max_device_need_gib "$plan")"; }
  printf '%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$profile" "$bin" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" "$extra" "$need" >>"$RESULT"
}
has forced && smoke forced forced "$FORCED_BIN" "$FORCED_TARGET_MIB" forced ''
has warp_tuned && smoke warp_tuned "$WARP_PROFILE" "$WARP_BIN" "$BUCKET_TARGET_MIB" warp "$WARP_PLAN"
has orbit_tuned && smoke orbit_tuned "$ORBIT_PROFILE" "$ORBIT_BIN" "$BUCKET_TARGET_MIB" orbit "$ORBIT_PLAN"

selection="$(python3 - "$RESULT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));ok=[r for r in rows if r['status']=='ok']
if not ok:raise SystemExit('no successful profiled n27 candidate')
res={r['residue'] for r in ok}
if len(res)!=1:raise SystemExit('FATAL residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])):
 print('CANDIDATE',r['backend'],'profile='+r['profile'],'wall_s='+r['wall_s'],'mc_avg='+r['mc_avg_pct'],'extra_meta_mib='+r['extra_metadata_mib'],file=sys.stderr)
b=min(ok,key=lambda x:float(x['wall_s']))
print('\t'.join([b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$selection"
echo "PROFILED SELECTED backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL residue=$BEST_RES" >&2
cat "$RESULT" >&2

SHA12="$(sha256sum "$BEST_BIN" | awk '{print substr($1,1,12)}')"
BEST_WORK="$WORK_ROOT/b300_exact_profiled_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; mkdir -p "$BEST_WORK"
python3 - "$BEST_WORK" "$BEST_BIN" "$PRIME" "$BEST_RES" "$BEST_WALL" <<'PY'
import hashlib,json,sys
from pathlib import Path
work=Path(sys.argv[1]);binary=Path(sys.argv[2]).resolve();p=int(sys.argv[3]);r=int(sys.argv[4]);wall=float(sys.argv[5])
h=hashlib.sha256()
with binary.open('rb') as f:
 for z in iter(lambda:f.read(1<<20),b''):h.update(z)
fp={'schema':2,'binary_sha256':h.hexdigest()};cp=work/'checkpoint.json';res={}
if cp.exists():
 old=json.loads(cp.read_text())
 if int(old.get('n',-1))!=27 or old.get('solver_fingerprint')!=fp:raise SystemExit(f'checkpoint incompatible: {cp}')
 res=dict(old.get('residues',{}))
if str(p) in res and int(res[str(p)]['residue'])!=r:raise SystemExit('smoke/checkpoint residue disagreement')
res[str(p)]={'residue':r,'wall_s':wall};tmp=cp.with_suffix('.json.tmp')
tmp.write_text(json.dumps({'n':27,'solver_fingerprint':fp,'residues':res},indent=2,sort_keys=True)+'\n');tmp.replace(cp)
print(f'seeded {cp} cached_residues={len(res)}',file=sys.stderr)
PY

if [[ "$SELECT_ONLY" == 1 ]]; then echo "SELECT_ONLY=1: selected $BEST/$BEST_PROFILE; CRT not continued" >&2; exit 0; fi
if [[ "$BEST" == forced ]]; then
  export GRIDFP_THREADS="$FORCED_THREADS"; RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$BEST" == warp_tuned ]]; then
  export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; RUN_TARGET="$BUCKET_TARGET_MIB"
else
  export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"
  unset BUCKET_ORBITCTA_FLAT_BLOCKS
  if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]]; then
    export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM"
  else
    unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM
  fi
  RUN_TARGET="$BUCKET_TARGET_MIB"
fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BEST_BIN" --target-mib "$RUN_TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$BEST_WORK" "$@"
