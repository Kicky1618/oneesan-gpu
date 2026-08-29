#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo "HBM9 selector currently targets n=27" >&2; exit 2; }
PRIME="${SMOKE_PRIME:-4294967291}"
ARCH="${ARCH:-native}"
MAX_WINDOW="${MAX_WINDOW:-14}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"
BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
THREADS="${BUCKET_THREADS:-256}"
FORCED_THREADS="${GRIDFP_THREADS:-256}"
WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"
ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
COL_ILP="${COL_ILP:-2}"
PAIR_MLP="${PAIR_MLP:-1}"
WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"
PM_ACCUM="${PM_ACCUM:-1}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
REBUILD="${REBUILD:-1}"
SELECT_ONLY="${SELECT_ONLY:-0}"
CANDIDATES="${CANDIDATES:-forced forced_ilp3 forced_ilp4 forced_high forced_lowrec warp_dense warp_sparse orbit_dense orbit_sparse}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_hbm9_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
PTXAS="${PTXAS:-${PREFIX}_ptxas.tsv}"
WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

for x in PAIR_MLP WINDOW4 PM_ACCUM CPASYNC_PAIR REBUILD SELECT_ONLY; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1,2,4" >&2; exit 2;; esac
if [[ "$PAIR_MLP" == 1 ]]; then
  [[ "$WINDOW4" == 1 ]] || { echo "PAIR_MLP requires WINDOW4=1" >&2; exit 2; }
  [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo "PAIR_MLP requires COL_ILP=2/4" >&2; exit 2; }
fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || { echo "CPASYNC_PAIR requires PAIR_MLP=1" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo "need 8 visible GPUs" >&2; exit 2; }

has(){ [[ " $CANDIDATES " == *" $1 "* ]]; }
FORCED_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_n27"
FORCED_ILP3_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_ilp3_n27"
FORCED_ILP4_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_ilp4_n27"
FORCED_HIGH_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_high_n27"
FORCED_LOWREC_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_forced_lowrec_n27"
WARP_DENSE_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_warp_dense_n27"
WARP_SPARSE_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_warp_sparse_n27"
ORBIT_DENSE_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_orbit_dense_n27"
ORBIT_SPARSE_BIN="$ONEESAN_BUILD_DIR/b300_hbm9_orbit_sparse_n27"

build_forced(){
  local mode="$1" bin="$2" ilp="$3" high="$4" rec="$5"
  [[ "$REBUILD" == 1 || ! -x "$bin" ]] || return 0
  echo "=== build $mode ilp=$ilp high=$high lowrec=$rec ===" >&2
  N=27 ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP="$ilp" HIGH_DROP_CHUNK="$high" LOW_MAIN_RECURRENCE="$rec" \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
    LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
  grep -Fq "main_pull_ilp=$ilp" "$LOGDIR/${mode}.build.out"
  grep -Fq "high_drop_chunk=$high" "$LOGDIR/${mode}.build.out"
  grep -Fq "low_main_recurrence=$rec" "$LOGDIR/${mode}.build.out"
}
build_warp(){
  local mode="$1" sparse="$2" bin="$3"
  [[ "$REBUILD" == 1 || ! -x "$bin" ]] || return 0
  N=27 ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" MLP_WINDOW4="$WINDOW4" \
    DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" CPASYNC_PAIR="$CPASYNC_PAIR" \
    PM_ACCUM="$PM_ACCUM" SORTED=0 PRECTX_FORWARD=0 PREFETCH_NEXT=0 FORCE7=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-max.sh" >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
}
build_orbit(){
  local mode="$1" sparse="$2" bin="$3"
  [[ "$REBUILD" == 1 || ! -x "$bin" ]] || return 0
  N=27 ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$sparse" \
    ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" \
    RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
}

if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
    >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { cat "$LOGDIR/cpasync-peer.out" >&2; cat "$LOGDIR/cpasync-peer.err" >&2; exit 4; }
fi
has forced && build_forced forced "$FORCED_BIN" 2 0 0
has forced_ilp3 && build_forced forced_ilp3 "$FORCED_ILP3_BIN" 3 0 0
has forced_ilp4 && build_forced forced_ilp4 "$FORCED_ILP4_BIN" 4 0 0
has forced_high && build_forced forced_high "$FORCED_HIGH_BIN" 2 1 0
has forced_lowrec && build_forced forced_lowrec "$FORCED_LOWREC_BIN" 2 0 1
has warp_dense && build_warp warp_dense 0 "$WARP_DENSE_BIN"
has warp_sparse && build_warp warp_sparse 1 "$WARP_SPARSE_BIN"
has orbit_dense && build_orbit orbit_dense 0 "$ORBIT_DENSE_BIN"
has orbit_sparse && build_orbit orbit_sparse 1 "$ORBIT_SPARSE_BIN"

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$PTXAS"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
for mode in forced forced_ilp3 forced_ilp4 forced_high forced_lowrec warp_dense warp_sparse orbit_dense orbit_sparse; do
  has "$mode" || continue
  [[ -f "$LOGDIR/$mode.build.err" ]] || continue
  if [[ "$mode" == forced* ]]; then
    python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains main_pull_kernel --contains block_pull_kernel >>"$PTXAS" || true
  else
    python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$PTXAS" || true
  fi
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

for spec in "warp_dense:$WARP_DENSE_BIN" "warp_sparse:$WARP_SPARSE_BIN" "orbit_dense:$ORBIT_DENSE_BIN" "orbit_sparse:$ORBIT_SPARSE_BIN"; do
  mode="${spec%%:*}"; bin="${spec#*:}"; has "$mode" || continue
  "$bin" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 --plan-only >"$LOGDIR/$mode.plan.out" 2>"$LOGDIR/$mode.plan.err" || true
done

printf 'backend\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
smoke(){
  local mode="$1" bin="$2" target="$3" family="$4"
  local so="$LOGDIR/$mode.smoke.out" se="$LOGDIR/$mode.smoke.err" dm="$LOGDIR/$mode.smoke.dmon"
  echo "=== HBM9 smoke $mode ===" >&2
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e
  if [[ "$family" == forced ]]; then
    GRIDFP_THREADS="$FORCED_THREADS" "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  elif [[ "$family" == warp ]]; then
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" \
      BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  else
    BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
      BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$bin" 27 "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  fi
  local rc=$?
  set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  if ((rc)); then printf '%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$mode" "$bin" "$rc" >>"$RESULT"; return 0; fi
  local line
  if [[ "$family" == forced ]]; then
    line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  else
    line="$(grep '^residue=' "$so" | tail -n1 || true)"
  fi
  if [[ -z "$line" ]]; then printf '%s\t%s\tfailed:no_residue\tNA\tNA\tNA\tNA\t0\n' "$mode" "$bin" >>"$RESULT"; return 0; fi
  local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
  printf '%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$bin" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}

has forced && smoke forced "$FORCED_BIN" "$FORCED_TARGET_MIB" forced
has forced_ilp3 && smoke forced_ilp3 "$FORCED_ILP3_BIN" "$FORCED_TARGET_MIB" forced
has forced_ilp4 && smoke forced_ilp4 "$FORCED_ILP4_BIN" "$FORCED_TARGET_MIB" forced
has forced_high && smoke forced_high "$FORCED_HIGH_BIN" "$FORCED_TARGET_MIB" forced
has forced_lowrec && smoke forced_lowrec "$FORCED_LOWREC_BIN" "$FORCED_TARGET_MIB" forced
has warp_dense && smoke warp_dense "$WARP_DENSE_BIN" "$BUCKET_TARGET_MIB" warp
has warp_sparse && smoke warp_sparse "$WARP_SPARSE_BIN" "$BUCKET_TARGET_MIB" warp
has orbit_dense && smoke orbit_dense "$ORBIT_DENSE_BIN" "$BUCKET_TARGET_MIB" orbit
has orbit_sparse && smoke orbit_sparse "$ORBIT_SPARSE_BIN" "$BUCKET_TARGET_MIB" orbit

selection="$(python3 - "$RESULT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r['status']=='ok']
if not ok:raise SystemExit('no successful HBM9 candidate')
res={r['residue'] for r in ok}
if len(res)!=1:raise SystemExit('FATAL residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])):
 print('CANDIDATE',r['backend'],'wall_s='+r['wall_s'],'mc_avg='+r['mc_avg_pct'],'mc_max='+r['mc_max_pct'],file=sys.stderr)
b=min(ok,key=lambda x:float(x['wall_s']))
print('\t'.join([b['backend'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST BEST_BIN BEST_RES BEST_WALL <<<"$selection"
echo "HBM9 SELECTED backend=$BEST wall_s=$BEST_WALL residue=$BEST_RES" >&2
cat "$RESULT" >&2

BEST_WORK="$WORK_ROOT/b300_exact_hbm9_${BEST}_n27"
mkdir -p "$BEST_WORK"
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

if [[ "$SELECT_ONLY" == 1 ]]; then
  echo "SELECT_ONLY=1: selected $BEST; CRT not continued" >&2
  exit 0
fi

if [[ "$BEST" == forced* ]]; then
  export GRIDFP_THREADS="$FORCED_THREADS"; RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$BEST" == warp_* ]]; then
  export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; RUN_TARGET="$BUCKET_TARGET_MIB"
else
  export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; RUN_TARGET="$BUCKET_TARGET_MIB"
fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BEST_BIN" --target-mib "$RUN_TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$BEST_WORK" "$@"
