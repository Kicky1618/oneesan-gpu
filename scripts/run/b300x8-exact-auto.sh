#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo "auto selector currently targets n=27" >&2; exit 2; }
NGPU=8
PRIME="${SMOKE_PRIME:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
ORBIT_TARGET_MIB="${ORBIT_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
GRIDFP_THREADS="${GRIDFP_THREADS:-256}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_ORBITCTA_GRID_Y="${BUCKET_ORBITCTA_GRID_Y:-128}"
BUCKET_LOW_GRID_X="${BUCKET_LOW_GRID_X:-16}"
BUCKET_LOW_GRID_Y="${BUCKET_LOW_GRID_Y:-8}"
BUCKET_TRANSPOSE_CHUNK_MIB="${BUCKET_TRANSPOSE_CHUNK_MIB:-1024}"
BUCKET_RESERVE_MIB="${BUCKET_RESERVE_MIB:-8192}"
REBUILD="${REBUILD:-1}"
CANDIDATES="${CANDIDATES:-forced orbit_dense orbit_sparse}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_auto_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
SELECT_TSV="${SELECT_TSV:-${PREFIX}_select.tsv}"
FINAL_WORK_DIR="${WORK_DIR:-$ONEESAN_ROOT/work/b300_exact_auto_selected_n${N}}"
mkdir -p "$LOGDIR" "$(dirname "$SELECT_TSV")" "$FINAL_WORK_DIR"

command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= 8 )) || { echo "need 8 visible GPUs, got $visible" >&2; exit 2; }
[[ "$REBUILD" == 0 || "$REBUILD" == 1 ]] || { echo "REBUILD must be 0 or 1" >&2; exit 2; }

FORCED_BIN="$ONEESAN_BUILD_DIR/b300_auto_forced_n${N}"
DENSE_BIN="$ONEESAN_BUILD_DIR/b300_auto_orbit_dense_n${N}"
SPARSE_BIN="$ONEESAN_BUILD_DIR/b300_auto_orbit_sparse_n${N}"

has_candidate(){ [[ " $CANDIDATES " == *" $1 "* ]]; }

if has_candidate forced && [[ "$REBUILD" == 1 || ! -x "$FORCED_BIN" ]]; then
  echo "=== build forced full-pull ===" >&2
  N="$N" ARCH="$ARCH" OUT="$FORCED_BIN" MAIN_PULL_ILP=2 HIGH_DROP_CHUNK=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" \
    >"$LOGDIR/forced.build.out" 2>"$LOGDIR/forced.build.err"
fi
if has_candidate orbit_dense && [[ "$REBUILD" == 1 || ! -x "$DENSE_BIN" ]]; then
  echo "=== build orbit dense64 ===" >&2
  N="$N" ARCH="$ARCH" OUT="$DENSE_BIN" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/orbit_dense.build.out" 2>"$LOGDIR/orbit_dense.build.err"
fi
if has_candidate orbit_sparse && [[ "$REBUILD" == 1 || ! -x "$SPARSE_BIN" ]]; then
  echo "=== build orbit sparse64 ===" >&2
  N="$N" ARCH="$ARCH" OUT="$SPARSE_BIN" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
    >"$LOGDIR/orbit_sparse.build.out" 2>"$LOGDIR/orbit_sparse.build.err"
fi

export GRIDFP_THREADS BUCKET_THREADS BUCKET_ORBITCTA_GRID_Y BUCKET_TRANSPOSE_CHUNK_MIB BUCKET_RESERVE_MIB
export BUCKET_GRID_X="$BUCKET_LOW_GRID_X" BUCKET_GRID_Y="$BUCKET_LOW_GRID_Y"
export BUCKET_LOW_GRID_X BUCKET_LOW_GRID_Y

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
summarize_dmon(){ python3 - "$1" <<'PY'
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

# Preflight orbit candidates without allocating authoritative state. A broken or
# non-fitting backend may fail its smoke run and is then excluded, but a residue
# disagreement between successful candidates is fatal.
for spec in "orbit_dense:$DENSE_BIN" "orbit_sparse:$SPARSE_BIN"; do
  mode="${spec%%:*}"; bin="${spec#*:}"
  has_candidate "$mode" || continue
  echo "=== plan $mode ===" >&2
  "$bin" "$N" "$ORBIT_TARGET_MIB" "$MAX_WINDOW" 8 --plan-only \
    >"$LOGDIR/${mode}.plan.out" 2>"$LOGDIR/${mode}.plan.err" || {
      echo "warning: $mode plan-only failed; candidate will still be smoke-tested" >&2
    }
done

printf 'backend\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$SELECT_TSV"

smoke(){
  local mode="$1" bin="$2" target="$3"
  local so="$LOGDIR/${mode}.smoke.out" se="$LOGDIR/${mode}.smoke.err" dm="$LOGDIR/${mode}.smoke.dmon"
  echo "=== smoke $mode mod=$PRIME ===" >&2
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!
  set +e
  "$bin" "$N" "$target" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  local rc=$?
  set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  if ((rc!=0)); then
    printf '%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$mode" "$bin" "$rc" >>"$SELECT_TSV"
    echo "candidate $mode failed rc=$rc; excluding" >&2
    return 0
  fi
  local line residue wall avg mx ns
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  if [[ -z "$line" ]]; then
    printf '%s\t%s\tfailed:no_residue\tNA\tNA\tNA\tNA\t0\n' "$mode" "$bin" >>"$SELECT_TSV"
    echo "candidate $mode produced no residue; excluding" >&2
    return 0
  fi
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  IFS=$'\t' read -r avg mx ns <<<"$(summarize_dmon "$dm")"
  printf '%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$bin" "$residue" "$wall" "$avg" "$mx" "$ns" >>"$SELECT_TSV"
}

has_candidate forced && smoke forced "$FORCED_BIN" "$TARGET_MIB"
has_candidate orbit_dense && smoke orbit_dense "$DENSE_BIN" "$ORBIT_TARGET_MIB"
has_candidate orbit_sparse && smoke orbit_sparse "$SPARSE_BIN" "$ORBIT_TARGET_MIB"

selection="$(python3 - "$SELECT_TSV" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r['status']=='ok']
if not ok:raise SystemExit('no successful backend candidate')
res={r['residue'] for r in ok}
if len(res)!=1:raise SystemExit('FATAL RESIDUE MISMATCH: '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])):
    print('CANDIDATE',r['backend'],'wall_s='+r['wall_s'],'mc_avg_pct='+r['mc_avg_pct'],'mc_max_pct='+r['mc_max_pct'],file=sys.stderr)
best=min(ok,key=lambda x:float(x['wall_s']))
print('\t'.join([best['backend'],best['binary'],best['residue'],best['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST_MODE BEST_BIN BEST_RESIDUE BEST_WALL <<<"$selection"
echo "SELECTED backend=$BEST_MODE wall_s=$BEST_WALL residue=$BEST_RESIDUE" >&2

# Seed the exact-batch checkpoint with the already-paid smoke residue. The
# checkpoint fingerprint is deliberately computed exactly like
# solve_b300_exact_batch.py, preventing cross-binary reuse.
python3 - "$FINAL_WORK_DIR" "$N" "$BEST_BIN" "$PRIME" "$BEST_RESIDUE" "$BEST_WALL" <<'PY'
import hashlib,json,sys
from pathlib import Path
work=Path(sys.argv[1]);n=int(sys.argv[2]);binary=Path(sys.argv[3]).resolve();p=int(sys.argv[4]);r=int(sys.argv[5]);wall=float(sys.argv[6])
h=hashlib.sha256()
with binary.open('rb') as f:
    for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
fp={'schema':2,'binary_sha256':h.hexdigest()}
work.mkdir(parents=True,exist_ok=True)
cp=work/'checkpoint.json'
cp.write_text(json.dumps({'n':n,'solver_fingerprint':fp,'residues':{str(p):{'residue':r,'wall_s':wall}}},indent=2,sort_keys=True)+'\n')
print(f'seeded_checkpoint={cp} binary_sha256={fp["binary_sha256"]}',file=sys.stderr)
PY

# Both backend families accept the same multi-modulus CLI. Keep both environment
# families exported; unused variables are harmless for the selected executable.
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" "$N" \
  --binary "$BEST_BIN" \
  --target-mib "$([[ "$BEST_MODE" == forced ]] && echo "$TARGET_MIB" || echo "$ORBIT_TARGET_MIB")" \
  --max-window "$MAX_WINDOW" \
  --gpus 8 \
  --work-dir "$FINAL_WORK_DIR" \
  "$@"
