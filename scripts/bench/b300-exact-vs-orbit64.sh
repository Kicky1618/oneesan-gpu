#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"
REPEATS="${REPEATS:-1}"
THREADS="${THREADS:-256}"
ORBIT_GY="${ORBIT_GY:-128}"
LOW_GX="${LOW_GX:-16}"
LOW_GY="${LOW_GY:-8}"
MAIN_PULL_ILP="${MAIN_PULL_ILP:-2}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
EXPECT="${EXPECT:-}"
if [[ -z "$EXPECT" && "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516; fi

command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
[[ "$MAIN_PULL_ILP" == 1 || "$MAIN_PULL_ILP" == 2 ]] || { echo "MAIN_PULL_ILP must be 1 or 2" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_vs_orbit64_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

EXACT_BIN="$ONEESAN_BUILD_DIR/b300_exact_vs_orbit64_exact_n${N}"
ORBIT_BIN="$ONEESAN_BUILD_DIR/b300_exact_vs_orbit64_orbit_n${N}"

N="$N" ARCH="$ARCH" OUT="$EXACT_BIN" MAIN_PULL_ILP="$MAIN_PULL_ILP" \
  HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" \
  >"$LOGDIR/exact.build.out" 2>"$LOGDIR/exact.build.err"

N="$N" ARCH="$ARCH" OUT="$ORBIT_BIN" DIRECTGATHER64=1 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" \
  >"$LOGDIR/orbit64.build.out" 2>"$LOGDIR/orbit64.build.err"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$LOGDIR/exact.build.err" --label exact \
  --contains main_pull_kernel --contains block_pull_kernel >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/orbit64.build.err" --label orbit64 >>"$RESOURCE" || true

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

summarize_dmon(){
  python3 - "$1" <<'PY'
import sys
p=sys.argv[1]
vals=[]
try:
    for line in open(p,encoding='utf-8',errors='replace'):
        s=line.strip()
        if not s or s.startswith('#'): continue
        a=s.split()
        if len(a)<3: continue
        try: vals.append(float(a[2])) # dmon -s u: gpu sm mem ...
        except ValueError: pass
except FileNotFoundError:
    pass
# Drop one startup sample per GPU when enough observations exist; that sample
# is commonly taken before the CUDA process has reached its hot phase.
if len(vals)>16:
    vals=vals[8:]
if not vals:
    print('NA\tNA\t0'); raise SystemExit
print(f'{sum(vals)/len(vals):.3f}\t{max(vals):.3f}\t{len(vals)}')
PY
}

printf 'backend\trepeat\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"

run_case(){
  local name="$1" bin="$2" orbit="$3" rep="$4"
  local so="$LOGDIR/${name}_r${rep}.out" se="$LOGDIR/${name}_r${rep}.err" dm="$LOGDIR/${name}_r${rep}.dmon"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null &
  local dpid=$!
  set +e
  if [[ "$orbit" == 1 ]]; then
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
    BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  else
    GRIDFP_THREADS="$THREADS" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  fi
  local rc=$?
  set -e
  kill "$dpid" 2>/dev/null || true
  wait "$dpid" 2>/dev/null || true
  (( rc == 0 )) || { echo "$name failed rc=$rc; see $se" >&2; exit "$rc"; }
  local line residue wall stats avg mx ns
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ -n "$wall" ]] || wall=NA
  IFS=$'\t' read -r avg mx ns <<<"$(summarize_dmon "$dm")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rep" "$residue" "$wall" "$avg" "$mx" "$ns" >>"$RESULT"
}

for ((r=1;r<=REPEATS;++r)); do
  run_case exact "$EXACT_BIN" 0 "$r"
  run_case orbit64 "$ORBIT_BIN" 1 "$r"
done

python3 - "$RESULT" "$EXPECT" <<'PY'
import csv,statistics,sys
path,expect=sys.argv[1],sys.argv[2]
rows=list(csv.DictReader(open(path),delimiter='\t'))
by={}
for r in rows: by.setdefault(r['backend'],[]).append(r)
if set(by)!={'exact','orbit64'}:
    raise SystemExit(f'missing backend rows: {sorted(by)}')
for b,g in by.items():
    residues={r['residue'] for r in g}
    if len(residues)!=1: raise SystemExit(f'{b} residue changed across repeats: {residues}')
res={b:next(iter({r['residue'] for r in g})) for b,g in by.items()}
if res['exact']!=res['orbit64']:
    raise SystemExit(f'RESIDUE MISMATCH exact={res["exact"]} orbit64={res["orbit64"]}')
if expect and res['exact']!=expect:
    raise SystemExit(f'EXPECTED MISMATCH got={res["exact"]} expected={expect}')
print('residue_match=OK',res['exact'])
summary={}
for b,g in by.items():
    walls=[float(r['wall_s']) for r in g if r['wall_s']!='NA']
    avgs=[float(r['mc_avg_pct']) for r in g if r['mc_avg_pct']!='NA']
    maxs=[float(r['mc_max_pct']) for r in g if r['mc_max_pct']!='NA']
    summary[b]={
        'wall':statistics.median(walls) if walls else float('nan'),
        'mc_avg':statistics.median(avgs) if avgs else float('nan'),
        'mc_max':max(maxs) if maxs else float('nan'),
    }
for b in ('exact','orbit64'):
    z=summary[b]
    print(b,f"wall_s={z['wall']:.6f}",f"mc_avg_pct={z['mc_avg']:.3f}",f"mc_max_pct={z['mc_max']:.3f}")
if summary['orbit64']['wall']>0:
    print(f"orbit64_speedup={summary['exact']['wall']/summary['orbit64']['wall']:.6f}")
print(f"mc_avg_delta_pp={summary['orbit64']['mc_avg']-summary['exact']['mc_avg']:.3f}")
PY

echo "result=$RESULT ptxas=$RESOURCE logs=$LOGDIR" >&2
