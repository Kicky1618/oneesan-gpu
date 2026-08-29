#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
MAIN_PULL_ILP="${MAIN_PULL_ILP:-2}"; HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
THREAD_LIST="${THREAD_LIST:-128 256 512}"; REPEATS="${REPEATS:-1}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_exact_batch_thread_sweep_n${N}_ilp${MAIN_PULL_ILP}}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_batch_thread_sweep_n${N}_ilp${MAIN_PULL_ILP}}"
LOGDIR="${PREFIX}_logs"; RESULT="${PREFIX}.tsv"; RESOURCE="${PREFIX}_ptxas.tsv"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }

N="$N" ARCH="$ARCH" OUT="$BIN" MAIN_PULL_ILP="$MAIN_PULL_ILP" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"
python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/build.err" --label exact --header \
  --contains main_pull_kernel --contains block_pull_kernel >"$RESOURCE" || true

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
summarize_dmon(){ python3 - "$1" <<'PY'
import sys
v=[]
for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
    s=line.strip()
    if not s or s.startswith('#'):continue
    a=s.split()
    if len(a)>=3:
        try:v.append(float(a[2]))
        except ValueError:pass
if len(v)>16:v=v[8:]
if v:print(f'{sum(v)/len(v):.3f}\t{max(v):.3f}\t{len(v)}')
else:print('NA\tNA\t0')
PY
}

printf 'threads\trepeat\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"
for threads in $THREAD_LIST; do
  (( threads>=64 && threads<=1024 && threads%32==0 )) || { echo "bad thread count $threads" >&2; exit 2; }
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/t${threads}_r${r}.out"; se="$LOGDIR/t${threads}_r${r}.err"; dm="$LOGDIR/t${threads}_r${r}.dmon"
    nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & dp=$!
    set +e
    GRIDFP_THREADS="$threads" "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    rc=$?
    set -e
    kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
    ((rc==0)) || { echo "threads=$threads failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "missing residue t=$threads" >&2; exit 3; }
    residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; IFS=$'\t' read -r avg mx ns <<<"$(summarize_dmon "$dm")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$threads" "$r" "$residue" "${wall:-NA}" "$avg" "$mx" "$ns" >>"$RESULT"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));by={}
for r in rows:by.setdefault(r['threads'],[]).append(r)
res={t:{x['residue'] for x in g} for t,g in by.items()}
if any(len(x)!=1 for x in res.values()) or len({next(iter(x)) for x in res.values()})!=1:raise SystemExit(f'RESIDUE MISMATCH {res}')
print('residue_match=OK',next(iter(next(iter(res.values())))))
out=[]
for t,g in by.items():
    w=statistics.median(float(x['wall_s']) for x in g)
    a=[float(x['mc_avg_pct']) for x in g if x['mc_avg_pct']!='NA'];m=[float(x['mc_max_pct']) for x in g if x['mc_max_pct']!='NA']
    out.append((w,int(t),statistics.median(a) if a else float('nan'),max(m) if m else float('nan')))
for w,t,a,m in sorted(out):print(f'threads={t}',f'wall_s={w:.6f}',f'mc_avg_pct={a:.3f}',f'mc_max_pct={m:.3f}')
print('BEST_THREADS',min(out)[1])
PY

echo "result=$RESULT ptxas=$RESOURCE logs=$LOGDIR" >&2
