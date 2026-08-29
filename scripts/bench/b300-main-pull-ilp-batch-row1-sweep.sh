#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27
MOD="${MOD:-4294967291}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
GPUS=8
ARCH="${ARCH:-native}"
ROWS="${ROWS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256}"
ILP_LIST="${ILP_LIST:-1 2 3 4}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_main_pull_ilp_batch_row${ROWS}_sweep}"
mkdir -p "$(dirname "$PREFIX")" "$ONEESAN_BUILD_DIR"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo "ROWS must be 1..28" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; ((visible>=GPUS)) || { echo "need $GPUS GPUs, visible=$visible" >&2; exit 2; }

for ilp in $ILP_LIST; do
  case "$ilp" in 1|2|3|4) ;; *) echo "ILP_LIST supports only 1 2 3 4" >&2; exit 2;; esac
  bin="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n27_ilp${ilp}_rowsweep"
  bout="${PREFIX}.ilp${ilp}.build.out"; berr="${PREFIX}.ilp${ilp}.build.err"
  echo "=== build ILP=$ilp ===" >&2
  N="$N" ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP="$ilp" \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    FAST_SHARD_ADDRESS8=1 LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 \
    HIGH_DROP_CHUNK=0 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$bout" 2>"$berr"
  grep -Fq "main_pull_ilp=$ilp" "$bout" || { echo "build metadata lost ILP=$ilp" >&2; exit 3; }
  grep -Fq 'batch_row_limit_env=B300_ROW_LIMIT' "$bout" || { echo "build lost batch row-limit support" >&2; exit 3; }
done

python3 - "$PREFIX" $ILP_LIST <<'PY'
import re,sys,pathlib
prefix=sys.argv[1]
for ilp in map(int,sys.argv[2:]):
    p=pathlib.Path(f'{prefix}.ilp{ilp}.build.err')
    text=p.read_text(errors='replace') if p.exists() else ''
    target='main_pull_kernel' if ilp==1 else f'main_pull_kernel_ilp{ilp}'
    lines=text.splitlines(); regs=[]; ss=[]; sl=[]; active=False
    for line in lines:
        if 'Function properties for' in line:
            active=target in line; continue
        if active:
            m=re.search(r'Used\s+(\d+)\s+registers',line)
            if m: regs.append(int(m.group(1)))
            m=re.search(r'(\d+)\s+bytes spill stores',line)
            if m: ss.append(int(m.group(1)))
            m=re.search(r'(\d+)\s+bytes spill loads',line)
            if m: sl.append(int(m.group(1)))
    print(f'b300_ilp{ilp}_ptxas_kernel={target}')
    print(f'b300_ilp{ilp}_ptxas_registers={max(regs) if regs else -1}')
    print(f'b300_ilp{ilp}_ptxas_spill_store_bytes={max(ss) if ss else -1}')
    print(f'b300_ilp{ilp}_ptxas_spill_load_bytes={max(sl) if sl else -1}')
PY

run_one(){
  local ilp="$1" threads="$2"
  local bin="$ONEESAN_BUILD_DIR/b300_hbm32_batch_n27_ilp${ilp}_rowsweep"
  local tag="ilp${ilp}_t${threads}" out="${PREFIX}.${tag}.out" err="${PREFIX}.${tag}.err" tele="${PREFIX}.${tag}.gpu.csv"
  : >"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total \
    --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null &
  local mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$GPUS" "$MOD" >"$out" 2>"$err"
  local rc=$?
  set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 100 "$err" >&2 || true; return "$rc"; }
  local line
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing result line" >&2; return 4; }
  grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) " <<<"$line" || { echo "$tag result lacks expected row calibration metadata" >&2; echo "$line" >&2; return 5; }
  local residue wall
  residue="$(sed -nE 's/.* residue=([^[:space:]]+).*/\1/p' <<<"$line")"
  wall="$(sed -nE 's/.* wall_s=([^[:space:]]+).*/\1/p' <<<"$line")"
  local stats
  stats="$(python3 - "$tele" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
with open(sys.argv[1],newline='') as f:
    for r in csv.reader(f):
        if len(r)<5: continue
        try:s=float(r[2].strip());m=float(r[3].strip());p=float(r[4].strip())
        except ValueError: continue
        sm.append(s);mem.append(m);power.append(p)
        if s>=50: busy.append(m)
def avg(x):return sum(x)/len(x) if x else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)} {len(busy)}')
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ilp" "$threads" "$residue" "$wall" "$tag" "$stats"
}

TSV="${PREFIX}.summary.tsv"
printf 'ilp\tthreads\tresidue\twall_s\ttag\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\tbusy_samples\n' >"$TSV"
for threads in $THREADS_LIST; do
  if [[ "$threads" == "128" ]]; then order="$ILP_LIST"; else order="$(tr ' ' '\n' <<<"$ILP_LIST" | tac | tr '\n' ' ')"; fi
  for ilp in $order; do
    echo "=== run ILP=$ilp threads=$threads rows=$ROWS ===" >&2
    run_one "$ilp" "$threads" >>"$TSV"
  done
done

python3 - "$TSV" <<'PY'
import csv,sys
p=sys.argv[1]
with open(p,newline='') as f: rows=list(csv.DictReader(f,delimiter='\t'))
if not rows: raise SystemExit('no ILP sweep rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('ILP sweep residue mismatch: '+repr(sorted(res)))
for r in rows:r['wall']=float(r['wall_s']);r['mem']=float(r['mem_busy_avg_pct']);r['sm']=float(r['sm_avg_pct'])
best=min(rows,key=lambda r:r['wall']);base=min((r for r in rows if r['ilp']=='1'),key=lambda r:r['wall'],default=None)
print('b300_ilp_sweep_exact_intermediate_match=1')
print(f'b300_ilp_sweep_residue={next(iter(res))}')
print(f'b300_ilp_sweep_best_ilp={best["ilp"]}')
print(f'b300_ilp_sweep_best_threads={best["threads"]}')
print(f'b300_ilp_sweep_best_wall_s={best["wall"]:.9f}')
print(f'b300_ilp_sweep_best_memctl_busy_avg_pct={best["mem"]:.3f}')
print(f'b300_ilp_sweep_best_sm_avg_pct={best["sm"]:.3f}')
if base: print(f'b300_ilp_sweep_speedup_vs_best_ilp1={base["wall"]/best["wall"]:.9f}x')
print('b300_ilp_sweep_rows:')
for r in sorted(rows,key=lambda x:(int(x['ilp']),int(x['threads']))):
    print(f'  ilp={r["ilp"]} threads={r["threads"]} wall_s={r["wall"]:.9f} mem_busy={r["mem"]:.3f}% sm={r["sm"]:.3f}%')
PY
printf 'b300_ilp_sweep_rows=%s\n' "$ROWS"
printf 'b300_ilp_sweep_summary=%s\n' "$TSV"
printf 'b300_ilp_sweep_note=partial-row residues are calibration-only; equality is used only for A/B correctness\n'
