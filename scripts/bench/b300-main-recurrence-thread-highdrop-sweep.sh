#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; GPUS=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_thread_highdrop_row${ROWS}}"
mkdir -p "$(dirname "$PREFIX")" "$ONEESAN_BUILD_DIR"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= GPUS )) || { echo "need $GPUS GPUs" >&2; exit 2; }

for h in $HIGHDROP_LIST; do
  [[ "$h" == 0 || "$h" == 1 ]] || { echo 'HIGHDROP_LIST supports 0 1' >&2; exit 2; }
  bin="$ONEESAN_BUILD_DIR/b300_mainrec_high${h}_threadsweep_n27"
  bout="${PREFIX}.high${h}.build.out"; berr="${PREFIX}.high${h}.build.err"
  echo "=== build MAIN_RECURRENCE=1 HIGH_DROP_CHUNK=$h ===" >&2
  N=27 ARCH="$ARCH" OUT="$bin" MAIN_PULL_ILP=2 MAIN_RECURRENCE=1 HIGH_MAIN_RECURRENCE=0 LOW_MAIN_RECURRENCE=0 HIGH_DROP_CHUNK="$h" \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1 \
    LOW_DROP_CACHE=1 LOW_DROP_CHUNK=1 LOW_BLOCK_CACHE=1 RUNTIME_THREADS=1 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >"$bout" 2>"$berr"
  grep -Fq 'main_pull_ilp=2' "$bout"
  grep -Fq 'main_recurrence=1' "$bout"
  grep -Fq "high_drop_chunk=$h" "$bout"
  grep -Fq 'high_recurrence_p_range=27..15 high_symbol_range=14..27' "$bout"
done

# ptxas resource cost is independent of runtime thread count. Parse the ILP2
# recurrent kernel once per high-drop build so occupancy regressions are visible.
python3 - "$PREFIX" $HIGHDROP_LIST <<'PY'
import pathlib,re,sys
prefix=sys.argv[1]
for h in map(int,sys.argv[2:]):
    text=pathlib.Path(f'{prefix}.high{h}.build.err').read_text(errors='replace')
    target='main_pull_kernel_ilp2';regs=[];stores=[];loads=[];active=False
    for line in text.splitlines():
        if 'Function properties for' in line:
            active=target in line;continue
        if not active:continue
        m=re.search(r'Used\s+(\d+)\s+registers',line)
        if m:regs.append(int(m.group(1)))
        m=re.search(r'(\d+)\s+bytes spill stores',line)
        if m:stores.append(int(m.group(1)))
        m=re.search(r'(\d+)\s+bytes spill loads',line)
        if m:loads.append(int(m.group(1)))
    print(f'b300_mainrec_high{h}_ptxas_registers={max(regs) if regs else -1}')
    print(f'b300_mainrec_high{h}_ptxas_spill_store_bytes={max(stores) if stores else -1}')
    print(f'b300_mainrec_high{h}_ptxas_spill_load_bytes={max(loads) if loads else -1}')
PY

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
run_one(){
  local high="$1" threads="$2"
  local bin="$ONEESAN_BUILD_DIR/b300_mainrec_high${high}_threadsweep_n27"
  local tag="high${high}_t${threads}" out="${PREFIX}.${tag}.out" err="${PREFIX}.${tag}.err" tele="${PREFIX}.${tag}.gpu.csv"
  : >"$tele"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$out" 2>"$err"
  local rc=$?; set -e
  kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; tail -n 120 "$err" >&2 || true; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing result" >&2; return 4; }
  grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) " <<<"$line" || { echo "$tag row metadata mismatch" >&2; return 5; }
  local residue wall hg hf
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; hg="$(field high_rec_groups "$line")"; hf="$(field high_rec_fallback_groups "$line")"
  [[ "$hg" =~ ^[0-9]+$ ]] && ((hg>0)) || { echo "$tag zero high recurrence coverage hg=$hg hf=$hf" >&2; return 6; }
  local stats
  stats="$(python3 - "$tele" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5:continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError:continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50:busy.append(m)
def avg(v):return sum(v)/len(v) if v else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)}')
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$high" "$threads" "$residue" "$wall" "$hg" "$hf" "$stats"
}

TSV="${PREFIX}.summary.tsv"
printf 'high_drop_chunk\tthreads\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$TSV"
for t in $THREADS_LIST; do
  for h in $HIGHDROP_LIST; do
    echo "=== run MAIN_RECURRENCE high=$h threads=$t rows=$ROWS ===" >&2
    run_one "$h" "$t" >>"$TSV"
  done
done

python3 - "$TSV" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows:raise SystemExit('no sweep rows')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL recurrence sweep residue mismatch '+repr(sorted(res)))
for r in rows:
    r['wall']=float(r['wall_s']);r['busy']=float(r['mem_busy_avg_pct']);r['sm']=float(r['sm_avg_pct'])
b=min(rows,key=lambda r:r['wall'])
print('b300_mainrec_sweep_exact_intermediate_match=1')
print(f'b300_mainrec_sweep_residue={next(iter(res))}')
print(f'b300_mainrec_sweep_best_high_drop_chunk={b["high_drop_chunk"]}')
print(f'b300_mainrec_sweep_best_threads={b["threads"]}')
print(f'b300_mainrec_sweep_best_wall_s={b["wall"]:.9f}')
print(f'b300_mainrec_sweep_best_memctl_busy_avg_pct={b["busy"]:.3f}')
print(f'b300_mainrec_sweep_best_sm_avg_pct={b["sm"]:.3f}')
print(f'b300_mainrec_sweep_best_high_rec_groups={b["high_rec_groups"]}')
print(f'b300_mainrec_sweep_best_high_rec_fallback_groups={b["high_rec_fallback_groups"]}')
for r in sorted(rows,key=lambda x:(int(x['high_drop_chunk']),int(x['threads']))):
    print(f'  high={r["high_drop_chunk"]} threads={r["threads"]} wall_s={r["wall"]:.9f} mem_busy={r["busy"]:.3f}% sm={r["sm"]:.3f}% rec={r["high_rec_groups"]}/{r["high_rec_fallback_groups"]}')
PY
printf 'b300_mainrec_sweep_rows=%s\n' "$ROWS"
printf 'b300_mainrec_sweep_summary=%s\n' "$TSV"
printf 'b300_mainrec_sweep_note=run full exact only after row-limited residue equality and wall-time selection\n'
