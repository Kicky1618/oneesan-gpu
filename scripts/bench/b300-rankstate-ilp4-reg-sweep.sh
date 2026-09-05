#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; [[ "$N" == 27 ]] || { echo 'ILP4 register sweep targets n=27' >&2; exit 2; }
MOD="${MOD:-4294967291}"; ARCH="${ARCH:-native}"; TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; THREADS_LIST="${THREADS_LIST:-128 256}"; CAPS="${CAPS:-0 96 112 128 144}"; QUADS="${QUADS:-0 1}"; HOT="${HOT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankstate_ilp4_reg_row${ROWS}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR"
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }; command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
[[ "$HOT" == 0 || "$HOT" == 1 ]] || { echo 'HOT must be 0/1' >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-ilp4-partition-proof.sh"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'variant\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
: >"$LOGDIR/binaries.tsv"
for q in $QUADS; do
  [[ "$q" == 0 || "$q" == 1 ]] || { echo "bad closure quad=$q" >&2; exit 2; }
  for cap in $CAPS; do
    [[ "$cap" =~ ^[0-9]+$ ]] && ((cap==0||(cap>=32&&cap<=255))) || { echo "bad cap=$cap" >&2; exit 2; }
    tag="q${q}_r${cap}"; bin="$ONEESAN_BUILD_DIR/b300_rankstate_ilp4_${tag}_n27"
    echo "=== build ILP4 closure_quad=$q maxrregcount=$cap ===" >&2
    N=27 ARCH="$ARCH" OUT="$bin" FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
      MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=0 RANK_STATE_ILP4=1 BLOCK_CLOSURE_QUAD="$q" \
      HOT_DELTA_TABLE="$HOT" CONCURRENT_GROUP_IO=1 MAXRREGCOUNT="$cap" PTXAS_VERBOSE=1 \
      bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/$tag.build.out" 2>"$LOGDIR/$tag.build.err"
    printf '%s\t%s\n' "$tag" "$bin" >>"$LOGDIR/binaries.tsv"
    python3 "$PARSER" "$LOGDIR/$tag.build.err" --label "$tag" --contains rankstate_ilp4 >>"$RESOURCE" || true
  done
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
stats(){ python3 - "$1" <<'PY'
import csv,sys
m=[];b=[];s=[]
for r in csv.reader(open(sys.argv[1])):
 if len(r)<4:continue
 try:sm=float(r[2]);mc=float(r[3])
 except:continue
 s.append(sm);m.append(mc)
 if sm>=50:b.append(mc)
def a(x):return sum(x)/len(x) if x else float('nan')
print(f'{a(m):.6f} {max(m) if m else float("nan"):.6f} {a(b):.6f} {a(s):.6f} {len(m)}')
PY
}
printf 'closure_quad\tcap\tthreads\tresidue\twall_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tsamples\n' >"$RESULT"
for q in $QUADS; do for cap in $CAPS; do
  tag="q${q}_r${cap}"; bin="$(awk -F '\t' -v t="$tag" '$1==t{print $2}' "$LOGDIR/binaries.tsv")"
  for th in $THREADS_LIST; do
    out="$LOGDIR/${tag}_t${th}.out"; err="$LOGDIR/${tag}_t${th}.err"; tele="$LOGDIR/${tag}_t${th}.gpu.csv"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & mon=$!; sleep 1
    set +e; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$th" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"; rc=$?; set -e
    kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
    ((rc==0)) || { echo "$tag threads=$th failed rc=$rc" >&2; tail -n 100 "$err" >&2; exit "$rc"; }
    line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1)"; residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; z="$(stats "$tele")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$q" "$cap" "$th" "$residue" "$wall" "$z" >>"$RESULT"
  done
done; done

python3 - "$RESULT" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if len({x['residue'] for x in r})!=1:raise SystemExit('FATAL residue mismatch')
for x in r:x['wall']=float(x['wall_s']);x['busy']=float(x['mem_busy_avg_pct'])
r.sort(key=lambda x:x['wall'])
for x in r:print(f'q={x["closure_quad"]} cap={x["cap"]} threads={x["threads"]} wall_s={x["wall"]:.9f} mem_busy={x["busy"]:.3f}%')
b=r[0]
print('b300_ilp4_reg_residue_match=1')
print(f'b300_ilp4_reg_best_closure_quad={b["closure_quad"]}')
print(f'b300_ilp4_reg_best_cap={b["cap"]}')
print(f'b300_ilp4_reg_best_threads={b["threads"]}')
print(f'b300_ilp4_reg_best_wall_s={b["wall"]:.9f}')
print(f'b300_ilp4_reg_best_mem_busy_pct={b["busy"]:.3f}')
PY
cat "$RESOURCE"
echo "b300-rankstate-ilp4-reg-sweep OK result=$RESULT resource=$RESOURCE" >&2
