#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516; else echo "EXPECT must be set when N/MOD differ from the n=21 reference" >&2; exit 2; fi
fi
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
REPEATS="${REPEATS:-3}"
THREADS_LIST="${THREADS_LIST:-32 64 128 256 512}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_warpstriped_thread_sweep_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/sweep_depthcode_warpstriped_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if (( NGPU != 8 )); then echo "warpstriped sweep requires NGPU=8" >&2; exit 2; fi
if (( REPEATS < 1 )); then echo "REPEATS must be >=1" >&2; exit 2; fi
for t in $THREADS_LIST; do if (( t < 32 || t > 1024 || t % 32 != 0 )); then echo "invalid thread count $t" >&2; exit 2; fi; done
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

N="$N" OUT="$BIN" HIGH_CTX=warpstriped DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
printf 'threads\twarps\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\tdecode_load_skip_fraction\n' >"$RESULT"
for t in $THREADS_LIST; do
  for ((rep=1; rep<=REPEATS; ++rep)); do
    so="$LOGDIR/t${t}_r${rep}.out"; se="$LOGDIR/t${t}_r${rep}.err"
    echo "=== warpstriped threads=$t repeat=$rep/$REPEATS ===" >&2
    BUCKET_THREADS="$t" "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "threads=$t missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "threads=$t residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    builder="$(grep 'pattern10_depthcode direct_build=1' "$se" | tail -n1 || true)"; [[ -n "$builder" ]] || { echo "threads=$t missing builder stats" >&2; exit 5; }
    rewritten="$(field rewritten_ops "$builder")"; none="$(field none_ops "$builder")"
    skipfrac="$(python3 - "$rewritten" "$none" <<'PY'
import sys
n=int(sys.argv[1]);s=int(sys.argv[2]);print(f"{(s/n if n else 0.0):.9f}")
PY
)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$((t/32))" "$rep" "$residue" "$(field wall_s "$line")" "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" "$(field transpose_s "$detail")" "$skipfrac" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src,newline=''),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
out=[]
for t in sorted({int(r['threads']) for r in rows}):
    g=[r for r in rows if int(r['threads'])==t]
    row={'threads':str(t),'warps':g[0]['warps'],'repeats':str(len(g)),'decode_load_skip_fraction':g[0]['decode_load_skip_fraction']}
    for m in metrics: row[m]=f"{statistics.median(float(r[m]) for r in g):.9f}"
    out.append(row)
fields=('threads','warps','repeats',*metrics,'decode_load_skip_fraction')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(out)
best=min(out,key=lambda r:float(r['forward_high_s'])+float(r['reverse_high_s']))
print(f"best_high_threads={best['threads']}")
print(f"best_high_sum_s={float(best['forward_high_s'])+float(best['reverse_high_s']):.9f}")
print(f"best_wall_threads={min(out,key=lambda r:float(r['wall_s']))['threads']}")
print(f"summary={dst}")
PY

echo "depthcode-warpstriped-thread-sweep OK n=$N repeats=$REPEATS decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
