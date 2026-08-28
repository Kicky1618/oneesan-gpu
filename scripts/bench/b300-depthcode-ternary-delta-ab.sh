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
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-3}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_ternary_delta_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCES="${RESOURCES:-${PREFIX}_ptxas.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if (( BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 )); then echo "ternary-delta A/B requires BUCKET_THREADS multiple of 32 in [32,1024]" >&2; exit 2; fi
if (( BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then echo "BUCKET_GRID_X/Y must be >=1" >&2; exit 2; fi
if (( NGPU != 8 )); then echo "ternary-delta A/B requires NGPU=8" >&2; exit 2; fi
if (( REPEATS < 1 )); then echo "REPEATS must be >=1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCES"
build_one(){
  local ctx="$1" bin="$2"
  local log="$LOGDIR/${ctx}.ptxas.log"
  echo "=== build high_ctx=$ctx ===" >&2
  N="$N" OUT="$bin" HIGH_CTX="$ctx" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${ctx}.build.out" 2>"$log"
  python3 "$PARSER" "$log" --label "$ctx" >>"$RESOURCES"
}

printf 'high_ctx\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\tbinary\n' >"$RESULT"
run_one(){
  local ctx="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${ctx}_r${rep}.out" se="$LOGDIR/${ctx}_r${rep}.err"
  echo "=== run high_ctx=$ctx repeat=$rep/$REPEATS ===" >&2
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$ctx missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$ctx residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ctx" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" "$bin" >>"$RESULT"
}

for ctx in warpstriped warpstriped_delta; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_${ctx}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  build_one "$ctx" "$bin"
  for ((rep=1; rep<=REPEATS; ++rep)); do run_one "$ctx" "$bin" "$rep"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src,newline=""),delimiter="\t"))
metrics=("wall_s","forward_high_s","reverse_high_s","forward_low_s","reverse_low_s","transpose_s")
out=[]
for ctx in ("warpstriped","warpstriped_delta"):
    g=[r for r in rows if r["high_ctx"]==ctx]
    row={"high_ctx":ctx,"repeats":str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!="NA"]
        row[m]=f"{statistics.median(xs):.9f}" if xs else "NA"
    out.append(row)
fields=("high_ctx","repeats",*metrics)
with open(dst,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter="\t");w.writeheader();w.writerows(out)
q={r["high_ctx"]:r for r in out}
for m in ("wall_s","forward_high_s","reverse_high_s"):
    a=q["warpstriped"][m]; b=q["warpstriped_delta"][m]
    if a!="NA" and b!="NA": print(f"ternary_delta_{m}_speedup_vs_mateid={float(a)/float(b):.6f}x")
print(f"summary={dst}")
PY
cat "$RESOURCES"

echo "depthcode-ternary-delta-ab OK n=$N repeats=$REPEATS threads=$BUCKET_THREADS gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT resources=$RESOURCES" >&2
