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
HIGH_CTX="${HIGH_CTX:-warp}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-global}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
REPEATS="${REPEATS:-3}"
RUN="${RUN:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_source_key_ab_n${N}_${HIGH_CTX}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_key4${TERNARY_KEY4}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCES="${RESOURCES:-${PREFIX}_ptxas.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

case "$HIGH_CTX" in thread|resolved|warp) ;; *) echo "HIGH_CTX must be thread, resolved, or warp" >&2; exit 2;; esac
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if [[ "$RUN" != 0 && "$RUN" != 1 ]]; then echo "RUN must be 0 or 1" >&2; exit 2; fi
if (( REPEATS < 1 )); then echo "REPEATS must be >=1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if [[ "$RUN" == 1 ]]; then
  if (( NGPU != 8 )); then echo "depthcode source-key A/B requires NGPU=8" >&2; exit 2; fi
  if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
  visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCES"
build_one(){
  local mode="$1"
  local bin="$2"
  local label="depthcode_${HIGH_CTX}_source_${mode}"
  local bout="$LOGDIR/${label}.build.out"
  local blog="$LOGDIR/${label}.ptxas.log"
  echo "=== build $label ===" >&2
  N="$N" OUT="$bin" HIGH_CTX="$HIGH_CTX" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" DEPTHCODE_SOURCE_KEY="$mode" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" >"$bout" 2>"$blog"
  python3 "$PARSER" "$blog" --label "$label" >>"$RESOURCES"
}

printf 'source_key\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tmetadata_mib_per_gpu\tforward_attach_mib\treverse_attach_mib\tbinary\n' >"$RESULT"
run_one(){
  local mode="$1"
  local bin="$2"
  local rep="$3"
  local so="$LOGDIR/${mode}_r${rep}.out"
  local se="$LOGDIR/${mode}_r${rep}.err"
  echo "=== run source_key=$mode repeat=$rep/$REPEATS ===" >&2
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail plan residue wall fh fl rl rh ts meta fattach rattach
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  plan="$(grep 'backend=gridfp-b300-bucket-snake-onepass-graph-batch' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  meta="$(field metadata_mib_per_gpu "$plan")"; fattach="$(field forward_attach_mib "$plan")"; rattach="$(field reverse_attach_mib "$plan")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$residue" "${wall:-NA}" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" "${meta:-NA}" "${fattach:-NA}" "${rattach:-NA}" "$bin" >>"$RESULT"
}

for mode in scalar key4; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_${HIGH_CTX}_source_${mode}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  build_one "$mode" "$bin"
  if [[ "$RUN" == 1 ]]; then
    for ((rep=1; rep<=REPEATS; ++rep)); do run_one "$mode" "$bin" "$rep"; done
  fi
done

cat "$RESOURCES"
if [[ "$RUN" == 0 ]]; then
  echo "depthcode-source-key-ab compile-only OK n=$N high_ctx=$HIGH_CTX transpose=$TRANSPOSE_MODE resources=$RESOURCES" >&2
  exit 0
fi

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
metrics=("wall_s","forward_high_s","forward_low_s","reverse_low_s","reverse_high_s","transpose_s")
rows=list(csv.DictReader(open(src,newline=""),delimiter="\t"))
out=[]
for mode in ("scalar","key4"):
    g=[r for r in rows if r["source_key"]==mode]
    row={"source_key":mode,"repeats":str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!="NA"]
        row[m]=f"{statistics.median(xs):.9f}" if xs else "NA"
    for m in ("metadata_mib_per_gpu","forward_attach_mib","reverse_attach_mib"):
        row[m]=next((r[m] for r in g if r[m]!="NA"),"NA")
    out.append(row)
fields=("source_key","repeats",*metrics,"metadata_mib_per_gpu","forward_attach_mib","reverse_attach_mib")
with open(dst,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(out)
q={r["source_key"]:r for r in out}
for m in ("wall_s","forward_high_s","forward_low_s","reverse_low_s","reverse_high_s"):
    if q["scalar"][m]!="NA" and q["key4"][m]!="NA":
        print(f"key4_{m}_speedup_vs_scalar={float(q['scalar'][m])/float(q['key4'][m]):.6f}x")
print(f"summary={dst}")
PY
echo "depthcode-source-key-ab OK n=$N high_ctx=$HIGH_CTX repeats=$REPEATS transpose=$TRANSPOSE_MODE result=$RESULT resources=$RESOURCES" >&2
