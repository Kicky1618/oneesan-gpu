#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
EXPECT="${EXPECT:-998035516}"
NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
REPEATS="${REPEATS:-3}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depth4_depthcode_graph_ab_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

if (( NGPU != 8 )); then echo "depth4/depthcode Graph A/B requires NGPU=8" >&2; exit 2; fi
if (( REPEATS < 1 )); then echo "REPEATS must be >=1" >&2; exit 2; fi
case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_backend(){
  local backend="$1" bin="$2"
  case "$backend" in
    depth4_lut_resolved)
      N="$N" OUT="$bin" P10_DECODE=lut HIGH_CTX=resolved TRANSPOSE_MODE="$TRANSPOSE_MODE" \
        PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
        bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depth4-graph-batch.sh" ;;
    depthcode_payload_thread)
      N="$N" OUT="$bin" HIGH_CTX=thread TRANSPOSE_MODE="$TRANSPOSE_MODE" \
        PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
        bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" ;;
    depthcode_payload_resolved)
      N="$N" OUT="$bin" HIGH_CTX=resolved TRANSPOSE_MODE="$TRANSPOSE_MODE" \
        PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
        bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" ;;
    *) echo "unknown backend $backend" >&2; exit 2;;
  esac
}

run_one(){
  local backend="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${backend}_r${rep}.out" se="$LOGDIR/${backend}_r${rep}.err"
  echo "=== run $backend repeat=$rep/$REPEATS ===" >&2
  "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail plan residue wall fh fl rl rh ts meta fattach rattach
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$backend missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || { echo "$backend residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  plan="$(grep 'backend=gridfp-b300-bucket-snake-onepass-graph-batch' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"; rh="$(field reverse_high_s "$detail")"; ts="$(field transpose_s "$detail")"
  meta="$(field metadata_mib_per_gpu "$plan")"; fattach="$(field forward_attach_mib "$plan")"; rattach="$(field reverse_attach_mib "$plan")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$backend" "$rep" "$residue" "${wall:-NA}" "${fh:-NA}" "${fl:-NA}" "${rl:-NA}" "${rh:-NA}" "${ts:-NA}" \
    "${meta:-NA}" "${fattach:-NA}" "${rattach:-NA}" "$bin" >>"$RESULT"
}

printf 'backend\trepeat\tresidue\twall_s\tforward_high_s\tforward_low_s\treverse_low_s\treverse_high_s\ttranspose_s\tmetadata_mib_per_gpu\tforward_attach_mib\treverse_attach_mib\tbinary\n' >"$RESULT"
BACKENDS=(depth4_lut_resolved depthcode_payload_thread depthcode_payload_resolved)
for backend in "${BACKENDS[@]}"; do
  bin="$ONEESAN_BUILD_DIR/ab_${backend}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build $backend ===" >&2
  build_backend "$backend" "$bin"
  for ((rep=1; rep<=REPEATS; ++rep)); do run_one "$backend" "$bin" "$rep"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
metrics=("wall_s","forward_high_s","forward_low_s","reverse_low_s","reverse_high_s","transpose_s")
backends=("depth4_lut_resolved","depthcode_payload_thread","depthcode_payload_resolved")
with open(src,newline="") as f: rows=list(csv.DictReader(f,delimiter="\t"))
out=[]
for b in backends:
    g=[r for r in rows if r["backend"]==b]
    row={"backend":b,"repeats":str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!="NA"]
        row[m]=f"{statistics.median(xs):.9f}" if xs else "NA"
    for m in ("metadata_mib_per_gpu","forward_attach_mib","reverse_attach_mib"):
        row[m]=next((r[m] for r in g if r[m]!="NA"),"NA")
    out.append(row)
fields=("backend","repeats",*metrics,"metadata_mib_per_gpu","forward_attach_mib","reverse_attach_mib")
with open(dst,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(out)
q={r["backend"]:r for r in out}
def ratio(a,b,m):
    if q[a][m]!="NA" and q[b][m]!="NA":
        print(f"{b}_{m}_speedup_vs_{a}={float(q[a][m])/float(q[b][m]):.6f}x")
for m in ("wall_s","forward_high_s","forward_low_s","reverse_low_s","reverse_high_s"):
    ratio("depth4_lut_resolved","depthcode_payload_thread",m)
    ratio("depthcode_payload_thread","depthcode_payload_resolved",m)
    ratio("depth4_lut_resolved","depthcode_payload_resolved",m)
print(f"summary={dst}")
PY

echo "depth4-depthcode-graph-ab OK n=$N repeats=$REPEATS transpose=$TRANSPOSE_MODE result=$RESULT" >&2
