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
REPEATS="${REPEATS:-3}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_warpstriped_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_t${BUCKET_THREADS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "TRANSPOSE_MODE must be sync, events, or pipeline" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if (( BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 )); then echo "warpstriped requires BUCKET_THREADS multiple of 32 in [32,1024]" >&2; exit 2; fi
if (( NGPU != 8 )); then echo "warpstriped A/B requires NGPU=8" >&2; exit 2; fi
if (( REPEATS < 1 )); then echo "REPEATS must be >=1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local ctx="$1" bin="$2"
  echo "=== build high_ctx=$ctx ===" >&2
  N="$N" OUT="$bin" HIGH_CTX="$ctx" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${ctx}.build.out" 2>"$LOGDIR/${ctx}.build.err"
}

printf 'high_ctx\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\trewritten_ops\tnone_ops\tdecode_load_skip_fraction\tbinary\n' >"$RESULT"
run_one(){
  local ctx="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${ctx}_r${rep}.out" se="$LOGDIR/${ctx}_r${rep}.err"
  echo "=== run high_ctx=$ctx repeat=$rep/$REPEATS ===" >&2
  BUCKET_THREADS="$BUCKET_THREADS" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail builder residue wall fh rh fl rl ts rewritten none skipfrac
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$ctx missing residue line" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$ctx residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  builder="$(grep 'pattern10_depthcode direct_build=1' "$se" | tail -n1 || true)"; [[ -n "$builder" ]] || { echo "$ctx missing depthcode builder stats" >&2; exit 5; }
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  rewritten="$(field rewritten_ops "$builder")"; none="$(field none_ops "$builder")"
  skipfrac="$(python3 - "$rewritten" "$none" <<'PY'
import sys
n=int(sys.argv[1]); s=int(sys.argv[2]); print(f"{(s/n if n else 0.0):.9f}")
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ctx" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" "$rewritten" "$none" "$skipfrac" "$bin" >>"$RESULT"
}

for ctx in warp warpstriped; do
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
for ctx in ("warp","warpstriped"):
    g=[r for r in rows if r["high_ctx"]==ctx]
    row={"high_ctx":ctx,"repeats":str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!="NA"]
        row[m]=f"{statistics.median(xs):.9f}" if xs else "NA"
    for m in ("rewritten_ops","none_ops","decode_load_skip_fraction"):
        row[m]=next((r[m] for r in g if r[m]!="NA"),"NA")
    out.append(row)
fields=("high_ctx","repeats",*metrics,"rewritten_ops","none_ops","decode_load_skip_fraction")
with open(dst,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter="\t");w.writeheader();w.writerows(out)
q={r["high_ctx"]:r for r in out}
for m in ("wall_s","forward_high_s","reverse_high_s"):
    if q["warp"][m]!="NA" and q["warpstriped"][m]!="NA":
        print(f"warpstriped_{m}_speedup_vs_warp={float(q['warp'][m])/float(q['warpstriped'][m]):.6f}x")
print(f"decode_load_skip_fraction={q['warp']['decode_load_skip_fraction']}")
print(f"summary={dst}")
PY

echo "depthcode-warpstriped-ab OK n=$N repeats=$REPEATS threads=$BUCKET_THREADS decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
