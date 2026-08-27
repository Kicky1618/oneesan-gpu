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
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
THREADS_LIST="${THREADS_LIST:-64 128 256}"
GRID_Y_LIST="${GRID_Y_LIST:-1 2 4 8}"
REPEATS="${REPEATS:-2}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_warpstriped_geometry_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_gx${BUCKET_GRID_X}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/sweep_depthcode_warpstriped_geometry_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if (( NGPU != 8 || BUCKET_GRID_X < 1 || REPEATS < 1 )); then echo "invalid NGPU/BUCKET_GRID_X/REPEATS" >&2; exit 2; fi
for t in $THREADS_LIST; do if (( t < 32 || t > 1024 || t % 32 != 0 )); then echo "invalid thread count $t" >&2; exit 2; fi; done
for gy in $GRID_Y_LIST; do if (( gy < 1 )); then echo "invalid grid_y $gy" >&2; exit 2; fi; done
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
if ! command -v nvidia-smi >/dev/null; then echo "nvidia-smi not found" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; if (( visible < NGPU )); then echo "requested $NGPU GPUs, but only $visible are visible" >&2; exit 2; fi
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

N="$N" OUT="$BIN" HIGH_CTX=warpstriped DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
  TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
  bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
printf 'threads\twarps\tgrid_x\tgrid_y\tvirtual_orbit_lanes\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
for t in $THREADS_LIST; do
  for gy in $GRID_Y_LIST; do
    for ((rep=1; rep<=REPEATS; ++rep)); do
      so="$LOGDIR/t${t}_gy${gy}_r${rep}.out"; se="$LOGDIR/t${t}_gy${gy}_r${rep}.err"
      echo "=== warpstriped threads=$t grid_y=$gy repeat=$rep/$REPEATS ===" >&2
      BUCKET_THREADS="$t" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$gy" \
        "$BIN" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
      line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "t=$t gy=$gy missing residue" >&2; exit 3; }
      residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "t=$t gy=$gy residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
      detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; [[ -n "$detail" ]] || { echo "t=$t gy=$gy missing phase timing" >&2; exit 5; }
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$t" "$((t/32))" "$BUCKET_GRID_X" "$gy" "$((gy*t/32))" "$rep" "$residue" \
        "$(field wall_s "$line")" "$(field forward_high_s "$detail")" "$(field reverse_high_s "$detail")" \
        "$(field forward_low_s "$detail")" "$(field reverse_low_s "$detail")" "$(field transpose_s "$detail")" >>"$RESULT"
    done
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src,newline=''),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
keys=sorted({(int(r['threads']),int(r['grid_y'])) for r in rows})
out=[]
for t,gy in keys:
    g=[r for r in rows if int(r['threads'])==t and int(r['grid_y'])==gy]
    row={'threads':str(t),'warps':g[0]['warps'],'grid_x':g[0]['grid_x'],'grid_y':str(gy),'virtual_orbit_lanes':g[0]['virtual_orbit_lanes'],'repeats':str(len(g))}
    for m in metrics: row[m]=f"{statistics.median(float(r[m]) for r in g):.9f}"
    row['high_sum_s']=f"{float(row['forward_high_s'])+float(row['reverse_high_s']):.9f}"
    out.append(row)
fields=('threads','warps','grid_x','grid_y','virtual_orbit_lanes','repeats',*metrics,'high_sum_s')
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(out)
best_high=min(out,key=lambda r:float(r['high_sum_s']))
best_wall=min(out,key=lambda r:float(r['wall_s']))
print(f"best_high_threads={best_high['threads']}")
print(f"best_high_grid_y={best_high['grid_y']}")
print(f"best_high_virtual_orbit_lanes={best_high['virtual_orbit_lanes']}")
print(f"best_high_sum_s={best_high['high_sum_s']}")
print(f"best_wall_threads={best_wall['threads']}")
print(f"best_wall_grid_y={best_wall['grid_y']}")
print(f"best_wall_s={best_wall['wall_s']}")
print(f"summary={dst}")
PY

echo "depthcode-warpstriped-geometry-sweep OK n=$N repeats=$REPEATS grid_x=$BUCKET_GRID_X decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
