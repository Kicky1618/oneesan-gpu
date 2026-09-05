#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516; else echo "EXPECT must be set when N/MOD differ from the n=21 reference" >&2; exit 2; fi
fi
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-16}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-3}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_direct_resolve_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then echo "invalid launch/A-B parameters" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"; (( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

N="$N" bash "$ONEESAN_ROOT/scripts/bench/closure-ternary-delta-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/cross5-automaton-proof.sh"
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local ctx="$1" bin="$2"
  N="$N" OUT="$bin" HIGH_CTX="$ctx" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${ctx}.build.out" 2>"$LOGDIR/${ctx}.build.err"
}

printf 'high_ctx\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
run_one(){
  local ctx="$1" bin="$2" rep="$3" so="$LOGDIR/${ctx}_r${rep}.out" se="$LOGDIR/${ctx}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"
  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$ctx missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$ctx residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"; ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ctx" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

for ctx in warpstriped_delta_cross5 warpstriped_delta_direct_cross5; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_${ctx}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build $ctx ===" >&2; build_one "$ctx" "$bin"
  for ((r=1;r<=REPEATS;++r)); do echo "=== run $ctx $r/$REPEATS ===" >&2; run_one "$ctx" "$bin" "$r"; done
done
cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
metrics=('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
out=[]
for ctx in ('warpstriped_delta_cross5','warpstriped_delta_direct_cross5'):
    g=[r for r in rows if r['high_ctx']==ctx]
    z={'high_ctx':ctx,'repeats':str(len(g))}
    for m in metrics:
        xs=[float(r[m]) for r in g if r[m]!='NA']; z[m]=f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('high_ctx','repeats',*metrics),delimiter='\t');w.writeheader();w.writerows(out)
q={r['high_ctx']:r for r in out}
base='warpstriped_delta_cross5'; opt='warpstriped_delta_direct_cross5'
for m in ('wall_s','forward_high_s','reverse_high_s'):
    if q[base][m]!='NA' and q[opt][m]!='NA':
        print(f'direct_resolve_{m}_speedup={float(q[base][m])/float(q[opt][m]):.6f}x')
if all(q[x][m]!='NA' for x in (base,opt) for m in ('forward_high_s','reverse_high_s')):
    b=float(q[base]['forward_high_s'])+float(q[base]['reverse_high_s'])
    o=float(q[opt]['forward_high_s'])+float(q[opt]['reverse_high_s'])
    print(f'direct_resolve_total_high_speedup={b/o:.6f}x')
print('intermediate_plan_local_descriptors_base=1')
print('intermediate_plan_local_descriptors_direct=0')
print(f'summary={dst}')
PY

echo "depthcode-direct-resolve-ab OK n=$N repeats=$REPEATS threads=$BUCKET_THREADS gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
