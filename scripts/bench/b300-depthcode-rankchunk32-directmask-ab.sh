#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then EXPECT=998035516
  else echo "EXPECT must be set when N/MOD differ" >&2; exit 2
  fi
fi

ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"
RANKCHUNK32_BYTEPACK="${RANKCHUNK32_BYTEPACK:-0}"
RANKCHUNK32_ALIGN32="${RANKCHUNK32_ALIGN32:-1}"
RANKCHUNK32_BLOCK64="${RANKCHUNK32_BLOCK64:-0}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-3}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankchunk32_directmask_ab_n${N}_${TRANSPOSE_MODE}_align${RANKCHUNK32_ALIGN32}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
for x in RANKCHUNK32_ONESHFL RANKCHUNK32_FUSED16 RANKCHUNK32_BYTEPACK RANKCHUNK32_ALIGN32 RANKCHUNK32_BLOCK64 PM_ACCUM TERNARY_KEY4 PTXAS_VERBOSE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if [[ "$RANKCHUNK32_BLOCK64" == 1 && "$RANKCHUNK32_BYTEPACK" == 1 ]]; then
  echo "RANKCHUNK32_BLOCK64 requires BYTEPACK=0" >&2; exit 2
fi
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then
  echo "invalid launch/A-B parameters" >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2; exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/cross5-rankmask-shape-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/cross5-rankstream-projection-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

build_one() {
  local directmask="$1" bin="$2"
  N="$N" ARCH="$ARCH" OUT="$bin" HIGH_CTX=warpstriped_delta_direct_affine_rankchunk32_cross5 \
    DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
    RANKCHUNK32_DIRECT3=0 RANKCHUNK32_DIRECTMASK="$directmask" \
    RANKCHUNK32_BYTEPACK="$RANKCHUNK32_BYTEPACK" RANKCHUNK32_ALIGN32="$RANKCHUNK32_ALIGN32" \
    RANKCHUNK32_BLOCK64="$RANKCHUNK32_BLOCK64" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/directmask_${directmask}.build.out" 2>"$LOGDIR/directmask_${directmask}.build.err"
}

printf 'directmask\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

run_one() {
  local directmask="$1" bin="$2" rep="$3"
  local so="$LOGDIR/directmask_${directmask}_r${rep}.out" se="$LOGDIR/directmask_${directmask}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"

  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "directmask=$directmask missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || {
    echo "directmask=$directmask residue mismatch got=$residue expected=$EXPECT" >&2; exit 4
  }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"; rl="$(field reverse_low_s "$detail")"
  ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$directmask" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" \
    "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

for directmask in 0 1; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_rankchunk32_directmask_${directmask}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build rankchunk32 directmask=$directmask ===" >&2
  build_one "$directmask" "$bin"
  for ((r = 1; r <= REPEATS; ++r)); do
    echo "=== run rankchunk32 directmask=$directmask $r/$REPEATS ===" >&2
    run_one "$directmask" "$bin" "$r"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv, statistics, sys
src, dst = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
metrics = ('wall_s','forward_high_s','reverse_high_s','forward_low_s','reverse_low_s','transpose_s')
out=[]
for mode in ('0','1'):
    group=[r for r in rows if r['directmask']==mode]
    z={'directmask':mode,'repeats':str(len(group))}
    for metric in metrics:
        xs=[float(r[metric]) for r in group if r[metric] != 'NA']
        z[metric]=f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)
with open(dst,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=('directmask','repeats',*metrics),delimiter='\t')
    w.writeheader(); w.writerows(out)
q={r['directmask']:r for r in out}
def speedup(metric):
    if q['0'][metric]=='NA' or q['1'][metric]=='NA': return
    print(f'rankchunk32_directmask_{metric}_speedup={float(q["0"][metric])/float(q["1"][metric]):.6f}x')
for metric in ('wall_s','forward_high_s','reverse_high_s'): speedup(metric)
if all(q[x][m] != 'NA' for x in ('0','1') for m in ('forward_high_s','reverse_high_s')):
    old=float(q['0']['forward_high_s'])+float(q['0']['reverse_high_s'])
    new=float(q['1']['forward_high_s'])+float(q['1']['reverse_high_s'])
    print(f'rankchunk32_directmask_total_high_speedup={old/new:.6f}x')
print('directmask_runtime_cross5_lut=0')
print('directmask_runtime_cross5_state=0')
print('directmask_zero_skips_rankchunk_meta=1')
print('directmask_layout=depth_major_coalesced_bytes')
print(f'summary={dst}')
PY

echo "b300-depthcode-rankchunk32-directmask-ab OK n=$N repeats=$REPEATS align32=$RANKCHUNK32_ALIGN32 result=$RESULT" >&2
