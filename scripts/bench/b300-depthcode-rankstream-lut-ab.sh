#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
if [[ -z "${EXPECT+x}" ]]; then
  if [[ "$N" == 21 && "$MOD" == 4294967291 ]]; then
    EXPECT=998035516
  else
    echo "EXPECT must be set when N/MOD differ" >&2
    exit 2
  fi
fi

TARGET_MIB="${TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-5}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"
RANK_BACKEND="${RANK_BACKEND:-rankstream}"

case "$RANK_BACKEND" in
  rankstream)
    CTX=warpstriped_delta_direct_affine_prekey_rankstream_cross5
    SELFTEST="$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankstream-cross5-selftest.sh"
    ;;
  rankchunk32)
    CTX=warpstriped_delta_direct_affine_rankchunk32_cross5
    SELFTEST="$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh"
    ;;
  *) echo "RANK_BACKEND must be rankstream or rankchunk32" >&2; exit 2;;
esac

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_${RANK_BACKEND}_lut_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
MODES=(constant ldg ldg256)

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if [[ "$RUN_SELFTEST" != 0 && "$RUN_SELFTEST" != 1 ]]; then echo "RUN_SELFTEST must be 0 or 1" >&2; exit 2; fi
if (( NGPU != 8 || REPEATS < 1 || BUCKET_THREADS < 32 || BUCKET_THREADS > 1024 || BUCKET_THREADS % 32 != 0 || BUCKET_GRID_X < 1 || BUCKET_GRID_Y < 1 )); then
  echo "invalid launch/A-B parameters" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( visible >= NGPU )) || { echo "need $NGPU GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/cross5-rankmask-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/cross5-rankstream-projection-proof.sh"
if [[ "$RANK_BACKEND" == rankchunk32 ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
fi
if [[ "$RUN_SELFTEST" == 1 ]]; then
  for mode in "${MODES[@]}"; do
    if [[ "$RANK_BACKEND" == rankstream ]]; then
      LUT_LOAD="$mode" PM_ACCUM="$PM_ACCUM" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
        bash "$SELFTEST" >"$LOGDIR/selftest_${mode}.out" 2>"$LOGDIR/selftest_${mode}.err"
    else
      RANKSTREAM_LUT_LOAD="$mode" PM_ACCUM="$PM_ACCUM" DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
        bash "$SELFTEST" >"$LOGDIR/selftest_${mode}.out" 2>"$LOGDIR/selftest_${mode}.err"
    fi
  done
fi

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

build_one() {
  local mode="$1" bin="$2"
  N="$N" OUT="$bin" HIGH_CTX="$CTX" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    RANKSTREAM_LUT_LOAD="$mode" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
}

printf 'lut_load\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

run_one() {
  local mode="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"

  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || {
    echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2
    exit 4
  }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"
  rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"
  ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" \
    "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

for mode in "${MODES[@]}"; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_${CTX}_${mode}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build $RANK_BACKEND LUT $mode ===" >&2
  build_one "$mode" "$bin"
  for ((r = 1; r <= REPEATS; ++r)); do
    echo "=== run $RANK_BACKEND LUT $mode $r/$REPEATS ===" >&2
    run_one "$mode" "$bin" "$r"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RANK_BACKEND" <<'PY'
import csv
import statistics
import sys

src, dst, backend = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
metrics = ('wall_s', 'forward_high_s', 'reverse_high_s', 'forward_low_s', 'reverse_low_s', 'transpose_s')
modes = ('constant', 'ldg', 'ldg256')
out = []
for mode in modes:
    group = [r for r in rows if r['lut_load'] == mode]
    z = {'lut_load': mode, 'repeats': str(len(group))}
    for metric in metrics:
        xs = [float(r[metric]) for r in group if r[metric] != 'NA']
        z[metric] = f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)

with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=('lut_load', 'repeats', *metrics), delimiter='\t')
    w.writeheader()
    w.writerows(out)

q = {r['lut_load']: r for r in out}
prefix = 'rankstream_lut' if backend == 'rankstream' else 'rankchunk32_lut'
def speedup(base, opt, metric, suffix):
    if q[base][metric] == 'NA' or q[opt][metric] == 'NA':
        return
    print(f'{prefix}_{suffix}_{metric}_speedup={float(q[base][metric]) / float(q[opt][metric]):.6f}x')

for metric in ('wall_s', 'forward_high_s', 'reverse_high_s'):
    speedup('constant', 'ldg', metric, 'ldg')
    speedup('ldg', 'ldg256', metric, 'ldg256_vs_ldg')
    speedup('constant', 'ldg256', metric, 'ldg256_vs_constant')

for base, opt, suffix in (
    ('constant', 'ldg', 'ldg'),
    ('ldg', 'ldg256', 'ldg256_vs_ldg'),
    ('constant', 'ldg256', 'ldg256_vs_constant'),
):
    if all(q[x][m] != 'NA' for x in (base, opt) for m in ('forward_high_s', 'reverse_high_s')):
        b = float(q[base]['forward_high_s']) + float(q[base]['reverse_high_s'])
        o = float(q[opt]['forward_high_s']) + float(q[opt]['reverse_high_s'])
        print(f'{prefix}_{suffix}_total_high_speedup={b / o:.6f}x')

print(f'rank_backend={backend}')
print('rankstream_lut_logical_bytes=6561')
print('rankstream_lut_constant_physical_bytes=6561')
print('rankstream_lut_ldg_physical_bytes=6561')
print('rankstream_lut_ldg256_physical_bytes=6899')
print('rankstream_lut_ldg256_state_stride=256')
print('rankstream_lut_ldg256_state_row_cachelines_128b=2')
print('production_state_checks=0')
print('production_fallback=0')
if backend == 'rankchunk32':
    print('rankchunk32_cross_runtime_divmod=0')
    print('rankchunk32_block_base_loads_per_warp_max=3')
print(f'summary={dst}')
PY

echo "depthcode-rankstream-lut-ab OK backend=$RANK_BACKEND n=$N repeats=$REPEATS modes=constant,ldg,ldg256 threads=$BUCKET_THREADS gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
