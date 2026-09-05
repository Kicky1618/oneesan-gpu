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
RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKCHUNK32_ONESHFL="${RANKCHUNK32_ONESHFL:-1}"
RANKCHUNK32_FUSED16="${RANKCHUNK32_FUSED16:-0}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-16}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
REPEATS="${REPEATS:-5}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"
RUN_PTXAS="${RUN_PTXAS:-1}"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankchunk32_bytepack_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_${RANKSTREAM_LUT_LOAD}_shfl${RANKCHUNK32_ONESHFL}_fused${RANKCHUNK32_FUSED16}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
case "$RANKSTREAM_LUT_LOAD" in constant|ldg|ldg256) ;; *) echo "invalid RANKSTREAM_LUT_LOAD" >&2; exit 2;; esac
for x in RANKCHUNK32_ONESHFL RANKCHUNK32_FUSED16 PM_ACCUM TERNARY_KEY4 RUN_SELFTEST RUN_PTXAS; do
  v="${!x}"
  if [[ "$v" != 0 && "$v" != 1 ]]; then echo "$x must be 0 or 1" >&2; exit 2; fi
done
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

bash "$ONEESAN_ROOT/scripts/bench/cross5-rankstream-projection-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-warpbase-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-bytepack-proof.sh"

if [[ "$RUN_SELFTEST" == 1 ]]; then
  for bytepack in 0 1; do
    RUN_LAYOUT_PROOF=0 RANKCHUNK32_BYTEPACK="$bytepack" \
      RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" \
      RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" PM_ACCUM="$PM_ACCUM" \
      DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankchunk32-cross5-selftest.sh" \
      >"$LOGDIR/selftest_bytepack${bytepack}.out" 2>"$LOGDIR/selftest_bytepack${bytepack}.err"
  done
fi

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

build_one() {
  local bytepack="$1" bin="$2"
  N="$N" OUT="$bin" HIGH_CTX=warpstriped_delta_direct_affine_rankchunk32_cross5 \
    DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" RANKSTREAM_LUT_LOAD="$RANKSTREAM_LUT_LOAD" \
    RANKCHUNK32_ONESHFL="$RANKCHUNK32_ONESHFL" RANKCHUNK32_FUSED16="$RANKCHUNK32_FUSED16" \
    RANKCHUNK32_BYTEPACK="$bytepack" TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/bytepack${bytepack}.build.out" 2>"$LOGDIR/bytepack${bytepack}.build.err"
}

printf 'bytepack\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"
if [[ "$RUN_PTXAS" == 1 ]]; then
  printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
fi

run_one() {
  local bytepack="$1" bin="$2" rep="$3"
  local so="$LOGDIR/bytepack${bytepack}_r${rep}.out" se="$LOGDIR/bytepack${bytepack}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"

  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "bytepack=$bytepack missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || {
    echo "bytepack=$bytepack residue mismatch got=$residue expected=$EXPECT" >&2
    exit 4
  }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"
  rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"
  ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$bytepack" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" \
    "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

for bytepack in 0 1; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_rankchunk32_bytepack${bytepack}_${RANKSTREAM_LUT_LOAD}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build rankchunk32 bytepack=$bytepack ===" >&2
  build_one "$bytepack" "$bin"
  if [[ "$RUN_PTXAS" == 1 ]]; then
    python3 "$PARSER" "$LOGDIR/bytepack${bytepack}.build.err" --label "bytepack${bytepack}" >>"$RESOURCE"
  fi
  for ((r = 1; r <= REPEATS; ++r)); do
    echo "=== run rankchunk32 bytepack=$bytepack $r/$REPEATS ===" >&2
    run_one "$bytepack" "$bin" "$r"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" <<'PY'
import csv
import statistics
import sys

src, dst, resource, run_ptxas = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
metrics = ('wall_s', 'forward_high_s', 'reverse_high_s', 'forward_low_s', 'reverse_low_s', 'transpose_s')
out = []
for mode in ('0', '1'):
    group = [r for r in rows if r['bytepack'] == mode]
    z = {'bytepack': mode, 'repeats': str(len(group))}
    for metric in metrics:
        xs = [float(r[metric]) for r in group if r[metric] != 'NA']
        z[metric] = f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)
with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=('bytepack', 'repeats', *metrics), delimiter='\t')
    w.writeheader(); w.writerows(out)
q = {r['bytepack']: r for r in out}
for metric in ('wall_s', 'forward_high_s', 'reverse_high_s'):
    if q['0'][metric] != 'NA' and q['1'][metric] != 'NA':
        print(f'rankchunk32_bytepack_{metric}_speedup={float(q["0"][metric])/float(q["1"][metric]):.6f}x')
if all(q[x][m] != 'NA' for x in ('0','1') for m in ('forward_high_s','reverse_high_s')):
    a=float(q['0']['forward_high_s'])+float(q['0']['reverse_high_s'])
    b=float(q['1']['forward_high_s'])+float(q['1']['reverse_high_s'])
    print(f'rankchunk32_bytepack_total_high_speedup={a/b:.6f}x')
if run_ptxas == '1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for mode in ('0','1'):
        g=[r for r in rr if r['backend']==f'bytepack{mode}' and 'high' in r['kernel'].lower()]
        def vals(k): return [int(r[k]) for r in g if r[k]!='NA']
        regs=vals('registers'); ss=vals('spill_store_bytes'); sl=vals('spill_load_bytes')
        print(f'rankchunk32_bytepack{mode}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'rankchunk32_bytepack{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'rankchunk32_bytepack{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
print('rankchunk32_bytepack0_model=chunk23+prefix9+block32')
print('rankchunk32_bytepack1_model=chunk24+prefix8+block32')
print('rankchunk32_bytepack1_byte_aligned_chunks=1')
print('rankchunk32_bytepack1_max_l_per_legal_code=7')
print('rankchunk32_bytepack1_max_prefix=217')
print('rankchunk32_metadata_bytes_per_code_equal=1')
print('rankchunk32_block_table_equal=1')
print('rankchunk32_cross_runtime_divmod=0')
print(f'summary={dst}')
PY

echo "b300-depthcode-rankchunk32-bytepack-ab OK n=$N repeats=$REPEATS lut=$RANKSTREAM_LUT_LOAD decode_load=$DEPTHCODE_DECODE_LOAD oneshfl=$RANKCHUNK32_ONESHFL fused16=$RANKCHUNK32_FUSED16 transpose=$TRANSPOSE_MODE result=$RESULT resource=$RESOURCE" >&2
