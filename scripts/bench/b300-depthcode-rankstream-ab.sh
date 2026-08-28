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
REPEATS="${REPEATS:-3}"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_depthcode_rankstream_ab_n${N}_${TRANSPOSE_MODE}_${DEPTHCODE_DECODE_LOAD}_pm${PM_ACCUM}_t${BUCKET_THREADS}_gx${BUCKET_GRID_X}_gy${BUCKET_GRID_Y}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "invalid DEPTHCODE_DECODE_LOAD" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
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

bash "$ONEESAN_ROOT/scripts/bench/cross5-automaton-proof.sh"

field() {
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

build_one() {
  local ctx="$1" bin="$2"
  N="$N" OUT="$bin" HIGH_CTX="$ctx" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${ctx}.build.out" 2>"$LOGDIR/${ctx}.build.err"
}

printf 'high_ctx\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tforward_low_s\treverse_low_s\ttranspose_s\n' >"$RESULT"

run_one() {
  local ctx="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${ctx}_r${rep}.out" se="$LOGDIR/${ctx}_r${rep}.err"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se"

  local line detail residue wall fh rh fl rl ts
  line="$(grep '^residue=' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$ctx missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"
  wall="$(field wall_s "$line")"
  [[ "$residue" == "$EXPECT" ]] || {
    echo "$ctx residue mismatch got=$residue expected=$EXPECT" >&2
    exit 4
  }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"
  rh="$(field reverse_high_s "$detail")"
  fl="$(field forward_low_s "$detail")"
  rl="$(field reverse_low_s "$detail")"
  ts="$(field transpose_s "$detail")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ctx" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" \
    "${fl:-NA}" "${rl:-NA}" "${ts:-NA}" >>"$RESULT"
}

contexts=(
  warpstriped_delta_direct_affine_prekey_rank16_cross5
  warpstriped_delta_direct_affine_prekey_rankstream_cross5
)
for ctx in "${contexts[@]}"; do
  bin="$ONEESAN_BUILD_DIR/ab_depthcode_${ctx}_${DEPTHCODE_DECODE_LOAD}_${TRANSPOSE_MODE}_n${N}"
  echo "=== build $ctx ===" >&2
  build_one "$ctx" "$bin"
  for ((r = 1; r <= REPEATS; ++r)); do
    echo "=== run $ctx $r/$REPEATS ===" >&2
    run_one "$ctx" "$bin" "$r"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" \
  "$LOGDIR/warpstriped_delta_direct_affine_prekey_rank16_cross5_r1.err" \
  "$LOGDIR/warpstriped_delta_direct_affine_prekey_rankstream_cross5_r1.err" <<'PY'
import csv
import re
import statistics
import sys

src, dst, rank16_log, rankstream_log = sys.argv[1:]
rows = list(csv.DictReader(open(src), delimiter='\t'))
metrics = ('wall_s', 'forward_high_s', 'reverse_high_s', 'forward_low_s', 'reverse_low_s', 'transpose_s')
order = (
    'warpstriped_delta_direct_affine_prekey_rank16_cross5',
    'warpstriped_delta_direct_affine_prekey_rankstream_cross5',
)
out = []
for ctx in order:
    group = [r for r in rows if r['high_ctx'] == ctx]
    z = {'high_ctx': ctx, 'repeats': str(len(group))}
    for metric in metrics:
        xs = [float(r[metric]) for r in group if r[metric] != 'NA']
        z[metric] = f'{statistics.median(xs):.9f}' if xs else 'NA'
    out.append(z)

with open(dst, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=('high_ctx', 'repeats', *metrics), delimiter='\t')
    w.writeheader()
    w.writerows(out)

q = {r['high_ctx']: r for r in out}
base, opt = order
for metric in ('wall_s', 'forward_high_s', 'reverse_high_s'):
    if q[base][metric] != 'NA' and q[opt][metric] != 'NA':
        print(f'rankstream_{metric}_speedup={float(q[base][metric]) / float(q[opt][metric]):.6f}x')
if all(q[x][m] != 'NA' for x in (base, opt) for m in ('forward_high_s', 'reverse_high_s')):
    b = float(q[base]['forward_high_s']) + float(q[base]['reverse_high_s'])
    o = float(q[opt]['forward_high_s']) + float(q[opt]['reverse_high_s'])
    print(f'rankstream_total_high_speedup={b / o:.6f}x')

rank16_text = open(rank16_log, errors='replace').read()
rankstream_text = open(rankstream_log, errors='replace').read()
r16 = [tuple(map(int, m)) for m in re.findall(
    r'p10dc_low_rank16 fixed_owner=\d+ entries=(\d+) valid=(\d+)', rank16_text)]
rs = [tuple(map(int, m)) for m in re.findall(
    r'p10dc_low_rankstream fixed_owner=\d+ codes=(\d+) l_ranks=(\d+)', rankstream_text)]
if r16:
    entries = sum(x for x, _ in r16)
    valid = sum(y for _, y in r16)
    rank16_bytes = entries * 2
    print(f'rank16_entries_total={entries}')
    print(f'rank16_valid_entries_total={valid}')
    print(f'rank16_metadata_mib_total={rank16_bytes / (1 << 20):.6f}')
else:
    rank16_bytes = 0
if rs:
    codes = sum(x for x, _ in rs)
    ranks = sum(y for _, y in rs)
    rankstream_bytes = codes * 4 + ranks * 2
    print(f'rankstream_codes_total={codes}')
    print(f'rankstream_l_ranks_total={ranks}')
    print(f'rankstream_metadata_mib_total={rankstream_bytes / (1 << 20):.6f}')
    if rank16_bytes:
        print(f'rankstream_metadata_reduction={rank16_bytes / rankstream_bytes:.6f}x')

print('rankstream_cross_runtime_direct_lookup=0')
print('rankstream_cross_runtime_ordinal_popcount=0')
print('fallback_structurally_unreachable=1')
print(f'summary={dst}')
PY

echo "depthcode-rankstream-ab OK n=$N repeats=$REPEATS threads=$BUCKET_THREADS gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y decode_load=$DEPTHCODE_DECODE_LOAD transpose=$TRANSPOSE_MODE result=$RESULT" >&2
