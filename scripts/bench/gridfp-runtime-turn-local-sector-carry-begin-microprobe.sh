#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-128}"
REPEATS="${REPEATS:-7}"
WARMUP="${WARMUP:-2}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
if (( BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 ||
      REPEATS < 1 || WARMUP < 0 )); then
  echo "invalid turn local-sector carry microprobe dimensions" >&2; exit 2
fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then
  echo "nvcc and nvidia-smi are required" >&2; exit 2
fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_runtime_turn_local_sector_carry_begin_microprobe_b${BLOCKS}_t${THREADS}_i${ITERS}}"
RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-turn-local-sector-carry-begin-proof.sh" \
  >"$LOGDIR/proof.out" 2>"$LOGDIR/proof.err"
SRC="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_turn_local_sector_carry_begin_microprobe.cu"
BIN0="$ONEESAN_BUILD_DIR/gridfp_runtime_turn_local_sector_carry_begin_microprobe0"
BIN1="$ONEESAN_BUILD_DIR/gridfp_runtime_turn_local_sector_carry_begin_microprobe1"
PTXAS_FLAGS=(); [[ "$PTXAS_VERBOSE" == 1 ]] && PTXAS_FLAGS+=("-Xptxas=-v")

build_one() {
  local carry="$1" bin="$2"
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    "${PTXAS_FLAGS[@]}" -DPROBE_TURN_LOCAL_CARRY_BEGIN="$carry" "$SRC" -o "$bin" \
    >"$LOGDIR/carry${carry}.build.out" 2>"$LOGDIR/carry${carry}.build.err"
}
build_one 0 "$BIN0"; build_one 1 "$BIN1"

printf 'carry\trepeat\tkernel_ms\tns_per_call\tchecksum\n' >"$RESULT"
run_one() {
  local carry="$1" bin="$2" rep="$3"
  local out="$LOGDIR/carry${carry}_run${rep}.out" err="$LOGDIR/carry${carry}_run${rep}.err"
  "$bin" "$BLOCKS" "$THREADS" "$ITERS" "$WARMUP" >"$out" 2>"$err"
  local line
  line="$(grep '^gridfp-runtime-turn-local-sector-carry-begin-microprobe ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2 || true; cat "$err" >&2 || true; exit 3; }
  grep -Fq " carry=$carry " <<<"$line" || { echo "unexpected carry mode: $line" >&2; exit 4; }
  local ms ns checksum
  ms="$(sed -nE 's/.* kernel_ms=([^[:space:]]+).*/\1/p' <<<"$line")"
  ns="$(sed -nE 's/.* ns_per_call=([^[:space:]]+).*/\1/p' <<<"$line")"
  checksum="$(sed -nE 's/.* checksum=([^[:space:]]+).*/\1/p' <<<"$line")"
  [[ -n "$ms" && -n "$ns" && -n "$checksum" ]] || exit 5
  printf '%s\t%s\t%s\t%s\t%s\n' "$carry" "$rep" "$ms" "$ns" "$checksum" >>"$RESULT"
}

for ((r=1;r<=REPEATS;++r)); do
  if ((r&1)); then order=(0 1); else order=(1 0); fi
  for carry in "${order[@]}"; do
    [[ "$carry" == 0 ]] && bin="$BIN0" || bin="$BIN1"
    echo "=== turn local-sector carry=$carry run $r/$REPEATS ===" >&2
    run_one "$carry" "$bin" "$r"
  done
done
cat "$RESULT"

python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,dst=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]; cs={}
for mode in ('0','1'):
 rs=[r for r in rows if r['carry']==mode]
 if not rs: raise SystemExit(f'missing carry={mode}')
 ss={r['checksum'] for r in rs}
 if len(ss)!=1: raise SystemExit(f'nondeterministic checksum carry={mode}: {sorted(ss)}')
 cs[mode]=next(iter(ss)); ms=[float(r['kernel_ms']) for r in rs]; ns=[float(r['ns_per_call']) for r in rs]
 out.append({'carry':mode,'repeats':len(rs),'kernel_ms_median':f'{statistics.median(ms):.9f}','ns_per_call_median':f'{statistics.median(ns):.9f}','checksum':cs[mode]})
if cs['0']!=cs['1']: raise SystemExit(f'checksum mismatch old={cs["0"]} carry={cs["1"]}')
with open(dst,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0].keys(),delimiter='\t'); w.writeheader(); w.writerows(out)
q={r['carry']:r for r in out}; old=float(q['0']['ns_per_call_median']); new=float(q['1']['ns_per_call_median'])
print(f'turn_local_sector_carry_begin_microprobe_speedup={old/new:.6f}x')
print(f'turn_local_sector_carry_begin_microprobe_delta_pct={(new/old-1)*100:.4f}%')
print('turn_local_sector_carry_begin_max_table_loads_old=5')
print('turn_local_sector_carry_begin_max_table_loads_new=4')
print('turn_local_sector_carry_begin_reload_eliminated=1')
print('turn_local_sector_carry_begin_production_integrated=0')
print(f'checksum={cs["0"]}')
print(f'summary={dst}')
PY

if [[ "$PTXAS_VERBOSE" == 1 ]]; then
  echo '--- ptxas turn generic begin reload ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/carry0.build.err" >&2 || true
  echo '--- ptxas turn begin carry ---' >&2
  grep -E 'Used .* registers|bytes smem|bytes cmem' "$LOGDIR/carry1.build.err" >&2 || true
fi

echo "gridfp-runtime-turn-local-sector-carry-begin-microprobe OK repeats=$REPEATS result=$RESULT" >&2
