#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"; ARCH="${ARCH:-native}"; PTX_ARCH="${PTX_ARCH:-sm_80}"
BLOCKS="${BLOCKS:-256}"; THREADS="${THREADS:-256}"; ITERS="${ITERS:-8192}"; RUNS="${RUNS:-12}"
command -v "$NVCC" >/dev/null || exit 2
command -v nvidia-smi >/dev/null || exit 2
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-u32addr-w28-ngpu8-proof.sh"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-u32addr-w28-ngpu8-ptx-proof.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/shard_owner_hi32_u32addr_w28_ngpu8_microprobe.cu"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_shard_owner_hi32_u32addr_w28_ngpu8_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
BINS=()
for m in 0 1 2; do
  BINS[$m]="$ONEESAN_BUILD_DIR/b300_hi32_u32addr_mode${m}"
  "$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_HI32_U32ADDR_MODE="$m" "$SRC" -o "${BINS[$m]}"
done
printf 'mode\trun\tmedian_ms\tGaddr_s\tchecksum\n' >"$RESULT"
run_one(){
  local m="$1" r="$2" line out="$LOGDIR/m${m}_r${r}.out"
  "${BINS[$m]}" "$BLOCKS" "$THREADS" "$ITERS" 1 >"$out"
  line="$(grep '^gridfp-b300-shard-owner-hi32-u32addr-w28-ngpu8-microprobe OK ' "$out" | tail -n1)"
  grep -Fq "mode=$m" <<<"$line"; grep -Fq 'exact=OK' <<<"$line"
  printf '%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$(sed -nE 's/.* median_ms=([^ ]+).*/\1/p' <<<"$line")" "$(sed -nE 's/.* Gaddr_s=([^ ]+).*/\1/p' <<<"$line")" "$(sed -nE 's/.* checksum=([^ ]+).*/\1/p' <<<"$line")" >>"$RESULT"
}
for ((r=1;r<=RUNS;++r)); do case $(((r-1)%3)) in 0) order=(0 1 2);;1) order=(1 2 0);;*) order=(2 0 1);;esac; for m in "${order[@]}"; do run_one "$m" "$r"; done; done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
R=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); q={}
for m in '012':
 x=[r for r in R if r['mode']==m]; c={r['checksum'] for r in x}; assert len(c)==1
 q[m]=statistics.median(float(r['median_ms']) for r in x)
assert len({r['checksum'] for r in R})==1
print(f'b300_shard_owner_hi32_u64corr_speedup={q["0"]/q["1"]:.6f}x')
print(f'b300_shard_owner_hi32_full_u32_speedup={q["0"]/q["2"]:.6f}x')
print(f'b300_shard_owner_hi32_u32addr_best_mode={min(q,key=q.get)}')
print('mode0=three_compare_subtract mode1=hi32_u64corr mode2=hi32_fully_u32')
PY
