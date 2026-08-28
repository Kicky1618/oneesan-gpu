#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";GPUS="${GPUS:-8}";ITERS="${ITERS:-4096}";REPEATS="${REPEATS:-9}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_group_meta_symbol_copy_microprobe}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_group_meta_symbol_copy_ab}";RESULT="${RESULT:-${PREFIX}.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"
(( GPUS>=1 && GPUS<=8 && ITERS>=1 && REPEATS>=1 )) || exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 group metadata symbol-copy benchmark"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)";(( visible>=GPUS )) || { echo "requested $GPUS GPUs, visible=$visible" >&2; exit 2; }
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/group_meta_symbol_copy_microprobe.cu"
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" "$SRC" -o "$BIN" >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"
printf 'gpu\tmeta_bytes\tunpacked_calls\tpacked_calls\tunpacked_us_per_group\tpacked_us_per_group\tspeedup\n' >"$RESULT"
for ((d=0;d<GPUS;++d)); do
  out="$LOGDIR/gpu_${d}.out";"$BIN" "$ITERS" "$REPEATS" "$d" >"$out"
  line="$(grep '^b300-group-meta-symbol-copy-microprobe OK ' "$out" | tail -n1 || true)";[[ -n "$line" ]] || { cat "$out" >&2; exit 3; }
  grep -Fq 'meta_bytes=13936' <<<"$line";grep -Fq 'unpacked_calls=6 packed_calls=1' <<<"$line";grep -Fq 'call_reduction=6x pageable_host=1 exact=OK' <<<"$line"
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$line"; }
  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$(field meta_bytes)" "$(field unpacked_calls)" "$(field packed_calls)" "$(field unpacked_us_per_group)" "$(field packed_us_per_group)" "$(field speedup)" >>"$RESULT"
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));s=[float(x['speedup']) for x in r];u=[float(x['unpacked_us_per_group']) for x in r];p=[float(x['packed_us_per_group']) for x in r]
print(f'b300_group_meta_copy_gpus={len(r)}')
print(f'b300_group_meta_unpacked_us_median={statistics.median(u):.6f}')
print(f'b300_group_meta_packed_us_median={statistics.median(p):.6f}')
print(f'b300_group_meta_copy_speedup_median={statistics.median(s):.6f}x')
print(f'b300_group_meta_copy_speedup_min={min(s):.6f}x')
print(f'b300_group_meta_copy_speedup_max={max(s):.6f}x')
print('b300_group_meta_bytes=13936 symbol_copy_calls_old=6 symbol_copy_calls_new=1')
PY
