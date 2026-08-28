#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-native}";GPUS="${GPUS:-8}";REPEATS="${REPEATS:-9}";MAIN_ELEMS="${MAIN_ELEMS:-16777216}";BLOCK_ELEMS="${BLOCK_ELEMS:-5872025}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_vmm_concurrent_io_microprobe}";PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_concurrent_io_ab}";RESULT="${RESULT:-${PREFIX}.tsv}";LOGDIR="${LOGDIR:-${PREFIX}_logs}"
(( GPUS>=1&&GPUS<=8&&REPEATS>=1&&MAIN_ELEMS>=GPUS*1024&&BLOCK_ELEMS>=GPUS*1024 ))||exit 2
require_nvcc_version_at_least "$NVCC" 13 0 "B300 concurrent VMM I/O benchmark";command -v nvidia-smi >/dev/null||exit 2
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)";((visible>=GPUS))||{ echo "requested $GPUS GPUs, visible=$visible" >&2;exit 2;};mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/vmm_concurrent_io_microprobe.cu"
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" "$SRC" -lcuda -o "$BIN" >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"
printf 'src_gpu\tmain_elems\tblock_elems\tblock_main_ratio\tserial_ms\tconcurrent_ms\tspeedup\n' >"$RESULT"
for((d=0;d<GPUS;++d));do
 out="$LOGDIR/gpu_${d}.out";"$BIN" "$GPUS" "$d" "$REPEATS" "$MAIN_ELEMS" "$BLOCK_ELEMS" >"$out"
 line="$(grep '^b300-vmm-concurrent-io-microprobe OK ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ cat "$out" >&2;exit 3; }
 grep -Fq 'serial_dependency=1 concurrent_overlap=1 exact=OK' <<<"$line"
 field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p"<<<"$line"; }
 printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$(field main_elems)" "$(field block_elems)" "$(field block_main_ratio)" "$(field serial_ms)" "$(field concurrent_ms)" "$(field speedup)" >>"$RESULT"
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));s=[float(x['speedup']) for x in r]
print(f'b300_concurrent_io_sources={len(r)}')
print(f'b300_concurrent_io_speedup_median={statistics.median(s):.6f}x')
print(f'b300_concurrent_io_speedup_min={min(s):.6f}x')
print(f'b300_concurrent_io_speedup_max={max(s):.6f}x')
print(f'b300_concurrent_io_block_main_ratio={r[0]["block_main_ratio"]}')
print('b300_concurrent_io_exact=1')
PY
