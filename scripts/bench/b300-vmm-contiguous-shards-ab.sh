#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
PTX_ARCH="${PTX_ARCH:-sm_103}"
GPUS="${GPUS:-8}"
CHUNK_ELEMS="${CHUNK_ELEMS:-1048576}"
BLOCKS="${BLOCKS:-256}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-4096}"
REPEATS="${REPEATS:-9}"
ALL_SRC_GPUS="${ALL_SRC_GPUS:-1}"
SRC_GPU="${SRC_GPU:-0}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/b300_vmm_contiguous_shards_microprobe}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_vmm_contiguous_shards_ab}"
RESULT="${RESULT:-${PREFIX}.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"

command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }
if (( GPUS < 1 || GPUS > 8 || CHUNK_ELEMS < 1024 || BLOCKS < 1 || THREADS < 1 || THREADS > 1024 || ITERS < 1 || REPEATS < 1 )); then
  echo "invalid benchmark parameters" >&2
  exit 2
fi
visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < GPUS )); then echo "requested $GPUS GPUs, visible=$visible" >&2; exit 2; fi

SRC="$ONEESAN_ROOT/src/cuda/b300/probes/vmm_contiguous_shards_microprobe.cu"
mkdir -p "$(dirname "$RESULT")" "$LOGDIR"
ARCH="$PTX_ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-base-source-ptx-proof.sh" \
  >"$LOGDIR/base_source_ptx.out" 2>"$LOGDIR/base_source_ptx.err"
"$NVCC" -O3 -std=c++17 -lineinfo -arch="$ARCH" "$SRC" -lcuda -o "$BIN" \
  >"$LOGDIR/build.out" 2>"$LOGDIR/build.err"

nvidia-smi -L >&2
nvidia-smi topo -m >&2 || true
printf 'src_gpu\tgpus\tchunk_elems\tchunk_bytes\tgranularity\ttotal_mib\told_ms\tsymbol_ms\targ_ms\told_vs_symbol_speedup\targ_vs_symbol_speedup\told_Gload_s\tsymbol_Gload_s\targ_Gload_s\n' >"$RESULT"
if [[ "$ALL_SRC_GPUS" == 1 ]]; then mapfile -t SOURCES < <(seq 0 $((GPUS-1))); else SOURCES=("$SRC_GPU"); fi
for src in "${SOURCES[@]}"; do
  out="$LOGDIR/src_gpu_${src}.out"
  "$BIN" "$GPUS" "$CHUNK_ELEMS" "$BLOCKS" "$THREADS" "$ITERS" "$REPEATS" "$src" >"$out"
  line="$(grep '^gridfp-b300-vmm-contiguous-shards-microprobe OK ' "$out" | tail -n1 || true)"
  [[ -n "$line" ]] || { cat "$out" >&2; exit 3; }
  grep -Fq 'exact=OK' <<<"$line" || exit 4
  grep -Fq 'owner_ops_vmm=0' <<<"$line" || exit 5
  grep -Fq 'dynamic_ptr_index_vmm=0' <<<"$line" || exit 6
  grep -Fq 'owner_ops_arg=0 dynamic_ptr_index_arg=0' <<<"$line" || exit 7
  grep -Fq 'base_source_symbol=constant base_source_arg=kernel_param' <<<"$line" || exit 8
  field(){ sed -nE "s/.* $1=([^[:space:]]+).*/\\1/p" <<<"$line"; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$src" "$(field gpus)" "$(field chunk_elems)" "$(field chunk_bytes)" "$(field granularity)" "$(field total_mib)" \
    "$(field old_ms)" "$(field symbol_ms)" "$(field arg_ms)" "$(field speedup)" "$(field arg_speedup_vs_symbol)" \
    "$(field old_Gload_s)" "$(field symbol_Gload_s)" "$(field arg_Gload_s)" >>"$RESULT"
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
old=[float(x['old_vs_symbol_speedup']) for x in r]
arg=[float(x['arg_vs_symbol_speedup']) for x in r]
print(f'b300_vmm_sources={len(r)}')
print(f'b300_vmm_symbol_speedup_vs_old_median={statistics.median(old):.6f}x')
print(f'b300_vmm_symbol_speedup_vs_old_min={min(old):.6f}x')
print(f'b300_vmm_symbol_speedup_vs_old_max={max(old):.6f}x')
print(f'b300_vmm_arg_speedup_vs_symbol_median={statistics.median(arg):.6f}x')
print(f'b300_vmm_arg_speedup_vs_symbol_min={min(arg):.6f}x')
print(f'b300_vmm_arg_speedup_vs_symbol_max={max(arg):.6f}x')
print('b300_vmm_symbol_owner_ops=0 b300_vmm_arg_owner_ops=0 b300_vmm_arg_dynamic_ptr_index=0 base_source_ptx_gated=1')
PY
