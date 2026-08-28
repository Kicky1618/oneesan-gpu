#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_103}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/vmm_base_source_ptx_probe.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_vmm_base_source_ptx}"
PTX="$OUTDIR/probe.ptx"
mkdir -p "$OUTDIR"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" "$SRC" -o "$PTX"
extract(){ local k="$1" o="$2"; awk -v k="$k" '$0 ~ "\\.entry[[:space:]]+" k {inside=1} inside{print} inside&&/^}/{exit}' "$PTX" >"$o"; [[ -s "$o" ]] || { echo "missing PTX body $k" >&2; exit 3; }; }
metric(){ grep -Ec "$1" "$2" || true; }
SYM="$OUTDIR/symbol.body.ptx"; ARG="$OUTDIR/arg.body.ptx"
extract b300_vmm_symbol_base_probe "$SYM"
extract b300_vmm_arg_base_probe "$ARG"
sym_const="$(metric '\bld\.const\.u64\b' "$SYM")"
arg_const="$(metric '\bld\.const\.u64\b' "$ARG")"
arg_param="$(metric '\bld\.param\.u64\b' "$ARG")"
sym_div="$(metric '\b(div|rem)\.u64\b' "$SYM")"; arg_div="$(metric '\b(div|rem)\.u64\b' "$ARG")"
sym_mul="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$SYM")"; arg_mul="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$ARG")"
(( sym_const >= 1 )) || { echo "symbol probe missing ld.const.u64" >&2; cat "$SYM" >&2; exit 4; }
(( arg_const == 0 )) || { echo "arg probe unexpectedly uses ld.const.u64=$arg_const" >&2; cat "$ARG" >&2; exit 5; }
(( arg_param >= 1 )) || { echo "arg probe missing ld.param.u64" >&2; cat "$ARG" >&2; exit 6; }
(( sym_div == 0 && arg_div == 0 && sym_mul == 0 && arg_mul == 0 )) || { echo "unexpected owner-style u64 arithmetic" >&2; exit 7; }
printf 'b300_vmm_symbol_ldconst_u64=%s\n' "$sym_const"
printf 'b300_vmm_arg_ldconst_u64=%s\n' "$arg_const"
printf 'b300_vmm_arg_ldparam_u64=%s\n' "$arg_param"
echo "b300-vmm-base-source-ptx-proof OK arch=$ARCH symbol_base=constant arg_base=kernel_param arg_const_load=0 owner_div64=0 owner_mul64=0"
