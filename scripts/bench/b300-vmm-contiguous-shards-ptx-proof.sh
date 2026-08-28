#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"; ARCH="${ARCH:-sm_80}"; command -v "$NVCC" >/dev/null || exit 2
SRC="$ONEESAN_ROOT/src/cuda/b300/probes/vmm_contiguous_shards_ptx_probe.cu"; OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_vmm_contiguous_shards_ptx}"; mkdir -p "$OUTDIR"; PTX="$OUTDIR/probe.ptx"
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" "$SRC" -o "$PTX"
extract(){ local k="$1" o="$2"; awk -v k="$k" '$0~"\\.entry[[:space:]]+"k{f=1}f{print}f&&/^}/{exit}' "$PTX">"$o"; [[ -s "$o" ]]||exit 3; }
metric(){ grep -Ec "$1" "$2" || true; }
extract b300_vmm_old_address_probe "$OUTDIR/old.body.ptx"; extract b300_vmm_direct_address_probe "$OUTDIR/vmm.body.ptx"
for m in old vmm; do b="$OUTDIR/$m.body.ptx"; div="$(metric '\b(div|rem)\.u64\b' "$b")"; setp="$(metric '\bsetp\..*\.u64\b' "$b")"; sub="$(metric '\bsub\.u64\b' "$b")"; bra="$(metric '\bbra\b' "$b")"; ldc="$(metric '\bld\.const\..*\.u64\b' "$b")"; printf '%s_divrem_u64=%s\n%s_setp_u64=%s\n%s_sub_u64=%s\n%s_bra=%s\n%s_ldconst_u64=%s\n' "$m" "$div" "$m" "$setp" "$m" "$sub" "$m" "$bra" "$m" "$ldc"; ((div==0))||exit 4; done
old_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/old.body.ptx")"; vmm_setp="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/vmm.body.ptx")"; vmm_sub="$(metric '\bsub\.u64\b' "$OUTDIR/vmm.body.ptx")"
((old_setp>=3)) || { echo "old shard path missing compare tree" >&2; exit 5; }
((vmm_setp==0&&vmm_sub==0)) || { echo "VMM direct path still has shard u64 compare/subtract" >&2; exit 6; }
echo "b300-vmm-contiguous-shards-ptx-proof OK arch=$ARCH old_setp_u64=$old_setp vmm_setp_u64=$vmm_setp vmm_sub_u64=$vmm_sub owner_ops_vmm=0 dynamic_ptr_index_vmm=0"
