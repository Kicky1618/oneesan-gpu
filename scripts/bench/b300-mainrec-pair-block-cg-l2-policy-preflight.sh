#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-load-policy.py"
OGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-cg-l2-policy.py"
python3 -m py_compile "$NGEN" "$OGEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageo-l2.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
src="$tmp/in.cu"
{
  cat <<'EOF'
using Count=unsigned int; using MateID=unsigned long long; using Code=unsigned long long;
__global__ void main_pull_kernel_ilp2(const Count* in,MateID* mates,const Count* in_block){
  bool hp0=true,hb0=true,hp1=true,hb1=true; Code pj0=0,bj0=0,pj1=1,bj1=1,i0=0,i1=1;
  const MateID m0=mates[i0]; const MateID m1=mates[i1];
  const Count pair0=hp0?in[pj0]:Count(0); const Count pair1=hp1?in[pj1]:Count(0);
  const Count block0=hb0?in_block[bj0]:Count(0); const Count block1=hb1?in_block[bj1]:Count(0);
  const Count self0=in[i0]; const Count self1=in[i1]; (void)m0;(void)m1;(void)pair0;(void)pair1;(void)block0;(void)block1;(void)self0;(void)self1;
}
__global__ void main_pull_kernel_ilp8_hybrid(const Count* in,MateID* mates,const Count* in_block){
  bool hp0=true,hb0=true,hp1=true,hb1=true,hp2=true,hb2=true,hp3=true,hb3=true,hp4=true,hb4=true,hp5=true,hb5=true,hp6=true,hb6=true,hp7=true,hb7=true;
  Code pj0=0,bj0=0,pj1=1,bj1=1,pj2=2,bj2=2,pj3=3,bj3=3,pj4=4,bj4=4,pj5=5,bj5=5,pj6=6,bj6=6,pj7=7,bj7=7,i0=0,i1=1,i2=2,i3=3,i4=4,i5=5,i6=6,i7=7;
  const MateID m0=mates[i0]; const MateID m1=mates[i1]; const MateID m2=mates[i2]; const MateID m3=mates[i3]; const MateID m4=mates[i4]; const MateID m5=mates[i5]; const MateID m6=mates[i6]; const MateID m7=mates[i7];
EOF
  for k in {0..7}; do echo "  const Count pair$k=hp$k?in[pj$k]:Count(0);"; done
  for k in {0..7}; do echo "  const Count block$k=hb$k?in_block[bj$k]:Count(0);"; done
  for k in {0..7}; do echo "  const Count self$k=in[i$k];"; done
  cat <<'EOF'
  (void)m0;(void)m7;(void)pair0;(void)pair7;(void)block0;(void)block7;(void)self0;(void)self7;
}
EOF
} >"$src"

make_n(){ local pair="$1" block="$2" base="$3" out="$4"; python3 "$NGEN" "$src" "$out" "$pair" "$block" "$base" >/dev/null; }
make_n cg default 128 "$tmp/n-pcg.cu"
for bytes in 0 64 128 256; do
  out="$tmp/o-pcg-$bytes.cu"; python3 "$OGEN" "$tmp/n-pcg.cu" "$out" "$bytes" 0 >"$tmp/o-pcg-$bytes.log"
  grep -Fq "pair_l2_bytes=$bytes block_l2_bytes=0" "$tmp/o-pcg-$bytes.log" || exit 3
  if [[ "$bytes" == 0 ]]; then grep -Fq 'ld.global.cg.u32' "$out" || exit 3; else grep -Fq "ld.global.cg.L2::${bytes}B.u32" "$out" || exit 3; fi
  [[ "$(grep -o 'b300_mainrec_stageo_pair_load_cg(in+pj' "$out" | wc -l)" == 10 ]] || { echo 'Stage-O pair call count mismatch' >&2; exit 3; }
  grep -Fq 'const Count block7=hb7?in_block[bj7]:Count(0);' "$out" || { echo 'Stage-O changed non-CG block axis' >&2; exit 3; }
  grep -Fq 'const Count self7=in[i7];' "$out" || exit 3; grep -Fq 'const MateID m7=mates[i7];' "$out" || exit 3
done

make_n default cg 64 "$tmp/n-bcg.cu"
for bytes in 0 64 128 256; do
  out="$tmp/o-bcg-$bytes.cu"; python3 "$OGEN" "$tmp/n-bcg.cu" "$out" 0 "$bytes" >/dev/null
  [[ "$(grep -o 'b300_mainrec_stageo_block_load_cg(in_block+bj' "$out" | wc -l)" == 10 ]] || { echo 'Stage-O block call count mismatch' >&2; exit 3; }
  grep -Fq 'const Count pair7=hp7?in[pj7]:Count(0);' "$out" || { echo 'Stage-O changed non-CG pair axis' >&2; exit 3; }
done

make_n cg cg 128 "$tmp/n-both.cu"
python3 "$OGEN" "$tmp/n-both.cu" "$tmp/o-both.cu" 64 256 >"$tmp/o-both.log"
grep -Fq 'pair_l2_bytes=64 block_l2_bytes=256 base_l2_bytes=128' "$tmp/o-both.log" || exit 3
grep -Fq 'ld.global.cg.L2::64B.u32' "$tmp/o-both.cu" || exit 3
grep -Fq 'ld.global.cg.L2::256B.u32' "$tmp/o-both.cu" || exit 3

make_n default cs 0 "$tmp/n-none.cu"
set +e
python3 "$OGEN" "$tmp/n-none.cu" "$tmp/o-none.cu" 0 0 >/dev/null 2>"$tmp/none.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'neither Stage-N axis uses cg' "$tmp/none.err" || { echo 'Stage O accepted non-CG Stage N' >&2; exit 3; }

set +e
python3 "$OGEN" "$tmp/n-pcg.cu" "$tmp/o-bad.cu" 64 64 >/dev/null 2>"$tmp/bad.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'BLOCK_L2_BYTES must be 0 when block policy is not cg' "$tmp/bad.err" || { echo 'Stage O accepted L2 bytes on non-CG block axis' >&2; exit 3; }

set +e
python3 "$OGEN" "$tmp/o-both.cu" "$tmp/o-double.cu" 64 256 >/dev/null 2>"$tmp/double.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'already contains Stage-O' "$tmp/double.err" || { echo 'Stage O double transform not rejected' >&2; exit 3; }

echo 'b300-mainrec-pair-block-cg-l2-policy-preflight OK stage=O pair_cg_sizes=0,64,128,256 block_cg_sizes=0,64,128,256 independent_axes=1 ilp2_calls=2 ilp8_calls=8 self_unchanged=1 mate_unchanged=1 noncg_axis_rejected=1 double_transform_rejected=1 gpu_work=0'
