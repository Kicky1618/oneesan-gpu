#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-load-policy.py"
python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagen-load.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
src="$tmp/in.cu"
{
  cat <<'EOF'
using Count=unsigned int; using MateID=unsigned long long; using Code=unsigned long long;
__device__ Count b300_mainrec_random_load_cg(const Count* p){return *p;}
__global__ void main_pull_kernel_ilp2(const Count* in,MateID* mates,const Count* in_block){
  bool hp0=true,hb0=true,hp1=true,hb1=true; Code pj0=0,bj0=0,pj1=1,bj1=1,i0=0,i1=1;
  const MateID m0=mates[i0]; const MateID m1=mates[i1];
  const Count pair0=hp0?b300_mainrec_random_load_cg(in+pj0):Count(0);
  const Count pair1=hp1?b300_mainrec_random_load_cg(in+pj1):Count(0);
  const Count block0=hb0?b300_mainrec_random_load_cg(in_block+bj0):Count(0);
  const Count block1=hb1?b300_mainrec_random_load_cg(in_block+bj1):Count(0);
  const Count self0=in[i0]; const Count self1=in[i1]; (void)m0;(void)m1;(void)pair0;(void)pair1;(void)block0;(void)block1;(void)self0;(void)self1;
}
__global__ void main_pull_kernel_ilp8_hybrid(const Count* in,MateID* mates,const Count* in_block){
  bool hp0=true,hb0=true,hp1=true,hb1=true,hp2=true,hb2=true,hp3=true,hb3=true,hp4=true,hb4=true,hp5=true,hb5=true,hp6=true,hb6=true,hp7=true,hb7=true;
  Code pj0=0,bj0=0,pj1=1,bj1=1,pj2=2,bj2=2,pj3=3,bj3=3,pj4=4,bj4=4,pj5=5,bj5=5,pj6=6,bj6=6,pj7=7,bj7=7,i0=0,i1=1,i2=2,i3=3,i4=4,i5=5,i6=6,i7=7;
  const MateID m0=mates[i0]; const MateID m1=mates[i1]; const MateID m2=mates[i2]; const MateID m3=mates[i3]; const MateID m4=mates[i4]; const MateID m5=mates[i5]; const MateID m6=mates[i6]; const MateID m7=mates[i7];
EOF
  for k in {0..7}; do echo "  const Count pair$k=hp$k?b300_mainrec_random_load_cg(in+pj$k):Count(0);"; done
  for k in {0..7}; do echo "  const Count block$k=hb$k?b300_mainrec_random_load_cg(in_block+bj$k):Count(0);"; done
  for k in {0..7}; do echo "  const Count self$k=in[i$k];"; done
  cat <<'EOF'
  (void)m0;(void)m7;(void)pair0;(void)pair7;(void)block0;(void)block7;(void)self0;(void)self7;
}
EOF
} >"$src"

for pair in default cg cs; do
  for block in default cg cs; do
    out="$tmp/${pair}-${block}.cu"; log="$tmp/${pair}-${block}.log"
    python3 "$GEN" "$src" "$out" "$pair" "$block" 128 >"$log"
    grep -Fq "pair_policy=$pair block_policy=$block cg_l2_bytes=128" "$log" || exit 3
    grep -Fq "b300_mainrec_stagen_pair_block_policy=1 pair=$pair block=$block cg_l2_bytes=128" "$out" || exit 3
    grep -Fq 'const Count self0=in[i0];' "$out" || { echo 'Stage-N changed self load' >&2; exit 3; }
    grep -Fq 'const MateID m7=mates[i7];' "$out" || { echo 'Stage-N changed mate load' >&2; exit 3; }
    case "$pair" in
      default) grep -Fq 'const Count pair7=hp7?in[pj7]:Count(0);' "$out";;
      cg) grep -Fq 'const Count pair7=hp7?b300_mainrec_stagen_load_cg(in+pj7):Count(0);' "$out";;
      cs) grep -Fq 'const Count pair7=hp7?b300_mainrec_stagen_load_cs(in+pj7):Count(0);' "$out";;
    esac
    case "$block" in
      default) grep -Fq 'const Count block7=hb7?in_block[bj7]:Count(0);' "$out";;
      cg) grep -Fq 'const Count block7=hb7?b300_mainrec_stagen_load_cg(in_block+bj7):Count(0);' "$out";;
      cs) grep -Fq 'const Count block7=hb7?b300_mainrec_stagen_load_cs(in_block+bj7):Count(0);' "$out";;
    esac
    if [[ "$pair" == cg || "$block" == cg ]]; then grep -Fq 'ld.global.cg.L2::128B.u32' "$out" || exit 3; fi
    if [[ "$pair" == cs || "$block" == cs ]]; then grep -Fq '__ldcs(p)' "$out" || exit 3; fi
  done
done

set +e
python3 "$GEN" "$tmp/cg-cs.cu" "$tmp/double.cu" cg cs 128 >"$tmp/double.out" 2>"$tmp/double.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'already contains Stage-N' "$tmp/double.err" || { echo 'Stage-N double transform not rejected' >&2; exit 3; }
set +e
python3 "$GEN" "$src" "$tmp/bad.cu" bad cg 0 >/dev/null 2>"$tmp/bad.err"; rc=$?
set -e
((rc!=0)) || { echo 'Stage-N invalid policy accepted' >&2; exit 3; }

echo 'b300-mainrec-pair-block-load-policy-preflight OK stage=N pair_policies=default,cg,cs block_policies=default,cg,cs combinations=9 ilp2=1 ilp8=1 self_unchanged=1 mate_unchanged=1 cg_l2=1 double_transform_rejected=1 gpu_work=0'
