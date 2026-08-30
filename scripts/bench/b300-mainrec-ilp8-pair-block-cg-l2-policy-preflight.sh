#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py"
python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageq-l2.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT

make_src(){
  local out="$1" with_o="$2" with_p="$3"
  {
    cat <<'EOF'
using Count=unsigned int; using MateID=unsigned long long; using Code=unsigned long long;
__device__ __forceinline__ Count b300_mainrec_stagen_load_cg(const Count* p){return *p;}
__device__ __forceinline__ Count b300_mainrec_stagen_load_cs(const Count* p){return *p;}
EOF
    if [[ "$with_o" == 1 ]]; then
      cat <<'EOF'
__device__ __forceinline__ Count b300_mainrec_stageo_pair_load_cg(const Count* p){return *p;}
__device__ __forceinline__ Count b300_mainrec_stageo_block_load_cg(const Count* p){return *p;}
EOF
    fi
    if [[ "$with_p" == 1 ]]; then
      cat <<'EOF'
__device__ __forceinline__ MateID b300_mainrec_hybrid8_mate_load_policy_cg(const MateID* p){return *p;}
EOF
    fi
    cat <<EOF
__global__ void main_pull_kernel_ilp2(const Count* in,MateID* mates,const Count* in_block){
  bool hp0=true,hb0=true,hp1=true,hb1=true; Code pj0=0,bj0=0,pj1=1,bj1=1,i0=0,i1=1;
  const MateID m0=mates[i0]; const MateID m1=mates[i1];
EOF
    if [[ "$with_o" == 1 ]]; then
      echo '  const Count pair0=hp0?b300_mainrec_stageo_pair_load_cg(in+pj0):Count(0); const Count pair1=hp1?b300_mainrec_stageo_pair_load_cg(in+pj1):Count(0);'
      echo '  const Count block0=hb0?b300_mainrec_stageo_block_load_cg(in_block+bj0):Count(0); const Count block1=hb1?b300_mainrec_stageo_block_load_cg(in_block+bj1):Count(0);'
    else
      echo '  const Count pair0=hp0?b300_mainrec_stagen_load_cg(in+pj0):Count(0); const Count pair1=hp1?b300_mainrec_stagen_load_cg(in+pj1):Count(0);'
      echo '  const Count block0=hb0?b300_mainrec_stagen_load_cg(in_block+bj0):Count(0); const Count block1=hb1?b300_mainrec_stagen_load_cg(in_block+bj1):Count(0);'
    fi
    cat <<'EOF'
  const Count self0=in[i0]; const Count self1=in[i1]; (void)m0;(void)m1;(void)pair0;(void)pair1;(void)block0;(void)block1;(void)self0;(void)self1;
}
__global__ void main_pull_kernel_ilp8_hybrid(const Count* in,MateID* mates,const Count* in_block){
  bool hp0=true,hb0=true,hp1=true,hb1=true,hp2=true,hb2=true,hp3=true,hb3=true,hp4=true,hb4=true,hp5=true,hb5=true,hp6=true,hb6=true,hp7=true,hb7=true;
  Code pj0=0,bj0=0,pj1=1,bj1=1,pj2=2,bj2=2,pj3=3,bj3=3,pj4=4,bj4=4,pj5=5,bj5=5,pj6=6,bj6=6,pj7=7,bj7=7,i0=0,i1=1,i2=2,i3=3,i4=4,i5=5,i6=6,i7=7;
EOF
    if [[ "$with_p" == 1 ]]; then
      for k in {0..7}; do echo "  const MateID m$k=b300_mainrec_hybrid8_mate_load_policy_cg(mates+i$k);"; done
    else
      for k in {0..7}; do echo "  const MateID m$k=mates[i$k];"; done
    fi
    if [[ "$with_o" == 1 ]]; then
      for k in {0..7}; do echo "  const Count pair$k=hp$k?b300_mainrec_stageo_pair_load_cg(in+pj$k):Count(0);"; done
      for k in {0..7}; do echo "  const Count block$k=hb$k?b300_mainrec_stageo_block_load_cg(in_block+bj$k):Count(0);"; done
    else
      for k in {0..7}; do echo "  const Count pair$k=hp$k?b300_mainrec_stagen_load_cg(in+pj$k):Count(0);"; done
      for k in {0..7}; do echo "  const Count block$k=hb$k?b300_mainrec_stagen_load_cg(in_block+bj$k):Count(0);"; done
    fi
    for k in {0..7}; do echo "  const Count self$k=in[i$k];"; done
    cat <<'EOF'
  mates[i7]=b300_high_state_advance; (void)m0;(void)m7;(void)pair0;(void)pair7;(void)block0;(void)block7;(void)self0;(void)self7;
}
// b300_mainrec_stagen_pair_block_policy=1 pair=cg block=cg cg_l2_bytes=128
EOF
    [[ "$with_o" == 1 ]] && echo '// b300_mainrec_stageo_pair_block_cg_l2=1 pair_policy=cg block_policy=cg pair_l2_bytes=64 block_l2_bytes=256 base_l2_bytes=128'
    [[ "$with_p" == 1 ]] && echo '// b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=64 scope=ilp8_mate_reads_only'
  } >"$out"
}

make_src "$tmp/op.cu" 1 1
python3 "$GEN" "$tmp/op.cu" "$tmp/q-op.cu" 0 64 >"$tmp/q-op.log"
grep -Fq 'pair_l2_bytes=0 block_l2_bytes=64 upstream_pair_l2_bytes=64 upstream_block_l2_bytes=256' "$tmp/q-op.log" || exit 3
grep -Fq 'stagep_preserved=1' "$tmp/q-op.log" || exit 3
grep -Fq 'ld.global.cg.u32' "$tmp/q-op.cu" || exit 3
grep -Fq 'ld.global.cg.L2::64B.u32' "$tmp/q-op.cu" || exit 3
[[ "$(grep -o 'b300_mainrec_stageq_ilp8_pair_load_cg(in+pj' "$tmp/q-op.cu" | wc -l)" == 8 ]] || exit 3
[[ "$(grep -o 'b300_mainrec_stageq_ilp8_block_load_cg(in_block+bj' "$tmp/q-op.cu" | wc -l)" == 8 ]] || exit 3
[[ "$(grep -o 'b300_mainrec_stageo_pair_load_cg(in+pj' "$tmp/q-op.cu" | wc -l)" == 2 ]] || { echo 'Stage Q changed ILP2 pair stream' >&2; exit 3; }
[[ "$(grep -o 'b300_mainrec_stageo_block_load_cg(in_block+bj' "$tmp/q-op.cu" | wc -l)" == 2 ]] || { echo 'Stage Q changed ILP2 block stream' >&2; exit 3; }
[[ "$(grep -o 'b300_mainrec_hybrid8_mate_load_policy_cg(mates+i' "$tmp/q-op.cu" | wc -l)" == 8 ]] || { echo 'Stage Q changed Stage-P mate stream' >&2; exit 3; }
grep -Fq 'b300_mainrec_stagep_mate_cg_l2=1 l2_bytes=64' "$tmp/q-op.cu" || exit 3

make_src "$tmp/n.cu" 0 0
python3 "$GEN" "$tmp/n.cu" "$tmp/q-n.cu" 256 0 >"$tmp/q-n.log"
grep -Fq 'upstream_pair_l2_bytes=128 upstream_block_l2_bytes=128' "$tmp/q-n.log" || exit 3
[[ "$(grep -o 'b300_mainrec_stageq_ilp8_pair_load_cg(in+pj' "$tmp/q-n.cu" | wc -l)" == 8 ]] || exit 3
[[ "$(grep -o 'b300_mainrec_stageq_ilp8_block_load_cg(in_block+bj' "$tmp/q-n.cu" | wc -l)" == 8 ]] || exit 3
[[ "$(grep -o 'b300_mainrec_stagen_load_cg(in+pj' "$tmp/q-n.cu" | wc -l)" == 2 ]] || exit 3
[[ "$(grep -o 'b300_mainrec_stagen_load_cg(in_block+bj' "$tmp/q-n.cu" | wc -l)" == 2 ]] || exit 3
grep -Fq 'ld.global.cg.L2::256B.u32' "$tmp/q-n.cu" || exit 3

set +e
python3 "$GEN" "$tmp/q-op.cu" "$tmp/double.cu" 0 64 >"$tmp/double.out" 2>"$tmp/double.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'already contains Stage-Q' "$tmp/double.err" || { echo 'Stage Q double transform accepted' >&2; exit 3; }

cp "$tmp/n.cu" "$tmp/noncg.cu"
sed -i 's/pair=cg block=cg/pair=cs block=default/' "$tmp/noncg.cu"
set +e
python3 "$GEN" "$tmp/noncg.cu" "$tmp/noncg-out.cu" 0 0 >"$tmp/noncg.out" 2>"$tmp/noncg.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'neither Stage-N axis uses cg' "$tmp/noncg.err" || { echo 'Stage Q accepted non-CG Stage N' >&2; exit 3; }

echo 'b300-mainrec-ilp8-pair-block-cg-l2-policy-preflight OK stage=Q ilp8_only=1 ilp2_exact_upstream=1 stagep_preserved=1 stagen_direct=1 stageo_composed=1 pair_block_independent=1 sizes=0,64,128,256 double_transform_rejected=1 noncg_rejected=1 gpu_work=0'
