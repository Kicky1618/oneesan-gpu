#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
RGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-load-policy.py"
SGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py"
python3 -m py_compile "$RGEN" "$SGEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stages-l2.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
base="$tmp/base.cu"
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
  const Count self0=in[0]; const MateID m0=mates[0];
EOF
for k in {0..7}; do echo "  const Count pair$k=in[$k]; const Count block$k=in_block[$k];"; done
cat <<'EOF'
  (void)self0;(void)m0;(void)pair0;(void)pair7;(void)block0;(void)block7;
}
// b300_mainrec_stagen_pair_block_policy=1 pair=cg block=cg cg_l2_bytes=128
// b300_mainrec_stageq_ilp8_pair_block_cg_l2=1 pair_policy=cg block_policy=cg pair_l2_bytes=256 block_l2_bytes=64 upstream_pair_l2_bytes=128 upstream_block_l2_bytes=128 stagep_preserved=0
EOF
} >"$base"
r="$tmp/r.cu"; python3 "$RGEN" "$base" "$r" cg cg >"$tmp/r.log"
grep -Fq 'b300_mainrec_stager_ilp2_pair_block_policy=1 pair_policy=cg block_policy=cg' "$tmp/r.log"
python3 - "$r" >"$tmp/ilp8.sha" <<'PY'
from pathlib import Path
import hashlib,re,sys
s=Path(sys.argv[1]).read_text(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0; e=None
for i in range(b,len(s)):
    if s[i]=='{': d+=1
    elif s[i]=='}':
        d-=1
        if d==0: e=i+1; break
print(hashlib.sha256(s[m.start():e].encode()).hexdigest())
PY
for pl2 in 0 64 128 256; do
  for bl2 in 0 64 128 256; do
    out="$tmp/s-${pl2}-${bl2}.cu"; log="$tmp/s-${pl2}-${bl2}.log"
    python3 "$SGEN" "$r" "$out" "$pl2" "$bl2" >"$log"
    grep -Fq "pair_l2_bytes=$pl2 block_l2_bytes=$bl2" "$log" || exit 3
    grep -Fq "b300_mainrec_stages_ilp2_pair_block_cg_l2=1 pair_policy=cg block_policy=cg pair_l2_bytes=$pl2 block_l2_bytes=$bl2" "$out" || exit 3
    grep -Fq 'high_pair=cg high_block=cg high_pair_l2=256 high_block_l2=64 stageq_preserved=1' "$out" || { echo 'Stage-S lost high-state provenance' >&2; exit 3; }
    for k in 0 1; do
      grep -Fq "b300_mainrec_stages_ilp2_pair_load_cg(in+pj$k)" "$out" || exit 3
      grep -Fq "b300_mainrec_stages_ilp2_block_load_cg(in_block+bj$k)" "$out" || exit 3
    done
    grep -Fq 'const Count self0=in[i0];' "$out" || { echo 'Stage-S changed self load' >&2; exit 3; }
    grep -Fq 'const MateID m0=mates[i0];' "$out" || { echo 'Stage-S changed mate load' >&2; exit 3; }
    pairq='ld.global.cg.u32'; [[ "$pl2" == 0 ]] || pairq="ld.global.cg.L2::${pl2}B.u32"
    blockq='ld.global.cg.u32'; [[ "$bl2" == 0 ]] || blockq="ld.global.cg.L2::${bl2}B.u32"
    grep -Fq "asm volatile(\"$pairq %0, [%1];\"" "$out" || { echo "Stage-S pair qualifier mismatch $pl2" >&2; exit 3; }
    grep -Fq "asm volatile(\"$blockq %0, [%1];\"" "$out" || { echo "Stage-S block qualifier mismatch $bl2" >&2; exit 3; }
    python3 - "$out" "$tmp/ilp8.sha" <<'PY'
from pathlib import Path
import hashlib,re,sys
s=Path(sys.argv[1]).read_text(); want=Path(sys.argv[2]).read_text().strip(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0; e=None
for i in range(b,len(s)):
    if s[i]=='{': d+=1
    elif s[i]=='}':
        d-=1
        if d==0: e=i+1; break
got=hashlib.sha256(s[m.start():e].encode()).hexdigest()
if got!=want: raise SystemExit('Stage-S changed ILP8 kernel')
PY
  done
done
# Non-CG axes must force their hint to zero.
rpc="$tmp/r-pc.cu"; python3 "$RGEN" "$base" "$rpc" cg cs >/dev/null
python3 "$SGEN" "$rpc" "$tmp/pc-ok.cu" 128 0 >/dev/null
set +e; python3 "$SGEN" "$rpc" "$tmp/pc-bad.cu" 128 64 >"$tmp/pc-bad.out" 2>"$tmp/pc-bad.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'BLOCK_L2_BYTES must be 0' "$tmp/pc-bad.err" || { echo 'Stage-S accepted L2 hint on non-CG block axis' >&2; exit 3; }
# Double transform and no-CG input must fail closed.
set +e; python3 "$SGEN" "$tmp/s-128-64.cu" "$tmp/double.cu" 128 64 >/dev/null 2>"$tmp/double.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'already contains Stage-S' "$tmp/double.err" || { echo 'Stage-S double transform not rejected' >&2; exit 3; }
rcc="$tmp/r-cc.cu"; python3 "$RGEN" "$base" "$rcc" cs cs >/dev/null
set +e; python3 "$SGEN" "$rcc" "$tmp/no-cg.cu" 0 0 >/dev/null 2>"$tmp/no-cg.err"; rc=$?; set -e
((rc!=0)) && grep -Fq 'neither Stage-R ILP2 axis uses cg' "$tmp/no-cg.err" || { echo 'Stage-S no-CG input not rejected' >&2; exit 3; }
echo 'b300-mainrec-ilp2-pair-block-cg-l2-policy-preflight OK stage=S combinations=16 low_pair_l2=0,64,128,256 low_block_l2=0,64,128,256 stage_r_policy_preserved=1 stage_q_high_provenance=1 ilp8_byte_locked=1 self_unchanged=1 mate_unchanged=1 invalid_non_cg_hint_rejected=1 double_transform_rejected=1 gpu_work=0'
