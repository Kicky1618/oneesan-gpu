#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-load-policy.py"
OGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-pair-block-cg-l2-policy.py"
QGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py"
RGEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-load-policy-stager.py"
python3 -m py_compile "$NGEN" "$OGEN" "$QGEN" "$RGEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stager-ilp2.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
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
for k in {0..7}; do echo " const Count pair$k=hp$k?in[pj$k]:Count(0);"; done
for k in {0..7}; do echo " const Count block$k=hb$k?in_block[bj$k]:Count(0);"; done
for k in {0..7}; do echo " const Count self$k=in[i$k];"; done
cat <<'EOF'
 (void)m0;(void)m7;(void)pair0;(void)pair7;(void)block0;(void)block7;(void)self0;(void)self7;
 mates[i7]=0;
}
EOF
} >"$src"
python3 "$NGEN" "$src" "$tmp/n.cu" cg cg 128 >/dev/null
python3 "$OGEN" "$tmp/n.cu" "$tmp/o.cu" 64 256 >/dev/null
python3 "$QGEN" "$tmp/o.cu" "$tmp/q.cu" 256 64 >/dev/null
python3 - "$tmp/q.cu" "$tmp/q.ilp8" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0
for i in range(b,len(s)):
 d+=(s[i]=='{')-(s[i]=='}')
 if d==0: Path(sys.argv[2]).write_text(s[m.start():i+1]); break
PY
python3 "$RGEN" "$tmp/q.cu" "$tmp/r.cu" cs default >"$tmp/r.log"
grep -Fq 'pair_policy=cs block_policy=default' "$tmp/r.log"
grep -Fq 'high_pair_l2=256 high_block_l2=64 stageq_preserved=1' "$tmp/r.log"
for k in 0 1; do
 grep -Fq "const Count pair$k=hp$k?b300_mainrec_stager_ilp2_load_cs(in+pj$k):Count(0);" "$tmp/r.cu"
 grep -Fq "const Count block$k=hb$k?in_block[bj$k]:Count(0);" "$tmp/r.cu"
done
python3 - "$tmp/r.cu" "$tmp/q.ilp8" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text(); want=Path(sys.argv[2]).read_text(); m=re.search(r'__global__\s+void\s+main_pull_kernel_ilp8_hybrid\s*\(',s); assert m
b=s.find('{',m.end()); d=0
for i in range(b,len(s)):
 d+=(s[i]=='{')-(s[i]=='}')
 if d==0:
  if s[m.start():i+1]!=want: raise SystemExit('Stage R changed Stage-Q ILP8 bytes')
  break
PY
grep -Fq '// b300_mainrec_stageq_ilp8_pair_block_cg_l2=1' "$tmp/r.cu"
set +e
python3 "$RGEN" "$tmp/r.cu" "$tmp/double.cu" cs default >/dev/null 2>"$tmp/double.err"; rc=$?
set -e
((rc!=0)) && grep -Fq 'already contains Stage-R' "$tmp/double.err" || exit 3

echo 'b300-mainrec-ilp2-pair-block-load-policy-stager-preflight OK stage=R ilp2_lanes=2 low_pair_policies=default,cg,cs low_block_policies=default,cg,cs stageq_high_l2_provenance=1 ilp8_byte_identical=1 gpu_work=0'
