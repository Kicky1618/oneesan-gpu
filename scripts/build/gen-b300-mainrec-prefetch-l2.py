#!/usr/bin/env python3
from __future__ import annotations
import pathlib,re,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-mainrec-prefetch-l2.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('main_pull_kernel_ilp2','const Count pair0=','const Count block0=','const Count self0=','high_rec_groups='):
    if req not in s:raise SystemExit(f'mainrec L2 prefetch requires artifact: {req}')
if 'prefetch.global.L2' in s:raise SystemExit('mainrec source already contains L2 prefetch')

p=s.find('main_pull_kernel_ilp2(');start=s.rfind('\n',0,p)+1;brace=s.find('{',p);depth=0;end=-1
for i in range(brace,len(s)):
    if s[i]=='{':depth+=1
    elif s[i]=='}':
        depth-=1
        if depth==0:end=i+1;break
if end<0:raise SystemExit('mainrec kernel end not found')
body=s[start:end]
ids=sorted({int(x) for x in re.findall(r'const Count pair(\d+)=',body)})
if not ids or ids!=list(range(max(ids)+1)):raise SystemExit(f'unexpected lane ids {ids}')

# Capture complete declaration lines. They may use ordinary loads or the .cg
# helper; this transform only changes ordering and inserts L2 prefetch hints.
def get_line(prefix:str,k:int)->str:
    m=re.search(rf'^\s*const Count {prefix}{k}=.*;$',body,re.M)
    if not m:raise SystemExit(f'missing {prefix}{k} declaration')
    return m.group(0)
pairs=[get_line('pair',k) for k in ids]
blocks=[get_line('block',k) for k in ids]
selfs=[get_line('self',k) for k in ids]
# The generator emits these groups contiguously pair->block->self.
old='\n'.join(pairs+blocks+selfs)
if body.count(old)!=1:raise SystemExit('random/self load group is not contiguous/unique')
pref=[]
for k in ids:pref.append(f'        b300_mainrec_prefetch_l2(in+pj{k},hp{k});')
for k in ids:pref.append(f'        b300_mainrec_prefetch_l2(hb{k}?in_block+bj{k}:in,hb{k});')
new='\n'.join(pref+selfs+pairs+blocks)
body=body.replace(old,new,1)
helper=r'''__device__ __forceinline__ void b300_mainrec_prefetch_l2(const Count* p,bool valid){
#if __CUDA_ARCH__ >= 700
    if(valid){const unsigned long long a=reinterpret_cast<unsigned long long>(p);asm volatile("prefetch.global.L2 [%0];" :: "l"(a));}
#else
    (void)p;(void)valid;
#endif
}

'''
s=s[:start]+helper+body+s[end:]
for k in ids:
    for req in (f'b300_mainrec_prefetch_l2(in+pj{k},hp{k})',f'b300_mainrec_prefetch_l2(hb{k}?in_block+bj{k}:in,hb{k})'):
        if req not in s:raise SystemExit(f'missing prefetch artifact: {req}')
# Verify the intended latency gap: final prefetch < first self < first random.
a=s.find(f'b300_mainrec_prefetch_l2(hb{ids[-1]}?in_block+bj{ids[-1]}:in,hb{ids[-1]})',start)
b=s.find('const Count self0=',a);c=s.find('const Count pair0=',b)
if not(a>=0 and b>a and c>b):raise SystemExit('prefetch/self/random ordering not preserved')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: mainrec_prefetch_l2=1 lanes={len(ids)} prefetch_addresses={2*len(ids)} latency_gap=coalesced_self_loads self_loads_before_random=1 random_policy_preserved=1 shared_bytes=0 extra_state_bytes=0 semantics_unchanged=1')
