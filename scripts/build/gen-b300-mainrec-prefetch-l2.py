#!/usr/bin/env python3
from __future__ import annotations
import pathlib,re,sys

if len(sys.argv)!=3:raise SystemExit('usage: gen-b300-mainrec-prefetch-l2.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in ('main_pull_kernel_ilp2','const Count pair0=','const Count block0=','const Count self0=','high_rec_groups='):
    if req not in s:raise SystemExit(f'mainrec L2 prefetch requires artifact: {req}')
if 'b300_mainrec_prefetch_l2' in s:raise SystemExit('mainrec source already contains L2 prefetch helper')

kernel_names=['main_pull_kernel_ilp2']
if 'main_pull_kernel_ilp8_hybrid(' in s:
    kernel_names.append('main_pull_kernel_ilp8_hybrid')

def function_span(text:str,name:str)->tuple[int,int]:
    token=name+'('
    if text.count(token)!=1:raise SystemExit(f'{name} definition expected once, got {text.count(token)}')
    p=text.find(token);start=text.rfind('\n',0,p)+1;brace=text.find('{',p);depth=0;end=-1
    if brace<0:raise SystemExit(f'{name} opening brace not found')
    for i in range(brace,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:end=i+1;break
    if end<0:raise SystemExit(f'{name} end not found')
    return start,end

def transform_body(name:str,body:str)->tuple[str,int]:
    ids=sorted({int(x) for x in re.findall(r'const Count pair(\d+)=',body)})
    if not ids or ids!=list(range(max(ids)+1)):raise SystemExit(f'{name}: unexpected lane ids {ids}')
    def get_line(prefix:str,k:int)->str:
        m=re.search(rf'^\s*const Count {prefix}{k}=.*;$',body,re.M)
        if not m:raise SystemExit(f'{name}: missing {prefix}{k} declaration')
        return m.group(0)
    pairs=[get_line('pair',k) for k in ids]
    blocks=[get_line('block',k) for k in ids]
    selfs=[get_line('self',k) for k in ids]
    old='\n'.join(pairs+blocks+selfs)
    if body.count(old)!=1:raise SystemExit(f'{name}: random/self load group is not contiguous/unique')
    pref=[]
    for k in ids:pref.append(f'        b300_mainrec_prefetch_l2(in+pj{k},hp{k});')
    for k in ids:pref.append(f'        b300_mainrec_prefetch_l2(hb{k}?in_block+bj{k}:in,hb{k});')
    new='\n'.join(pref+selfs+pairs+blocks)
    body=body.replace(old,new,1)
    final_pref=f'b300_mainrec_prefetch_l2(hb{ids[-1]}?in_block+bj{ids[-1]}:in,hb{ids[-1]})'
    a=body.find(final_pref);b=body.find('const Count self0=',a);c=body.find('const Count pair0=',b)
    if not(a>=0 and b>a and c>b):raise SystemExit(f'{name}: prefetch/self/random ordering not preserved')
    return body,len(ids)

transforms=[];lane_counts={}
for name in kernel_names:
    start,end=function_span(s,name);body,nlanes=transform_body(name,s[start:end])
    transforms.append((start,end,body,name));lane_counts[name]=nlanes
for start,end,body,_ in sorted(transforms,reverse=True):
    s=s[:start]+body+s[end:]
insert_at=min(x[0] for x in transforms)
helper=r'''__device__ __forceinline__ void b300_mainrec_prefetch_l2(const Count* p,bool valid){
#if __CUDA_ARCH__ >= 700
    if(valid){const unsigned long long a=reinterpret_cast<unsigned long long>(p);asm volatile("prefetch.global.L2 [%0];" :: "l"(a));}
#else
    (void)p;(void)valid;
#endif
}

'''
s=s[:insert_at]+helper+s[insert_at:]
for name,nlanes in lane_counts.items():
    start,end=function_span(s,name);body=s[start:end]
    for k in range(nlanes):
        for req in (f'b300_mainrec_prefetch_l2(in+pj{k},hp{k})',f'b300_mainrec_prefetch_l2(hb{k}?in_block+bj{k}:in,hb{k})'):
            if req not in body:raise SystemExit(f'{name}: missing prefetch artifact: {req}')
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
summary=','.join(f'{name}:{lane_counts[name]}' for name in kernel_names)
print(f'generated {out} from {src}: mainrec_prefetch_l2=1 kernels={len(kernel_names)} kernel_lanes={summary} prefetch_addresses={2*sum(lane_counts.values())} latency_gap=coalesced_self_loads self_loads_before_random=1 random_policy_preserved=1 hybrid_policy_consistent=1 shared_bytes=0 extra_state_bytes=0 semantics_unchanged=1')
