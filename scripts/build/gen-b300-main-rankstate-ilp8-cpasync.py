#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-main-rankstate-ilp8-cpasync.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
for req in (
    'b300_main_rankstate_ilp8_blocks','base+=8*grid',
    'b300_main_pull_rankstate_ilp4_kernel','const Code pj7=', 'const Code bj7=',
    'rank_state[i7]=b300_pack_rank_state','const Count pair7=', 'const Count block7='
):
    if req not in s:raise SystemExit(f'ILP8 cp.async requires artifact: {req}')

def replace_function(text:str,name:str,new:str)->str:
    p=text.find(name+'(')
    if p<0:raise SystemExit(f'function not found: {name}')
    start=text.rfind('\n',0,p)+1
    brace=text.find('{',p);depth=0;end=-1
    for k in range(brace,len(text)):
        if text[k]=='{':depth+=1
        elif text[k]=='}':
            depth-=1
            if depth==0:end=k+1;break
    if end<0:raise SystemExit(f'function end not found: {name}')
    return text[:start]+new+text[end:]

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker not found')
helper=r'''

__device__ __forceinline__ void b300_cpasync_count_ca(
    Count* shared_dst,const Count* global_src,bool valid
){
#if __CUDA_ARCH__ >= 800
    static_assert(sizeof(Count)==4,"ILP8 cp.async assumes 32-bit Count");
    const uint32_t smem=static_cast<uint32_t>(__cvta_generic_to_shared(shared_dst));
    const unsigned long long gmem=reinterpret_cast<unsigned long long>(global_src);
    const uint32_t src_bytes=valid?4u:0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
                 "r"(smem),"l"(gmem),"r"(src_bytes));
#else
    *shared_dst=valid?*global_src:Count(0);
#endif
}
__device__ __forceinline__ void b300_cpasync_commit(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
__device__ __forceinline__ void b300_cpasync_wait_all(){
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
'''
s=s.replace(marker,helper+marker,1)

L=[];A=L.append
A('__global__ void b300_main_pull_rankstate_ilp4_kernel(')
A('    const Count* __restrict__ in,const MateID* __restrict__ mates,Code n,')
A('    const Count* __restrict__ in_block,Code nblock,')
A('    Count* __restrict__ out_main,int p,RankState* __restrict__ rank_state')
A('){')
A('    extern __shared__ Count b300_cpasync_smem[];')
A('    Count* const my_smem=b300_cpasync_smem+size_t(threadIdx.x)*16u;')
A('    const Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;')
A('    const Code grid=Code(gridDim.x)*blockDim.x;')
A('    for(Code base=tid;base<n;base+=8*grid){')
for k in range(8): A(f'        const Code i{k}=base+Code({k})*grid;')
A('        const bool v0=true;')
for k in range(1,8): A(f'        const bool v{k}=i{k}<n;')
A('        const MateID m0=mates[i0];')
for k in range(1,8): A(f'        const MateID m{k}=v{k}?mates[i{k}]:MateID(0);')
A('        const RankState s0=rank_state[i0];')
for k in range(1,8): A(f'        const RankState s{k}=v{k}?rank_state[i{k}]:RankState(0);')
for k in range(8): A(f'        const RankDelta rd{k}=v{k}?b300_unpack_rank_delta(s{k}):RankDelta(0);')
for k in range(8): A(f'        const int h{k}=v{k}?b300_unpack_rank_height(s{k}):0;')
for k in range(8): A(f'        const MateValuePair pp{k}=v{k}?mpair(m{k},p):NN;')
for k in range(8): A(f'        const bool hp{k}=v{k}&&(pp{k}==LR||pp{k}==NR||pp{k}==NL);')
for k in range(8): A(f'        const Code pj{k}=hp{k}?main_pull_direct_pair_source_rank(i{k},m{k},p,h{k}):Code(0);')
for k in range(8): A(f'        const MateValue mv{k}=v{k}?mget(m{k},p):N;')
for k in range(8): A(f'        const bool hb{k}=v{k}&&nblock&&mv{k}==N;')
for k in range(8): A(f'        const Code bj{k}=hb{k}?b300_add_rank_delta(i{k},rd{k}):Code(0);')
A('')
A('        // Two uniform groups of eight Count copies. Invalid candidates use')
A('        // src-size=0, zero-filling their four-byte shared destinations.')
for k in range(8): A(f'        b300_cpasync_count_ca(my_smem+{k},hp{k}?in+pj{k}:in,hp{k});')
A('        b300_cpasync_commit();')
for k in range(8): A(f'        b300_cpasync_count_ca(my_smem+{8+k},(hb{k}&&bj{k}<nblock)?in_block+bj{k}:in,hb{k}&&bj{k}<nblock);')
A('        b300_cpasync_commit();')
A('')
A('        // Do independent state work and coalesced self reads while both')
A('        // random global->shared async groups are outstanding.')
A('        rank_state[i0]=b300_pack_rank_state(rd0+b300_rank_delta_step(mv0,p,h0),b300_rank_height_advance(h0,mv0));')
for k in range(1,8): A(f'        if(v{k})rank_state[i{k}]=b300_pack_rank_state(rd{k}+b300_rank_delta_step(mv{k},p,h{k}),b300_rank_height_advance(h{k},mv{k}));')
A('        const Count self0=in[i0];')
for k in range(1,8): A(f'        const Count self{k}=v{k}?in[i{k}]:Count(0);')
A('        b300_cpasync_wait_all();')
for k in range(8): A(f'        const Count pair{k}=my_smem[{k}];')
for k in range(8): A(f'        const Count block{k}=my_smem[{8+k}];')
A('        const uint64_t mod=D_MOD;')
for k in range(8):
    stmt=f'uint64_t a=uint64_t(self{k})+pair{k}+block{k};if(a>=mod)a-=mod;if(a>=mod)a-=mod;out_main[i{k}]=Count(a);'
    A(('        {' if k==0 else f'        if(v{k})'+'{')+stmt+'}')
A('    }')
A('}')
new='\n'.join(L)
s=replace_function(s,'b300_main_pull_rankstate_ilp4_kernel',new)

old='b300_main_pull_rankstate_ilp4_kernel<<<b300_main_rankstate_ilp8_blocks(ms.size,threads),threads,0,c.sMain>>>'
new_launch='b300_main_pull_rankstate_ilp4_kernel<<<b300_main_rankstate_ilp8_blocks(ms.size,threads),threads,size_t(threads)*16u*sizeof(Count),c.sMain>>>'
if s.count(old)!=1:raise SystemExit(f'ILP8 launch anchor expected one match got {s.count(old)}')
s=s.replace(old,new_launch,1)

for req in (
    'cp.async.ca.shared.global','cp.async.commit_group','cp.async.wait_group 0',
    'size_t(threadIdx.x)*16u','my_smem+15','const Count pair7=my_smem[7]',
    'const Count block7=my_smem[15]','size_t(threads)*16u*sizeof(Count)',
    'rank_state[i7]=b300_pack_rank_state','const Count self7='
):
    if req not in s:raise SystemExit(f'missing ILP8 cp.async artifact: {req}')
if s.count('b300_cpasync_commit();') != 2:
    raise SystemExit(f'expected exactly two async groups, got {s.count("b300_cpasync_commit();")} commits')
for stale in ('const Count pair7=hp7?', 'const Count block7=(hb7'):
    if stale in s:raise SystemExit(f'stale synchronous ILP8 gather remains: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_main_rankstate_ilp8_cpasync=1 cp_bytes=4 cache_operator=ca async_random_copies_per_thread=16 async_groups=2 copies_per_group=8 dynamic_shared_bytes_per_thread=64 overlap=rank_state+self_load zero_fill_invalid=1 block_path_unchanged=1 closure_path_unchanged=1')
