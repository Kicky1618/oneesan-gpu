#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-block-pull.py INPUT.cu OUTPUT.cu')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()

if 'main_pull_kernel' not in s or 'main_to_block_kernel' not in s:
    raise SystemExit('block-pull transform requires gen-b300-main-pull.py first')

marker = '\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:
    raise SystemExit('rank_full marker not found')
insert = r'''

__device__ __forceinline__ void pull_add_mod(Count& acc,Count v){
    Count mod=D_MOD;
    acc=(acc>=mod-v)?acc-(mod-v):acc+v;
}

__device__ __forceinline__ void block_pull_add_rank(
    Count& acc,const Count*in_main,Code j
){
    pull_add_mod(acc,in_main[j]);
}

// Exact inverse of rank_drop_n_t for a full-width state whose inserted symbol
// at p is N. Only the high prefix changes coordinates.
template<int WIDTH>
__device__ __forceinline__ Code rank_lift_n_t(Code block_rank,MateID blocked,int p){
    MateID m=minsert(blocked,p,N);
    Code a=0,b=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>p;--pos){
        MateValue v=mget(m,pos);
        if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))a+=D_MAIN_DP[pos][h];
        if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))a+=D_MAIN_DP[pos][h-1];
        int q=pos-1;
        if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];
        if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];
        if(v==R)--h;else if(v==L)++h;
    }
    return a>=b?block_rank+(a-b):block_rank-(b-a);
}

__device__ __forceinline__ void block_pull_add_same_rank(
    Count& acc,const Count*in_main,Code base_rank,MateID base,MateID x
){
    Code j=rank_same_t<TARGET_W>(base_rank,base,x,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
    pull_add_mod(acc,in_main[j]);
}

__device__ __forceinline__ uint32_t block_pull_endpoint_mask(MateID mate){
    uint64_t x=(uint64_t(mate)|(uint64_t(mate)>>1))&0x5555555555555555ULL;
    x=(x|(x>>1))&0x3333333333333333ULL;
    x=(x|(x>>2))&0x0f0f0f0f0f0f0f0fULL;
    x=(x|(x>>4))&0x00ff00ff00ff00ffULL;
    x=(x|(x>>8))&0x0000ffff0000ffffULL;
    x=(x|(x>>16))&0x00000000ffffffffULL;
    uint32_t out=uint32_t(x);
    if constexpr(TARGET_W<32)out&=(uint32_t(1)<<TARGET_W)-1u;
    return out;
}

__global__ void block_pull_kernel(const Count*in_main,Code n,Count*out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
        Count acc=0;
        MateValue look=mget(b,p-1);
        if(look==R||look==L){
            // NR/NL removes physical p. Invert rank_drop_n_t directly from the
            // blocked destination rank; no full main rank walk is needed.
            Code j=rank_lift_n_t<TARGET_W>(i,b,p);
            block_pull_add_rank(acc,in_main,j);
        }else if(look==N){
            // Closure removes physical p-1. Lift the NN destination rank once;
            // all RL/LL/RR preimages differ only on a local span and therefore
            // use rank_same_t rather than complete rank_group_t walks.
            MateID d=minsert(b,p-1,N);
            Code base_rank=rank_lift_n_t<TARGET_W>(i,b,p-1);

            if(height_before_rank_pos<TARGET_W>(d,p)>0){
                MateID x=msetpair(d,p,RL);
                block_pull_add_same_rank(acc,in_main,base_rank,d,x);
            }

            // N contributes zero to the closure balance. Compress the MateID to
            // a one-bit endpoint mask and scan only L/R positions. This preserves
            // the exact legacy stopping condition while removing all N iterations.
            const uint32_t endpoints=block_pull_endpoint_mask(d);
            uint32_t left=endpoints&((uint32_t(1)<<(p-1))-1u);
            int bal=0;
            while(left){
                int q=31-__clz(left);MateValue v=mget(d,q);
                if(bal==0&&v==L){
                    MateID x=msetpair(d,p,LL);x=mset(x,q,R);
                    block_pull_add_same_rank(acc,in_main,base_rank,d,x);
                }
                if(v==L)++bal;else --bal;
                left^=uint32_t(1)<<q;
                if(bal<0)break;
            }
            uint32_t right=endpoints&~((uint32_t(1)<<(p+1))-1u);
            bal=0;
            while(right){
                int q=__ffs(right)-1;MateValue v=mget(d,q);
                if(bal==0&&v==R){
                    MateID x=msetpair(d,p,RR);x=mset(x,q,L);
                    block_pull_add_same_rank(acc,in_main,base_rank,d,x);
                }
                if(v==R)++bal;else --bal;
                right&=right-1u;
                if(bal<0)break;
            }
        }
        out_block[i]=acc;
    }
}
'''
s = s.replace(marker, insert + marker, 1)

old = '''        if(p>1){
            if(ds.size)ck(cudaMemsetAsync(dnext,0,size_t(ds.size)*sizeof(Count),c.sBlock),"clear next D pull");
            if(ms.size){
                if(useMate)main_pull_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p);
                if(ds.size){
                    if(useMate)main_to_block_kernel<true><<<bm,threads,0,c.sBlock>>>(cur,c.dMate,ms.size,dnext,p);
                    else main_to_block_kernel<false><<<bm,threads,0,c.sBlock>>>(cur,nullptr,ms.size,dnext,p);
                }
            }
            ck(cudaGetLastError(),"doubleD pull transition");
'''
new = '''        if(p>1){
            if(ms.size){
                if(useMate)main_pull_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p);
            }
            if(ds.size)block_pull_kernel<<<bd,threads,0,c.sBlock>>>(cur,ds.size,dnext,p);
            ck(cudaGetLastError(),"doubleD full pull transition");
'''
if old not in s:
    raise SystemExit('main-pull p>1 loop anchor not found')
s = s.replace(old, new, 1)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_block_pull=1 p_scope=2..Wm1 block_atomic=0 block_memset=0 deferred_insert=p endpoint_rank=direct_lift closure_base_rank=direct_lift closure_candidate_rank=local_delta rl_validity_gate=1 closure_scan=endpoint_setbits full_group_rank_calls=0')
