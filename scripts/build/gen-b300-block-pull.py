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

template<int WIDTH>
__device__ __forceinline__ void block_pull_add_source(
    Count& acc,const Count*in_main,MateID x
){
    Code j=rank_group_t<WIDTH>(x,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
    pull_add_mod(acc,in_main[j]);
}

__global__ void block_pull_kernel(const Count*in_main,Code n,Count*out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
        Count acc=0;
        MateValue look=mget(b,p-1);
        if(look==R||look==L){
            // NR/NL has N at physical p. The push path's rank_drop_n_t(...,p)
            // is exactly rank(mshrink(source,p)), so reconstruct the source by
            // reinserting N at p while retaining the endpoint at p-1.
            MateID x=minsert(b,p,N);
            block_pull_add_source<TARGET_W>(acc,in_main,x);
        }else if(look==N){
            // LL/RR/RL closure removes physical p-1 after producing NN.
            MateID d=minsert(b,p-1,N);
            MateID x=msetpair(d,p,RL);
            block_pull_add_source<TARGET_W>(acc,in_main,x);

            int bal=0;
            for(int q=p-2;q>=0;--q){
                MateValue v=mget(d,q);
                if(bal==0&&v==L){
                    x=msetpair(d,p,LL);x=mset(x,q,R);
                    block_pull_add_source<TARGET_W>(acc,in_main,x);
                }
                if(v==L)++bal;else if(v==R)--bal;
                if(bal<0)break;
            }
            bal=0;
            for(int q=p+1;q<TARGET_W;++q){
                MateValue v=mget(d,q);
                if(bal==0&&v==R){
                    x=msetpair(d,p,RR);x=mset(x,q,L);
                    block_pull_add_source<TARGET_W>(acc,in_main,x);
                }
                if(v==R)++bal;else if(v==L)--bal;
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
print(f'generated {out} from {src}: b300_block_pull=1 p_scope=2..Wm1 block_atomic=0 block_memset=0 deferred_insert=p')
