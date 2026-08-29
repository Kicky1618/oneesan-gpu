#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-main-pull.py INPUT.cu OUTPUT.cu')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()

if 'template<bool CACHED_MATE>\n__global__ void main_group_kernel' not in s:
    raise SystemExit('main-pull transform requires gen-b300-main-mate-cache.py first')

marker = '\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:
    raise SystemExit('rank_full marker not found')
insert = r'''

__device__ __forceinline__ Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p){
    const int h=height_before_rank_pos<TARGET_W>(m,p);
    switch(mpair(m,p)){
        case LR:{
            // Destination LR <- source NN. NN precedes LR by the complete
            // N branch, the R branch at p, and the N branch at p-1 after L.
            const Code delta=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0)+D_MAIN_DP[p-1][h+1];
            return dst_rank-delta;
        }
        case NR:{
            // Destination NR <- source RN.
            const Code delta=D_MAIN_DP[p][h]-D_MAIN_DP[p-1][h];
            return dst_rank+delta;
        }
        case NL:{
            // Destination NL <- source LN.
            const Code a=D_MAIN_DP[p][h]+(h?D_MAIN_DP[p][h-1]:0);
            const Code b=D_MAIN_DP[p-1][h]+(h?D_MAIN_DP[p-1][h-1]:0);
            return dst_rank+(a-b);
        }
        default:return dst_rank;
    }
}

template<bool CACHED_MATE>
__global__ void main_pull_kernel(const Count*in,const MateID*mates,Code n,const Count*in_block,Code nblock,Count*out_main,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID m;
        if constexpr(CACHED_MATE)m=mates[i];
        else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
        uint64_t acc=in[i];
        const MateValuePair pair=mpair(m,p);
        if(pair==LR||pair==NR||pair==NL){
            const Code j=main_pull_direct_pair_source_rank(i,m,p);
            acc+=in[j];
        }
        // d[p]==N means d is exactly the N-lift of its blocked predecessor.
        // Reuse the production direct drop-rank delta instead of recomputing a
        // full W-1 grouped rank from mshrink(d,p).
        if(nblock&&mget(m,p)==N){Code j=rank_drop_n_t<TARGET_W>(i,m,p);if(j<nblock)acc+=in_block[j];}
        uint64_t mod=D_MOD;if(acc>=mod)acc-=mod;if(acc>=mod)acc-=mod;out_main[i]=Count(acc);
    }
}

template<bool CACHED_MATE>
__global__ void main_to_block_kernel(const Count*in,const MateID*mates,Code n,Count*out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        Count c=in[i];if(!c)continue;
        MateID m;
        if constexpr(CACHED_MATE)m=mates[i];
        else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
        MateValuePair w=mpair(m,p);
        switch(w){
            case NR:case NL:{Code j=rank_drop_n_t<TARGET_W>(i,m,p);atomic_add_mod(out_block+j,c);break;}
            case LL:{MateID t=msetpair(m,p,NN);int q=p-1,s=1;while(s){--q;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}t=mset(t,q,L);t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);break;}
            case RR:{MateID t=msetpair(m,p,NN);int q=p,s=1;while(s){++q;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}t=mset(t,q,R);t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);break;}
            case RL:{MateID t=msetpair(m,p,NN);t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);break;}
            default:break;
        }
    }
}
'''
s = s.replace(marker, insert + marker, 1)

start_marker = '    for(int p=wp.p_hi;p>=wp.p_lo;--p){\n'
end_marker = '    ck(cudaStreamSynchronize(c.sMain),"main sync");'
a = s.find(start_marker)
if a < 0:
    raise SystemExit('process_group loop start not found')
b = s.find(end_marker, a)
if b < 0:
    raise SystemExit('process_group loop end not found')
new_loop = r'''    for(int p=wp.p_hi;p>=wp.p_lo;--p){
        if(p>1){
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
        }else{
            if(ms.size)ck(cudaMemcpyAsync(nxt,cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToDevice,c.sMain),"identity async");
            if(ds.size)ck(cudaMemsetAsync(dnext,0,size_t(ds.size)*sizeof(Count),c.sBlock),"clear next D");
            ck(cudaEventRecord(c.copyDone,c.sMain),"record copy");ck(cudaEventRecord(c.clearDone,c.sBlock),"record clear");
            ck(cudaStreamWaitEvent(c.sMain,c.clearDone,0),"main wait clear");ck(cudaStreamWaitEvent(c.sBlock,c.copyDone,0),"block wait copy");
            if(ms.size){if(useMate)main_group_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,nxt,dnext,p);else main_group_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,nxt,dnext,p);}
            if(ds.size)blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,nxt,p);
            ck(cudaGetLastError(),"doubleD transition");
        }
        ck(cudaEventRecord(c.mainDone,c.sMain),"record main");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block");
        ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block");ck(cudaStreamWaitEvent(c.sBlock,c.mainDone,0),"block wait main");
        std::swap(cur,nxt);std::swap(dcur,dnext);
    }
'''
s = s[:a] + new_loop + s[b:]

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_main_pull=1 p_scope=2..Wm1 main_atomic=0 main_identity_copy=0 blocked_to_main_scatter=0 block_rank_guard=1 blocked_preimage_rank=direct_drop pair_preimage_rank=direct_dp rank_same_calls=0')
