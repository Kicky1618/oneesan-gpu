#pragma once
// Included after the factorized solver's rank/unrank and reverse-step helpers.
// Same-popcount inactive occupancy masks are conjugate under gap relocation.
// Their local rank order and predecessor multiplicities coincide. See
// docs/results/frontier-compiled-2026-09-05.md for the proof and limits.
namespace factor_transfer {
template<class Emit> __device__ __forceinline__ void main_sources(MateID t,int p,Code n,Emit emit){
    emit(factor_rank_main(t));
    if(p==1){
        oneesan::gridfp::reverse_boundary_main_predecessors(t,TARGET_W,[&](oneesan::gridfp::MateID m){emit(factor_rank_main(m));});
        if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N)emit(n+factor_rank_block(oneesan::gridfp::mshrink(t,p)));
        return;
    }
    MateID pred=0;bool have=true;
    switch(oneesan::gridfp::mpair(t,p)){
    case oneesan::gridfp::LR:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);break;
    case oneesan::gridfp::NR:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);break;
    case oneesan::gridfp::NL:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);break;
    default:have=false;
    }
    if(have)emit(factor_rank_main(pred));
    if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N)
        emit(n+factor_rank_block(oneesan::gridfp::mshrink(t,p)));
}
template<class Emit> __device__ __forceinline__ void block_sources(MateID b,int p,Emit emit){
    oneesan::gridfp::reverse_block_predecessors<GRIDFP_SPARSE_REVERSE>(b,TARGET_W,p,
        [&](oneesan::gridfp::MateID m){emit(factor_rank_main(m));});
}
template<class Emit> __device__ __forceinline__ void one_step_sources(Code i,Code n,int p,Emit emit){
    if(i<n)main_sources(factor_unrank_main(i),p,n,emit);
    else if(p>1)block_sources(factor_unrank_block(i-n),p,emit);
}
template<class Emit> __device__ __forceinline__ void sources(Code i,Code n,int p,bool paired,Emit emit){
    if(!paired){one_step_sources(i,n,p,emit);return;}
    one_step_sources(i,n,p-1,[&](Code j){one_step_sources(j,n,p,emit);});
}

// Every main row starts with its identity edge. Store that edge implicitly;
// keep all other duplicates, since they count distinct edge choices.
__global__ void counts_kernel(Code n,Code total,int p,bool paired,uint32_t* offsets){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<=total;i+=Code(gridDim.x)*blockDim.x){
        Code count=0;if(i<total)sources(i,n,p,paired,[&](Code){++count;});offsets[i]=count-(i<n?1:0);
    }
}
__global__ void fill_kernel(Code n,Code total,int p,bool paired,const uint32_t* offsets,uint32_t* columns){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=Code(gridDim.x)*blockDim.x){
        Code pos=offsets[i];bool first=i<n;
        sources(i,n,p,paired,[&](Code j){if(first){first=false;return;}columns[pos++]=uint32_t(j);});
    }
}
__global__ void verify_kernel(Code n,Code total,int p,bool paired,const uint32_t* offsets,const uint32_t* columns,Code* errors){
    Code bad=0;
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=Code(gridDim.x)*blockDim.x){
        Code pos=offsets[i],end=offsets[i+1];bool first=i<n;
        sources(i,n,p,paired,[&](Code j){if(first){if(j!=i)++bad;first=false;return;}if(pos>=end||j>=total||columns[pos]!=j)++bad;++pos;});
        if(pos!=end)++bad;
    }
    if(bad)atomicAdd(errors,bad);
}
__global__ void apply_kernel(const Count* __restrict__ in,const Count* __restrict__ din,Count* __restrict__ out,Count* __restrict__ dout,Code n,Code total,const uint32_t* __restrict__ offsets,const uint32_t* __restrict__ columns){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<total;i+=Code(gridDim.x)*blockDim.x){
        Code sum=i<n?in[i]:0;
        for(Code e=offsets[i];e<offsets[i+1];++e){Code j=columns[e];sum+=j<n?in[j]:din[j-n];}
        // At most 6*TARGET_W source occurrences: sum < 2^40 at W<=28.
        Count value=oneesan::invariant_divmod(sum,D_MOD,D_MOD_RECIPROCAL).remainder;
        if(i<n)out[i]=value;else dout[i-n]=value;
    }
}
struct Step { int p=0;bool paired=false;uint32_t* offsets=nullptr;uint32_t* columns=nullptr;Code edges=0; };
}
