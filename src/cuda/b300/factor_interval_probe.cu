#define main factor_original_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_reverse_mate_batch.cu"
#undef main
#include <limits>

struct IStat { uint64_t intervals=0, elems=0, maxlen=0; };
static inline void addrun(IStat& s, uint64_t len){ if(!len)return; ++s.intervals; s.elems+=len; s.maxlen=std::max(s.maxlen,len); }
static Code main_row_base(uint32_t hc,int he,int cv){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    uint32_t har=G_FACTOR.high_packed_rank[hc]>>H;
    Code r=G_FACTOR.high_main_base[G_FACTOR.high_all_off[he]+har];
    if(cv>int(N))r+=H_DP[L][he];
    if(cv>int(R)&&he>0)r+=H_DP[L][he-1];
    return r;
}
static Code block_row_base(uint32_t hc,int he){
    constexpr int H=HIGH_LUT_K;
    uint32_t har=G_FACTOR.high_packed_rank[hc]>>H;
    return G_FACTOR.high_block_base[G_FACTOR.high_all_off[he]+har];
}
static IStat factor_main_stats(bool fixLow,uint32_t mask){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=FactorTablesHost::STRIDE;
    IStat st; auto bs=make_factor_main_blocks(fixLow,mask);
    for(auto const& x:bs){ if(!x.stride||x.end==x.off)continue;
        uint32_t hb=fixLow?G_FACTOR.high_all_off[x.he]:G_FACTOR.high_mask_off[size_t(mask)*S+x.he];
        uint32_t hc=uint32_t((x.end-x.off)/x.stride);
        if(!fixLow){ for(uint32_t hr=0;hr<hc;++hr)addrun(st,x.stride); }
        else {
            uint32_t lb=G_FACTOR.low_mask_off[size_t(mask)*S+x.hs];
            for(uint32_t hr=0;hr<hc;++hr){
                (void)main_row_base(G_FACTOR.high_all_codes[hb+hr],x.he,x.c);
                uint64_t run=0;uint32_t prev=0;
                for(uint32_t lr=0;lr<x.stride;++lr){uint32_t code=G_FACTOR.low_mask_codes[lb+lr],ar=G_FACTOR.low_packed_rank[code]>>L;if(run&&ar==prev+1){++run;}else{addrun(st,run);run=1;}prev=ar;}addrun(st,run);
            }
        }
    } return st;
}
static IStat factor_block_stats(bool fixLow,uint32_t mask){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=FactorTablesHost::STRIDE;
    IStat st; auto bs=make_factor_block_blocks(fixLow,mask);
    for(auto const& x:bs){ if(!x.stride||x.end==x.off)continue;
        uint32_t hb=fixLow?G_FACTOR.high_all_off[x.he]:G_FACTOR.high_mask_off[size_t(mask)*S+x.he];
        uint32_t hc=uint32_t((x.end-x.off)/x.stride);
        if(!fixLow){for(uint32_t hr=0;hr<hc;++hr)addrun(st,x.stride);}
        else {uint32_t lb=G_FACTOR.low_mask_off[size_t(mask)*S+x.hs];for(uint32_t hr=0;hr<hc;++hr){(void)block_row_base(G_FACTOR.high_all_codes[hb+hr],x.he);uint64_t run=0;uint32_t prev=0;for(uint32_t lr=0;lr<x.stride;++lr){uint32_t code=G_FACTOR.low_mask_codes[lb+lr],ar=G_FACTOR.low_packed_rank[code]>>L;if(run&&ar==prev+1)++run;else{addrun(st,run);run=1;}prev=ar;}addrun(st,run);}}
    } return st;
}
int main(){build_full_dp();G_FACTOR=build_factor_tables();constexpr int W=TARGET_W;const int ranges[2][2]={{W-1,LOW_LUT_K+1},{LOW_LUT_K,1}};for(auto&r:ranges){int hi=r[0],lo=r[1];auto fp=window_candidates(W,hi,lo);bool fixLow=hi>LOW_LUT_K;uint64_t GI=0,GE=0,GM=0,DI=0,DE=0,DM=0;double minavg=1e100,maxavg=0;for(uint32_t g=0;g<(1u<<fp.size());++g){uint32_t mf,mo,bf,bo;window_masks(W,hi,lo,fp,g,mf,mo,bf,bo);uint32_t mask=fixLow?(mo&((1u<<LOW_LUT_K)-1u)):((mo>>(LOW_LUT_K+1))&((1u<<HIGH_LUT_K)-1u));auto a=factor_main_stats(fixLow,mask),b=factor_block_stats(fixLow,mask);GI+=a.intervals;GE+=a.elems;GM=std::max(GM,a.maxlen);DI+=b.intervals;DE+=b.elems;DM=std::max(DM,b.maxlen);if(a.intervals){double av=double(a.elems)/a.intervals;minavg=std::min(minavg,av);maxavg=std::max(maxavg,av);}}
std::cout<<"range="<<hi<<".."<<lo<<" fixLow="<<fixLow<<" groups="<<(1u<<fp.size())<<" main intervals="<<GI<<" elems="<<GE<<" avg="<<double(GE)/GI<<" max="<<GM<<" block intervals="<<DI<<" elems="<<DE<<" avg="<<double(DE)/DI<<" max="<<DM<<" group_main_avg_min="<<minavg<<" max="<<maxavg<<"\n";}}
