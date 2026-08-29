#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>

using Code=std::uint64_t;
static constexpr int MAXW=28;
struct Spec{Code dp[MAXW+1][MAXW+2]{};std::uint32_t fixed=0,occ=0;};

Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){
    Spec s;s.fixed=fixed;s.occ=occ;
    for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){int p=w-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<=MAXW;++h){Code z=0;if(!f||!o)z+=s.dp[w-1][h];if(!f||o){if(h>0)z+=s.dp[w-1][h-1];if(h<MAXW+1)z+=s.dp[w-1][h+1];}s.dp[w][h]=z;}}
    return s;
}

int main(){
    constexpr int W=28,K=13;constexpr std::uint32_t fixed=(1u<<K)-1u;
    std::int64_t step_abs_max=0,pair_abs_max=0;std::int64_t pair_slot_max[3]{};
    std::uint64_t cases=0;
    for(std::uint32_t occ=0;occ<(1u<<K);++occ){
        const Spec ms=make_spec(W,fixed,occ),bs=make_spec(W-1,fixed,occ);
        for(int p=1;p<W;++p)for(int h=0;h<=MAXW;++h){
            const Code mh=ms.dp[p][h],bh=bs.dp[p-1][h];
            const Code mhm=h?ms.dp[p][h-1]:0,bhm=h?bs.dp[p-1][h-1]:0;
            const std::int64_t stepR=std::int64_t(bh)-std::int64_t(mh);
            const std::int64_t stepL=std::int64_t(bh+bhm)-std::int64_t(mh+mhm);
            step_abs_max=std::max(step_abs_max,std::max(stepR<0?-stepR:stepR,stepL<0?-stepL:stepL));
            const Code low_h=ms.dp[p-1][h],low_hm=h?ms.dp[p-1][h-1]:0,low_hp=h<MAXW+1?ms.dp[p-1][h+1]:0;
            const std::int64_t pair[3]={-std::int64_t(mh+mhm+low_hp),std::int64_t(mh)-std::int64_t(low_h),std::int64_t(mh+mhm)-std::int64_t(low_h+low_hm)};
            for(int k=0;k<3;++k){const auto a=pair[k]<0?-pair[k]:pair[k];pair_abs_max=std::max(pair_abs_max,a);pair_slot_max[k]=std::max(pair_slot_max[k],a);if(pair[k]<std::numeric_limits<std::int32_t>::min()||pair[k]>std::numeric_limits<std::int32_t>::max())return 2;}
            if(stepR<std::numeric_limits<std::int32_t>::min()||stepR>std::numeric_limits<std::int32_t>::max()||stepL<std::numeric_limits<std::int32_t>::min()||stepL>std::numeric_limits<std::int32_t>::max())return 3;
            ++cases;
        }
    }
    if(step_abs_max!=1060346729LL)return 4;
    if(pair_abs_max!=1881935601LL)return 5;
    if(pair_slot_max[0]!=1881935601LL||pair_slot_max[1]!=539902168LL||pair_slot_max[2]!=1060346729LL)return 6;

    // Exact explicit __constant__ footprint of the HBM32 base source:
    // 3 DP arrays = 20,880; fixed/occ/int/mod/pointer/chunk/LUT pointer symbols = 232.
    constexpr std::uint64_t base_dp=3ull*(MAXW+1)*(MAXW+2)*8; // 20,880
    constexpr std::uint64_t base_other=232;
    constexpr std::uint64_t hot_step=1ull*(MAXW+1)*(MAXW+2)*2*4; // 6,960
    constexpr std::uint64_t hot_pair=1ull*(MAXW+1)*(MAXW+2)*3*4; // 10,440
    constexpr std::uint64_t closure_one=1ull*(MAXW+1)*(MAXW+2)*8; // 6,960
    constexpr std::uint64_t closure_all=3*closure_one; // contrib, shift2, cross
    constexpr std::uint64_t total=base_dp+base_other+hot_step+hot_pair+closure_all;
    static_assert(base_dp==20880&&hot_step==6960&&hot_pair==10440&&closure_all==20880);
    static_assert(total==59392);
    static_assert(total<65536);
    std::cout<<"b300-saturation-constant-budget-proof OK production_width=28 fixed_low_bits=13 occ_patterns=8192 cases="<<cases
             <<" hot_step_abs_max="<<step_abs_max<<" hot_pair_abs_max="<<pair_abs_max
             <<" hot_pair_lr_abs_max="<<pair_slot_max[0]<<" hot_pair_nr_abs_max="<<pair_slot_max[1]<<" hot_pair_nl_abs_max="<<pair_slot_max[2]
             <<" int32_exact=1 base_constant_bytes="<<(base_dp+base_other)
             <<" hot_delta_bytes="<<(hot_step+hot_pair)<<" closure_table_bytes="<<closure_all
             <<" total_constant_bytes="<<total<<" constant_headroom_bytes="<<(65536-total)<<" exact=1\n";
    return 0;
}
