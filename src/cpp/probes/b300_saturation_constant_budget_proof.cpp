#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using Code=std::uint64_t;
static constexpr int MAXW=28;
struct Spec{Code dp[MAXW+1][MAXW+2]{};std::uint32_t fixed=0,occ=0;};

Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){
    Spec s;s.fixed=fixed;s.occ=occ;
    for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){int p=w-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<=MAXW;++h){Code z=0;if(!f||!o)z+=s.dp[w-1][h];if(!f||o){if(h>0)z+=s.dp[w-1][h-1];if(h<MAXW+1)z+=s.dp[w-1][h+1];}s.dp[w][h]=z;}}
    return s;
}

struct Bounds{std::int64_t step=0,pair=0;std::array<std::int64_t,3> slot{};std::uint64_t cases=0,patterns=0;};
Bounds scan_window(const std::vector<int>&fixed_pos,bool high_window){
    Bounds z;std::uint32_t mf=0;for(int q:fixed_pos)mf|=1u<<q;
    const std::uint32_t total=1u<<fixed_pos.size();
    for(std::uint32_t bits=0;bits<total;++bits){
        std::uint32_t mo=0;for(std::size_t k=0;k<fixed_pos.size();++k)if((bits>>k)&1u)mo|=1u<<fixed_pos[k];
        std::uint32_t bf=0,bo=0;
        for(int q:fixed_pos){const int bq=high_window?q-1:q;bf|=1u<<bq;if((mo>>q)&1u)bo|=1u<<bq;}
        const Spec ms=make_spec(28,mf,mo),bs=make_spec(27,bf,bo);++z.patterns;
        for(int p=1;p<28;++p)for(int h=0;h<=MAXW;++h){
            const Code mh=ms.dp[p][h],bh=bs.dp[p-1][h];
            const Code mhm=h?ms.dp[p][h-1]:0,bhm=h?bs.dp[p-1][h-1]:0;
            const std::int64_t stepR=std::int64_t(bh)-std::int64_t(mh);
            const std::int64_t stepL=std::int64_t(bh+bhm)-std::int64_t(mh+mhm);
            z.step=std::max(z.step,std::max(stepR<0?-stepR:stepR,stepL<0?-stepL:stepL));
            if(stepR<std::numeric_limits<std::int32_t>::min()||stepR>std::numeric_limits<std::int32_t>::max()||stepL<std::numeric_limits<std::int32_t>::min()||stepL>std::numeric_limits<std::int32_t>::max())std::exit(2);
            const Code low_h=ms.dp[p-1][h],low_hm=h?ms.dp[p-1][h-1]:0,low_hp=h<MAXW+1?ms.dp[p-1][h+1]:0;
            const std::int64_t pair[3]={-std::int64_t(mh+mhm+low_hp),std::int64_t(mh)-std::int64_t(low_h),std::int64_t(mh+mhm)-std::int64_t(low_h+low_hm)};
            for(int k=0;k<3;++k){const auto a=pair[k]<0?-pair[k]:pair[k];z.pair=std::max(z.pair,a);z.slot[k]=std::max(z.slot[k],a);if(pair[k]<std::numeric_limits<std::int32_t>::min()||pair[k]>std::numeric_limits<std::int32_t>::max())std::exit(3);}
            ++z.cases;
        }
    }
    return z;
}

int main(){
    std::vector<int> low13,high14;for(int q=0;q<13;++q)low13.push_back(q);for(int q=14;q<28;++q)high14.push_back(q);
    const Bounds low=scan_window(low13,false),high=scan_window(high14,true);
    if(low.step!=1060346729LL||low.pair!=1881935601LL||low.slot[0]!=1881935601LL||low.slot[1]!=539902168LL||low.slot[2]!=1060346729LL)return 4;
    if(high.step!=1060346729LL||high.pair!=2019358161LL||high.slot[0]!=2019358161LL||high.slot[1]!=392805229LL||high.slot[2]!=770685743LL)return 5;
    const auto step=std::max(low.step,high.step),pair=std::max(low.pair,high.pair);
    if(step!=1060346729LL||pair!=2019358161LL)return 6;

    // Exact explicit __constant__ footprint of the HBM32 base source plus both
    // experimental table families. Base non-DP symbols are 232 bytes.
    constexpr std::uint64_t base_dp=3ull*(MAXW+1)*(MAXW+2)*8;
    constexpr std::uint64_t base_other=232;
    constexpr std::uint64_t hot_step=1ull*(MAXW+1)*(MAXW+2)*2*4;
    constexpr std::uint64_t hot_pair=1ull*(MAXW+1)*(MAXW+2)*3*4;
    constexpr std::uint64_t closure_one=1ull*(MAXW+1)*(MAXW+2)*8;
    constexpr std::uint64_t closure_all=3*closure_one;
    constexpr std::uint64_t total=base_dp+base_other+hot_step+hot_pair+closure_all;
    static_assert(base_dp==20880&&base_other==232&&hot_step==6960&&hot_pair==10440&&closure_all==20880);
    static_assert(total==59392&&total<65536);
    std::cout<<"b300-saturation-constant-budget-proof OK production_width=28"
             <<" low_fixed_bits=13 low_occ_patterns="<<low.patterns
             <<" high_fixed_bits=14 high_occ_patterns="<<high.patterns
             <<" low_hot_step_abs_max="<<low.step<<" low_hot_pair_abs_max="<<low.pair
             <<" high_hot_step_abs_max="<<high.step<<" high_hot_pair_abs_max="<<high.pair
             <<" high_hot_pair_lr_abs_max="<<high.slot[0]<<" high_hot_pair_nr_abs_max="<<high.slot[1]<<" high_hot_pair_nl_abs_max="<<high.slot[2]
             <<" hot_step_abs_max="<<step<<" hot_pair_abs_max="<<pair
             <<" int32_exact=1 base_constant_bytes="<<(base_dp+base_other)
             <<" hot_delta_bytes="<<(hot_step+hot_pair)<<" closure_table_bytes="<<closure_all
             <<" total_constant_bytes="<<total<<" constant_headroom_bytes="<<(65536-total)<<" exact=1\n";
    return 0;
}
