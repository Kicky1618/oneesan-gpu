#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

using Code=std::uint64_t;
enum MateValue:std::uint8_t{N=0,R=1,L=2,X=3};
static constexpr int MAXW=28;

struct Spec{
    int width=0;std::uint32_t fixed=0,occ=0;
    Code dp[MAXW+1][MAXW+2]{};
};

bool allowed(std::uint32_t fixed,std::uint32_t occ,int pos,MateValue v){
    if(!((fixed>>pos)&1u))return v!=X;
    return ((occ>>pos)&1u)?(v==R||v==L):(v==N);
}

Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){
    Spec s;s.width=width;s.fixed=fixed;s.occ=occ;
    for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){
        int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;
        for(int h=0;h<=MAXW;++h){
            Code z=0;
            if(!f||!o)z+=s.dp[w-1][h];
            if(!f||o){if(h>0)z+=s.dp[w-1][h-1];if(h<MAXW+1)z+=s.dp[w-1][h+1];}
            s.dp[w][h]=z;
        }
    }
    return s;
}

Code old_contrib(const Spec&s,MateValue v,int pos,int h){
    if(h<0||h>MAXW+1)return 0;
    Code z=0;
    if(v>N&&allowed(s.fixed,s.occ,pos,N))z+=s.dp[pos][h];
    if(v>R&&h>0&&allowed(s.fixed,s.occ,pos,R))z+=s.dp[pos][h-1];
    return z;
}

std::array<Code,2> table_entry(const Spec&s,int pos,int h){
    Code nbranch=0,rbranch=0;
    if(h>=0&&h<=MAXW+1&&allowed(s.fixed,s.occ,pos,N))nbranch=s.dp[pos][h];
    if(h>0&&h<=MAXW+1&&allowed(s.fixed,s.occ,pos,R))rbranch=s.dp[pos][h-1];
    return{nbranch,nbranch+rbranch};
}

int main(){
    std::mt19937_64 rng(0x636c6f7375726574ULL);
    std::uint64_t cases=0,specs=0;
    for(int W=4;W<=12;++W){
        const std::uint32_t mask=(1u<<W)-1u;
        int k=std::min(W,6);
        for(std::uint32_t fsmall=0;fsmall<(1u<<k);++fsmall){
            for(std::uint32_t osmall=0;osmall<(1u<<k);++osmall){
                std::uint32_t fixed=fsmall,occ=osmall&fsmall;
                if(W>k){
                    std::uint32_t hi=std::uint32_t(rng())&mask&~((1u<<k)-1u);
                    std::uint32_t ho=std::uint32_t(rng())&hi;
                    fixed|=hi;occ|=ho;
                }
                Spec s=make_spec(W,fixed,occ);++specs;
                for(int pos=0;pos<W;++pos)for(int h=0;h<=MAXW+1;++h){
                    auto a=table_entry(s,pos,h),b=table_entry(s,pos,h+2);
                    if(old_contrib(s,N,pos,h)!=0)return 2;
                    if(old_contrib(s,R,pos,h)!=a[0])return 3;
                    if(old_contrib(s,L,pos,h)!=a[1])return 4;
                    const std::int64_t dr=std::int64_t(b[0])-std::int64_t(a[0]);
                    const std::int64_t dl=std::int64_t(b[1])-std::int64_t(a[1]);
                    if(dr!=std::int64_t(old_contrib(s,R,pos,h+2))-std::int64_t(old_contrib(s,R,pos,h)))return 5;
                    if(dl!=std::int64_t(old_contrib(s,L,pos,h+2))-std::int64_t(old_contrib(s,L,pos,h)))return 6;
                    const std::int64_t xll=std::int64_t(b[0])-std::int64_t(a[1]);
                    const std::int64_t xrr=std::int64_t(a[1])-std::int64_t(a[0]);
                    if(xll!=std::int64_t(old_contrib(s,R,pos,h+2))-std::int64_t(old_contrib(s,L,pos,h)))return 7;
                    if(xrr!=std::int64_t(old_contrib(s,L,pos,h))-std::int64_t(old_contrib(s,R,pos,h)))return 8;
                    cases+=7;
                }
            }
        }
    }

    // Production n=27 planner split: MAX_WINDOW=14 fixes low 13 positions.
    // Exhaust all 8192 occupancy masks and establish the exact packed bounds.
    constexpr int PW=28,PK=13;
    constexpr std::uint32_t fixed=(1u<<PK)-1u;
    Code prod_contrib_max=0;
    std::int64_t prod_shift_abs_max=0,prod_cross_abs_max=0;
    std::int64_t prod_cross_ll_min=0,prod_cross_ll_max=0;
    std::int64_t prod_cross_rr_min=0,prod_cross_rr_max=0;
    bool first=true;
    std::uint64_t prod_entries=0;
    for(std::uint32_t occ=0;occ<(1u<<PK);++occ){
        Spec s=make_spec(PW,fixed,occ);
        for(int pos=0;pos<PW;++pos)for(int h=0;h<=MAXW+1;++h){
            const auto a=table_entry(s,pos,h),b=table_entry(s,pos,h+2);
            for(int k=0;k<2;++k){
                prod_contrib_max=std::max(prod_contrib_max,a[k]);
                const std::int64_t d=std::int64_t(b[k])-std::int64_t(a[k]);
                prod_shift_abs_max=std::max(prod_shift_abs_max,d<0?-d:d);
                if(a[k]>std::numeric_limits<std::uint32_t>::max())return 9;
                if(d<std::numeric_limits<std::int32_t>::min()||d>std::numeric_limits<std::int32_t>::max())return 10;
                ++prod_entries;
            }
            const std::int64_t xll=std::int64_t(b[0])-std::int64_t(a[1]);
            const std::int64_t xrr=std::int64_t(a[1])-std::int64_t(a[0]);
            if(xll<std::numeric_limits<std::int32_t>::min()||xll>std::numeric_limits<std::int32_t>::max())return 11;
            if(xrr<std::numeric_limits<std::int32_t>::min()||xrr>std::numeric_limits<std::int32_t>::max())return 12;
            prod_cross_abs_max=std::max(prod_cross_abs_max,std::max(xll<0?-xll:xll,xrr<0?-xrr:xrr));
            if(first){prod_cross_ll_min=prod_cross_ll_max=xll;prod_cross_rr_min=prod_cross_rr_max=xrr;first=false;}
            else{
                prod_cross_ll_min=std::min(prod_cross_ll_min,xll);prod_cross_ll_max=std::max(prod_cross_ll_max,xll);
                prod_cross_rr_min=std::min(prod_cross_rr_min,xrr);prod_cross_rr_max=std::max(prod_cross_rr_max,xrr);
            }
        }
    }
    if(prod_contrib_max!=1615814681ULL)return 13;
    if(prod_shift_abs_max!=928513911LL)return 14;
    if(prod_cross_abs_max!=1029686463LL)return 15;
    if(prod_cross_ll_min!=-1029686463LL)return 16;
    if(prod_cross_rr_max!=821588872LL)return 17;

    constexpr std::uint64_t dp_bytes=3ull*(MAXW+1)*(MAXW+2)*sizeof(Code);
    constexpr std::uint64_t one_table_bytes=1ull*(MAXW+1)*(MAXW+2)*sizeof(std::uint64_t);
    constexpr std::uint64_t contrib_bytes=one_table_bytes;
    constexpr std::uint64_t shift_bytes=one_table_bytes;
    constexpr std::uint64_t cross_bytes=one_table_bytes;
    constexpr std::uint64_t table_bytes=contrib_bytes+shift_bytes+cross_bytes;
    constexpr std::uint64_t known=dp_bytes+table_bytes;
    static_assert(known<65536,"closure contribution tables exceed constant memory");
    std::cout<<"b300-closure-contrib-table-proof OK specs="<<specs
             <<" cases="<<cases
             <<" semantics=lexicographic_prefix_mass exact=1"
             <<" packed32=1 shift2_i32=1 cross_i32=1"
             <<" production_width=28 production_fixed_low_bits=13 production_occ_patterns=8192"
             <<" production_entries="<<prod_entries
             <<" production_contrib_max="<<prod_contrib_max
             <<" production_shift_abs_max="<<prod_shift_abs_max
             <<" production_cross_abs_max="<<prod_cross_abs_max
             <<" production_cross_ll_min="<<prod_cross_ll_min
             <<" production_cross_ll_max="<<prod_cross_ll_max
             <<" production_cross_rr_min="<<prod_cross_rr_min
             <<" production_cross_rr_max="<<prod_cross_rr_max
             <<" dp_bytes="<<dp_bytes<<" contrib_bytes="<<contrib_bytes
             <<" shift_bytes="<<shift_bytes<<" cross_bytes="<<cross_bytes
             <<" table_bytes="<<table_bytes
             <<" known_constant_bytes="<<known
             <<" known_headroom_bytes="<<(65536-known)<<'\n';
    return 0;
}
