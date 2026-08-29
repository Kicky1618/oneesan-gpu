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
                    auto t=table_entry(s,pos,h);
                    if(old_contrib(s,N,pos,h)!=0)return 2;
                    if(old_contrib(s,R,pos,h)!=t[0])return 3;
                    if(old_contrib(s,L,pos,h)!=t[1])return 4;
                    const auto t2=table_entry(s,pos,h+2);
                    const std::int64_t dr=std::int64_t(t2[0])-std::int64_t(t[0]);
                    const std::int64_t dl=std::int64_t(t2[1])-std::int64_t(t[1]);
                    if(dr!=std::int64_t(old_contrib(s,R,pos,h+2))-std::int64_t(old_contrib(s,R,pos,h)))return 5;
                    if(dl!=std::int64_t(old_contrib(s,L,pos,h+2))-std::int64_t(old_contrib(s,L,pos,h)))return 6;
                    cases+=5;
                }
            }
        }
    }

    // Production n=27 planner split: MAX_WINDOW=14 fixes the low 13 physical
    // positions in the first wide window. Exhaust all 2^13 occupancy patterns
    // and prove both packed representations fit their runtime types.
    constexpr int PW=28,PK=13;
    constexpr std::uint32_t fixed=(1u<<PK)-1u;
    Code prod_contrib_max=0;std::int64_t prod_shift_abs_max=0;
    std::uint64_t prod_entries=0;
    for(std::uint32_t occ=0;occ<(1u<<PK);++occ){
        Spec s=make_spec(PW,fixed,occ);
        for(int pos=0;pos<PW;++pos)for(int h=0;h<=MAXW+1;++h){
            const auto a=table_entry(s,pos,h),b=table_entry(s,pos,h+2);
            for(int k=0;k<2;++k){
                prod_contrib_max=std::max(prod_contrib_max,a[k]);
                const std::int64_t d=std::int64_t(b[k])-std::int64_t(a[k]);
                prod_shift_abs_max=std::max(prod_shift_abs_max,d<0?-d:d);
                if(a[k]>std::numeric_limits<std::uint32_t>::max())return 7;
                if(d<std::numeric_limits<std::int32_t>::min()||d>std::numeric_limits<std::int32_t>::max())return 8;
                ++prod_entries;
            }
        }
    }
    if(prod_contrib_max!=1615814681ULL)return 9;
    if(prod_shift_abs_max!=928513911LL)return 10;

    constexpr std::uint64_t dp_bytes=3ull*(MAXW+1)*(MAXW+2)*sizeof(Code);
    constexpr std::uint64_t contrib_bytes=1ull*(MAXW+1)*(MAXW+2)*sizeof(std::uint64_t);
    constexpr std::uint64_t shift_bytes=contrib_bytes;
    constexpr std::uint64_t table_bytes=contrib_bytes+shift_bytes;
    constexpr std::uint64_t known=dp_bytes+table_bytes;
    static_assert(known<65536,"closure contribution tables exceed constant memory");
    std::cout<<"b300-closure-contrib-table-proof OK specs="<<specs
             <<" cases="<<cases
             <<" semantics=lexicographic_prefix_mass exact=1"
             <<" packed32=1 shift2_i32=1"
             <<" production_width=28 production_fixed_low_bits=13 production_occ_patterns=8192"
             <<" production_entries="<<prod_entries
             <<" production_contrib_max="<<prod_contrib_max
             <<" production_shift_abs_max="<<prod_shift_abs_max
             <<" dp_bytes="<<dp_bytes<<" contrib_bytes="<<contrib_bytes
             <<" shift_bytes="<<shift_bytes<<" table_bytes="<<table_bytes
             <<" known_constant_bytes="<<known
             <<" known_headroom_bytes="<<(65536-known)<<'\n';
    return 0;
}
