#pragma once

#include "gridfp_transition.hpp"
#include <cstdint>

#if defined(__CUDACC__)
#define ONEESAN_PAT10_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_PAT10_HD inline
#endif

namespace oneesan::gridfp {

static constexpr std::uint16_t CLOSURE_PATTERN10_NONE = 1023u;

// Number of length-m binary masks with no adjacent 1 bits = F_{m+2}.
ONEESAN_PAT10_HD constexpr std::uint16_t closure_fib_count(int m) {
    std::uint16_t a=1,b=2; // m=0,m=1
    for(int i=0;i<m;++i){if(i==0)continue;std::uint16_t c=std::uint16_t(a+b);a=b;b=c;}
    return m==0?1:b;
}

// Rank/unrank no-adjacent masks in MSB-lexicographic order. The rank range is
// [0,F_{m+2}).
ONEESAN_PAT10_HD std::uint16_t closure_noadj_rank(std::uint16_t mask,int m){
    std::uint16_t rank=0;bool prev=false;
    for(int i=m-1;i>=0;--i){
        bool bit=((mask>>i)&1u)!=0;
        if(prev){prev=false;continue;}
        if(bit){rank=std::uint16_t(rank+closure_fib_count(i));prev=true;}
    }
    return rank;
}
ONEESAN_PAT10_HD std::uint16_t closure_noadj_unrank(std::uint16_t rank,int m){
    std::uint16_t mask=0;bool prev=false;
    for(int i=m-1;i>=0;--i){
        if(prev){prev=false;continue;}
        std::uint16_t n0=closure_fib_count(i);
        if(rank>=n0){rank=std::uint16_t(rank-n0);mask=std::uint16_t(mask|(std::uint16_t(1u)<<i));prev=true;}
    }
    return mask;
}

// Encode the ordinary LL/RR remote-candidate positions of a destination whose
// active pair is NN. Bit 0 of each side mask denotes the position nearest the
// pair. RL is implicit and therefore consumes no pattern bit.
ONEESAN_PAT10_HD std::uint16_t closure_pattern10_encode(MateID d,int len,int p){
    if(p<=0||p>=len||mpair(d,p)!=NN)return CLOSURE_PATTERN10_NONE;
    std::uint16_t lm=0,rm=0;int bal=0,bi=0;
    for(int q=p-2;q>=0;--q,++bi){MateValue v=mget(d,q);if(bal==0&&v==L)lm=std::uint16_t(lm|(std::uint16_t(1u)<<bi));if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}
    bal=0;bi=0;
    for(int q=p+1;q<len;++q,++bi){MateValue v=mget(d,q);if(bal==0&&v==R)rm=std::uint16_t(rm|(std::uint16_t(1u)<<bi));if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}
    int l=p-1,r=len-p-1;std::uint16_t rc=closure_fib_count(r);
    std::uint32_t z=std::uint32_t(closure_noadj_rank(lm,l))*rc+closure_noadj_rank(rm,r);
    return z<CLOSURE_PATTERN10_NONE?std::uint16_t(z):CLOSURE_PATTERN10_NONE;
}

ONEESAN_PAT10_HD void closure_pattern10_decode(std::uint16_t z,int len,int p,std::uint16_t&lm,std::uint16_t&rm){
    lm=rm=0;if(z==CLOSURE_PATTERN10_NONE||p<=0||p>=len)return;
    int l=p-1,r=len-p-1;std::uint16_t rc=closure_fib_count(r);
    lm=closure_noadj_unrank(std::uint16_t(z/rc),l);
    rm=closure_noadj_unrank(std::uint16_t(z%rc),r);
}

ONEESAN_PAT10_HD constexpr std::uint16_t closure_pattern10_count(int len,int p){
    return std::uint16_t(closure_fib_count(p-1)*closure_fib_count(len-p-1));
}

} // namespace oneesan::gridfp

#undef ONEESAN_PAT10_HD
