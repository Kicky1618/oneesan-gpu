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
    if(m<=0)return 1;
    std::uint16_t a=1,b=2; // m=0,m=1
    for(int i=1;i<m;++i){std::uint16_t c=std::uint16_t(a+b);a=b;b=c;}
    return b;
}

// For a mask of length m, return F_{m+1} and F_m in the same shifted
// sequence used by closure_fib_count.  Walking these counts backwards lets
// rank/unrank remain O(m), rather than calling closure_fib_count(i) inside an
// O(m) loop and becoming O(m^2) per pattern decode.
ONEESAN_PAT10_HD void closure_noadj_top_counts(int m,std::uint16_t&hi,std::uint16_t&lo){
    if(m<=0){hi=lo=0;return;}
    hi=1;lo=0; // m=1: count at bit 0 is closure_fib_count(0)=1
    if(m==1)return;
    lo=1;hi=2; // m=2: counts for bit 1 / bit 0
    for(int i=2;i<m;++i){std::uint16_t next=std::uint16_t(lo+hi);lo=hi;hi=next;}
}

// Rank/unrank no-adjacent masks in MSB-lexicographic order. The rank range is
// [0,F_{m+2}).
ONEESAN_PAT10_HD std::uint16_t closure_noadj_rank(std::uint16_t mask,int m){
    if(m<=0)return 0;
    std::uint16_t hi=0,lo=0;closure_noadj_top_counts(m,hi,lo);
    std::uint16_t rank=0;bool prev=false;
    for(int i=m-1;i>=0;--i){
        bool bit=((mask>>i)&1u)!=0;
        if(prev)prev=false;
        else if(bit){rank=std::uint16_t(rank+hi);prev=true;}
        if(i>0){std::uint16_t next_hi=lo,next_lo=std::uint16_t(hi-lo);hi=next_hi;lo=next_lo;}
    }
    return rank;
}
ONEESAN_PAT10_HD std::uint16_t closure_noadj_unrank(std::uint16_t rank,int m){
    if(m<=0)return 0;
    std::uint16_t hi=0,lo=0;closure_noadj_top_counts(m,hi,lo);
    std::uint16_t mask=0;bool prev=false;
    for(int i=m-1;i>=0;--i){
        if(prev)prev=false;
        else if(rank>=hi){rank=std::uint16_t(rank-hi);mask=std::uint16_t(mask|(std::uint16_t(1u)<<i));prev=true;}
        if(i>0){std::uint16_t next_hi=lo,next_lo=std::uint16_t(hi-lo);hi=next_hi;lo=next_lo;}
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
