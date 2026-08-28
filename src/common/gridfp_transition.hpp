#pragma once
#include <cstdint>

#ifndef ONEESAN_FAST_CLOSURE_NONN_SCAN
#define ONEESAN_FAST_CLOSURE_NONN_SCAN 1
#endif
static_assert(ONEESAN_FAST_CLOSURE_NONN_SCAN == 0 ||
              ONEESAN_FAST_CLOSURE_NONN_SCAN == 1,
              "ONEESAN_FAST_CLOSURE_NONN_SCAN must be 0 or 1");

#if defined(__CUDACC__)
#define ONEESAN_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_HD inline
#endif

namespace oneesan::gridfp {
using MateID = std::uint64_t;

enum MateValue : std::uint8_t { N=0, R=1, L=2, X=3 };
enum MateValuePair : std::uint8_t {
    NN=0x0, NR=0x1, NL=0x2, NX=0x3,
    RN=0x4, RR=0x5, RL=0x6, RX=0x7,
    LN=0x8, LR=0x9, LL=0xa, LX=0xb,
    XN=0xc, XR=0xd, XL=0xe, XX=0xf
};

ONEESAN_HD MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
ONEESAN_HD MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
ONEESAN_HD MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
ONEESAN_HD MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
ONEESAN_HD MateID mshrink(MateID m,int k){MateID mask=(MateID(1)<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);}
ONEESAN_HD MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((MateID(1)<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}

ONEESAN_HD std::uint32_t mate_non_n_mask(MateID mate,int width){
    std::uint64_t x=(mate|(mate>>1))&0x5555555555555555ULL;
    x=(x|(x>>1))&0x3333333333333333ULL;
    x=(x|(x>>2))&0x0f0f0f0f0f0f0f0fULL;
    x=(x|(x>>4))&0x00ff00ff00ff00ffULL;
    x=(x|(x>>8))&0x0000ffff0000ffffULL;
    x=(x|(x>>16))&0x00000000ffffffffULL;
    std::uint32_t out=static_cast<std::uint32_t>(x);
    if(width<32)out&=(std::uint32_t(1)<<width)-1u;
    return out;
}

ONEESAN_HD int mate_msb_index32(std::uint32_t x){
#if defined(__CUDA_ARCH__)
    return 31-__clz(x);
#elif defined(__GNUC__) || defined(__clang__)
    return 31-__builtin_clz(x);
#else
    int q=0;while(x>>1){x>>=1;++q;}return q;
#endif
}
ONEESAN_HD int mate_lsb_index32(std::uint32_t x){
#if defined(__CUDA_ARCH__)
    return __ffs(x)-1;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctz(x);
#else
    int q=0;while((x&1u)==0){x>>=1;++q;}return q;
#endif
}

ONEESAN_HD int closure_match_left(MateID t,int p){
#if ONEESAN_FAST_CLOSURE_NONN_SCAN
    std::uint32_t mask=p<=1?0u:
        (mate_non_n_mask(t,p-1)&((std::uint32_t(1)<<(p-1))-1u));
    int s=1;
    while(mask){
        const int q=mate_msb_index32(mask);const MateValue v=mget(t,q);
        if(v==L)++s;else if(v==R)--s;
        if(!s)return q;
        mask^=std::uint32_t(1)<<q;
    }
    return -1;
#else
    int q=p-1,s=1;
    while(s){--q;if(q<0)return -1;const MateValue v=mget(t,q);if(v==L)++s;else if(v==R)--s;}
    return q;
#endif
}

ONEESAN_HD int closure_match_right(MateID t,int width,int p){
#if ONEESAN_FAST_CLOSURE_NONN_SCAN
    std::uint32_t mask=mate_non_n_mask(t,width);
    const std::uint32_t low=p+1>=32?~0u:((std::uint32_t(1)<<(p+1))-1u);
    mask&=~low;
    int s=1;
    while(mask){
        const int q=mate_lsb_index32(mask);const MateValue v=mget(t,q);
        if(v==L)--s;else if(v==R)++s;
        if(!s)return q;
        mask&=mask-1u;
    }
    return -1;
#else
    int q=p,s=1;
    while(s){++q;if(q>=width)return -1;const MateValue v=mget(t,q);if(v==L)--s;else if(v==R)++s;}
    return q;
#endif
}

struct IncludeResult {
    MateID mate = 0;
    bool valid = false;
    bool blocked = false;
};

// Semantics of the INCLUDED horizontal-edge branch in the production Grid-FP
// kernel. The excluded branch is identity for a main state.
ONEESAN_HD IncludeResult include_horizontal(MateID m,int width,int p){
    IncludeResult z{};MateID t=m;const MateValuePair pair=mpair(m,p);
    switch(pair){
    case NN:
        z.mate=msetpair(m,p,LR);z.valid=true;return z;
    case NR: case NL:
        if(p==1){z.mate=msetpair(m,p,pair==NR?RN:LN);z.valid=true;return z;}
        z.mate=mshrink(m,p);z.valid=true;z.blocked=true;return z;
    case RN:
        z.mate=msetpair(m,p,NR);z.valid=true;return z;
    case LN:
        z.mate=msetpair(m,p,NL);z.valid=true;return z;
    case LL:{
        t=msetpair(m,p,NN);const int q=closure_match_left(t,p);if(q<0)return z;
        t=mset(t,q,L);
        if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RR:{
        t=msetpair(m,p,NN);const int q=closure_match_right(t,width,p);if(q<0)return z;
        t=mset(t,q,R);
        if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RL:
        t=msetpair(m,p,NN);
        if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    default:
        return z;
    }
}

ONEESAN_HD MateID blocked_exclude(MateID compressed,int p){return minsert(compressed,p,N);}
ONEESAN_HD bool is_endpoint(MateValue v){return v==L||v==R;}

} // namespace oneesan::gridfp

#undef ONEESAN_HD
