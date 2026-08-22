#pragma once
#include <cstdint>

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

struct IncludeResult {
    MateID mate = 0;
    bool valid = false;
    bool blocked = false;
};

// Semantics of the INCLUDED horizontal-edge branch in the production Grid-FP
// kernel.  The excluded branch is identity for a main state.  A blocked state
// has no included branch and its excluded branch is blocked_exclude().
ONEESAN_HD IncludeResult include_horizontal(MateID m,int width,int p){
    IncludeResult z{};
    MateID t=m;
    switch(mpair(m,p)){
    case NN:
        z.mate=msetpair(m,p,LR);z.valid=true;return z;
    case NR: case NL:
        if(p==1){z.mate=msetpair(m,p,mpair(m,p)==NR?RN:LN);z.valid=true;return z;}
        z.mate=mshrink(m,p);z.valid=true;z.blocked=true;return z;
    case RN:
        z.mate=msetpair(m,p,NR);z.valid=true;return z;
    case LN:
        z.mate=msetpair(m,p,NL);z.valid=true;return z;
    case LL:{
        t=msetpair(m,p,NN);int q=p-1,s=1;
        while(s){--q;if(q<0)return z;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}
        t=mset(t,q,L);
        if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RR:{
        t=msetpair(m,p,NN);int q=p,s=1;
        while(s){++q;if(q>=width)return z;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}
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
