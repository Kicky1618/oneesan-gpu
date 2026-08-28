#include "../../common/gridfp_transition.hpp"

#include <cstdint>
#include <iostream>
#include <random>

namespace {
using namespace oneesan::gridfp;

std::uint32_t non_n_mask(MateID mate, int width) {
    std::uint64_t x = (mate | (mate >> 1)) & 0x5555555555555555ULL;
    x = (x | (x >> 1)) & 0x3333333333333333ULL;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00ffULL;
    x = (x | (x >> 8)) & 0x0000ffff0000ffffULL;
    x = (x | (x >> 16)) & 0x00000000ffffffffULL;
    std::uint32_t out = static_cast<std::uint32_t>(x);
    if (width < 32) out &= (std::uint32_t(1) << width) - 1u;
    return out;
}

int legacy_left(MateID t, int p) {
    int q = p - 1, s = 1;
    while (s) {
        --q;
        if (q < 0) return -1;
        const auto v = mget(t, q);
        if (v == L) ++s; else if (v == R) --s;
    }
    return q;
}
int fast_left(MateID t, int p) {
    std::uint32_t mask = p <= 1 ? 0u :
        (non_n_mask(t, p - 1) & ((std::uint32_t(1) << (p - 1)) - 1u));
    int s = 1;
    while (mask) {
        const int q = 31 - __builtin_clz(mask);
        const auto v = mget(t, q);
        if (v == L) ++s; else if (v == R) --s;
        if (!s) return q;
        mask ^= std::uint32_t(1) << q;
    }
    return -1;
}
int legacy_right(MateID t, int width, int p) {
    int q = p, s = 1;
    while (s) {
        ++q;
        if (q >= width) return -1;
        const auto v = mget(t, q);
        if (v == L) --s; else if (v == R) ++s;
    }
    return q;
}
int fast_right(MateID t, int width, int p) {
    std::uint32_t mask = non_n_mask(t, width);
    const std::uint32_t low = p + 1 >= 32 ? ~0u :
        ((std::uint32_t(1) << (p + 1)) - 1u);
    mask &= ~low;
    int s = 1;
    while (mask) {
        const int q = __builtin_ctz(mask);
        const auto v = mget(t, q);
        if (v == L) --s; else if (v == R) ++s;
        if (!s) return q;
        mask &= mask - 1u;
    }
    return -1;
}

IncludeResult legacy_include(MateID m,int width,int p) {
    IncludeResult z{}; MateID t=m; const auto pair=mpair(m,p);
    switch(pair) {
    case NN: z.mate=msetpair(m,p,LR);z.valid=true;return z;
    case NR: case NL:
        if(p==1){z.mate=msetpair(m,p,pair==NR?RN:LN);z.valid=true;return z;}
        z.mate=mshrink(m,p);z.valid=true;z.blocked=true;return z;
    case RN: z.mate=msetpair(m,p,NR);z.valid=true;return z;
    case LN: z.mate=msetpair(m,p,NL);z.valid=true;return z;
    case LL:{
        t=msetpair(m,p,NN);const int q=legacy_left(t,p);if(q<0)return z;
        t=mset(t,q,L);if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RR:{
        t=msetpair(m,p,NN);const int q=legacy_right(t,width,p);if(q<0)return z;
        t=mset(t,q,R);if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RL:
        t=msetpair(m,p,NN);if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    default:return z;
    }
}
IncludeResult fast_include(MateID m,int width,int p) {
    IncludeResult z{}; MateID t=m; const auto pair=mpair(m,p);
    switch(pair) {
    case NN: z.mate=msetpair(m,p,LR);z.valid=true;return z;
    case NR: case NL:
        if(p==1){z.mate=msetpair(m,p,pair==NR?RN:LN);z.valid=true;return z;}
        z.mate=mshrink(m,p);z.valid=true;z.blocked=true;return z;
    case RN: z.mate=msetpair(m,p,NR);z.valid=true;return z;
    case LN: z.mate=msetpair(m,p,NL);z.valid=true;return z;
    case LL:{
        t=msetpair(m,p,NN);const int q=fast_left(t,p);if(q<0)return z;
        t=mset(t,q,L);if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RR:{
        t=msetpair(m,p,NN);const int q=fast_right(t,width,p);if(q<0)return z;
        t=mset(t,q,R);if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    }
    case RL:
        t=msetpair(m,p,NN);if(p==1){z.mate=t;z.valid=true;return z;}
        z.mate=mshrink(t,p-1);z.valid=true;z.blocked=true;return z;
    default:return z;
    }
}
bool same(const IncludeResult&a,const IncludeResult&b){return a.mate==b.mate&&a.valid==b.valid&&a.blocked==b.blocked;}
}

int main(){
    std::uint64_t exhaustive=0;
    for(int width=2;width<=8;++width){
        const MateID limit=MateID(1)<<(2*width);
        for(MateID m=0;m<limit;++m)for(int p=1;p<width;++p){
            ++exhaustive;
            MateID t=msetpair(m,p,NN);
            if(legacy_left(t,p)!=fast_left(t,p)||legacy_right(t,width,p)!=fast_right(t,width,p)||!same(legacy_include(m,width,p),fast_include(m,width,p))){
                std::cerr<<"mismatch width="<<width<<" p="<<p<<" mate="<<m<<'\n';return 2;
            }
        }
    }
    std::mt19937_64 rng(0x636c6f737572656eULL);constexpr std::uint64_t RANDOM=1000000;
    for(std::uint64_t i=0;i<RANDOM;++i){
        const int width=2+int(rng()%27),p=1+int(rng()%(width-1));
        const MateID m=rng()&((MateID(1)<<(2*width))-1);MateID t=msetpair(m,p,NN);
        if(legacy_left(t,p)!=fast_left(t,p)||legacy_right(t,width,p)!=fast_right(t,width,p)||!same(legacy_include(m,width,p),fast_include(m,width,p)))return 3;
    }
    std::cout<<"gridfp-closure-nonn-scan-proof OK exhaustive_width_max=8 exhaustive_cases="<<exhaustive<<" random_cases="<<RANDOM<<" production_width_max=28 non_n_scan_exact=1 include_exact=1\n";
    return 0;
}
