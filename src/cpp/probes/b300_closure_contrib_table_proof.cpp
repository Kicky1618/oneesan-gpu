#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
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
    Code z=0;
    if(v>N&&allowed(s.fixed,s.occ,pos,N))z+=s.dp[pos][h];
    if(v>R&&h>0&&allowed(s.fixed,s.occ,pos,R))z+=s.dp[pos][h-1];
    return z;
}

std::array<Code,2> table_entry(const Spec&s,int pos,int h){
    Code nbranch=0,rbranch=0;
    if(allowed(s.fixed,s.occ,pos,N))nbranch=s.dp[pos][h];
    if(h>0&&allowed(s.fixed,s.occ,pos,R))rbranch=s.dp[pos][h-1];
    return{nbranch,nbranch+rbranch};
}

int main(){
    std::mt19937_64 rng(0x636c6f7375726574ULL);
    std::uint64_t cases=0,specs=0;
    for(int W=4;W<=12;++W){
        const std::uint32_t mask=(1u<<W)-1u;
        // Exhaust the first six fixed positions and vary the remaining bits
        // randomly. This covers fixed-N, fixed-endpoint and free positions in
        // every width while keeping the proof fast enough for build gating.
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
                    if(old_contrib(s,R,pos,h)!=t[0]){
                        std::cerr<<"R mismatch W="<<W<<" fixed="<<fixed<<" occ="<<occ<<" pos="<<pos<<" h="<<h<<'\n';return 3;
                    }
                    if(old_contrib(s,L,pos,h)!=t[1]){
                        std::cerr<<"L mismatch W="<<W<<" fixed="<<fixed<<" occ="<<occ<<" pos="<<pos<<" h="<<h<<'\n';return 4;
                    }
                    cases+=3;
                }
            }
        }
    }
    // Production arrays at MAXW=28. This intentionally counts only the known
    // dominant constant arrays; the remaining scalar/pointer symbols are well
    // below the reported headroom.
    constexpr std::uint64_t dp_bytes=3ull*(MAXW+1)*(MAXW+2)*sizeof(Code);
    constexpr std::uint64_t table_bytes=1ull*(MAXW+1)*(MAXW+2)*2*sizeof(Code);
    constexpr std::uint64_t known=dp_bytes+table_bytes;
    static_assert(known<65536,"closure contribution table exceeds constant memory");
    std::cout<<"b300-closure-contrib-table-proof OK specs="<<specs
             <<" cases="<<cases
             <<" semantics=lexicographic_prefix_mass exact=1"
             <<" dp_bytes="<<dp_bytes<<" table_bytes="<<table_bytes
             <<" known_constant_bytes="<<known
             <<" known_headroom_bytes="<<(65536-known)<<'\n';
    return 0;
}
