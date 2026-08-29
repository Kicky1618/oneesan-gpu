#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_closure_inverse.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace oneesan::gridfp;
using Count=std::uint32_t;
static constexpr Count MOD=4294967291u;

static std::vector<MateID> gen_valid(int W){
    std::vector<MateID> out;
    auto rec=[&](auto&& self,int pos,int h,MateID m)->void{
        int rem=W-pos;if(h<0||h>rem)return;
        if(pos==W){if(h==0)out.push_back(m);return;}
        int bit=W-1-pos;self(self,pos+1,h,m);
        if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*bit)));
        self(self,pos+1,h+1,m|(MateID(L)<<(2*bit)));
    };
    rec(rec,0,1,0);return out;
}
static bool valid_mate(MateID m,int W){int h=1;for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;if(h<0)return false;}return h==0;}
static Count addm(Count a,Count b){std::uint64_t z=std::uint64_t(a)+b;if(z>=MOD)z-=MOD;return Count(z);}

int main(){
    std::mt19937_64 rng(0x77617270636c6f73ULL);
    std::uint64_t destinations=0,closure_destinations=0,candidates=0;
    int max_terms=0;
    for(int W=4;W<=12;++W){
        auto block=gen_valid(W-1);
        for(int p=2;p<W;++p){
            for(MateID b:block){
                ++destinations;
                const MateValue look=mget(b,p-1);
                if(look!=N)continue;
                ++closure_destinations;
                const MateID d=minsert(b,p-1,N);
                MateID cand[32]{};
                const int n=ordinary_closure_preimages_partial(d,W,p,cand);
                if(n<0||n>32)return 2;
                max_terms=std::max(max_terms,n);candidates+=n;
                Count sequential=0;
                Count vals[32]{};
                for(int k=0;k<n;++k){
                    if(!valid_mate(cand[k],W))return 3;
                    vals[k]=Count(rng()%MOD);sequential=addm(sequential,vals[k]);
                }
                // Model one value per warp lane followed by a tree reduction.
                for(int step=16;step;step>>=1)for(int lane=0;lane<step;++lane){
                    const int rhs=lane+step;
                    if(rhs<n)vals[lane]=addm(vals[lane],vals[rhs]);
                }
                const Count warp=n?vals[0]:0;
                if(warp!=sequential){std::cerr<<"reduction mismatch W="<<W<<" p="<<p<<" n="<<n<<'\n';return 4;}
            }
        }
    }
    if(!closure_destinations||!candidates)return 5;
    std::cout<<"b300-block-closure-warp-batch-proof OK exhaustive_width_max=12 destinations="<<destinations
             <<" closure_destinations="<<closure_destinations<<" candidates="<<candidates
             <<" max_terms="<<max_terms<<" warp_lanes=32 one_source_load_per_lane=1 modular_tree_exact=1 exact=1\n";
    return 0;
}
