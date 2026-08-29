#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_closure_inverse.hpp"

#include <array>
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

static bool valid_mate(MateID m,int W){
    int h=1;
    for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;if(h<0)return false;}
    return h==0;
}
static int endpoint_count(MateID m,int W){int n=0;for(int p=0;p<W;++p){auto v=mget(m,p);n+=(v==R||v==L);}return n;}
static Count addm(Count a,Count b){std::uint64_t z=std::uint64_t(a)+b;if(z>=MOD)z-=MOD;return Count(z);}

int main(){
    constexpr std::array<int,4> THRESH{0,4,8,12};
    std::array<std::uint64_t,THRESH.size()> scalar{},warp{};
    std::uint64_t closures=0,candidates=0,rejected=0;
    std::mt19937_64 rng(0x6879627269647761ULL);
    for(int W=4;W<=12;++W){
        auto block=gen_valid(W-1);
        for(int p=2;p<W;++p){
            for(MateID b:block){
                if(mget(b,p-1)!=N)continue;
                ++closures;
                const MateID d=minsert(b,p-1,N);
                const int ep=endpoint_count(d,W);
                MateID cand[32]{};const int n=ordinary_closure_preimages_partial(d,W,p,cand);
                if(n<0||n>32)return 2;
                Count seq=0;std::array<Count,32> vals{};int nv=0;
                for(int k=0;k<n;++k){
                    if(!valid_mate(cand[k],W)){++rejected;continue;}
                    vals[nv]=Count(rng()%MOD);seq=addm(seq,vals[nv]);++nv;++candidates;
                }
                for(std::size_t ti=0;ti<THRESH.size();++ti){
                    const int t=THRESH[ti];const bool use_warp=(t==0)||ep>=t;
                    if(use_warp)++warp[ti];else ++scalar[ti];
                    Count got=0;
                    if(use_warp){
                        auto tmp=vals;
                        for(int step=16;step;step>>=1)for(int lane=0;lane<step;++lane){int rhs=lane+step;if(rhs<nv)tmp[lane]=addm(tmp[lane],tmp[rhs]);}
                        got=nv?tmp[0]:0;
                    }else{
                        for(int k=0;k<nv;++k)got=addm(got,vals[k]);
                    }
                    if(got!=seq){std::cerr<<"hybrid sum mismatch W="<<W<<" p="<<p<<" threshold="<<t<<"\n";return 3;}
                }
            }
        }
    }
    if(!closures||!candidates||!rejected)return 4;
    bool mixed=false;for(std::size_t i=1;i<THRESH.size();++i)mixed|=scalar[i]&&warp[i];
    if(!mixed)return 5;
    std::cout<<"b300-block-closure-warp-hybrid-proof OK exhaustive_width_max=12 closures="<<closures
             <<" valid_candidates="<<candidates<<" rejected_candidates="<<rejected;
    for(std::size_t i=0;i<THRESH.size();++i)std::cout<<" t"<<THRESH[i]<<"_scalar="<<scalar[i]<<" t"<<THRESH[i]<<"_warp="<<warp[i];
    std::cout<<" disjoint_partition=1 scalar_sum_exact=1 warp_sum_exact=1 rl_filter=valid_source_only exact=1\n";
    return 0;
}
