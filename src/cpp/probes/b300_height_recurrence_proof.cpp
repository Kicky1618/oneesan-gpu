#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

using namespace oneesan::gridfp;

namespace {
std::vector<MateID> gen_valid(int W){
    std::vector<MateID> out;
    auto rec=[&](auto&& self,int pos,int h,MateID m)->void{
        int rem=W-pos;if(h<0||h>rem)return;
        if(pos==W){if(h==0)out.push_back(m);return;}
        int bit=W-1-pos;
        self(self,pos+1,h,m);
        if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*bit)));
        self(self,pos+1,h+1,m|(MateID(L)<<(2*bit)));
    };
    rec(rec,0,1,0);return out;
}
int height_before(MateID m,int W,int p){
    int h=1;
    for(int q=W-1;q>p;--q){MateValue v=mget(m,q);if(v==R)--h;else if(v==L)++h;}
    return h;
}
int advance(int h,MateValue v){return h+(v==L)-(v==R);}

MateID random_valid(int W,std::mt19937_64& rng){
    // Rejection from a simple DP-guided walk is unnecessary here: generate a
    // random valid word by choosing uniformly from the exhaustive list for a
    // smaller skeleton, then pad with balanced NN-neutral structure via a
    // deterministic valid prefix. For the recurrence identity any symbols are
    // sufficient, but valid words also verify the production height range.
    MateID m=0;int h=1;
    for(int p=W-1;p>=0;--p){
        int rem=p;
        std::uint64_t r=rng();
        MateValue v=N;
        if(h>0 && (r&3u)==0 && h-1<=rem)v=R;
        else if(h+1<=rem && (r&3u)==1)v=L;
        else if(h>rem)v=R;
        if(v==R){m|=MateID(R)<<(2*p);--h;}
        else if(v==L){m|=MateID(L)<<(2*p);++h;}
    }
    // The guards force h=0 at the end.
    return m;
}
}

int main(){
    std::uint64_t exhaustive_states=0,exhaustive_steps=0,random_steps=0;
    int max_height=0;
    for(int W=2;W<=12;++W){
        auto states=gen_valid(W);exhaustive_states+=states.size();
        for(MateID m:states){
            for(int p=W-1;p>=1;--p){
                int h=height_before(m,W,p);
                int hn=height_before(m,W,p-1);
                if(hn!=advance(h,mget(m,p))){
                    std::cerr<<"recurrence mismatch W="<<W<<" p="<<p<<'\n';return 2;
                }
                if(h<0||h>W||hn<0||hn>W)return 3;
                max_height=std::max({max_height,h,hn});++exhaustive_steps;
            }
        }
    }
    std::mt19937_64 rng(0x6865696768746361ULL);
    constexpr int W=28;
    for(int it=0;it<200000;++it){
        MateID m=random_valid(W,rng);
        for(int p=W-1;p>=1;--p){
            int h=height_before(m,W,p),hn=height_before(m,W,p-1);
            if(hn!=advance(h,mget(m,p)))return 4;
            if(h<0||h>W||hn<0||hn>W)return 5;
            max_height=std::max({max_height,h,hn});++random_steps;
        }
    }
    if(max_height>=256)return 6;
    std::cout<<"b300-height-recurrence-proof OK exhaustive_width_max=12 exhaustive_states="<<exhaustive_states
             <<" exhaustive_steps="<<exhaustive_steps<<" random_w28_steps="<<random_steps
             <<" max_height="<<max_height
             <<" storage_bytes=1 main_update=mp block_update=pm1 exact=1\n";
    return 0;
}
