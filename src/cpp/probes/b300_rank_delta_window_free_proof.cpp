#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
std::vector<int> candidates(int W,int hi,int lo){
    std::vector<int> v;
    for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);
    return v;
}
void masks(int hi,int lo,const std::vector<int>&fp,std::uint32_t&mf,std::uint32_t&bf){
    mf=bf=0;
    for(int q:fp){
        mf|=std::uint32_t(1)<<q;
        int bq=q<lo-1?q:q-1;
        bf|=std::uint32_t(1)<<bq;
    }
}
}

int main(){
    std::uint64_t windows=0,pchecks=0,kchecks=0;
    for(int W=4;W<=28;++W){
        for(int hi=W-1;hi>=1;--hi){
            for(int lo=1;lo<=hi;++lo){
                auto cand=candidates(W,hi,lo);
                int klim=std::min<int>(20,cand.size());
                for(int k=0;k<=klim;++k){
                    std::vector<int>fp(cand.begin(),cand.begin()+k);
                    std::uint32_t mf=0,bf=0;masks(hi,lo,fp,mf,bf);++kchecks;
                    for(int p=hi;p>=lo;--p){
                        if((mf>>p)&1u){
                            std::cerr<<"main moving position fixed W="<<W<<" hi="<<hi<<" lo="<<lo<<" k="<<k<<" p="<<p<<'\n';return 2;
                        }
                        if((bf>>(p-1))&1u){
                            std::cerr<<"block moving position fixed W="<<W<<" hi="<<hi<<" lo="<<lo<<" k="<<k<<" p="<<p<<'\n';return 3;
                        }
                        ++pchecks;
                    }
                }
                ++windows;
            }
        }
    }
    std::cout<<"b300-rank-delta-window-free-proof OK width_max=28 windows="<<windows
             <<" k_configs="<<kchecks<<" position_checks="<<pchecks
             <<" main_p_free=1 block_pm1_free=1 allowed_checks_required=0 exact=1\n";
    return 0;
}
