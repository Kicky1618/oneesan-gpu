#include "../../common/gridfp_closure_pattern10.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>

using namespace oneesan::gridfp;

static bool no_adj(std::uint16_t x){return (x&(x<<1))==0;}

int main(){
    for(int m=0;m<=13;++m){
        std::set<std::uint16_t> seen;
        for(std::uint32_t x=0;x<(1u<<m);++x){
            if(!no_adj(std::uint16_t(x)))continue;
            auto r=closure_noadj_rank(std::uint16_t(x),m);
            auto y=closure_noadj_unrank(r,m);
            if(y!=x){std::cerr<<"noadj codec mismatch m="<<m<<" x="<<x<<" rank="<<r<<" y="<<y<<'\n';return 1;}
            seen.insert(r);
        }
        if(seen.size()!=closure_fib_count(m)||(!seen.empty()&&*seen.rbegin()+1!=seen.size())){
            std::cerr<<"noadj rank range mismatch m="<<m<<" got="<<seen.size()<<" expected="<<closure_fib_count(m)<<'\n';return 2;
        }
    }

    // Exhaust all ternary destination words through length 10. Candidate masks
    // must be no-adjacent and encode/decode losslessly for every NN pair.
    for(int len=2;len<=10;++len){
        std::uint64_t total=1;for(int i=0;i<len;++i)total*=3;
        for(int p=1;p<len;++p){
            std::uint16_t maxid=0;
            for(std::uint64_t key=0;key<total;++key){
                std::uint64_t t=key;MateID d=0;
                for(int q=0;q<len;++q){auto v=MateValue(t%3);t/=3;d|=MateID(v)<<(2*q);}
                if(mpair(d,p)!=NN)continue;
                auto id=closure_pattern10_encode(d,len,p);if(id==CLOSURE_PATTERN10_NONE){std::cerr<<"unexpected pattern NONE len="<<len<<" p="<<p<<'\n';return 3;}
                std::uint16_t lm=0,rm=0;closure_pattern10_decode(id,len,p,lm,rm);
                if(!no_adj(lm)||!no_adj(rm)){std::cerr<<"decoded adjacent pattern len="<<len<<" p="<<p<<'\n';return 4;}
                if(closure_noadj_rank(lm,p-1)*closure_fib_count(len-p-1)+closure_noadj_rank(rm,len-p-1)!=id){std::cerr<<"combined rerank mismatch\n";return 5;}
                maxid=std::max(maxid,id);
            }
            if(closure_pattern10_count(len,p)>=CLOSURE_PATTERN10_NONE){std::cerr<<"pattern count overflow len="<<len<<" p="<<p<<'\n';return 6;}
            (void)maxid;
        }
    }

    std::uint16_t max15=0,max14=0;
    for(int p=1;p<15;++p)max15=std::max(max15,closure_pattern10_count(15,p));
    for(int p=1;p<14;++p)max14=std::max(max14,closure_pattern10_count(14,p));
    if(max15!=754||max14!=466||max15>=CLOSURE_PATTERN10_NONE){
        std::cerr<<"production pattern bound mismatch len15="<<max15<<" len14="<<max14<<'\n';return 7;
    }
    std::cout<<"gridfp-closure-pattern10-selftest OK max_len15="<<max15<<" max_len14="<<max14<<" sentinel="<<CLOSURE_PATTERN10_NONE<<"\n";
    return 0;
}
