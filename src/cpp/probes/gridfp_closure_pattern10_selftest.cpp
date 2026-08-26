#include "../../common/gridfp_closure_pattern10.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <utility>

using namespace oneesan::gridfp;

static bool no_adj(std::uint16_t x){return (x&(x<<1))==0;}
static std::pair<std::uint16_t,std::uint16_t> reference_masks(MateID d,int len,int p){
    std::uint16_t lm=0,rm=0;int bal=0,bi=0;
    for(int q=p-2;q>=0;--q,++bi){MateValue v=mget(d,q);if(bal==0&&v==L)lm=std::uint16_t(lm|(std::uint16_t(1u)<<bi));if(v==L)++bal;else if(v==R)--bal;if(bal<0)break;}
    bal=0;bi=0;
    for(int q=p+1;q<len;++q,++bi){MateValue v=mget(d,q);if(bal==0&&v==R)rm=std::uint16_t(rm|(std::uint16_t(1u)<<bi));if(v==R)++bal;else if(v==L)--bal;if(bal<0)break;}
    return {lm,rm};
}

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
    // from the independent balance scan must be no-adjacent and must survive
    // pattern10 encode/decode exactly.
    for(int len=2;len<=10;++len){
        std::uint64_t total=1;for(int i=0;i<len;++i)total*=3;
        for(int p=1;p<len;++p){
            std::uint16_t maxid=0;
            for(std::uint64_t key=0;key<total;++key){
                std::uint64_t t=key;MateID d=0;
                for(int q=0;q<len;++q){auto v=MateValue(t%3);t/=3;d|=MateID(v)<<(2*q);}
                if(mpair(d,p)!=NN)continue;
                auto [want_l,want_r]=reference_masks(d,len,p);
                if(!no_adj(want_l)||!no_adj(want_r)){std::cerr<<"reference mask adjacency violation len="<<len<<" p="<<p<<'\n';return 3;}
                auto id=closure_pattern10_encode(d,len,p);if(id==CLOSURE_PATTERN10_NONE){std::cerr<<"unexpected pattern NONE len="<<len<<" p="<<p<<'\n';return 4;}
                std::uint16_t got_l=0,got_r=0;closure_pattern10_decode(id,len,p,got_l,got_r);
                if(got_l!=want_l||got_r!=want_r){std::cerr<<"pattern decode mismatch len="<<len<<" p="<<p<<" id="<<id<<" got="<<got_l<<','<<got_r<<" want="<<want_l<<','<<want_r<<'\n';return 5;}
                maxid=std::max(maxid,id);
            }
            if(closure_pattern10_count(len,p)>=CLOSURE_PATTERN10_NONE){std::cerr<<"pattern count overflow len="<<len<<" p="<<p<<'\n';return 6;}
            if(maxid>=closure_pattern10_count(len,p)){std::cerr<<"pattern id range mismatch len="<<len<<" p="<<p<<'\n';return 7;}
        }
    }

    std::uint16_t max15=0,max14=0;
    for(int p=1;p<15;++p)max15=std::max(max15,closure_pattern10_count(15,p));
    for(int p=1;p<14;++p)max14=std::max(max14,closure_pattern10_count(14,p));
    if(max15!=754||max14!=466||max15>=CLOSURE_PATTERN10_NONE){
        std::cerr<<"production pattern bound mismatch len15="<<max15<<" len14="<<max14<<'\n';return 8;
    }
    std::cout<<"gridfp-closure-pattern10-selftest OK max_len15="<<max15<<" max_len14="<<max14<<" sentinel="<<CLOSURE_PATTERN10_NONE<<"\n";
    return 0;
}
