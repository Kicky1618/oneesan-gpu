#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_closure_inverse.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <vector>

using namespace oneesan::gridfp;
using Count = std::uint32_t;
using Wide = std::uint64_t;
static constexpr Count MOD = 4294967291u;

namespace {

std::vector<MateID> gen_valid(int W) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self,int pos,int h,MateID m)->void {
        const int rem=W-pos;
        if(h<0||h>rem)return;
        if(pos==W){if(h==0)out.push_back(m);return;}
        const int bit=W-1-pos;
        self(self,pos+1,h,m);
        if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*bit)));
        self(self,pos+1,h+1,m|(MateID(L)<<(2*bit)));
    };
    rec(rec,0,1,0);
    std::sort(out.begin(),out.end());
    return out;
}

bool valid_mate(MateID m,int W){
    int h=1;
    for(int pos=W-1;pos>=0;--pos){
        const MateValue v=mget(m,pos);
        if(v==X)return false;
        if(v==R)--h;else if(v==L)++h;
        if(h<0)return false;
    }
    return h==0;
}

bool forced_block_dest(MateID m,int W,int p,MateID& b){
    if(p<=1)return false;
    MateID t=m;
    switch(mpair(m,p)){
    case NR:case NL:
        b=mshrink(m,p-1);return true;
    case LL:{
        t=msetpair(m,p,NN);const int q=closure_match_left(t,p);if(q<0)return false;
        t=mset(t,q,L);b=mshrink(t,p-1);return true;
    }
    case RR:{
        t=msetpair(m,p,NN);const int q=closure_match_right(t,W,p);if(q<0)return false;
        t=mset(t,q,R);b=mshrink(t,p-1);return true;
    }
    case RL:
        t=msetpair(m,p,NN);b=mshrink(t,p-1);return true;
    default:return false;
    }
}

Count add_mod(Count a,Count b){Wide z=Wide(a)+b;if(z>=MOD)z-=MOD;return Count(z);}

std::map<MateID,Count> push_block(const std::vector<MateID>& main,const std::vector<Count>& mv,int W,int p){
    std::map<MateID,Count> out;
    for(std::size_t i=0;i<main.size();++i){
        MateID b=0;if(!forced_block_dest(main[i],W,p,b))continue;
        auto it=out.find(b);if(it==out.end())out.emplace(b,mv[i]);else it->second=add_mod(it->second,mv[i]);
    }
    return out;
}

std::map<MateID,Count> pull_block(const std::vector<MateID>& main,const std::vector<MateID>& block,const std::vector<Count>& mv,int W,int p,std::size_t& max_terms){
    std::map<MateID,std::size_t> mi;
    for(std::size_t i=0;i<main.size();++i)mi.emplace(main[i],i);
    std::map<MateID,Count> out;
    for(MateID b:block){
        Count acc=0;std::size_t terms=0;
        auto add_source=[&](MateID x){
            if(!valid_mate(x,W))return;
            MateID got=0;if(!forced_block_dest(x,W,p,got)||got!=b)return;
            auto it=mi.find(x);if(it==mi.end())std::exit(10);
            acc=add_mod(acc,mv[it->second]);++terms;
        };
        if(is_endpoint(mget(b,p-1))){
            // Old forced2window NR/NL channel removes physical p-1.
            add_source(minsert(b,p-1,N));
        }else if(mget(b,p-1)==N){
            // Closure channel also removes physical p-1 after producing NN.
            const MateID closure_dest=minsert(b,p-1,N);
            MateID cand[32]{};
            const int n=ordinary_closure_preimages_partial(closure_dest,W,p,cand);
            for(int i=0;i<n;++i)add_source(cand[i]);
        }
        max_terms=std::max(max_terms,terms);
        if(acc)out.emplace(b,acc);
    }
    return out;
}

} // namespace

int main(){
    std::mt19937_64 rng(0x626c6f636b70756cULL);
    std::uint64_t positions=0,destinations=0;std::size_t max_terms=0;
    for(int W=4;W<=11;++W){
        const auto main=gen_valid(W);const auto block=gen_valid(W-1);
        std::vector<Count> mv(main.size());for(auto&x:mv)x=Count(rng()%MOD);
        for(int p=2;p<W;++p){
            const auto push=push_block(main,mv,W,p);
            const auto pull=pull_block(main,block,mv,W,p,max_terms);
            if(push!=pull){std::cerr<<"mismatch W="<<W<<" p="<<p<<" push="<<push.size()<<" pull="<<pull.size()<<'\n';return 2;}
            ++positions;destinations+=block.size();
        }
    }
    std::cout<<"b300-block-pull-operator-proof OK exhaustive_width_max=11 positions="<<positions
             <<" blocked_destinations="<<destinations
             <<" p_scope=2..Wm1 block_memset_required=0 block_atomic_updates_required=0"
             <<" pull_terms_max="<<max_terms<<" exact=1\n";
    return 0;
}
