#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using oneesan::gridfp::MateID;
static std::vector<Mate> states(MateCodec const& mc){std::vector<Mate> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
static const std::unordered_set<MateID>* LEGAL=nullptr;
static void add_if(std::vector<MateID>& out, MateID c, MateID target, int W, int p){
    if(LEGAL && !LEGAL->count(c)) return;
    auto z=oneesan::gridfp::include_horizontal(c,W,p);
    if(z.valid&&z.blocked&&z.mate==target)out.push_back(c);
}
static std::vector<MateID> reverse_block(MateID b,int W,int p){
    std::vector<MateID> out;
    auto low=oneesan::gridfp::mget(b,p-1);
    if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
        MateID c=oneesan::gridfp::minsert(b,p,oneesan::gridfp::N); // insert the removed high N
        add_if(out,c,b,W,p);
    }else if(low==oneesan::gridfp::N){
        MateID u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N); // inverse of shrink(p-1), pair becomes NN
        add_if(out,oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL),b,W,p);
        for(int q=0;q<=p-2;++q) if(oneesan::gridfp::mget(u,q)==oneesan::gridfp::L){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::R);add_if(out,c,b,W,p);}
        for(int q=p+1;q<W;++q) if(oneesan::gridfp::mget(u,q)==oneesan::gridfp::R){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::L);add_if(out,c,b,W,p);}
    }
    std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return out;
}
int main(int argc,char**argv){msg=NONE;int W=argc>1?std::atoi(argv[1]):12;PathCounter<uint64_t> pc(W,W,false,false);auto ms=states(pc.mc),ds=states(pc.wc);std::unordered_set<MateID> legal;legal.reserve(ms.size()*2);for(auto m:ms)legal.insert(m.id());LEGAL=&legal;std::unordered_map<MateID,std::vector<MateID>> brute; brute.reserve(ds.size()*2);
for(int p=2;p<W;++p){brute.clear();for(auto m:ms){auto z=oneesan::gridfp::include_horizontal(m.id(),W,p);if(z.valid&&z.blocked)brute[z.mate].push_back(m.id());}uint64_t bad=0,maxg=0,maxb=0;for(auto t:ds){auto got=reverse_block(t.id(),W,p);auto exp=brute[t.id()];std::sort(exp.begin(),exp.end());exp.erase(std::unique(exp.begin(),exp.end()),exp.end());maxg=std::max<uint64_t>(maxg,got.size());maxb=std::max<uint64_t>(maxb,exp.size());if(got!=exp){if(bad<5){std::cerr<<"mismatch W="<<W<<" p="<<p<<" t="<<t<<" got="<<got.size()<<" exp="<<exp.size()<<"\n";}++bad;}}std::cout<<"W="<<W<<" p="<<p<<" bad="<<bad<<" max="<<maxg<<" bruteMax="<<maxb<<"\n";if(bad)return 1;} }
