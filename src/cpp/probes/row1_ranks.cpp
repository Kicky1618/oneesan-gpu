#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
using namespace oneesan::gridfp;
static std::string show(MateID m,int w){std::string s;for(int i=w-1;i>=0;--i)s += mget(m,i)==oneesan::gridfp::N?'N':mget(m,i)==oneesan::gridfp::R?'R':'L';return s;}
int main(int argc,char**argv){int W=argc>1?std::atoi(argv[1]):6;std::unordered_set<MateID> M,D,nM,nD;M.insert(MateID(oneesan::gridfp::R)<<(2*(W-1)));
for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto m:M){nM.insert(m);auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD.insert(z.mate);else nM.insert(z.mate);}}for(auto b:D)nM.insert(blocked_exclude(b,p));M.swap(nM);D.swap(nD);std::cerr<<"p="<<p<<" M="<<M.size()<<" D="<<D.size()<<"\n";}
std::vector<MateID> v(M.begin(),M.end());std::sort(v.begin(),v.end());PathCounter<uint64_t> pc(W,W,false,false); std::vector<unsigned long long> rs; for(auto m:v){Mate mm(m); rs.push_back(pc.mc.encode(mm));} std::sort(rs.begin(),rs.end()); bool cont=true; for(size_t i=1;i<rs.size();++i) if(rs[i]!=rs[i-1]+1) cont=false; std::cout<<"W="<<W<<" states="<<v.size()<<" rank_min="<<rs.front()<<" rank_max="<<rs.back()<<" contiguous="<<cont<<"\n"; for(size_t i=0;i<std::min<size_t>(v.size(),64);++i){Mate mm(v[i]);std::cout<<show(v[i],W)<<" rank="<<pc.mc.encode(mm)<<" 0x"<<std::hex<<v[i]<<std::dec<<"\n";}
}
