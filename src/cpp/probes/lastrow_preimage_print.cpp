#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>
using oneesan::gridfp::MateID;
static std::vector<MateID> states(MateCodec const& mc){std::vector<MateID> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=(b.mateL|b.mateR[i]).id();}return o;}
int main(int argc,char**argv){msg=NONE;int W=argc>1?std::atoi(argv[1]):8;PathCounter<uint64_t> pc(W,W,false,false);auto M=states(pc.mc),D=states(pc.wc);
 std::unordered_map<MateID,unsigned long long> nm,nd,om,od; nm.reserve(M.size()*2);nd.reserve(D.size()*2);om.reserve(M.size()*2);od.reserve(D.size()*2);
 MateID fin=MateID(oneesan::gridfp::R);nm[fin]=1;
 for(int p=1;p<W;++p){om.clear();od.clear();
   for(auto m:M){unsigned long long z=0;auto it=nm.find(m);if(it!=nm.end())z+=it->second;auto r=oneesan::gridfp::include_horizontal(m,W,p);if(r.valid){auto &mp=r.blocked?nd:nm;auto jt=mp.find(r.mate);if(jt!=mp.end())z+=jt->second;}if(z)om.emplace(m,z);}
   for(auto b:D){MateID t=oneesan::gridfp::blocked_exclude(b,p);auto it=nm.find(t);if(it!=nm.end()&&it->second)od.emplace(b,it->second);}
   nm.swap(om);nd.swap(od);unsigned long long mx=0;for(auto [_,x]:nm)mx=std::max(mx,x);for(auto [_,x]:nd)mx=std::max(mx,x);std::cerr<<"rev p="<<p<<" M="<<nm.size()<<" D="<<nd.size()<<" maxw="<<mx<<"\n";
 }
 unsigned long long mx=0,sum=0;for(auto [_,x]:nm){mx=std::max(mx,x);sum+=x;}std::cout<<"W="<<W<<" preimage_main="<<nm.size()<<" max_coeff="<<mx<<" coeff_sum="<<sum<<" blocked="<<nd.size()<<" expected2="<<(1ULL<<(W-1))<<"\n"; if(W<=6){std::vector<MateID> vv;for(auto [m,_]:nm)vv.push_back(m);std::sort(vv.begin(),vv.end());for(auto m:vv){for(int i=W-1;i>=0;--i){auto v=oneesan::gridfp::mget(m,i);std::cout<<(v==oneesan::gridfp::N?'N':v==oneesan::gridfp::R?'R':'L');}std::cout<<"\n";}}
}
