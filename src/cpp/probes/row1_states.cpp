#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>
using namespace oneesan::gridfp;
static std::string show(MateID m,int w){std::string s;for(int i=w-1;i>=0;--i)s += mget(m,i)==N?'N':mget(m,i)==R?'R':'L';return s;}
int main(int argc,char**argv){int W=argc>1?std::atoi(argv[1]):6;std::unordered_set<MateID> M,D,nM,nD;M.insert(MateID(R)<<(2*(W-1)));
for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto m:M){nM.insert(m);auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD.insert(z.mate);else nM.insert(z.mate);}}for(auto b:D)nM.insert(blocked_exclude(b,p));M.swap(nM);D.swap(nD);std::cerr<<"p="<<p<<" M="<<M.size()<<" D="<<D.size()<<"\n";}
std::vector<MateID> v(M.begin(),M.end());std::sort(v.begin(),v.end());std::cout<<"W="<<W<<" states="<<v.size()<<"\n";for(size_t i=0;i<std::min<size_t>(v.size(),128);++i)std::cout<<show(v[i],W)<<" 0x"<<std::hex<<v[i]<<std::dec<<"\n";
}
