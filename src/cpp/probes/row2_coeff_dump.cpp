#include "../../common/gridfp_transition.hpp"
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace oneesan::gridfp;
static std::string show(MateID m,int W){std::string s;for(int i=W-1;i>=0;--i){auto v=mget(m,i);s+=(v==N?'N':v==R?'R':'L');}return s;}
int main(int argc,char**argv){int maxW=argc>1?std::atoi(argv[1]):9;using C=unsigned long long;for(int W=2;W<=maxW;++W){std::unordered_map<MateID,C>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int row=0;row<2;++row){for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto [m,c]:M){nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto [b,c]:D)nM[blocked_exclude(b,p)]+=c;M.swap(nM);D.swap(nD);}}for(auto [m,c]:M)std::cout<<W<<' '<<show(m,W)<<' '<<c<<'\n';}
}
