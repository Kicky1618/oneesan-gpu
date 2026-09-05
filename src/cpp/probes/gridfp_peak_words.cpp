#include "../../common/gridfp_transition.hpp"
#include <iostream>
#include <unordered_map>
#include <string>
using namespace oneesan::gridfp; using C=unsigned long long;
static int peak(MateID m,int W){int h=1,p=h;for(int k=W-1;k>=0;--k){auto v=mget(m,k);if(v==R)--h;else if(v==L)++h;p=std::max(p,h);}return p;}
static std::string show(MateID m,int W){std::string s;for(int k=W-1;k>=0;--k){auto v=mget(m,k);s+=v==N?'N':v==R?'R':'L';}return s;}
int main(){int W=6;std::unordered_map<MateID,C>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto [m,c]:M){nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto[b,c]:D)nM[blocked_exclude(b,p)]+=c;M.swap(nM);D.swap(nD);}int n=0;for(auto [m,c]:M)if(peak(m,W)>=2&&n++<20)std::cout<<show(m,W)<<" peak="<<peak(m,W)<<" ways="<<c<<"\n";}
