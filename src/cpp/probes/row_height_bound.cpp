#include "../../common/gridfp_transition.hpp"
#include <cstdint>
#include <iostream>
#include <unordered_map>
using namespace oneesan::gridfp;
static int maxh(MateID m,int W){int h=1,mx=h;for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;if(h>mx)mx=h;}return mx;}
int main(int argc,char**argv){int W=argc>1?atoi(argv[1]):10,R=argc>2?atoi(argv[2]):5;using C=uint64_t;std::unordered_map<MateID,C>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int row=1;row<=R;++row){int mxM=0,mxD=0;for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto [m,c]:M){nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto [b,c]:D)nM[blocked_exclude(b,p)]+=c;M.swap(nM);D.swap(nD);for(auto [m,c]:M)mxM=std::max(mxM,maxh(m,W));for(auto [b,c]:D)mxD=std::max(mxD,maxh(b,W-1));std::cout<<"row="<<row<<" p="<<p<<" M="<<M.size()<<" D="<<D.size()<<" maxM="<<mxM<<" maxD="<<mxD<<"\n";}std::cout<<"END row="<<row<<" maxM="<<mxM<<" maxD="<<mxD<<"\n";}}
