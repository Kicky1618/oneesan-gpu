#include "../../common/gridfp_transition.hpp"
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <map>
using namespace oneesan::gridfp;
int main(int argc,char**argv){int W=argc>1?std::atoi(argv[1]):10;int R=argc>2?std::atoi(argv[2]):2;using C=unsigned long long;std::unordered_map<MateID,C>M,D,nM,nD;M[MateID(oneesan::gridfp::R)<<(2*(W-1))]=1;
for(int row=0;row<R;++row){for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto [m,c]:M){nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto [b,c]:D)nM[blocked_exclude(b,p)]+=c;M.swap(nM);D.swap(nD);}C mx=0,sum=0,multi=0;for(auto [_,c]:M){mx=std::max(mx,c);sum+=c;if(c>1)multi++;}std::cout<<"W="<<W<<" row="<<row+1<<" M="<<M.size()<<" multi="<<multi<<" max="<<mx<<" sum="<<sum<<" D="<<D.size()<<"\n"; if(row==1){std::map<C,C> h;for(auto [_,c]:M)h[c]++;for(auto [c,k]:h)std::cout<<c<<":"<<k<<" ";std::cout<<"\n";}}
}
