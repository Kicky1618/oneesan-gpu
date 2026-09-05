#include "../../common/gridfp_transition.hpp"
#include <cstdint>
#include <iostream>
#include <unordered_map>
using namespace oneesan::gridfp;
static constexpr uint64_t MOD=1000000007ull;
static inline void add(uint64_t& a,uint64_t b){ a+=b; if(a>=MOD)a-=MOD; }
static std::string show(MateID m,int W){std::string s;for(int i=W-1;i>=0;--i){auto v=mget(m,i);s+=(v==N?'N':v==R?'R':'L');}return s;}
int main(int argc,char**argv){int rows=argc>1?atoi(argv[1]):2,maxW=argc>2?atoi(argv[2]):14;for(int W=2;W<=maxW;++W){std::unordered_map<MateID,uint64_t>M,D,nM,nD;M[MateID(oneesan::gridfp::R)<<(2*(W-1))]=1;for(int row=0;row<rows;++row){for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto [m,c]:M){add(nM[m],c);auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)add(nD[z.mate],c);else add(nM[z.mate],c);}}for(auto [b,c]:D)add(nM[blocked_exclude(b,p)],c);M.swap(nM);D.swap(nD);}}for(auto [m,c]:M)std::cout<<W<<' '<<show(m,W)<<' '<<c<<'\n';}}
