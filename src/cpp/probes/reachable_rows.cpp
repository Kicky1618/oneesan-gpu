#include "../../common/gridfp_transition.hpp"
#include <cstdint>
#include <iostream>
#include <unordered_set>
using namespace oneesan::gridfp;
int main(int argc,char**argv){int W=argc>1?std::atoi(argv[1]):10;int R=argc>2?std::atoi(argv[2]):2;std::unordered_set<MateID>M,D,nM,nD;M.reserve(1<<20);D.reserve(1<<20);M.insert(MateID(oneesan::gridfp::R)<<(2*(W-1)));
for(int row=0;row<R;++row){for(int p=W-1;p>=1;--p){nM.clear();nD.clear();nM.reserve(M.size()*2+1);nD.reserve(M.size()/2+1);for(auto m:M){nM.insert(m);auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD.insert(z.mate);else nM.insert(z.mate);}}for(auto b:D)nM.insert(blocked_exclude(b,p));M.swap(nM);D.swap(nD);}std::cout<<"W="<<W<<" row="<<row+1<<" M="<<M.size()<<" D="<<D.size()<<"\n";}
}
