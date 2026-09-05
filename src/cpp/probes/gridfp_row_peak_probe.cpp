#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <vector>
using namespace oneesan::gridfp;
struct H{size_t operator()(MateID x)const noexcept{return x^(x>>33);}};
static int peak(MateID m,int W){int h=1,pk=h;for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;pk=std::max(pk,h);}return pk;}
static int endh(MateID m,int W){int h=1;for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;}return h;}
int main(int ac,char**av){int W=ac>1?atoi(av[1]):12,nrows=ac>2?atoi(av[2]):8;std::unordered_set<MateID,H>M,D,nM,nD;M.insert(MateID(oneesan::gridfp::R)<<(2*(W-1)));std::cout<<"row=0 states="<<M.size()<<" maxpeak="<<peak(*M.begin(),W)<<"\n";
for(int row=1;row<=nrows;++row){for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto m:M){nM.insert(m);auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD.insert(z.mate);else nM.insert(z.mate);}}for(auto b:D)nM.insert(blocked_exclude(b,p));M.swap(nM);D.swap(nD);}int mp=0,me=-99,mi=99,bad=0;for(auto m:M){mp=std::max(mp,peak(m,W));me=std::max(me,endh(m,W));mi=std::min(mi,endh(m,W));if(endh(m,W)!=0)++bad;}std::cout<<"row="<<row<<" states="<<M.size()<<" blocked="<<D.size()<<" maxpeak="<<mp<<" endh=["<<mi<<","<<me<<"] badend="<<bad<<"\n";}
}
