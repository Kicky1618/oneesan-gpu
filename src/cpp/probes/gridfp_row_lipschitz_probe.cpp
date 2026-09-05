#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
using namespace oneesan::gridfp;
struct H{size_t operator()(MateID x)const noexcept{return x^(x>>33);}};
static std::vector<int> prof(MateID m,int W){std::vector<int> a(W+1);int h=1;a[0]=h;for(int q=0,p=W-1;p>=0;--p,++q){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;a[q+1]=h;}return a;}
static std::unordered_set<MateID,H> rowstep(MateID start,int W){std::unordered_set<MateID,H>M,D,nM,nD;M.insert(start);for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto m:M){nM.insert(m);auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD.insert(z.mate);else nM.insert(z.mate);}}for(auto b:D)nM.insert(blocked_exclude(b,p));M.swap(nM);D.swap(nD);}return M;}
int main(int ac,char**av){int W=ac>1?atoi(av[1]):10,nrows=ac>2?atoi(av[2]):6;std::unordered_set<MateID,H> reach;reach.insert(MateID(oneesan::gridfp::R)<<(2*(W-1)));for(int row=1;row<=nrows;++row){std::unordered_set<MateID,H> next;long long pairs=0;int maxDiff=-999,minDiff=999,bad=0;for(auto x:reach){auto ax=prof(x,W);auto ys=rowstep(x,W);for(auto y:ys){++pairs;next.insert(y);auto ay=prof(y,W);for(int q=0;q<=W;++q){int d=ay[q]-ax[q];maxDiff=std::max(maxDiff,d);minDiff=std::min(minDiff,d);if(d>1){if(bad++<5)std::cerr<<"bad row="<<row<<" q="<<q<<" diff="<<d<<"\n";}}}}std::cout<<"row="<<row<<" inputs="<<reach.size()<<" outputs="<<next.size()<<" pairs="<<pairs<<" diff=["<<minDiff<<","<<maxDiff<<"] bad="<<bad<<"\n";reach.swap(next);} }
