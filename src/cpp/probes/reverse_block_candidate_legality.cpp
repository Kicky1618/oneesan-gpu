#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <array>
#include <iostream>
#include <unordered_set>
#include <vector>
static std::vector<Mate> states(MateCodec const& mc){std::vector<Mate> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
int main(int ac,char**av){msg=NONE;int W=ac>1?atoi(av[1]):14;PathCounter<uint64_t>pc(W,W,false,false);auto ms=states(pc.mc),ds=states(pc.wc);std::unordered_set<MateID>M;M.reserve(ms.size()*2);for(auto x:ms)M.insert(x.id());for(int p=2;p<W;++p){std::array<uint64_t,4> cand{},bad{};for(auto tb:ds){auto b=tb.id();auto low=oneesan::gridfp::mget(b,p-1);if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){auto c=oneesan::gridfp::minsert(b,p,oneesan::gridfp::N);cand[0]++;bad[0]+=!M.count(c);}else if(low==oneesan::gridfp::N){auto u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N);auto c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL);cand[1]++;bad[1]+=!M.count(c);for(int q=0;q<=p-2;++q)if(oneesan::gridfp::mget(u,q)==oneesan::gridfp::L){auto x=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);x=oneesan::gridfp::mset(x,q,oneesan::gridfp::R);cand[2]++;bad[2]+=!M.count(x);}for(int q=p+1;q<W;++q)if(oneesan::gridfp::mget(u,q)==oneesan::gridfp::R){auto x=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);x=oneesan::gridfp::mset(x,q,oneesan::gridfp::L);cand[3]++;bad[3]+=!M.count(x);}}}std::cout<<"W="<<W<<" p="<<p;const char*n[]={"NRNL","RL","LL","RR"};for(int k=0;k<4;++k)std::cout<<' '<<n[k]<<'='<<cand[k]<<'/'<<bad[k];std::cout<<'\n';}}
