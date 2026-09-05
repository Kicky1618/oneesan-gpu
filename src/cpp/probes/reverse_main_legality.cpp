#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <iostream>
#include <unordered_set>
#include <vector>
static std::vector<Mate> states(MateCodec const& mc){std::vector<Mate> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
int main(int ac,char**av){msg=NONE;int W=ac>1?atoi(av[1]):14;PathCounter<uint64_t>pc(W,W,false,false);auto ms=states(pc.mc),ds=states(pc.wc);std::unordered_set<MateID>M,D;for(auto x:ms)M.insert(x.id());for(auto x:ds)D.insert(x.id());for(int p=2;p<W;++p){uint64_t cand=0,bad=0,bcand=0,bbad=0;for(auto x:ms){auto t=x.id();oneesan::gridfp::MateID pred=0;bool h=false;switch(oneesan::gridfp::mpair(t,p)){case oneesan::gridfp::LR:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);h=true;break;case oneesan::gridfp::NR:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);h=true;break;case oneesan::gridfp::NL:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);h=true;break;default:break;}if(h){++cand;if(!M.count(pred))++bad;}if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){++bcand;auto b=oneesan::gridfp::mshrink(t,p);if(!D.count(b))++bbad;}}std::cout<<"W="<<W<<" p="<<p<<" mainCand="<<cand<<" bad="<<bad<<" blockCand="<<bcand<<" bad="<<bbad<<"\n";}}
