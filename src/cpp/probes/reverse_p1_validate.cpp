#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using oneesan::gridfp::MateID;
static std::vector<Mate> states(MateCodec const& mc){std::vector<Mate> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
static int hbefore(MateID m,int W,int p){int h=1;for(int q=W-1;q>p;--q){auto v=oneesan::gridfp::mget(m,q);if(v==oneesan::gridfp::R)--h;else if(v==oneesan::gridfp::L)++h;}return h;}
static std::vector<MateID> rev(MateID t,int W){namespace g=oneesan::gridfp;std::vector<MateID>o;auto w=g::mpair(t,1);switch(w){case g::LR:o.push_back(g::msetpair(t,1,g::NN));break;case g::RN:o.push_back(g::msetpair(t,1,g::NR));break;case g::LN:o.push_back(g::msetpair(t,1,g::NL));break;case g::NR:o.push_back(g::msetpair(t,1,g::RN));break;case g::NL:o.push_back(g::msetpair(t,1,g::LN));break;case g::NN:{if(hbefore(t,W,1)>0)o.push_back(g::msetpair(t,1,g::RL));int s=1;for(int q=2;q<W&&s>0;++q){auto v=g::mget(t,q);if(v==g::R&&s==1){auto c=g::msetpair(t,1,g::RR);c=g::mset(c,q,g::L);o.push_back(c);}if(v==g::L)--s;else if(v==g::R)++s;}break;}default:break;}std::sort(o.begin(),o.end());return o;}
int main(int ac,char**av){msg=NONE;int W=ac>1?atoi(av[1]):14;PathCounter<uint64_t>pc(W,W,false,false);auto ms=states(pc.mc);std::unordered_map<MateID,std::vector<MateID>>br;br.reserve(ms.size()*2);for(auto m:ms){auto z=oneesan::gridfp::include_horizontal(m.id(),W,1);if(z.valid&&!z.blocked)br[z.mate].push_back(m.id());}uint64_t bad=0;for(auto t:ms){auto a=rev(t.id(),W),b=br[t.id()];std::sort(b.begin(),b.end());if(a!=b){if(bad<5)std::cerr<<"bad target="<<t<<" got="<<a.size()<<" exp="<<b.size()<<"\n";++bad;}}std::cout<<"W="<<W<<" bad="<<bad<<"\n";return bad?1:0;}
