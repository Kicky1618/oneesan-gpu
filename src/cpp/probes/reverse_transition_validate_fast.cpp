#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>
using oneesan::gridfp::MateID;
static std::vector<Mate> states(MateCodec const& mc){std::vector<Mate> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
static int height_above(MateID m,int W,int p){int h=1;for(int q=W-1;q>p;--q){auto v=oneesan::gridfp::mget(m,q);if(v==oneesan::gridfp::R)--h;else if(v==oneesan::gridfp::L)++h;}return h;}
static std::vector<MateID> rev(MateID b,int W,int p){std::vector<MateID>o;auto low=oneesan::gridfp::mget(b,p-1);if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){o.push_back(oneesan::gridfp::minsert(b,p,oneesan::gridfp::N));}else if(low==oneesan::gridfp::N){MateID u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N);if(height_above(u,W,p)>0)o.push_back(oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL));int s=1;for(int q=p-2;q>=0&&s>0;--q){auto v=oneesan::gridfp::mget(u,q);if(v==oneesan::gridfp::L&&s==1){auto c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::R);o.push_back(c);}if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;}s=1;for(int q=p+1;q<W&&s>0;++q){auto v=oneesan::gridfp::mget(u,q);if(v==oneesan::gridfp::R&&s==1){auto c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::L);o.push_back(c);}if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;}}std::sort(o.begin(),o.end());return o;}
int main(int ac,char**av){msg=NONE;int W=ac>1?atoi(av[1]):14;PathCounter<uint64_t>pc(W,W,false,false);auto ms=states(pc.mc),ds=states(pc.wc);for(int p=2;p<W;++p){std::unordered_map<MateID,std::vector<MateID>>br;br.reserve(ds.size()*2);for(auto m:ms){auto z=oneesan::gridfp::include_horizontal(m.id(),W,p);if(z.valid&&z.blocked)br[z.mate].push_back(m.id());}uint64_t bad=0;for(auto t:ds){auto a=rev(t.id(),W,p),b=br[t.id()];std::sort(b.begin(),b.end());if(a!=b){if(bad<3)std::cerr<<"bad p="<<p<<" target="<<t<<" got="<<a.size()<<" exp="<<b.size()<<"\n";++bad;}}std::cout<<"W="<<W<<" p="<<p<<" bad="<<bad<<"\n";if(bad)return 1;}}
