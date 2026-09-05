#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>
using oneesan::gridfp::MateID;using C=unsigned long long;
static std::vector<MateID> states(MateCodec const&mc){std::vector<MateID>o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const&b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=(b.mateL|b.mateR[i]).id();}return o;}
static std::string show(MateID m,int W){std::string s;for(int i=W-1;i>=0;--i){auto v=oneesan::gridfp::mget(m,i);s+=(v==oneesan::gridfp::N?'N':v==oneesan::gridfp::R?'R':'L');}return s;}
int main(int argc,char**argv){msg=NONE;int rows=argc>1?atoi(argv[1]):2,maxW=argc>2?atoi(argv[2]):12;for(int W=2;W<=maxW;++W){PathCounter<uint64_t>pc(W,W,false,false);auto allM=states(pc.mc),allD=states(pc.wc);std::unordered_map<MateID,C>M,D,oM,oD;M[MateID(oneesan::gridfp::R)]=1;for(int row=0;row<rows;++row){for(int p=1;p<W;++p){oM.clear();oD.clear();for(auto m:allM){C z=0;if(auto it=M.find(m);it!=M.end())z+=it->second;auto r=oneesan::gridfp::include_horizontal(m,W,p);if(r.valid){auto &mp=r.blocked?D:M;if(auto jt=mp.find(r.mate);jt!=mp.end())z+=jt->second;}if(z)oM.emplace(m,z);}for(auto b:allD){MateID t=oneesan::gridfp::blocked_exclude(b,p);if(auto it=M.find(t);it!=M.end()&&it->second)oD.emplace(b,it->second);}M.swap(oM);D.swap(oD);}}for(auto [m,c]:M)std::cout<<W<<' '<<show(m,W)<<' '<<c<<'\n';}}
