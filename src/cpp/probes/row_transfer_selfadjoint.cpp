#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include "../../common/gridfp_transition.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
using oneesan::gridfp::MateID;
using C=unsigned long long;
static std::vector<MateID> states(MateCodec const& mc){std::vector<MateID> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=(b.mateL|b.mateR[i]).id();}return o;}
static void fwdrow(std::unordered_map<MateID,C>&M,std::unordered_map<MateID,C>&D,int W){std::unordered_map<MateID,C> nM,nD;for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto [m,c]:M){nM[m]+=c;auto z=oneesan::gridfp::include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto [b,c]:D)nM[oneesan::gridfp::blocked_exclude(b,p)]+=c;M.swap(nM);D.swap(nD);}}
static void revrow(std::unordered_map<MateID,C>&M,std::unordered_map<MateID,C>&D,int W,const std::vector<MateID>&allM,const std::vector<MateID>&allD){std::unordered_map<MateID,C> oM,oD;for(int p=1;p<W;++p){oM.clear();oD.clear();for(auto m:allM){C z=0;if(auto it=M.find(m);it!=M.end())z+=it->second;auto r=oneesan::gridfp::include_horizontal(m,W,p);if(r.valid){auto &mp=r.blocked?D:M;if(auto jt=mp.find(r.mate);jt!=mp.end())z+=jt->second;}if(z)oM.emplace(m,z);}for(auto b:allD){MateID t=oneesan::gridfp::blocked_exclude(b,p);if(auto it=M.find(t);it!=M.end()&&it->second)oD.emplace(b,it->second);}M.swap(oM);D.swap(oD);}}
static bool eq(const std::unordered_map<MateID,C>&A,const std::unordered_map<MateID,C>&B){if(A.size()!=B.size())return false;for(auto [k,v]:A){auto it=B.find(k);if(it==B.end()||it->second!=v)return false;}return true;}
int main(int argc,char**argv){msg=NONE;int W=argc>1?atoi(argv[1]):8;int K=argc>2?atoi(argv[2]):W;PathCounter<uint64_t> pc(W,W,false,false);auto allM=states(pc.mc),allD=states(pc.wc);std::unordered_map<MateID,C> FM,FD,RM,RD;FM[MateID(oneesan::gridfp::R)<<(2*(W-1))]=1;RM[MateID(oneesan::gridfp::R)]=1;for(int k=1;k<=K;++k){fwdrow(FM,FD,W);revrow(RM,RD,W,allM,allD);bool same=eq(FM,RM);C fsum=0,rsum=0,fmax=0,rmax=0;for(auto [_,v]:FM){fsum+=v;fmax=std::max(fmax,v);}for(auto [_,v]:RM){rsum+=v;rmax=std::max(rmax,v);}std::cout<<"W="<<W<<" k="<<k<<" fM="<<FM.size()<<" rM="<<RM.size()<<" same="<<same<<" fsum="<<fsum<<" rsum="<<rsum<<" fmax="<<fmax<<" rmax="<<rmax<<"\n";if(!same&&k<=3){int z=0;for(auto [m,v]:FM){auto it=RM.find(m);C r=it==RM.end()?0:it->second;if(v!=r&&z++<5)std::cerr<<std::hex<<m<<std::dec<<" f="<<v<<" r="<<r<<"\n";}}}}
