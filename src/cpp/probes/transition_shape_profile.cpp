#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <array>
#include <iomanip>
#include <iostream>
#include <vector>
static std::vector<Mate> states(MateCodec const& mc){std::vector<Mate> o(mc.codeSize());for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i)o[b.base+i]=b.mateL|b.mateR[i];}return o;}
int main(int argc,char**argv){msg=NONE;int W=argc>1?std::atoi(argv[1]):16;PathCounter<uint64_t> pc(W,W,false,false);auto ss=states(pc.mc);std::cout<<"W="<<W<<" states="<<ss.size()<<"\n";uint64_t all_cl=0,all_dist=0;for(int p=1;p<W;++p){std::array<uint64_t,16>cnt{};uint64_t dist=0,closures=0,maxd=0;for(auto m:ss){auto w=m.getPair(p);++cnt[w];if(w==LL){Mate t=m;t.setPair(p,NN);int q=p-1,s=1;while(s){--q;auto v=t.get(q);if(v==L)++s;else if(v==R)--s;}uint64_t d=(p-1)-q;dist+=d;maxd=std::max(maxd,d);++closures;}else if(w==RR){Mate t=m;t.setPair(p,NN);int q=p,s=1;while(s){++q;auto v=t.get(q);if(v==L)--s;else if(v==R)++s;}uint64_t d=q-p;dist+=d;maxd=std::max(maxd,d);++closures;}}all_cl+=closures;all_dist+=dist;auto frac=[&](int w){return 100.0*cnt[w]/ss.size();};std::cout<<"p="<<std::setw(2)<<p<<" NN="<<std::fixed<<std::setprecision(1)<<frac(NN)<<" NR/NL="<<frac(NR)+frac(NL)<<" RN/LN="<<frac(RN)+frac(LN)<<" LL/RR="<<frac(LL)+frac(RR)<<" RL="<<frac(RL)<<" closure_avg_dist="<<(closures?double(dist)/closures:0.0)<<" max="<<maxd<<"\n";}std::cout<<"overall closure fraction="<<100.0*all_cl/(ss.size()*(W-1))<<" avg_dist="<<(all_cl?double(all_dist)/all_cl:0.0)<<"\n";}
