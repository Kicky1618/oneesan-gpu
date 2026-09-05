#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>
using namespace oneesan::gridfp;
static int maxh(MateID m,int W){int h=1,mx=h;for(int p=W-1;p>=0;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;mx=std::max(mx,h);}return mx;}
static void genrec(int W,int pos,int h,MateID m,std::vector<MateID>&out){
  if(pos<0){if(h==0)out.push_back(m);return;}
  // N
  genrec(W,pos-1,h,m,out);
  if(h>0)genrec(W,pos-1,h-1,m|(MateID(R)<<(2*pos)),out);
  if(h+1<=W+1)genrec(W,pos-1,h+1,m|(MateID(L)<<(2*pos)),out);
}
static std::vector<MateID> gen(int W){std::vector<MateID>v;genrec(W,W-1,1,0,v);return v;}
static const char* pairname(MateValuePair x){switch(x){case NN:return"NN";case NR:return"NR";case NL:return"NL";case RN:return"RN";case RR:return"RR";case RL:return"RL";case LN:return"LN";case LR:return"LR";case LL:return"LL";default:return"XX";}}
int main(int argc,char**argv){int maxW=argc>1?atoi(argv[1]):12;for(int W=3;W<=maxW;++W){auto ms=gen(W),ds=gen(W-1);std::map<std::string,std::map<int,uint64_t>> hist;int maxinc=-99,maxb=-99;uint64_t valid=0;
  for(auto m:ms)for(int p=W-1;p>=1;--p){auto z=include_horizontal(m,W,p);if(!z.valid)continue;++valid;int d=maxh(z.mate,z.blocked?W-1:W)-maxh(m,W);hist[std::string(pairname(mpair(m,p)))+(z.blocked?"->D":"->M")][d]++;maxinc=std::max(maxinc,d);}
  for(auto b:ds)for(int p=W-1;p>=1;--p){auto t=blocked_exclude(b,p);int d=maxh(t,W)-maxh(b,W-1);hist["D->M"][d]++;maxb=std::max(maxb,d);}
  std::cout<<"W="<<W<<" M="<<ms.size()<<" D="<<ds.size()<<" valid="<<valid<<" max_include_delta="<<maxinc<<" max_D_delta="<<maxb<<"\n";
  for(auto const&[k,h]:hist){std::cout<<"  "<<k;for(auto [d,c]:h)std::cout<<" d"<<d<<"="<<c;std::cout<<"\n";}
}}
