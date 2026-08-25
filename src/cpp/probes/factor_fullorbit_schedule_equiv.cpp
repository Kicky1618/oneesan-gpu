#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "../../common/gridfp_transition.hpp"
using namespace oneesan::gridfp;
using Count=std::uint32_t;
static constexpr Count MOD=4294967291u;
static Count addmod(Count a,Count b){std::uint64_t z=std::uint64_t(a)+b;return Count(z>=MOD?z-MOD:z);}
static std::vector<MateID> enumerate_states(int width){std::vector<MateID>out;auto rec=[&](auto&&self,int pos,int h,MateID m)->void{if(pos<0){if(h==0)out.push_back(m);return;}if(h<0||h>pos+1)return;self(self,pos-1,h,m);if(h>0)self(self,pos-1,h-1,m|(MateID(R)<<(2*pos)));self(self,pos-1,h+1,m|(MateID(L)<<(2*pos)));};rec(rec,width-1,1,0);return out;}
static std::uint32_t occupancy(MateID m,int begin,int len){std::uint32_t z=0;for(int i=0;i<len;++i)if(mget(m,begin+i)!=N)z|=1u<<i;return z;}
struct Space{std::vector<MateID> ms,bs;std::unordered_map<MateID,std::size_t>mi,bi;};
static Space make_space(int W){Space s;s.ms=enumerate_states(W);s.bs=enumerate_states(W-1);s.mi.reserve(s.ms.size()*2+1);s.bi.reserve(s.bs.size()*2+1);for(std::size_t i=0;i<s.ms.size();++i)s.mi.emplace(s.ms[i],i);for(std::size_t i=0;i<s.bs.size();++i)s.bi.emplace(s.bs[i],i);return s;}
static bool standard_step(const Space&s,int W,int p,std::vector<Count>&m,std::vector<Count>&b){std::vector<Count>nm=m,nb(b.size(),0);for(std::size_t i=0;i<s.ms.size();++i){Count c=m[i];if(!c)continue;auto z=include_horizontal(s.ms[i],W,p);if(!z.valid)continue;if(z.blocked){auto it=s.bi.find(z.mate);if(it==s.bi.end())return false;nb[it->second]=addmod(nb[it->second],c);}else{auto it=s.mi.find(z.mate);if(it==s.mi.end())return false;nm[it->second]=addmod(nm[it->second],c);}}for(std::size_t i=0;i<s.bs.size();++i){Count c=b[i];if(!c)continue;MateID z=blocked_exclude(s.bs[i],p);auto it=s.mi.find(z);if(it==s.mi.end())return false;nm[it->second]=addmod(nm[it->second],c);}m.swap(nm);b.swap(nb);return true;}
static bool orbit_group_step(const Space&s,int W,int p,const std::vector<std::size_t>&mis,const std::vector<std::size_t>&bis,std::vector<Count>&m,std::vector<Count>&b){
 (void)bis;
 for(std::size_t gi:mis){MateID x=s.ms[gi];MateValuePair w=mpair(x,p);if(w!=NN&&w!=NR&&w!=NL)continue;MateValuePair cw=LR;if(w==NR)cw=RN;else if(w==NL)cw=LN;MateID companion=msetpair(x,p,cw),dropped=mshrink(x,p);auto jt=s.mi.find(companion);auto dt=s.bi.find(dropped);if(jt==s.mi.end()||dt==s.bi.end())return false;std::size_t j=jt->second,d=dt->second;Count c=m[gi],old_d=b[d];if(w==NN){m[j]=addmod(m[j],c);m[gi]=addmod(c,old_d);b[d]=0;}else{Count cc=m[j];Count all=addmod(addmod(c,cc),old_d);m[gi]=all;if(p==1){m[j]=addmod(c,cc);b[d]=0;}else b[d]=c;}}
 for(std::size_t gi:mis){MateID x=s.ms[gi];MateValuePair w=mpair(x,p);if(w!=LL&&w!=RR&&w!=RL)continue;Count c=m[gi];if(!c)continue;auto z=include_horizontal(x,W,p);if(!z.valid)continue;if(z.blocked){auto it=s.bi.find(z.mate);if(it==s.bi.end())return false;b[it->second]=addmod(b[it->second],c);}else{auto it=s.mi.find(z.mate);if(it==s.mi.end())return false;m[it->second]=addmod(m[it->second],c);}}
 return true;}
static bool same(const std::vector<Count>&a,const std::vector<Count>&b,const char*what,int row){if(a==b)return true;std::size_t i=0;while(i<a.size()&&a[i]==b[i])++i;std::cerr<<"fullorbit schedule mismatch "<<what<<" row="<<row<<" index="<<i<<" got="<<(i<a.size()?a[i]:0)<<" ref="<<(i<b.size()?b[i]:0)<<'\n';return false;}
int main(int argc,char**argv){int W=argc>1?std::atoi(argv[1]):10;int low=argc>2?std::atoi(argv[2]):5;int rows=argc>3?std::atoi(argv[3]):W;int high=W-1-low;if(W<4||W>13||low<1||high<1||rows<1){std::cerr<<"usage: factor_fullorbit_schedule_equiv [W<=13] [LOW] [rows]\n";return 1;}Space s=make_space(W);std::vector<std::vector<std::size_t>> high_groups(1u<<low),high_bgroups(1u<<low),low_groups(1u<<high),low_bgroups(1u<<high);for(std::size_t i=0;i<s.ms.size();++i){high_groups[occupancy(s.ms[i],0,low)].push_back(i);low_groups[occupancy(s.ms[i],low+1,high)].push_back(i);}for(std::size_t i=0;i<s.bs.size();++i){high_bgroups[occupancy(s.bs[i],0,low)].push_back(i);low_bgroups[occupancy(s.bs[i],low,high)].push_back(i);}
 std::vector<Count>refm(s.ms.size()),refb(s.bs.size()),gotm(s.ms.size()),gotb(s.bs.size());MateID init=MateID(R)<<(2*(W-1));auto it=s.mi.find(init);if(it==s.mi.end()){std::cerr<<"init missing\n";return 2;}refm[it->second]=gotm[it->second]=1;
 std::uint64_t group_steps=0;for(int row=0;row<rows;++row){for(int p=W-1;p>=low+1;--p)if(!standard_step(s,W,p,refm,refb))return 3;for(std::size_t mask=0;mask<high_groups.size();++mask)for(int p=W-1;p>=low+1;--p){if(!orbit_group_step(s,W,p,high_groups[mask],high_bgroups[mask],gotm,gotb))return 4;++group_steps;}if(!same(gotm,refm,"HIGH main",row)||!same(gotb,refb,"HIGH block",row))return 5;
  for(int p=low;p>=1;--p){ if(!standard_step(s,W,p,refm,refb)) return 6; }
  for(std::size_t mask=0;mask<low_groups.size();++mask){ for(int p=low;p>=1;--p){ if(!orbit_group_step(s,W,p,low_groups[mask],low_bgroups[mask],gotm,gotb)) return 7; ++group_steps; }}
  if(!same(gotm,refm,"LOW main",row)||!same(gotb,refb,"LOW block",row)) return 8;
 }
 MateID out=MateID(R);auto oi=s.mi.find(out);Count ans=oi==s.mi.end()?0:gotm[oi->second];std::cout<<"factor-fullorbit-schedule OK W="<<W<<" low="<<low<<" high="<<high<<" rows="<<rows<<" main_states="<<s.ms.size()<<" block_states="<<s.bs.size()<<" group_steps="<<group_steps<<" final_R="<<ans<<'\n';return 0;}
