#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
using namespace oneesan::gridfp;
using Code=std::uint64_t;
namespace {
constexpr int MW=11;
struct Spec{int w=0;std::uint32_t fixed=0,occ=0;Code dp[MW+1][MW+3]{};};
bool allowed(const Spec&s,int p,MateValue v){if(!((s.fixed>>p)&1u))return v!=X;bool o=(s.occ>>p)&1u;return o?(v==R||v==L):(v==N);}
Spec make_spec(int w,std::uint32_t fixed,std::uint32_t occ){Spec s;s.w=w;s.fixed=fixed;s.occ=occ;for(int h=0;h<=MW+2;++h)s.dp[0][h]=(h==0);for(int n=1;n<=w;++n){int p=n-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<=MW+1;++h){Code x=0;if(!f||!o)x+=s.dp[n-1][h];if(!f||o){if(h)x+=s.dp[n-1][h-1];x+=s.dp[n-1][h+1];}s.dp[n][h]=x;}}return s;}
std::vector<MateID> gen(int W){std::vector<MateID>o;auto rec=[&](auto&&self,int pos,int h,MateID m)->void{int rem=W-pos;if(h<0||h>rem)return;if(pos==W){if(!h)o.push_back(m);return;}int b=W-1-pos;self(self,pos+1,h,m);if(h)self(self,pos+1,h-1,m|(MateID(R)<<(2*b)));self(self,pos+1,h+1,m|(MateID(L)<<(2*b)));};rec(rec,0,1,0);return o;}
bool in_spec(MateID m,const Spec&s){for(int p=0;p<s.w;++p)if((s.fixed>>p)&1u){bool o=(s.occ>>p)&1u;MateValue v=mget(m,p);if(o?(v!=R&&v!=L):(v!=N))return false;}return true;}
Code rank(MateID m,const Spec&s){Code r=0;int h=1;for(int p=s.w-1;p>=0;--p){MateValue v=mget(m,p);if(v>N&&allowed(s,p,N))r+=s.dp[p][h];if(v>R&&h&&allowed(s,p,R))r+=s.dp[p][h-1];if(v==R)--h;else if(v==L)++h;}return r;}
int height_before(MateID m,int W,int p){int h=1;for(int q=W-1;q>p;--q){auto v=mget(m,q);if(v==R)--h;else if(v==L)++h;}return h;}
Code direct_source_rank(Code i,MateID d,int p,const Spec&s){int h=height_before(d,s.w,p);switch(mpair(d,p)){
case LR:{Code z=s.dp[p][h]+(h?s.dp[p][h-1]:0)+s.dp[p-1][h+1];return i-z;}
case NR:{Code z=s.dp[p][h]-s.dp[p-1][h];return i+z;}
case NL:{Code a=s.dp[p][h]+(h?s.dp[p][h-1]:0),b=s.dp[p-1][h]+(h?s.dp[p-1][h-1]:0);return i+(a-b);}
default:return i;}}
std::vector<int> cand(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
}
int main(){std::uint64_t checked=0;for(int W=4;W<=MW;++W){auto all=gen(W);for(int hi=W-1;hi>=2;--hi)for(int lo=2;lo<=hi;++lo){auto c=cand(W,hi,lo);int k=std::min<int>(4,c.size());for(std::uint32_t g=0;g<(1u<<k);++g){std::uint32_t f=0,o=0;for(int x=0;x<k;++x){int q=c[x];f|=1u<<q;if((g>>x)&1u)o|=1u<<q;}Spec s=make_spec(W,f,o);std::vector<MateID>v;for(auto m:all)if(in_spec(m,s))v.push_back(m);std::unordered_map<MateID,Code>idx;for(Code i=0;i<v.size();++i){if(rank(v[i],s)!=i)return 2;idx[v[i]]=i;}for(int p=lo;p<=hi;++p){if((f>>p&1u)||(f>>(p-1)&1u))return 3;for(Code i=0;i<v.size();++i){MateID d=v[i],src=0;bool has=true;switch(mpair(d,p)){case LR:src=msetpair(d,p,NN);break;case NR:src=msetpair(d,p,RN);break;case NL:src=msetpair(d,p,LN);break;default:has=false;}if(!has)continue;auto it=idx.find(src);if(it==idx.end())return 4;if(direct_source_rank(i,d,p,s)!=it->second){std::cerr<<"mismatch W="<<W<<" p="<<p<<" i="<<i<<'\n';return 5;}++checked;}}}}}
std::cout<<"b300-main-pull-direct-pair-rank-proof OK width_max="<<MW<<" checked="<<checked<<" rank_slice_calls=0 exact=1\n";}
