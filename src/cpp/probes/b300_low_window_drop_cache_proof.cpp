#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace oneesan::gridfp;
using Code=std::uint64_t;

namespace {
constexpr int MAXW=28;
Code H[MAXW+1][MAXW+3]{};
struct Spec{int w=0;std::uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+3]{};};
struct Entry{std::uint32_t code=0,base=0;int h=0;};
void build_full(){for(int h=0;h<=MAXW+2;++h)H[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW+1;++h)H[w][h]=H[w-1][h]+(h?H[w-1][h-1]:0)+H[w-1][h+1];}
bool allowed(const Spec&s,int p,MateValue v){if(!((s.fixed>>p)&1u))return v!=X;return ((s.occ>>p)&1u)?(v==R||v==L):(v==N);}
Spec make_spec(int w,std::uint32_t fixed,std::uint32_t occ){Spec s;s.w=w;s.fixed=fixed;s.occ=occ;for(int h=0;h<=MAXW+2;++h)s.dp[0][h]=(h==0);for(int n=1;n<=w;++n){int p=n-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<=MAXW+1;++h){Code x=0;if(!f||!o)x+=s.dp[n-1][h];if(!f||o){if(h)x+=s.dp[n-1][h-1];x+=s.dp[n-1][h+1];}s.dp[n][h]=x;}}return s;}
Code rank_group(MateID m,const Spec&s){Code r=0;int h=1;for(int p=s.w-1;p>=0;--p){auto v=mget(m,p);if(v>N&&allowed(s,p,N))r+=s.dp[p][h];if(v>R&&h&&allowed(s,p,R))r+=s.dp[p][h-1];if(v==R)--h;else if(v==L)++h;}return r;}
bool in_spec(MateID m,const Spec&s){for(int p=0;p<s.w;++p)if((s.fixed>>p)&1u){auto v=mget(m,p);bool o=(s.occ>>p)&1u;if(o?(v!=R&&v!=L):(v!=N))return false;}return true;}
std::vector<MateID> gen_valid(int W){std::vector<MateID>o;auto rec=[&](auto&&self,int pos,int h,MateID m)->void{int rem=W-pos;if(h<0||h>rem)return;if(pos==W){if(h==0)o.push_back(m);return;}int b=W-1-pos;self(self,pos+1,h,m);if(h)self(self,pos+1,h-1,m|(MateID(R)<<(2*b)));self(self,pos+1,h+1,m|(MateID(L)<<(2*b)));};rec(rec,0,1,0);return o;}
std::vector<Entry> high_entries(int K,std::uint32_t occ,int lowlen){std::vector<Entry>out;std::uint64_t base=0;auto rec=[&](auto&&self,int pos,int h,std::uint32_t code)->void{if(pos<0){Code cnt=H[lowlen][h];if(!cnt)return;if(base>0xffffffffULL||base+cnt>0x100000000ULL){std::cerr<<"base overflow\n";std::exit(2);}out.push_back({code,(std::uint32_t)base,h});base+=cnt;return;}if(!((occ>>pos)&1u)){self(self,pos-1,h,code);return;}if(h)self(self,pos-1,h-1,code|(std::uint32_t(R)<<(2*pos)));self(self,pos-1,h+1,code|(std::uint32_t(L)<<(2*pos)));};rec(rec,K-1,1,0);return out;}
long long local_delta(MateID m,int top,int p,int enter_h){long long d=0;int h=enter_h;for(int pos=top;pos>p;--pos){auto v=mget(m,pos);if(v==R){d+=static_cast<long long>(H[pos-1][h])-static_cast<long long>(H[pos][h]);--h;}else if(v==L){Code b=H[pos-1][h]+(h?H[pos-1][h-1]:0),a=H[pos][h]+(h?H[pos][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}}return d;}
int enter_height(MateID m,int W,int lowlen){int h=1;for(int p=W-1;p>=lowlen;--p){auto v=mget(m,p);if(v==R)--h;else if(v==L)++h;}return h;}
}

int main(){build_full();
    // Production W28/K13 packing bounds.
    constexpr int K=13;std::uint64_t paired=0;std::uint32_t max_mag=0;int max_h=0;
    for(std::uint32_t mask=0;mask<(1u<<K);++mask){auto a=high_entries(K,mask,15),b=high_entries(K,mask,14);std::size_t j=0;for(auto const&e:a){while(j<b.size()&&b[j].code<e.code)++j;if(j>=b.size()||b[j].code!=e.code){std::cerr<<"missing block prefix mask="<<mask<<" code="<<e.code<<'\n';return 3;}if(b[j].base>e.base){std::cerr<<"delta sign\n";return 4;}max_mag=std::max(max_mag,e.base-b[j].base);max_h=std::max(max_h,e.h);++paired;}}
    if(max_mag>=(1u<<30)||max_h>15){std::cerr<<"packing overflow mag="<<max_mag<<" h="<<max_h<<'\n';return 5;}

    // Exhaustive small-width proof of rank = main_rank - packed_base_delta +
    // free-low-prefix delta for every deletable N in the low window.
    std::uint64_t checked=0;
    for(int W=6;W<=10;++W){int k=std::min(3,W-3),lowlen=W-k,top=lowlen-1;auto allm=gen_valid(W),allb=gen_valid(W-1);for(std::uint32_t mask=0;mask<(1u<<k);++mask){std::uint32_t mf=((1u<<k)-1u)<<lowlen,mo=mask<<lowlen,bf=((1u<<k)-1u)<<(lowlen-1),bo=mask<<(lowlen-1);Spec ms=make_spec(W,mf,mo),bs=make_spec(W-1,bf,bo);std::vector<MateID>mv,bv;for(auto m:allm)if(in_spec(m,ms))mv.push_back(m);for(auto b:allb)if(in_spec(b,bs))bv.push_back(b);std::unordered_map<MateID,Code>bi;for(Code i=0;i<bv.size();++i){if(rank_group(bv[i],bs)!=i)return 6;bi[bv[i]]=i;}auto hm=high_entries(k,mask,lowlen),hb=high_entries(k,mask,lowlen-1);std::unordered_map<std::uint32_t,std::uint32_t>mb,bb;for(auto const&e:hm)mb[e.code]=e.base;for(auto const&e:hb)bb[e.code]=e.base;for(Code i=0;i<mv.size();++i){MateID m=mv[i];if(rank_group(m,ms)!=i)return 7;std::uint32_t code=std::uint32_t(m>>(2*lowlen));auto im=mb.find(code),ib=bb.find(code);if(im==mb.end()||ib==bb.end())return 8;std::uint32_t mag=im->second-ib->second;int eh=enter_height(m,W,lowlen);for(int p=top;p>=2;--p)if(mget(m,p)==N){MateID b=mshrink(m,p);auto it=bi.find(b);if(it==bi.end())continue;long long delta=-static_cast<long long>(mag)+local_delta(m,top,p,eh);Code got=delta>=0?i+Code(delta):i-Code(-delta);if(got!=it->second){std::cerr<<"decomp mismatch W="<<W<<" mask="<<mask<<" p="<<p<<'\n';return 9;}++checked;}}}}
    std::cout<<"b300-low-window-drop-cache-proof OK w28_high_prefixes="<<paired<<" max_base_delta="<<max_mag<<" max_entry_height="<<max_h<<" packed_bits=64 cache_extra_bytes=0 exhaustive_width_max=10 checked_drop_cases="<<checked<<" high_prefix_walk=0 low_drop_max_steps=12 exact=1\n";
}
