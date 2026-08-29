#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>
using namespace oneesan::gridfp;
using Code=std::uint64_t;
namespace {
constexpr int W=28,K=13,LOWB=14,LOWM=15,MAXW=28;
Code H[MAXW+1][MAXW+3]{};
struct Spec{int w=0;std::uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+3]{};};
struct E{std::uint32_t code=0,base=0;int h=0;};
void build_full(){for(int h=0;h<=MAXW+2;++h)H[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW+1;++h)H[w][h]=H[w-1][h]+(h?H[w-1][h-1]:0)+H[w-1][h+1];}
Spec make_spec(int w,std::uint32_t fixed,std::uint32_t occ){Spec s;s.w=w;s.fixed=fixed;s.occ=occ;for(int h=0;h<=MAXW+2;++h)s.dp[0][h]=(h==0);for(int n=1;n<=w;++n){int p=n-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<=MAXW+1;++h){Code x=0;if(!f||!o)x+=s.dp[n-1][h];if(!f||o){if(h)x+=s.dp[n-1][h-1];x+=s.dp[n-1][h+1];}s.dp[n][h]=x;}}return s;}
bool allowed(const Spec&s,int p,MateValue v){if(!((s.fixed>>p)&1u))return v!=X;return ((s.occ>>p)&1u)?(v==R||v==L):(v==N);}
Code rank_group(MateID m,const Spec&s){Code r=0;int h=1;for(int p=s.w-1;p>=0;--p){auto v=mget(m,p);if(v>N&&allowed(s,p,N))r+=s.dp[p][h];if(v>R&&h&&allowed(s,p,R))r+=s.dp[p][h-1];if(v==R)--h;else if(v==L)++h;}return r;}
MateID unrank_free(Code rank,int width,int h){MateID m=0;for(int p=width-1;p>=0;--p){Code z=H[p][h];if(rank<z)continue;rank-=z;if(h){z=H[p][h-1];if(rank<z){m|=MateID(R)<<(2*p);--h;continue;}rank-=z;}m|=MateID(L)<<(2*p);++h;}return m;}
std::vector<E> high_entries(std::uint32_t mask,int lowlen){std::vector<E>out;std::uint64_t base=0;auto rec=[&](auto&&self,int pos,int h,std::uint32_t code)->void{if(pos<0){Code cnt=H[lowlen][h];if(!cnt)return;if(base+cnt>0x100000000ULL){std::cerr<<"base overflow\n";std::exit(2);}out.push_back({code,std::uint32_t(base),h});base+=cnt;return;}if(!((mask>>pos)&1u)){self(self,pos-1,h,code);return;}if(h)self(self,pos-1,h-1,code|(std::uint32_t(R)<<(2*pos)));self(self,pos-1,h+1,code|(std::uint32_t(L)<<(2*pos)));};rec(rec,K-1,1,0);return out;}
long long local_drop_delta(MateID full,int p,int h){long long d=0;for(int pos=14;pos>p;--pos){auto v=mget(full,pos);if(v==R){d+=static_cast<long long>(H[pos-1][h])-static_cast<long long>(H[pos][h]);--h;}else if(v==L){Code b=H[pos-1][h]+(h?H[pos-1][h-1]:0),a=H[pos][h]+(h?H[pos][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}}return d;}
}
int main(){build_full();
    Code max_local=0,total_low=0;for(int h=0;h<MAXW+2;++h){if(H[LOWB][h])max_local=std::max(max_local,H[LOWB][h]-1);total_low+=H[LOWB][h];}
    std::uint32_t max_rel=0,max_delta=0;int max_h=0;std::uint64_t prefixes=0;
    std::mt19937_64 rng(0x6c6f77626c6f636bULL);std::uint64_t sampled_lifts=0;
    for(std::uint32_t mask=0;mask<(1u<<K);++mask){auto me=high_entries(mask,LOWM),be=high_entries(mask,LOWB);std::unordered_map<std::uint32_t,E> mm;mm.reserve(me.size()*2+1);for(auto e:me)mm.emplace(e.code,e);max_rel=std::max(max_rel,std::uint32_t(be.size()?be.size()-1:0));for(std::uint32_t rel=0;rel<be.size();++rel){auto b=be[rel];auto it=mm.find(b.code);if(it==mm.end()){std::cerr<<"missing matching main prefix mask="<<mask<<" code="<<b.code<<'\n';return 3;}auto m=it->second;if(m.base<b.base){std::cerr<<"base sign\n";return 4;}max_delta=std::max(max_delta,m.base-b.base);max_h=std::max(max_h,b.h);++prefixes;
            if(((rel+mask*17u)%97u)!=0u||H[LOWB][b.h]==0)continue;
            std::uint32_t mf=((1u<<K)-1u)<<LOWM,mo=mask<<LOWM,bf=((1u<<K)-1u)<<LOWB,bo=mask<<LOWB;Spec ms=make_spec(W,mf,mo),bs=make_spec(W-1,bf,bo);
            const Code cnt=H[LOWB][b.h];std::array<Code,3> rs={0,cnt/2,cnt-1};for(Code lr:rs){MateID low=unrank_free(lr,LOWB,b.h);MateID bm=low|(MateID(b.code)<<(2*LOWB));const Code br=Code(b.base)+lr;if(rank_group(bm,bs)!=br){std::cerr<<"block rank mismatch\n";return 5;}for(int p=2;p<=14;++p){MateID full=minsert(bm,p,N);Code mr=rank_group(full,ms);long long local=local_drop_delta(full,p,b.h);long long lift=static_cast<long long>(m.base-b.base)-local;Code got=lift>=0?br+Code(lift):br-Code(-lift);if(got!=mr){std::cerr<<"lift mismatch mask="<<mask<<" rel="<<rel<<" p="<<p<<" br="<<br<<" got="<<got<<" mr="<<mr<<'\n';return 6;}++sampled_lifts;}}}}
    if(max_local>=(1u<<18)||max_h>=16||max_rel>=(1u<<12)||max_delta>=(1u<<30))return 7;
    std::cout<<"b300-low-block-cache-proof OK prefixes="<<prefixes<<" max_low_local="<<max_local<<" total_low_decode="<<total_low<<" max_height="<<max_h<<" max_high_rel="<<max_rel<<" max_base_delta="<<max_delta<<" packed_bits=64 extra_per_state_bytes=0 sampled_lifts="<<sampled_lifts<<" exact=1\n";
}
