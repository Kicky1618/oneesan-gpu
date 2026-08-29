#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace oneesan::gridfp;
using Code=std::uint64_t;
using Delta=std::int64_t;

namespace {
constexpr int MAXW=11;
struct Spec{int width=0;std::uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+3]{};Code size=0;};
bool allowed(std::uint32_t f,std::uint32_t o,int p,MateValue v){if(!((f>>p)&1u))return v!=X;return ((o>>p)&1u)?(v==R||v==L):(v==N);}
Spec make_spec(int W,std::uint32_t f,std::uint32_t o){Spec s;s.width=W;s.fixed=f;s.occ=o;for(int h=0;h<=MAXW+2;++h)s.dp[0][h]=(h==0);for(int w=1;w<=W;++w){int p=w-1;bool ff=(f>>p)&1u,oo=(o>>p)&1u;for(int h=0;h<=MAXW+1;++h){Code x=0;if(!ff||!oo)x+=s.dp[w-1][h];if(!ff||oo){if(h>0)x+=s.dp[w-1][h-1];x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}s.size=s.dp[W][1];return s;}
std::vector<MateID> gen_valid(int W){std::vector<MateID>out;auto rec=[&](auto&&self,int pos,int h,MateID m)->void{int rem=W-pos;if(h<0||h>rem)return;if(pos==W){if(h==0)out.push_back(m);return;}int bit=W-1-pos;self(self,pos+1,h,m);if(h>0)self(self,pos+1,h-1,m|(MateID(R)<<(2*bit)));self(self,pos+1,h+1,m|(MateID(L)<<(2*bit)));};rec(rec,0,1,0);return out;}
bool in_spec(MateID m,int W,std::uint32_t f,std::uint32_t o){for(int p=0;p<W;++p)if((f>>p)&1u){MateValue v=mget(m,p);bool oo=(o>>p)&1u;if(oo?(v!=R&&v!=L):(v!=N))return false;}return true;}
Delta prefix_delta(MateID m,int W,int p,const Spec&ms,const Spec&bs){Code a=0,b=0;int h=1;for(int pos=W-1;pos>p;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(ms.fixed,ms.occ,pos,N))a+=ms.dp[pos][h];if(v>R&&h>0&&allowed(ms.fixed,ms.occ,pos,R))a+=ms.dp[pos][h-1];int q=pos-1;if(v>N&&allowed(bs.fixed,bs.occ,q,N))b+=bs.dp[q][h];if(v>R&&h>0&&allowed(bs.fixed,bs.occ,q,R))b+=bs.dp[q][h-1];if(v==R)--h;else if(v==L)++h;}return Delta(b)-Delta(a);}
std::vector<int> candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
void masks(int hi,int lo,const std::vector<int>&fp,std::uint32_t g,std::uint32_t&mf,std::uint32_t&mo,std::uint32_t&bf,std::uint32_t&bo){mf=mo=bf=bo=0;for(std::size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(g>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=q<lo-1?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
std::uint64_t pack(std::int32_t d,std::uint8_t h){return std::uint32_t(d)|(std::uint64_t(h)<<32);}
std::int32_t unpack_delta(std::uint64_t x){return std::int32_t(std::uint32_t(x));}
std::uint8_t unpack_height(std::uint64_t x){return std::uint8_t(x>>32);}
}

int main(){
    std::uint64_t checked=0;Delta max_abs=0;Code max_guard_size=0;
    for(int W=4;W<=MAXW;++W){auto all=gen_valid(W);for(int hi=W-1;hi>=2;--hi)for(int lo=1;lo<=hi;++lo){auto cand=candidates(W,hi,lo);int klim=std::min<int>(4,cand.size());for(int k=0;k<=klim;++k){std::vector<int>fp(cand.begin(),cand.begin()+k);for(std::uint32_t g=0;g<(1u<<k);++g){std::uint32_t mf,mo,bf,bo;masks(hi,lo,fp,g,mf,mo,bf,bo);Spec ms=make_spec(W,mf,mo),bs=make_spec(W-1,bf,bo);Code guard=std::max(ms.size,bs.size);max_guard_size=std::max(max_guard_size,guard);for(MateID m:all)if(in_spec(m,W,mf,mo)){for(int p=hi;p>=std::max(1,lo);--p){Delta d=prefix_delta(m,W,p,ms,bs);Delta ad=d<0?-d:d;max_abs=std::max(max_abs,ad);if(ad>Delta(guard))return 2;if(guard<=Code(std::numeric_limits<std::int32_t>::max())){std::int32_t sd=std::int32_t(d);std::uint8_t h=std::uint8_t((p+W)%29);auto x=pack(sd,h);if(unpack_delta(x)!=sd||unpack_height(x)!=h)return 3;}++checked;}}}}}}
    // Runtime safety argument: prefix ranks a,b are each local ranks inside
    // their group, so |b-a| <= max(main_group_size,block_group_size). Therefore
    // guarding both group sizes <= INT32_MAX is sufficient before packed use.
    std::cout<<"b300-rank-state-i32-bound-proof OK width_max="<<MAXW<<" checked="<<checked
             <<" max_abs_delta="<<max_abs<<" max_guard_size="<<max_guard_size
             <<" runtime_guard=max_group_size_le_int32max storage_bytes=8 delta_bits=32 height_bits=8"
             <<" pack_roundtrip=1 exact=1\n";
    return 0;
}
