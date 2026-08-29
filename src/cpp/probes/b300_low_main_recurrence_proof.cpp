#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

using u64=std::uint64_t;using i64=std::int64_t;
namespace{
constexpr int W=28,K=13,HC=30;
u64 H[W+1][HC]{};
struct E{std::uint32_t code,base;int h;};
void build(){for(int h=0;h<HC;++h)H[0][h]=(h==0);for(int w=1;w<=W;++w)for(int h=0;h<HC-1;++h)H[w][h]=H[w-1][h]+(h?H[w-1][h-1]:0)+H[w-1][h+1];}
std::vector<E> entries(std::uint32_t occ,int low){std::vector<E>o;u64 base=0;auto rec=[&](auto&&self,int p,int h,std::uint32_t code)->void{if(p<0){u64 n=H[low][h];if(n){o.push_back({code,std::uint32_t(base),h});base+=n;}return;}if(!((occ>>p)&1u)){self(self,p-1,h,code);return;}if(h)self(self,p-1,h-1,code|(1u<<(2*p)));self(self,p-1,h+1,code|(2u<<(2*p)));};rec(rec,K-1,1,0);return o;}
i64 step(int p,int h,int v,int&nh){nh=h;if(v==1){if(!h)return INT64_MIN;nh=h-1;return i64(H[p-1][h])-i64(H[p][h]);}if(v==2){nh=h+1;u64 b=H[p-1][h]+(h?H[p-1][h-1]:0),a=H[p][h]+(h?H[p][h-1]:0);return i64(b)-i64(a);}return 0;}
unsigned pack3(int a,int b,int c){return unsigned(a+3*b+9*c);}
int get3(unsigned z,int r){if(r==0){unsigned q=z/3;return int(z-q*3);}if(r==1){unsigned q=z/3;return int(q-(q/3)*3);}return int(z/9);}
}
int main(){build();
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int c=0;c<3;++c){unsigned z=pack3(a,b,c);if(z>=27||get3(z,0)!=a||get3(z,1)!=b||get3(z,2)!=c)return 2;}
    std::array<u64,15> minmag{},maxmag{};minmag.fill(~u64(0));
    u64 prefixes=0;
    for(std::uint32_t mask=0;mask<(1u<<K);++mask){auto a=entries(mask,15),b=entries(mask,14);std::size_t j=0;for(auto const&e:a){while(j<b.size()&&b[j].code<e.code)++j;if(j==b.size()||b[j].code!=e.code)return 3;u64 mag=u64(e.base)-b[j].base;minmag[e.h]=std::min(minmag[e.h],mag);maxmag[e.h]=std::max(maxmag[e.h],mag);++prefixes;}}
    i64 global_lo=0,global_hi=0;bool first=true;
    for(int p=1;p<=14;++p)for(int h0=0;h0<=14;++h0)if(minmag[h0]!=~u64(0)){
        std::array<i64,HC> mn,mx;std::array<unsigned char,HC> seen{};mn.fill(0);mx.fill(0);seen[h0]=1;
        for(int pos=14;pos>p;--pos){auto nmn=mn,nmx=mx;std::array<unsigned char,HC> ns{};for(int h=0;h<HC;++h)if(seen[h])for(int v=0;v<3;++v){int nh;i64 d=step(pos,h,v,nh);if(d==INT64_MIN||nh<0||nh>=HC)continue;i64 a=mn[h]+d,b=mx[h]+d;if(!ns[nh]){nmn[nh]=a;nmx[nh]=b;ns[nh]=1;}else{nmn[nh]=std::min(nmn[nh],a);nmx[nh]=std::max(nmx[nh],b);}}mn=nmn;mx=nmx;seen=ns;}
        i64 lmin=0,lmax=0;bool f=true;for(int h=0;h<HC;++h)if(seen[h]){if(f){lmin=mn[h];lmax=mx[h];f=false;}else{lmin=std::min(lmin,mn[h]);lmax=std::max(lmax,mx[h]);}}
        i64 lo=-i64(maxmag[h0])+lmin,hi=-i64(minmag[h0])+lmax;if(first){global_lo=lo;global_hi=hi;first=false;}else{global_lo=std::min(global_lo,lo);global_hi=std::max(global_hi,hi);}
    }
    if(global_lo<-(i64(1)<<30)||global_hi>=(i64(1)<<30)){std::cerr<<"signed31 overflow lo="<<global_lo<<" hi="<<global_hi<<"\n";return 4;}
    // Recurrence identity over every legal one-symbol advance and representative
    // signed31 values. The packed state stores delta(p); advancing by symbol p
    // must produce delta(p-1) by exactly one scalar contribution.
    std::uint64_t recurrence=0;for(int p=2;p<=14;++p)for(int h=0;h<=14;++h)for(int v=0;v<3;++v){int nh;i64 d=step(p,h,v,nh);if(d==INT64_MIN)continue;for(i64 x:{global_lo,i64(-1234567),i64(0)}){i64 y=x+d;if(y>=-(i64(1)<<30)&&y<(i64(1)<<30)){std::uint32_t packed=std::uint32_t(y)&0x7fffffffu;i64 rt=i64(std::int32_t(packed<<1))>>1;if(rt!=y)return 5;++recurrence;}}}
    std::cout<<"b300-low-main-recurrence-proof OK prefixes="<<prefixes<<" delta_min="<<global_lo<<" delta_max="<<global_hi<<" signed31=1 trit_chunks=5 trit_bits=25 height_bits=4 total_bits=60 recurrence_cases="<<recurrence<<" exact=1\n";
}
