#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using Code=std::uint64_t;using Delta=std::int64_t;
namespace{
constexpr int W=28,HC=30;
struct Spec{int width=0;std::uint32_t fixed=0,occ=0;Code dp[W+1][HC]{};};
Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){Spec s;s.width=width;s.fixed=fixed;s.occ=occ;for(int h=0;h<HC;++h)s.dp[0][h]=(h==0);for(int w=1;w<=width;++w){int p=w-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<HC-1;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h)x+=s.dp[w-1][h-1];x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}return s;}
Delta contrib(const Spec&m,const Spec&b,int pos,int h,int v,int&nh){nh=h;if(v==1){if(h<=0)return std::numeric_limits<Delta>::min();nh=h-1;return Delta(b.dp[pos-1][h])-Delta(m.dp[pos][h]);}if(v==2){if(h>=HC-1)return std::numeric_limits<Delta>::min();Code bb=b.dp[pos-1][h]+(h?b.dp[pos-1][h-1]:0),aa=m.dp[pos][h]+(h?m.dp[pos][h-1]:0);nh=h+1;return Delta(bb)-Delta(aa);}return 0;}
unsigned chunk3(int a,int b,int c){return unsigned(a+3*b+9*c);}
int get3(unsigned z,int r){if(r==0){unsigned q=z/3;return int(z-q*3);}if(r==1){unsigned q=z/3;return int(q-(q/3)*3);}return int(z/9);}
}
int main(){
    // Five 5-bit base-3 chunks cover positions 14..27 (14 trits + one N pad).
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int c=0;c<3;++c){unsigned z=chunk3(a,b,c);if(z>=27||get3(z,0)!=a||get3(z,1)!=b||get3(z,2)!=c)return 2;}

    Delta glo=0,ghi=0;bool first=true;std::uint64_t specs=0,states=0;
    // High-window planner candidates are positions 13,12,...,0. Prove every
    // prefix length k and every occupancy assignment, not just the selected k.
    for(int k=0;k<=14;++k){
        std::uint32_t fixed=0;for(int j=0;j<k;++j)fixed|=1u<<(13-j);
        const std::uint32_t groups=1u<<k;
        for(std::uint32_t g=0;g<groups;++g){
            std::uint32_t occ=0;for(int j=0;j<k;++j)if((g>>j)&1u)occ|=1u<<(13-j);
            Spec m=make_spec(28,fixed,occ),b=make_spec(27,fixed,occ);++specs;
            std::array<Delta,HC> mn{},mx{};std::array<unsigned char,HC> seen{};seen[1]=1;mn[1]=mx[1]=0;
            auto record=[&](){for(int h=0;h<HC;++h)if(seen[h]){if(first){glo=mn[h];ghi=mx[h];first=false;}else{glo=std::min(glo,mn[h]);ghi=std::max(ghi,mx[h]);}++states;}};
            record(); // current p=27: no high prefix yet
            for(int pos=27;pos>=15;--pos){
                std::array<Delta,HC> nmn{},nmx{};std::array<unsigned char,HC> ns{};
                for(int h=0;h<HC;++h)if(seen[h])for(int v=0;v<3;++v){int nh;Delta d=contrib(m,b,pos,h,v,nh);if(d==std::numeric_limits<Delta>::min())continue;Delta lo=mn[h]+d,hi=mx[h]+d;if(!ns[nh]){ns[nh]=1;nmn[nh]=lo;nmx[nh]=hi;}else{nmn[nh]=std::min(nmn[nh],lo);nmx[nh]=std::max(nmx[nh],hi);}}
                seen=ns;mn=nmn;mx=nmx;record();
            }
        }
    }
    constexpr Delta LIM=Delta(1)<<34; // signed 35-bit range
    if(glo < -LIM || ghi >= LIM){std::cerr<<"signed35 overflow min="<<glo<<" max="<<ghi<<"\n";return 3;}

    // Exhaustively verify 14-trit pack/decode mapping (3^14 states).
    std::uint64_t total=1;for(int i=0;i<14;++i)total*=3;for(std::uint64_t x=0;x<total;++x){std::uint64_t q=x,packed=0;int a[14]{};for(int i=0;i<14;++i){a[i]=int(q%3);q/=3;}for(int c=0;c<5;++c){int p=3*c;unsigned z=chunk3(p<14?a[p]:0,p+1<14?a[p+1]:0,p+2<14?a[p+2]:0);packed|=std::uint64_t(z)<<(5*c);}for(int i=0;i<14;++i){unsigned z=unsigned((packed>>(5*(i/3)))&31u);if(get3(z,i%3)!=a[i])return 4;}}

    std::cout<<"b300-high-main-recurrence-proof OK planner_specs="<<specs<<" propagated_states="<<states<<" delta_min="<<glo<<" delta_max="<<ghi<<" signed35=1 trit_positions=14 trit_chunks=5 trit_bits=25 height_bits=4 total_bits=64 exhaustive_trit_states="<<total<<" exact=1\n";
}
