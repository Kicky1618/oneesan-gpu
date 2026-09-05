#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

using Code=std::uint64_t;using Delta=std::int64_t;
namespace{
constexpr int W=28,HC=30,MIN_FIXED=7,TRITS=14,MAX_FIXED=14;
struct Spec{int width=0;std::uint32_t fixed=0,occ=0;Code dp[W+1][HC]{};};
Spec make_spec(int width,std::uint32_t fixed,std::uint32_t occ){Spec s;s.width=width;s.fixed=fixed;s.occ=occ;for(int h=0;h<HC;++h)s.dp[0][h]=(h==0);for(int w=1;w<=width;++w){int p=w-1;bool f=(fixed>>p)&1u,o=(occ>>p)&1u;for(int h=0;h<HC-1;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h)x+=s.dp[w-1][h-1];x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}return s;}
Delta contrib(const Spec&m,const Spec&b,int pos,int h,int v,int&nh){nh=h;if(v==1){if(h<=0)return std::numeric_limits<Delta>::min();nh=h-1;return Delta(b.dp[pos-1][h])-Delta(m.dp[pos][h]);}if(v==2){if(h>=HC-1)return std::numeric_limits<Delta>::min();Code bb=b.dp[pos-1][h]+(h?b.dp[pos-1][h-1]:0),aa=m.dp[pos][h]+(h?m.dp[pos][h-1]:0);nh=h+1;return Delta(bb)-Delta(aa);}return 0;}
unsigned chunk3(int a,int b,int c){return unsigned(a+3*b+9*c);}
int get3(unsigned z,int r){if(r==0){unsigned q=z/3;return int(z-q*3);}if(r==1){unsigned q=z/3;return int(q-(q/3)*3);}return int(z/9);}
}
int main(){
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int c=0;c<3;++c){unsigned z=chunk3(a,b,c);if(z>=27||get3(z,0)!=a||get3(z,1)!=b||get3(z,2)!=c)return 2;}
    Delta all_lo=0,all_hi=0,safe_lo=0,safe_hi=0;bool all_first=true,safe_first=true;std::uint64_t specs=0,safe_specs=0,states=0;
    std::array<Delta,MAX_FIXED+1> k_lo{},k_hi{};
    // forced2window high range is p=27..15; planner-fixed positions are 13..0.
    // Transition pairs therefore need exactly symbols 14..27.
    for(int k=0;k<=MAX_FIXED;++k){bool kfirst=true;std::uint32_t fixed=0;for(int j=0;j<k;++j)fixed|=1u<<(13-j);const std::uint32_t groups=1u<<k;
        for(std::uint32_t g=0;g<groups;++g){std::uint32_t occ=0;for(int j=0;j<k;++j)if((g>>j)&1u)occ|=1u<<(13-j);Spec m=make_spec(28,fixed,occ),b=make_spec(27,fixed,occ);++specs;if(k>=MIN_FIXED)++safe_specs;
            std::array<Delta,HC> mn{},mx{};std::array<unsigned char,HC> seen{};seen[1]=1;
            auto record=[&](){for(int h=0;h<HC;++h)if(seen[h]){Delta lo=mn[h],hi=mx[h];if(kfirst){k_lo[k]=lo;k_hi[k]=hi;kfirst=false;}else{k_lo[k]=std::min(k_lo[k],lo);k_hi[k]=std::max(k_hi[k],hi);}if(all_first){all_lo=lo;all_hi=hi;all_first=false;}else{all_lo=std::min(all_lo,lo);all_hi=std::max(all_hi,hi);}if(k>=MIN_FIXED){if(safe_first){safe_lo=lo;safe_hi=hi;safe_first=false;}else{safe_lo=std::min(safe_lo,lo);safe_hi=std::max(safe_hi,hi);}}++states;}};record();
            for(int pos=27;pos>=15;--pos){std::array<Delta,HC> nmn{},nmx{};std::array<unsigned char,HC> ns{};for(int h=0;h<HC;++h)if(seen[h])for(int v=0;v<3;++v){int nh;Delta d=contrib(m,b,pos,h,v,nh);if(d==std::numeric_limits<Delta>::min())continue;Delta lo=mn[h]+d,hi=mx[h]+d;if(!ns[nh]){ns[nh]=1;nmn[nh]=lo;nmx[nh]=hi;}else{nmn[nh]=std::min(nmn[nh],lo);nmx[nh]=std::max(nmx[nh],hi);}}seen=ns;mn=nmn;mx=nmx;record();}
        }
    }
    constexpr Delta LIM=Delta(1)<<34;
    if(safe_lo < -LIM || safe_hi >= LIM){std::cerr<<"signed35 gated overflow min="<<safe_lo<<" max="<<safe_hi<<"\n";return 3;}
    if(k_lo[MIN_FIXED] < -LIM || k_hi[MIN_FIXED] >= LIM)return 4;
    if(k_lo[MIN_FIXED-1] >= -LIM && k_hi[MIN_FIXED-1] < LIM){std::cerr<<"gate is not minimal; k=6 also fits\n";return 5;}
    std::uint64_t total=1;for(int i=0;i<TRITS;++i)total*=3;
    for(std::uint64_t x=0;x<total;++x){std::uint64_t q=x,packed=0;int a[TRITS]{};for(int i=0;i<TRITS;++i){a[i]=int(q%3);q/=3;}for(int c=0;c<5;++c){int p=3*c;unsigned z=chunk3(p<TRITS?a[p]:0,p+1<TRITS?a[p+1]:0,p+2<TRITS?a[p+2]:0);packed|=std::uint64_t(z)<<(5*c);}for(int i=0;i<TRITS;++i){unsigned z=unsigned((packed>>(5*(i/3)))&31u);if(get3(z,i%3)!=a[i])return 6;}}
    std::cout<<"b300-high-main-recurrence-proof OK min_fixed="<<MIN_FIXED<<" fallback_fixed_lt="<<MIN_FIXED<<" p_lo=15 symbol_lo=14 symbol_hi=27 all_delta_min="<<all_lo<<" safe_delta_min="<<safe_lo<<" k6_delta_min="<<k_lo[6]<<" k7_delta_min="<<k_lo[7]<<" signed35=1 trit_positions=14 trit_chunks=5 trit_bits=25 height_bits=4 total_bits=64 planner_specs="<<specs<<" safe_specs="<<safe_specs<<" exhaustive_trit_states="<<total<<" exact=1\n";
}
