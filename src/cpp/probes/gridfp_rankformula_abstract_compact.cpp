#define main gridfp_rankformula_abstract_lut_main_unused
#include "gridfp_rankformula_abstract_lut.cpp"
#undef main

static uint64_t choose_u64_compact(int n,int k){if(k<0||k>n)return 0;if(k>n-k)k=n-k;uint64_t z=1;for(int i=1;i<=k;++i)z=z*uint64_t(n-k+i)/uint64_t(i);return z;}
static uint8_t direct_select(uint32_t lp,int n,uint32_t depth){uint32_t state=depth,li=0;uint8_t sm=0;for(int ord=0;ord<n;++ord){if((lp>>ord)&1u){if(state==1u)sm|=uint8_t(1u<<li);++li;++state;}else{if(state==1u)break;--state;}}return sm;}
static uint8_t swar_select(uint32_t dpack,uint32_t depth){uint32_t x=dpack^(depth*0x01111111u);uint32_t y=x|(x>>1)|(x>>2)|(x>>3);uint32_t z=(~y)&0x01111111u;z=(z|(z>>3))&0x03030303u;z=(z|(z>>6))&0x000f000fu;z=(z|(z>>12))&0xffu;return uint8_t(z&0x7fu);}

int main(){
    constexpr uint32_t STATES=7060u,OVERFLOW_N=429u;
    std::array<uint16_t,STATES> depth03{},depth46{};
    std::array<uint32_t,STATES> src03{},src36{};
    std::array<uint16_t,OVERFLOW_N> src7{};
    uint32_t di=0,universal_selected=0;uint64_t production_selected=0;
    for(int n=0;n<=L;++n){
        const uint64_t weight=choose_u64_compact(L,n);
        for(int h=0;h<16;++h){
            const uint32_t cnt=ballot_suffix(n,h);
            for(uint32_t local=0;local<cnt;++local,++di){
                if(di>=STATES)return 2;
                const uint32_t lp=abstract_lpattern(n,h,local);if(lp==INVALID)return 3;
                const uint32_t lc=uint32_t(__builtin_popcount(lp));if(lc>7u)return 4;
                uint32_t dpack=0;
                for(uint32_t depth=1;depth<=13u;++depth){
                    const uint8_t sm=direct_select(lp,n,depth);
                    for(uint32_t li=0;li<lc;++li)if((sm>>li)&1u){const uint32_t old=(dpack>>(4u*li))&15u;if(old&&old!=depth)return 5;dpack|=depth<<(4u*li);}
                }
                depth03[di]=uint16_t(dpack);depth46[di]=uint16_t((dpack>>16)&0xfffu);
                uint32_t li=0;
                for(int ord=0;ord<n;++ord)if((lp>>ord)&1u){
                    const uint32_t sr=abstract_rank(n,h+2,lp&~(1u<<ord));if(sr==INVALID||sr>1000u)return 6;
                    if(li<3u)src03[di]|=sr<<(10u*li);
                    else if(li<6u)src36[di]|=sr<<(10u*(li-3u));
                    else{if(li!=6u||n!=14||h!=0||local>=OVERFLOW_N)return 7;src7[local]=uint16_t(sr);}++li;
                }
                for(uint32_t depth=1;depth<=13u;++depth){
                    const uint32_t rebuilt=uint32_t(depth03[di])|(uint32_t(depth46[di])<<16);
                    const uint8_t want=direct_select(lp,n,depth),got=swar_select(rebuilt,depth);if(got!=want)return 8;
                    uint32_t bits=got;
                    while(bits){const uint32_t sel=uint32_t(__builtin_ctz(bits));bits&=bits-1u;uint32_t ord=0,seen=0;for(;ord<uint32_t(n);++ord)if((lp>>ord)&1u){if(seen==sel)break;++seen;}if(ord>=uint32_t(n))return 9;const uint32_t expected=abstract_rank(n,h+2,lp&~(1u<<ord));const uint32_t decoded=sel<3u?((src03[di]>>(10u*sel))&1023u):sel<6u?((src36[di]>>(10u*(sel-3u)))&1023u):uint32_t(src7[local]);if(decoded!=expected)return 10;++universal_selected;production_selected+=weight;}
                }
                if(swar_select(uint32_t(depth03[di])|(uint32_t(depth46[di])<<16),14u)||swar_select(uint32_t(depth03[di])|(uint32_t(depth46[di])<<16),15u))return 11;
            }
        }
    }
    const uint64_t depth_bytes=uint64_t(STATES)*4ull,src_bytes=uint64_t(STATES)*8ull+uint64_t(OVERFLOW_N)*2ull,total_lut=480ull+depth_bytes+src_bytes;
    if(di!=STATES||universal_selected!=19273u||production_selected!=2492769ull||depth_bytes!=28240ull||src_bytes!=57338ull||total_lut!=86058ull)return 12;
    std::cout<<"gridfp-rankformula-abstract-compact OK states="<<di<<" universal_selected="<<universal_selected<<" production_selected="<<production_selected<<" depth_bytes="<<depth_bytes<<" source_bytes="<<src_bytes<<" total_lut_bytes="<<total_lut<<" depth4_split16=1 srcpack10_split32=1 exact_selected_sources=1 depth14_15_zero=1\n";
    return 0;
}
