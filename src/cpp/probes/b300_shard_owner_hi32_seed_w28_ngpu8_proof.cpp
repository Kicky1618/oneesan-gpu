#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

struct Case {
    const char* name;
    u64 total;
    u64 chunk;
    u32 hmax;
};

constexpr std::array<Case,2> CASES{{
    {"main",385719506620ULL,48214938328ULL,89},
    {"block",135015505407ULL,16876938176ULL,31},
}};

u32 seed_main(u32 h){ return (h*365u)>>12; }
u32 seed_main_shiftadd(u32 h){ return ((h<<8)+(h<<6)+(h<<5)+(h<<3)+(h<<2)+h)>>12; }
u32 seed_block(u32 h){ return h>>2; }

u64 masked_base(u32 owner,u64 chunk){
    const u64 u=owner;
    return ((u64(0)-(u&1ULL))&chunk)+
           ((u64(0)-((u>>1)&1ULL))&(chunk<<1))+
           ((u64(0)-((u>>2)&1ULL))&(chunk<<2));
}

u32 seed(const Case& c,u32 h){ return c.name[0]=='m'?seed_main(h):seed_block(h); }

bool check_case(const Case& c){
    if(((c.total-1)>>32)!=c.hmax || c.chunk<=(1ULL<<32)){
        std::fprintf(stderr,"%s basic bound mismatch\n",c.name); return false;
    }
    u32 max_seed=0,max_delta=0;
    for(u32 h=0;h<=c.hmax;++h){
        const u32 q0=seed(c,h);
        const u32 exact0=u32((u64(h)<<32)/c.chunk);
        if(q0!=exact0){
            std::fprintf(stderr,"%s seed failure h=%u got=%u exact=%u\n",c.name,h,q0,exact0); return false;
        }
        if(c.name[0]=='m' && seed_main_shiftadd(h)!=q0){
            std::fprintf(stderr,"main shiftadd seed failure h=%u\n",h); return false;
        }
        max_seed=std::max(max_seed,q0);
    }

    // Since chunk > 2^32, fixing h and varying the low word can cross at most
    // one shard boundary. q0 is the owner at low=0, so exact owner is q0 or q0+1.
    u64 cases=0, corrections=0;
    constexpr u64 SAMPLES=3000000;
    for(u64 q=0;q<8;++q){
        const u64 lo=q*c.chunk;
        if(lo>=c.total) break;
        const u64 hi=std::min((q+1)*c.chunk-1,c.total-1);
        for(int d=-4;d<=4;++d){
            const u64 xlo=(d<0 && lo<u64(-d))?0:lo+d;
            if(xlo<c.total){
                const u32 h=u32(xlo>>32),q0=seed(c,h);u64 r=xlo-masked_base(q0,c.chunk);u32 corr=u32(r>=c.chunk);u32 owner=q0+corr;if(corr)r-=c.chunk;
                const u32 ex=u32(xlo/c.chunk);if(owner!=ex||r!=xlo-u64(ex)*c.chunk){std::fprintf(stderr,"%s boundary failure g=%llu\n",c.name,(unsigned long long)xlo);return false;}max_delta=std::max(max_delta,owner-q0);corrections+=corr;++cases;
            }
            if(hi+u64(d)>=hi && hi+u64(d)<c.total){
                const u64 x=hi+u64(d);const u32 h=u32(x>>32),q0=seed(c,h);u64 r=x-masked_base(q0,c.chunk);u32 corr=u32(r>=c.chunk);u32 owner=q0+corr;if(corr)r-=c.chunk;
                const u32 ex=u32(x/c.chunk);if(owner!=ex||r!=x-u64(ex)*c.chunk){std::fprintf(stderr,"%s boundary-hi failure g=%llu\n",c.name,(unsigned long long)x);return false;}max_delta=std::max(max_delta,owner-q0);corrections+=corr;++cases;
            }
        }
    }
    for(u64 i=0;i<=SAMPLES;++i){
        const u64 g=u64((u128(i)*(c.total-1))/SAMPLES);
        const u32 h=u32(g>>32),q0=seed(c,h);u64 r=g-masked_base(q0,c.chunk);const u32 corr=u32(r>=c.chunk);const u32 owner=q0+corr;if(corr)r-=c.chunk;
        const u32 ex=u32(g/c.chunk);if(owner!=ex||r!=g-u64(ex)*c.chunk){std::fprintf(stderr,"%s dense failure i=%llu g=%llu\n",c.name,(unsigned long long)i,(unsigned long long)g);return false;}max_delta=std::max(max_delta,owner-q0);corrections+=corr;++cases;
    }
    std::printf("%s total=%llu chunk=%llu hmax=%u seed_max=%u chunk_gt_2p32=1 seed_exact=1 correction_max=%u correction_cases=%llu cases=%llu exact=1\n",
        c.name,(unsigned long long)c.total,(unsigned long long)c.chunk,c.hmax,max_seed,max_delta,(unsigned long long)corrections,(unsigned long long)cases);
    return true;
}

} // namespace

int main(){
    for(const auto& c:CASES) if(!check_case(c)) return 1;
    std::puts("b300-shard-owner-hi32-seed-w28-ngpu8-proof OK ngpu=8 main_seed=(hi32*365)>>12 main_seed_shiftadd_exact=1 block_seed=hi32>>2 correction_compare64=1 correction_max=1 device_div64=0 device_mul64=0 base_table=0 exact=1");
    return 0;
}
