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

u32 seed(const Case& c,u32 h){
    if(c.name[0]=='m') return ((h<<8)+(h<<6)+(h<<5)+(h<<3)+(h<<2)+h)>>12;
    return h>>2;
}

struct Pair32{u32 lo,hi;};

Pair32 add_pair(Pair32 a,Pair32 b){
    const u32 old=a.lo;
    a.lo+=b.lo;
    a.hi+=b.hi+u32(a.lo<old);
    return a;
}

Pair32 base_bits(u32 owner,u64 chunk){
    const std::array<u64,3> xs{{chunk,chunk<<1,chunk<<2}};
    Pair32 p{0,0};
    for(int bit=0;bit<3;++bit) if((owner>>bit)&1u){
        Pair32 x{u32(xs[bit]),u32(xs[bit]>>32)};
        p=add_pair(p,x);
    }
    return p;
}

struct Address{u32 owner;u32 lo,hi;};

Address address_u32(const Case& c,u64 g){
    const u32 glo=u32(g), ghi=u32(g>>32);
    u32 owner=seed(c,ghi);
    Pair32 b=base_bits(owner,c.chunk);
    u32 rlo=glo-b.lo;
    const u32 borrow=u32(glo<b.lo);
    u32 rhi=ghi-b.hi-borrow;
    const u32 clo=u32(c.chunk),chi=u32(c.chunk>>32);
    const u32 corr=u32(rhi>chi || (rhi==chi && rlo>=clo));
    const u32 mask=0u-corr;
    const u32 sublo=clo&mask,subhi=chi&mask;
    const u32 old=rlo;
    rlo-=sublo;
    rhi-=subhi+u32(old<sublo);
    owner+=corr;
    return{owner,rlo,rhi};
}

bool check_case(const Case& c){
    if(((c.total-1)>>32)!=c.hmax || c.chunk<=(1ULL<<32)) return false;
    u64 cases=0,corrections=0;
    u32 max_owner=0,max_local_hi=0;
    constexpr u64 SAMPLES=4000000;
    auto check=[&](u64 g)->bool{
        const Address a=address_u32(c,g);
        const u32 ex=u32(g/c.chunk);
        const u64 local=g-u64(ex)*c.chunk;
        if(a.owner!=ex || a.lo!=u32(local) || a.hi!=u32(local>>32)){
            std::fprintf(stderr,"%s failure g=%llu got=(%u,%u:%u) exact=(%u,%llu)\n",
                c.name,(unsigned long long)g,a.owner,a.hi,a.lo,ex,(unsigned long long)local);
            return false;
        }
        max_owner=std::max(max_owner,a.owner);
        max_local_hi=std::max(max_local_hi,a.hi);
        const u32 q0=seed(c,u32(g>>32));
        corrections+=u64(a.owner!=q0);
        ++cases;
        return true;
    };
    for(u32 q=0;q<8;++q){
        const u64 lo=u64(q)*c.chunk;
        if(lo>=c.total) break;
        const u64 hi=std::min<u64>(u64(q+1)*c.chunk-1,c.total-1);
        for(int d=-16;d<=16;++d){
            if(d>=0 || lo>=u64(-d)){
                const u64 x=lo+u64(d);
                if(x<c.total && !check(x)) return false;
            }
            if(d>=0){
                const u64 x=hi+u64(d);
                if(x>=hi && x<c.total && !check(x)) return false;
            }else if(hi>=u64(-d)){
                const u64 x=hi-u64(-d);
                if(!check(x)) return false;
            }
        }
    }
    for(u64 i=0;i<=SAMPLES;++i){
        const u64 g=u64((u128(i)*(c.total-1))/SAMPLES);
        if(!check(g)) return false;
    }
    std::printf("%s total=%llu chunk=%llu chunk_hi=%u chunk_lo=%u hmax=%u cases=%llu correction_cases=%llu max_owner=%u max_local_hi=%u exact=1\n",
        c.name,(unsigned long long)c.total,(unsigned long long)c.chunk,u32(c.chunk>>32),u32(c.chunk),c.hmax,
        (unsigned long long)cases,(unsigned long long)corrections,max_owner,max_local_hi);
    return true;
}

} // namespace

int main(){
    for(const auto& c:CASES) if(!check_case(c)) return 1;
    std::puts("b300-shard-owner-hi32-u32addr-w28-ngpu8-proof OK ngpu=8 seed_hi32=1 base_u32_bits=1 correction_u32_compare=1 local_u32_subborrow=1 device_div64=0 device_mul64=0 device_setp64=0 device_sub64=0 exact=1");
    return 0;
}
