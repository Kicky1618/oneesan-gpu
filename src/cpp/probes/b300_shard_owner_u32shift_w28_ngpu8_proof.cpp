#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

constexpr u32 MAIN_MLO = 2614578007u;
constexpr u32 MAIN_MHI = 45u;
constexpr u32 BLOCK_MLO = 2466947517u;
constexpr u32 BLOCK_MHI = 32u;
constexpr u64 MAIN_TOTAL = 385719506620ULL;
constexpr u64 MAIN_CHUNK = 48214938328ULL;
constexpr u64 BLOCK_TOTAL = 135015505407ULL;
constexpr u64 BLOCK_CHUNK = 16876938176ULL;

u32 mul_hi_u32(u32 a, u32 b) { return u32((u64(a) * b) >> 32); }

struct Product32 { u32 lo; u32 hi; };

Product32 mul45_shiftadd(u32 x) {
    u32 lo = x << 5;
    u32 hi = x >> 27;
    const std::array<unsigned,3> shifts{{3,2,0}};
    for (unsigned s : shifts) {
        const u32 add_lo = s ? (x << s) : x;
        const u32 add_hi = s ? (x >> (32 - s)) : 0;
        const u32 old = lo;
        lo += add_lo;
        hi += add_hi + u32(lo < old);
    }
    return {lo, hi};
}

Product32 mul32_shift(u32 x) { return {x << 5, x >> 27}; }

u64 masked_base(u64 owner, u64 chunk) {
    const u64 b0 = (u64(0) - (owner & 1ULL)) & chunk;
    const u64 b1 = (u64(0) - ((owner >> 1) & 1ULL)) & (chunk << 1);
    const u64 b2 = (u64(0) - ((owner >> 2) & 1ULL)) & (chunk << 2);
    return b0 + b1 + b2;
}

template<bool MAIN>
u64 owner_shiftadd(u64 g, u32& carry_out, u32& high64_out) {
    constexpr u32 MLO = MAIN ? MAIN_MLO : BLOCK_MLO;
    constexpr unsigned HIGH_SHIFT = MAIN ? 9 : 7;
    const u32 a0 = u32(g);
    const u32 a1 = u32(g >> 32);
    const u32 p00_hi = mul_hi_u32(a0, MLO);
    const Product32 p01 = MAIN ? mul45_shiftadd(a0) : mul32_shift(a0);
    const u64 p10 = u64(a1) * MLO;
    const u32 p10_lo = u32(p10);
    const u32 p10_hi = u32(p10 >> 32);
    const u32 p11 = MAIN ? ((a1 << 5) + (a1 << 3) + (a1 << 2) + a1)
                         : (a1 << 5);
    const u32 s0 = p00_hi + p01.lo;
    u32 carry = u32(s0 < p00_hi);
    const u32 s1 = s0 + p10_lo;
    carry += u32(s1 < s0);
    const u32 high64 = p01.hi + p10_hi + p11 + carry;
    carry_out = carry;
    high64_out = high64;
    return high64 >> HIGH_SHIFT;
}

bool check_small_multiplier() {
    constexpr u64 SAMPLES = 1000000;
    for (u64 i=0;i<=SAMPLES;++i) {
        const u32 x = u32((u128(i) * 0xffffffffULL) / SAMPLES);
        const Product32 a = mul45_shiftadd(x);
        const u64 exact45 = u64(x) * 45u;
        if (a.lo != u32(exact45) || a.hi != u32(exact45 >> 32)) {
            std::fprintf(stderr,"mul45 shift-add failure x=%u\n",x); return false;
        }
        const Product32 b = mul32_shift(x);
        const u64 exact32 = u64(x) * 32u;
        if (b.lo != u32(exact32) || b.hi != u32(exact32 >> 32)) {
            std::fprintf(stderr,"mul32 shift failure x=%u\n",x); return false;
        }
    }
    return true;
}

template<bool MAIN>
bool check_case(const char* name) {
    constexpr u64 TOTAL = MAIN ? MAIN_TOTAL : BLOCK_TOTAL;
    constexpr u64 CHUNK = MAIN ? MAIN_CHUNK : BLOCK_CHUNK;
    constexpr u64 MAGIC = MAIN ? 195888106327ULL : 139905900989ULL;
    constexpr unsigned SHIFT = MAIN ? 73 : 71;
    constexpr u64 SAMPLES = 2000000;
    u32 max_carry=0,max_high=0;
    u64 cases=0;
    for (u64 q=0;q<8;++q) {
        const u64 lo=q*CHUNK;
        if(lo>=TOTAL) break;
        const u64 hi=std::min((q+1)*CHUNK-1,TOTAL-1);
        const std::array<u64,6> xs{{lo,std::min(lo+1,hi),std::min(lo+2,hi),hi>lo+1?hi-2:lo,hi>lo?hi-1:hi,hi}};
        for(u64 g:xs) {
            u32 carry=0,high=0;
            const u64 owner=owner_shiftadd<MAIN>(g,carry,high);
            const u64 exact_high=u64((u128(g)*MAGIC)>>64);
            const u64 exact_owner=g/CHUNK;
            if(high!=exact_high||owner!=exact_owner||g-masked_base(owner,CHUNK)!=g-exact_owner*CHUNK){
                std::fprintf(stderr,"%s endpoint failure g=%llu\n",name,(unsigned long long)g); return false;
            }
            max_carry=std::max(max_carry,carry); max_high=std::max(max_high,high); ++cases;
        }
    }
    for(u64 i=0;i<=SAMPLES;++i) {
        const u64 g=u64((u128(i)*(TOTAL-1))/SAMPLES);
        u32 carry=0,high=0;
        const u64 owner=owner_shiftadd<MAIN>(g,carry,high);
        const u64 exact_high=u64((u128(g)*MAGIC)>>64);
        if(high!=exact_high||owner!=g/CHUNK){
            std::fprintf(stderr,"%s dense failure i=%llu g=%llu\n",name,(unsigned long long)i,(unsigned long long)g); return false;
        }
        max_carry=std::max(max_carry,carry); max_high=std::max(max_high,high); ++cases;
    }
    std::printf("%s total=%llu chunk=%llu magic=%llu shift=%u high_shift=%u magic_hi=%u shiftadd_hi=1 max_carry=%u max_high=%u cases=%llu exact=1\n",
        name,(unsigned long long)TOTAL,(unsigned long long)CHUNK,(unsigned long long)MAGIC,SHIFT,SHIFT-64,
        MAIN?MAIN_MHI:BLOCK_MHI,max_carry,max_high,(unsigned long long)cases);
    return true;
}

} // namespace

int main(){
    if(!check_small_multiplier()) return 1;
    if(!check_case<true>("main")||!check_case<false>("block")) return 1;
    std::puts("b300-shard-owner-u32shift-w28-ngpu8-proof OK ngpu=8 device_mul64=0 device_div64=0 device_table_load=0 umulhi_u32_per_owner=2 mullo_u32_per_owner=1 magic_hi_mul_replaced_by_shiftadd=1 masked_base_exact=1 dense_samples_per_case=2000001 exact=1");
    return 0;
}
