#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>

using u64=std::uint64_t;

namespace {
constexpr int NGPU=8;
constexpr u64 ELEM_BYTES=4;

struct Plan{
    u64 logical_bytes=0,mapped_bytes=0,units=0,gran=0;
    std::array<u64,NGPU> segment{};
    std::array<u64,NGPU+1> off{};
};

u64 ceil_div(u64 a,u64 b){return a/b+u64(a%b!=0);}

Plan make_plan(u64 elems,u64 gran,int rotation){
    Plan p;p.logical_bytes=elems*ELEM_BYTES;p.gran=gran;
    p.units=ceil_div(p.logical_bytes,gran);p.mapped_bytes=p.units*gran;
    const u64 q=p.units/NGPU,r=p.units%NGPU;
    p.off[0]=0;
    for(int d=0;d<NGPU;++d){
        const int rel=(d-rotation+NGPU)%NGPU;
        const u64 units=q+u64(rel<int(r));
        p.segment[d]=units*gran;
        p.off[d+1]=p.off[d]+p.segment[d];
    }
    return p;
}

bool check(u64 gran){
    constexpr u64 MAIN=385719506620ULL,BLOCK=135015505407ULL;
    if(!gran||gran%ELEM_BYTES)return false;
    const Plan m=make_plan(MAIN,gran,0);
    const int block_rotation=int(m.units%NGPU);
    const Plan b=make_plan(BLOCK,gran,block_rotation);
    if(m.off.back()!=m.mapped_bytes||b.off.back()!=b.mapped_bytes)return false;
    if(m.mapped_bytes-m.logical_bytes>=gran||b.mapped_bytes-b.logical_bytes>=gran)return false;

    std::array<u64,NGPU> combined{};
    for(int d=0;d<NGPU;++d)combined[d]=m.segment[d]+b.segment[d];
    const auto [lo,hi]=std::minmax_element(combined.begin(),combined.end());
    if(*hi-*lo>gran)return false;

    const u64 mc=ceil_div(MAIN,NGPU),bc=ceil_div(BLOCK,NGPU);
    for(int d=0;d<NGPU;++d){
        const u64 ms=u64(d)*mc,bs=u64(d)*bc;
        if(ms>=MAIN||bs>=BLOCK)return false;
        // Logical shard views are offsets in the continuous VA and deliberately
        // do not have to coincide with physical VMM allocation boundaries.
        if(ms*ELEM_BYTES>=m.logical_bytes||bs*ELEM_BYTES>=b.logical_bytes)return false;
    }

    std::printf("gran=%llu main_units=%llu block_units=%llu main_extra=%llu block_extra=%llu block_rotation=%d combined_min=%llu combined_max=%llu combined_imbalance=%llu main_padding=%llu block_padding=%llu logical_views_independent_of_physical_boundaries=1 exact=1\n",
        (unsigned long long)gran,
        (unsigned long long)m.units,(unsigned long long)b.units,
        (unsigned long long)(m.units%NGPU),(unsigned long long)(b.units%NGPU),
        block_rotation,(unsigned long long)*lo,(unsigned long long)*hi,
        (unsigned long long)(*hi-*lo),
        (unsigned long long)(m.mapped_bytes-m.logical_bytes),
        (unsigned long long)(b.mapped_bytes-b.logical_bytes));
    return true;
}
}

int main(){
    constexpr std::array<u64,8> GRANS{{65536,131072,262144,524288,1048576,2097152,4194304,16777216}};
    for(u64 g:GRANS)if(!check(g))return 1;
    std::puts("b300-vmm-balanced-physical-layout-proof OK ngpu=8 arrays=2 granularities=8 block_rotation_after_main_extras=1 combined_imbalance_le_one_granularity=1 internal_gap=0 logical_shard_views_preserved=1 exact=1");
    return 0;
}
