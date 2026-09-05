#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

using u64=std::uint64_t;
using u128=unsigned __int128;

namespace {
constexpr int NGPU=8;
constexpr u64 ELEM_BYTES=4;

struct Plan{u64 mapped_bytes=0,padding_bytes=0,min_segment=0,max_segment=0;std::array<u64,NGPU+1> off{};};

u64 ceil_div(u64 a,u64 b){return a/b+u64(a%b!=0);}

Plan make_plan(u64 elems,u64 gran){
    const u64 logical=elems*ELEM_BYTES;
    const u64 units=ceil_div(logical,gran);
    Plan p;p.mapped_bytes=units*gran;p.padding_bytes=p.mapped_bytes-logical;
    const u64 q=units/NGPU,r=units%NGPU;
    p.off[0]=0;p.min_segment=~u64(0);p.max_segment=0;
    for(int d=0;d<NGPU;++d){
        const u64 seg=(q+u64(d<int(r)))*gran;
        p.min_segment=std::min(p.min_segment,seg);p.max_segment=std::max(p.max_segment,seg);p.off[d+1]=p.off[d]+seg;
    }
    return p;
}

bool check(const char* name,u64 elems,u64 gran){
    if(gran==0||gran%ELEM_BYTES) return false;
    Plan p=make_plan(elems,gran);const u64 logical=elems*ELEM_BYTES;
    if(p.off[0]!=0||p.off[NGPU]!=p.mapped_bytes||p.mapped_bytes<logical||p.padding_bytes>=gran)return false;
    for(int d=0;d<NGPU;++d){if(p.off[d]%gran||p.off[d+1]%gran||p.off[d]>=p.off[d+1])return false;}
    if(p.max_segment-p.min_segment>gran)return false;
    // Every logical byte belongs to exactly one contiguous mapped segment; there are no internal gaps.
    u64 last=0;
    for(int d=0;d<NGPU;++d){if(p.off[d]!=last)return false;last=p.off[d+1];}
    // Check all segment boundaries and dense deterministic element samples. Direct base[g] stays byte offset 4*g.
    std::vector<u64> xs{0,1,elems-1};
    for(int d=1;d<NGPU;++d){u64 e=p.off[d]/ELEM_BYTES;if(e<elems)xs.push_back(e);if(e&&e-1<elems)xs.push_back(e-1);}
    constexpr u64 S=1000000;
    for(u64 i=0;i<=S;++i)xs.push_back(u64((u128(i)*(elems-1))/S));
    for(u64 g:xs){
        const u64 byte=g*ELEM_BYTES;if(byte>=logical)return false;
        int hits=0;for(int d=0;d<NGPU;++d)hits+=int(byte>=p.off[d]&&byte<p.off[d+1]);if(hits!=1)return false;
        if(byte!=g*4)return false;
    }
    std::printf("%s elems=%llu logical_bytes=%llu gran=%llu mapped_bytes=%llu padding=%llu min_segment=%llu max_segment=%llu imbalance=%llu direct_base_index=1 internal_padding=0 tail_padding_only=1 exact=1\n",name,(unsigned long long)elems,(unsigned long long)logical,(unsigned long long)gran,(unsigned long long)p.mapped_bytes,(unsigned long long)p.padding_bytes,(unsigned long long)p.min_segment,(unsigned long long)p.max_segment,(unsigned long long)(p.max_segment-p.min_segment));
    return true;
}
}

int main(){
    constexpr u64 MAIN=385719506620ULL,BLOCK=135015505407ULL;
    constexpr std::array<u64,8> GRANS{{65536,131072,262144,524288,1048576,2097152,4194304,16777216}};
    for(u64 g:GRANS){if(!check("main",MAIN,g)||!check("block",BLOCK,g))return 1;}
    std::puts("b300-vmm-contiguous-layout-proof OK ngpu=8 arrays=2 granularities=8 equal_unit_balance=1 max_segment_imbalance_one_granularity=1 internal_padding=0 tail_padding_lt_granularity=1 direct_base_index=1 exact=1");
    return 0;
}
