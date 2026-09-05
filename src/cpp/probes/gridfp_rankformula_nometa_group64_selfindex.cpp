#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

#include <limits>

static int bits_u32(uint32_t x) {
    int b = 0;
    do { ++b; x >>= 1; } while (x);
    return b;
}
static int bits_signed(int lo, int hi) {
    for (int b = 1; b < 31; ++b) {
        const int mn = -(1 << (b - 1));
        const int mx = (1 << (b - 1)) - 1;
        if (lo >= mn && hi <= mx) return b;
    }
    return 31;
}
static uint64_t pack59(uint32_t start,uint32_t n,int delta,uint32_t count,uint32_t gi) {
    if (start >= (1u<<16) || n >= (1u<<4) ||
        delta < -(1<<14) || delta >= (1<<14) ||
        count == 0 || count >= (1u<<10) || gi >= (1u<<14)) std::exit(20);
    const uint32_t draw = uint32_t(delta) & 0x7fffu;
    return uint64_t(start) |
           (uint64_t(n) << 16) |
           (uint64_t(draw) << 20) |
           (uint64_t(count) << 35) |
           (uint64_t(gi) << 45);
}
static int unpack_delta(uint64_t x) {
    uint32_t z = uint32_t((x >> 20) & 0x7fffu);
    if (z & 0x4000u) z |= 0xffff8000u;
    return int(int32_t(z));
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;
    std::vector<int32_t> base(size_t(NG) * S * LM, -1);
    auto bref = [&](int g,int h,uint32_t m)->int32_t& {
        return base[(size_t(g)*S + size_t(h))*LM + m];
    };
    std::array<std::array<uint32_t,S>,NG> ranks{};
    for (int h=0;h<=L+1;++h) {
        std::array<uint32_t,NG> next{};
        for (uint32_t m=0;m<LM;++m) {
            const int g=int(owner[m]);
            const uint32_t n=uint32_t(f.low_mask_h[size_t(m)*S+size_t(h)].size());
            if (n) bref(g,h,m)=int32_t(next[g]);
            next[g]+=n;
        }
        for(int g=0;g<NG;++g)ranks[g][h]=next[g];
    }

    uint32_t max_start=0,max_count=0,max_n=0,max_gi=0;
    int min_delta=std::numeric_limits<int>::max(),max_delta=std::numeric_limits<int>::min();
    uint64_t groups=0,exact=0;
    std::array<uint32_t,NG> owner_groups{};
    for(int g=0;g<NG;++g){
        uint32_t gi=0;
        for(int h=0;h<=L+1;++h){
            for(uint32_t m=0;m<LM;++m){
                if(owner[m]!=g)continue;
                const auto& v=f.low_mask_h[size_t(m)*S+size_t(h)];
                if(v.empty())continue;
                const uint32_t start=uint32_t(bref(g,h,m));
                const uint32_t count=uint32_t(v.size());
                const uint32_t n=uint32_t(__builtin_popcount(m));
                int delta=0;
                if(h+2<=L+1){
                    const int32_t b=bref(g,h+2,m);
                    if(b>=0)delta=int(b)-int(start);
                }
                max_start=std::max(max_start,start);
                max_count=std::max(max_count,count);
                max_n=std::max(max_n,n);
                max_gi=std::max(max_gi,gi);
                min_delta=std::min(min_delta,delta);
                max_delta=std::max(max_delta,delta);
                const uint64_t p=pack59(start,n,delta,count,gi);
                const uint32_t us=uint32_t(p&0xffffu);
                const uint32_t un=uint32_t((p>>16)&0xfu);
                const int ud=unpack_delta(p);
                const uint32_t uc=uint32_t((p>>35)&0x3ffu);
                const uint32_t ug=uint32_t((p>>45)&0x3fffu);
                if(us!=start||un!=n||ud!=delta||uc!=count||ug!=gi||(p>>59)!=0u)return 2;
                ++exact;++groups;++gi;
            }
        }
        owner_groups[g]=gi;
    }
    if(groups!=69632ull||exact!=groups||max_start>=65536u||max_count>=1024u||
       max_n>14u||min_delta<-(1<<14)||max_delta>=(1<<14)||max_gi>=16384u)return 3;
    uint32_t max_owner_groups=*std::max_element(owner_groups.begin(),owner_groups.end());
    std::cout<<"gridfp-rankformula-nometa-group64-selfindex OK"
             <<" groups="<<groups
             <<" max_start="<<max_start<<" start_bits="<<bits_u32(max_start)
             <<" max_n="<<max_n<<" n_bits="<<bits_u32(max_n)
             <<" min_delta="<<min_delta<<" max_delta="<<max_delta
             <<" delta_signed_bits="<<bits_signed(min_delta,max_delta)
             <<" max_count="<<max_count<<" count_bits="<<bits_u32(max_count)
             <<" max_group_index="<<max_gi<<" group_index_bits="<<bits_u32(max_gi)
             <<" max_owner_groups="<<max_owner_groups
             <<" packed_bits=59 spare_bits=5 exact="<<exact
             <<" warpshare_gi_shuffle_elidable=1\n";
    return 0;
}
