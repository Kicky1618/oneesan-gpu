#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

// Common-refinement layout for the two conserved occupancy coordinates.
//
// owner_H(maskH) partitions LOW-window work; owner_L(maskL) partitions
// HIGH-window work. A state belongs to bucket B[owner_H][owner_L]. Each bucket
// keeps the original factor-block order and HIGH-row x LOW-column orientation,
// but rows/columns are owner-local. H-major and L-major therefore differ only
// in which GPU owns a raw bucket blob; intra-bucket addresses never change.

static constexpr int BUCKET_NGPU = 8;
static constexpr int BUCKET_LOCAL_RANK_BITS = 15;
static constexpr uint32_t BUCKET_LOCAL_RANK_MASK = (1u << BUCKET_LOCAL_RANK_BITS) - 1u;
static constexpr int BUCKET_OWNER_SHIFT = BUCKET_LOCAL_RANK_BITS;
static constexpr int BUCKET_LOCATOR_BITS = 3 + BUCKET_LOCAL_RANK_BITS;
static_assert(BUCKET_LOCATOR_BITS == 18);

static inline uint32_t bucket_locator_pack(uint32_t owner, uint32_t rank) {
    if (owner >= BUCKET_NGPU || rank > BUCKET_LOCAL_RANK_MASK) {
        std::cerr << "bucket locator overflow owner=" << owner << " rank=" << rank << '\n';
        std::exit(180);
    }
    return (owner << BUCKET_OWNER_SHIFT) | rank;
}
static inline uint32_t bucket_locator_owner(uint32_t x) { return x >> BUCKET_OWNER_SHIFT; }
static inline uint32_t bucket_locator_rank(uint32_t x) { return x & BUCKET_LOCAL_RANK_MASK; }

struct BucketOwnerHost {
    static constexpr int S = StorageFactorHost::S;
    std::vector<uint8_t> high_mask_owner;
    std::vector<uint8_t> low_mask_owner;
    // Indexed exactly like storage.{high,low}_all_codes. Value is owner+local rank.
    std::vector<uint32_t> high_all_locator;
    std::vector<uint32_t> low_all_locator;
    std::array<std::array<uint32_t,S>,BUCKET_NGPU> high_count{};
    std::array<std::array<uint32_t,S>,BUCKET_NGPU> low_count{};
    std::array<uint64_t,BUCKET_NGPU> high_load{};
    std::array<uint64_t,BUCKET_NGPU> low_load{};
    uint32_t max_high_count = 0;
    uint32_t max_low_count = 0;
};

static std::vector<uint8_t> bucket_lpt_owner(
    const std::vector<uint64_t>& weight,
    std::array<uint64_t,BUCKET_NGPU>& load
) {
    std::vector<std::pair<uint64_t,uint32_t>> order;
    order.reserve(weight.size());
    for (uint32_t m=0;m<weight.size();++m) order.push_back({weight[m],m});
    std::sort(order.begin(),order.end(),[](auto a,auto b){
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    load.fill(0);
    std::vector<uint8_t> owner(weight.size());
    for (auto [w,m] : order) {
        int g=0;
        for (int j=1;j<BUCKET_NGPU;++j) if (load[size_t(j)] < load[size_t(g)]) g=j;
        owner[m]=uint8_t(g);
        load[size_t(g)] += w;
    }
    return owner;
}

static BucketOwnerHost build_bucket_owners(
    const FactorTablesHost& base, const StorageFactorHost& storage
) {
    constexpr int H=HIGH_LUT_K, L=LOW_LUT_K, S=StorageFactorHost::S;
    const uint32_t HM=1u<<H, LM=1u<<L;
    BucketOwnerHost out;

    auto high_mask_count = [&](uint32_t m,int h)->uint32_t {
        size_t ix=size_t(m)*S+size_t(h);
        return base.high_mask_off[ix+1]-base.high_mask_off[ix];
    };
    auto low_mask_count = [&](uint32_t m,int h)->uint32_t {
        size_t ix=size_t(m)*S+size_t(h);
        return base.low_mask_off[ix+1]-base.low_mask_off[ix];
    };
    auto high_total = [&](int h)->uint64_t {
        return h>=0&&h<S-1 ? uint64_t(storage.high_all_off[h+1]-storage.high_all_off[h]) : 0;
    };
    auto low_total = [&](int h)->uint64_t {
        return h>=0&&h<S-1 ? uint64_t(storage.low_all_off[h+1]-storage.low_all_off[h]) : 0;
    };

    std::vector<uint64_t> wh(HM),wl(LM);
    for (uint32_t m=0;m<HM;++m) {
        uint64_t z=0;
        for (int h=0;h<=H+1;++h)
            z += uint64_t(high_mask_count(m,h))
                * (2*low_total(h)+low_total(h-1)+low_total(h+1));
        wh[m]=z;
    }
    for (uint32_t m=0;m<LM;++m) {
        uint64_t z=0;
        for (int h=0;h<=H+1;++h)
            z += high_total(h)
                * (2*uint64_t(low_mask_count(m,h))
                   +uint64_t(h>0?low_mask_count(m,h-1):0)
                   +uint64_t(h+1<S?low_mask_count(m,h+1):0));
        wl[m]=z;
    }
    out.high_mask_owner=bucket_lpt_owner(wh,out.high_load);
    out.low_mask_owner=bucket_lpt_owner(wl,out.low_load);

    out.high_all_locator.assign(storage.high_all_codes.size(),0xffffffffu);
    for (int h=0;h<=H+1;++h) {
        std::array<uint32_t,BUCKET_NGPU> next{};
        for (uint32_t m=0;m<HM;++m) {
            uint32_t g=out.high_mask_owner[m];
            size_t ix=size_t(m)*S+size_t(h);
            uint32_t cnt=base.high_mask_off[ix+1]-base.high_mask_off[ix];
            uint32_t all0=storage.high_all_off[h]+storage.high_mask_begin[ix];
            for (uint32_t r=0;r<cnt;++r)
                out.high_all_locator[all0+r]=bucket_locator_pack(g,next[g]++);
        }
        for (int g=0;g<BUCKET_NGPU;++g) {
            out.high_count[g][h]=next[g];
            out.max_high_count=std::max(out.max_high_count,next[g]);
        }
    }

    out.low_all_locator.assign(storage.low_all_codes.size(),0xffffffffu);
    for (int h=0;h<=L+1;++h) {
        std::array<uint32_t,BUCKET_NGPU> next{};
        for (uint32_t m=0;m<LM;++m) {
            uint32_t g=out.low_mask_owner[m];
            size_t ix=size_t(m)*S+size_t(h);
            uint32_t cnt=base.low_mask_off[ix+1]-base.low_mask_off[ix];
            uint32_t all0=storage.low_all_off[h]+storage.low_mask_begin[ix];
            for (uint32_t r=0;r<cnt;++r)
                out.low_all_locator[all0+r]=bucket_locator_pack(g,next[g]++);
        }
        for (int g=0;g<BUCKET_NGPU;++g) {
            out.low_count[g][h]=next[g];
            out.max_low_count=std::max(out.max_low_count,next[g]);
        }
    }

    if (out.max_high_count > (1u<<BUCKET_LOCAL_RANK_BITS)
        || out.max_low_count > (1u<<BUCKET_LOCAL_RANK_BITS)) {
        std::cerr << "bucket owner-local rank exceeds 15 bits high="
                  << out.max_high_count << " low=" << out.max_low_count << '\n';
        std::exit(181);
    }
    std::cerr << "bucket_owners high_max=" << out.max_high_count
              << " low_max=" << out.max_low_count
              << " locator_bits=" << BUCKET_LOCATOR_BITS << '\n';
    return out;
}

struct BucketPhysicalBlock {
    Code off=0;
    uint32_t rows=0,cols=0;
    uint8_t he=0,hs=0,c=0,valid=0;
};
struct BucketPairLayout {
    std::vector<BucketPhysicalBlock> main_blocks;
    std::vector<BucketPhysicalBlock> block_blocks;
    Code size=0;
};
struct BucketPhysicalLayoutHost {
    std::array<std::array<BucketPairLayout,BUCKET_NGPU>,BUCKET_NGPU> pair;
    std::array<std::array<Code,BUCKET_NGPU>,BUCKET_NGPU> slot_capacity{};
    std::array<Code,BUCKET_NGPU> gpu_capacity{};
};

static BucketPhysicalLayoutHost build_bucket_physical_layout(
    const StorageLayout& layout, const BucketOwnerHost& owner
) {
    BucketPhysicalLayoutHost out;
    for (int a=0;a<BUCKET_NGPU;++a) for (int b=0;b<BUCKET_NGPU;++b) {
        BucketPairLayout q;
        q.main_blocks.resize(layout.main_blocks.size());
        q.block_blocks.resize(layout.block_blocks.size());
        Code off=0;
        for (size_t bid=0;bid<layout.main_blocks.size();++bid) {
            const StorageBlock& x=layout.main_blocks[bid];
            BucketPhysicalBlock y;
            y.off=off;y.he=x.he;y.hs=x.hs;y.c=x.c;y.valid=x.valid;
            if (x.valid) {
                y.rows=owner.high_count[a][x.he];
                y.cols=owner.low_count[b][x.hs];
                off += Code(y.rows)*y.cols;
            }
            q.main_blocks[bid]=y;
        }
        for (size_t bid=0;bid<layout.block_blocks.size();++bid) {
            const StorageBlock& x=layout.block_blocks[bid];
            BucketPhysicalBlock y;
            y.off=off;y.he=x.he;y.hs=x.hs;y.c=x.c;y.valid=x.valid;
            if (x.valid) {
                y.rows=owner.high_count[a][x.he];
                y.cols=owner.low_count[b][x.hs];
                off += Code(y.rows)*y.cols;
            }
            q.block_blocks[bid]=y;
        }
        q.size=off;
        out.pair[a][b]=std::move(q);
    }
    for (int a=0;a<BUCKET_NGPU;++a) for (int b=0;b<BUCKET_NGPU;++b) {
        out.slot_capacity[a][b]=std::max(out.pair[a][b].size,out.pair[b][a].size);
        out.gpu_capacity[a]+=out.slot_capacity[a][b];
    }
    return out;
}

struct BucketAddress {
    uint8_t owner_h=0,owner_l=0;
    Code off=0;
};

static BucketAddress bucket_rank_main_host(
    MateID m, const StorageFactorHost& storage, const StorageLayout& layout,
    const BucketOwnerHost& owner, const BucketPhysicalLayoutHost& buckets
) {
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    uint32_t lc=uint32_t(m)&LM;
    uint32_t hc=uint32_t((m>>(2*(L+1)))&HM);
    int he=seg_end_height_host(hc,H),cv=int(mget(m,L));
    const StorageBlock& oldb=layout.main_blocks[3*he+cv];
    uint32_t lp=storage.low_packed_rank[lc],hp=storage.high_packed_rank[hc];
    if (!oldb.valid||lp==0xffffffffu||hp==0xffffffffu) std::exit(182);
    uint32_t lr=lp>>L,hr=hp>>H;
    uint32_t hl=owner.high_all_locator[storage.high_all_off[he]+hr];
    uint32_t ll=owner.low_all_locator[storage.low_all_off[oldb.hs]+lr];
    uint32_t oh=bucket_locator_owner(hl),ol=bucket_locator_owner(ll);
    const BucketPhysicalBlock& b=buckets.pair[oh][ol].main_blocks[3*he+cv];
    uint32_t rH=bucket_locator_rank(hl),rL=bucket_locator_rank(ll);
    if (rH>=b.rows||rL>=b.cols) std::exit(183);
    return {uint8_t(oh),uint8_t(ol),b.off+Code(rH)*b.cols+rL};
}

static BucketAddress bucket_rank_block_host(
    MateID m, const StorageFactorHost& storage, const StorageLayout& layout,
    const BucketOwnerHost& owner, const BucketPhysicalLayoutHost& buckets
) {
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;
    uint32_t lc=uint32_t(m)&LM;
    uint32_t hc=uint32_t((m>>(2*L))&HM);
    int h=seg_end_height_host(hc,H);
    uint32_t lp=storage.low_packed_rank[lc],hp=storage.high_packed_rank[hc];
    if (lp==0xffffffffu||hp==0xffffffffu) std::exit(184);
    uint32_t lr=lp>>L,hr=hp>>H;
    uint32_t hl=owner.high_all_locator[storage.high_all_off[h]+hr];
    uint32_t ll=owner.low_all_locator[storage.low_all_off[h]+lr];
    uint32_t oh=bucket_locator_owner(hl),ol=bucket_locator_owner(ll);
    const BucketPhysicalBlock& b=buckets.pair[oh][ol].block_blocks[h];
    uint32_t rH=bucket_locator_rank(hl),rL=bucket_locator_rank(ll);
    if (rH>=b.rows||rL>=b.cols) std::exit(185);
    return {uint8_t(oh),uint8_t(ol),b.off+Code(rH)*b.cols+rL};
}
