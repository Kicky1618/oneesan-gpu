#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_bucket_layout.hpp"

static void bucket_enum_rec(int pos,int h,MateID m,std::vector<MateID>&out){
    if(pos<0){if(h==0)out.push_back(m);return;}
    bucket_enum_rec(pos-1,h,mset(m,pos,N),out);
    if(h>0)bucket_enum_rec(pos-1,h-1,mset(m,pos,R),out);
    bucket_enum_rec(pos-1,h+1,mset(m,pos,::L),out);
}
static std::vector<MateID> bucket_enum(int width){std::vector<MateID>v;bucket_enum_rec(width-1,1,0,v);return v;}

int main(){
    constexpr int W=TARGET_W;
    static_assert(W==LOW_LUT_K+HIGH_LUT_K+1);
    static_assert(W<=12,"bucket codec selftest intentionally uses small width");
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    BucketPhysicalLayoutHost buckets=build_bucket_physical_layout(layout,owner);

    auto ms=bucket_enum(W),bs=bucket_enum(W-1);
    if(ms.size()!=layout.main_size||bs.size()!=layout.block_size)return 2;
    std::array<std::array<std::vector<uint64_t>,BUCKET_NGPU>,BUCKET_NGPU> raw;
    for(int a=0;a<BUCKET_NGPU;++a)for(int b=0;b<BUCKET_NGPU;++b)
        raw[a][b].assign(size_t(buckets.pair[a][b].size),0);

    uint64_t marker=1;
    for(MateID m:ms){auto a=bucket_rank_main_host(m,storage,layout,owner,buckets);auto&v=raw[a.owner_h][a.owner_l];if(a.off>=v.size()||v[size_t(a.off)])return 3;v[size_t(a.off)]=marker++;}
    for(MateID m:bs){auto a=bucket_rank_block_host(m,storage,layout,owner,buckets);auto&v=raw[a.owner_h][a.owner_l];if(a.off>=v.size()||v[size_t(a.off)])return 4;v[size_t(a.off)]=marker++;}
    uint64_t sum=0;
    for(int a=0;a<BUCKET_NGPU;++a)for(int b=0;b<BUCKET_NGPU;++b){sum+=raw[a][b].size();for(uint64_t x:raw[a][b])if(!x){std::cerr<<"bucket hole "<<a<<','<<b<<'\n';return 5;}}
    if(sum!=ms.size()+bs.size()||marker-1!=sum)return 6;

    std::array<std::array<std::vector<uint64_t>,BUCKET_NGPU>,BUCKET_NGPU> slot;
    for(int a=0;a<BUCKET_NGPU;++a)for(int b=0;b<BUCKET_NGPU;++b){slot[a][b].assign(size_t(buckets.slot_capacity[a][b]),0);std::copy(raw[a][b].begin(),raw[a][b].end(),slot[a][b].begin());}
    std::array<int,BUCKET_NGPU> ring{};std::iota(ring.begin(),ring.end(),0);std::set<std::pair<int,int>> seen;
    for(int round=0;round<BUCKET_NGPU-1;++round){
        for(int i=0;i<BUCKET_NGPU/2;++i){int a=ring[i],b=ring[BUCKET_NGPU-1-i];if(a>b)std::swap(a,b);if(!seen.insert({a,b}).second)return 7;auto&A=slot[a][b];auto&B=slot[b][a];if(A.size()!=B.size())return 8;constexpr size_t CH=31;for(size_t off=0;off<A.size();off+=CH){size_t n=std::min(CH,A.size()-off);std::vector<uint64_t>ta(A.begin()+off,A.begin()+off+n),tb(B.begin()+off,B.begin()+off+n);std::copy(tb.begin(),tb.end(),A.begin()+off);std::copy(ta.begin(),ta.end(),B.begin()+off);}}
        int last=ring.back();for(int i=BUCKET_NGPU-1;i>=2;--i)ring[i]=ring[i-1];ring[1]=last;
    }
    if(seen.size()!=28)return 9;
    marker=1;
    for(MateID m:ms){auto a=bucket_rank_main_host(m,storage,layout,owner,buckets);if(slot[a.owner_l][a.owner_h][size_t(a.off)]!=marker++)return 10;}
    for(MateID m:bs){auto a=bucket_rank_block_host(m,storage,layout,owner,buckets);if(slot[a.owner_l][a.owner_h][size_t(a.off)]!=marker++)return 11;}

    Code total_capacity=0;for(int g=0;g<BUCKET_NGPU;++g)total_capacity+=buckets.gpu_capacity[g];
    std::cout<<"ramstream32-bucket-layout-selftest OK W="<<W
             <<" main="<<ms.size()<<" blocked="<<bs.size()
             <<" total="<<sum<<" pairs="<<seen.size()
             <<" high_max="<<owner.max_high_count
             <<" low_max="<<owner.max_low_count
             <<" locator_bits="<<BUCKET_LOCATOR_BITS
             <<" padded_states="<<total_capacity
             <<" padding_states="<<(total_capacity-sum)<<'\n';
    return 0;
}
