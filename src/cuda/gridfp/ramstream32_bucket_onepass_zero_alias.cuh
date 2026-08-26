#pragma once

#include "ramstream32_bucket_orbit_closure_zero.cuh"

struct BucketForwardClosureZeroHost{size_t bytes()const{return 0;}};

static void bucket_zero_release_forward_closure(BucketFusedHost&f){
    std::vector<BucketFusedDst>().swap(f.low_dst);std::vector<BucketFusedDst>().swap(f.high_dst);
    std::vector<uint32_t>().swap(f.low_off);std::vector<uint32_t>().swap(f.high_off);
    std::vector<uint32_t>().swap(f.low_local_src);std::vector<uint32_t>().swap(f.low_cross_op);
    std::vector<uint32_t>().swap(f.high_local_src);std::vector<uint32_t>().swap(f.high_cross_op);
}
static void bucket_zero_release_reverse_closure(ReverseBucketFusedHost&f){
    std::vector<BucketFusedDst>().swap(f.low_dst);std::vector<BucketFusedDst>().swap(f.high_dst);
    std::vector<uint32_t>().swap(f.low_off);std::vector<uint32_t>().swap(f.high_off);
    std::vector<uint32_t>().swap(f.low_local_src);std::vector<uint32_t>().swap(f.low_cross_op);
    std::vector<uint32_t>().swap(f.high_local_src);std::vector<uint32_t>().swap(f.high_cross_op);
}
static BucketForwardClosureZeroHost build_bucket_forward_closure_zero(
    const StorageLayout&,BucketOrbitStreamsHost&,BucketFusedHost&f
){bucket_zero_release_forward_closure(f);return {};}
static ReverseSplit54Host build_bucket_reverse_split54_zero_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&,BucketFusedHost&,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){ReverseSplit54Host out=build_reverse_split54(layout,rb,true);bucket_zero_release_reverse_closure(rf);return out;}

struct BucketForwardClosureZeroDeviceTables{void install(const BucketForwardClosureZeroHost&){}void release(){}};

struct BucketFusedZeroClosureTables{
    BucketFusedDeviceTables base;
    void install_metadata(const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f){base.install_metadata(layout,o,f);}
    void bind_owner(uint32_t fixed,const BucketPhysicalLayoutHost&buckets,const std::array<Count*,BUCKET_NGPU>&slot){base.bind_owner(fixed,buckets,slot);}
    void release(){base.release();}
};

struct ReverseBucketZeroTables{void install(const ReverseBucketAtomicHost&,const ReverseBucketFusedHost&){}void release(){}};
