#pragma once

#include "ramstream32_bucket_orbit_closure_inline8_hybrid18.cuh"
#include "ramstream32_bucket_reverse_split18_direct.hpp"

struct BucketForwardHybrid18Inline8Host{
    BucketForwardOrbitClosureAttach18Host attach;
    BucketForwardOnePassInline8Host inline8;
    size_t bytes()const{return attach.bytes()+inline8.bytes();}
};
struct BucketReverseHybrid18Inline8Host{
    ReverseSplit18Host split;
    BucketReverseOnePassInline8Host inline8;
    size_t bytes()const{return split.bytes()+inline8.bytes();}
};

static void bucket_inline8_release_forward_legacy(BucketFusedHost&f){
    std::vector<BucketFusedDst>().swap(f.low_dst);std::vector<BucketFusedDst>().swap(f.high_dst);
    std::vector<uint32_t>().swap(f.low_local_src);std::vector<uint32_t>().swap(f.low_cross_op);
    std::vector<uint32_t>().swap(f.high_local_src);std::vector<uint32_t>().swap(f.high_cross_op);
}
static void bucket_inline8_release_reverse_legacy(ReverseBucketFusedHost&f){
    std::vector<BucketFusedDst>().swap(f.low_dst);std::vector<BucketFusedDst>().swap(f.high_dst);
    std::vector<uint32_t>().swap(f.low_local_src);std::vector<uint32_t>().swap(f.low_cross_op);
    std::vector<uint32_t>().swap(f.high_local_src);std::vector<uint32_t>().swap(f.high_cross_op);
}
static BucketForwardHybrid18Inline8Host build_bucket_forward_hybrid18_inline8(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf
){
    BucketForwardHybrid18Inline8Host out;
    out.attach=build_bucket_forward_orbit_closure_attach18(layout,bo,bf);
    out.inline8=build_bucket_forward_onepass_inline8(bf);
    return out;
}
static BucketReverseHybrid18Inline8Host build_bucket_reverse_hybrid18_inline8_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    BucketReverseHybrid18Inline8Host out;
    out.inline8=build_bucket_reverse_onepass_inline8(rf);
    out.split=build_reverse_split18_direct_checked(layout,bo,bf,rb,rf,true);
    bucket_inline8_release_forward_legacy(bf);
    bucket_inline8_release_reverse_legacy(rf);
    return out;
}

struct BucketForwardHybrid18Inline8DeviceTables{
    BucketForwardOrbitClosureAttach18DeviceTables attach;
    BucketForwardOnePassInline8DeviceTables inline8;
    void install(const BucketForwardHybrid18Inline8Host&h){attach.install(h.attach);inline8.install(h.inline8);}
    void release(){inline8.release();attach.release();}
};
struct BucketReverseHybrid18Inline8DeviceTables{
    ReverseSplit18DeviceTables split;
    BucketReverseOnePassInline8DeviceTables inline8;
    void install(const BucketReverseHybrid18Inline8Host&h){split.install(h.split);inline8.install(h.inline8);}
    void release(){inline8.release();split.release();}
};

struct BucketFusedHybrid18Inline8Tables{
    BucketFusedDeviceTables base;
    void install_metadata(const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f){
        base.install_metadata(layout,o,f);
        if(base.low_dst)cudaFree(base.low_dst);base.low_dst=nullptr;
        if(base.high_dst)cudaFree(base.high_dst);base.high_dst=nullptr;
        if(base.low_local_src)cudaFree(base.low_local_src);base.low_local_src=nullptr;
        if(base.low_cross_op)cudaFree(base.low_cross_op);base.low_cross_op=nullptr;
        if(base.high_local_src)cudaFree(base.high_local_src);base.high_local_src=nullptr;
        if(base.high_cross_op)cudaFree(base.high_cross_op);base.high_cross_op=nullptr;
    }
    void bind_owner(uint32_t fixed,const BucketPhysicalLayoutHost&buckets,const std::array<Count*,BUCKET_NGPU>&slot){base.bind_owner(fixed,buckets,slot);}
    void release(){base.release();}
};

// Only reverse closure offsets are needed after split18+inline8 conversion.
struct ReverseBucketInline8OffsetTables{
    uint32_t *low_off=nullptr,*high_off=nullptr;
    static void cp(uint32_t*&d,const std::vector<uint32_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(uint32_t)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),w);}
    void install(const ReverseBucketAtomicHost&,const ReverseBucketFusedHost&f){cp(low_off,f.low_off,"inline8 reverse low off");cp(high_off,f.high_off,"inline8 reverse high off");ck(cudaMemcpyToSymbol(D_RBF_LOW_OFF,&low_off,sizeof(low_off)),"inline8 reverse low off ptr");ck(cudaMemcpyToSymbol(D_RBF_HIGH_OFF,&high_off,sizeof(high_off)),"inline8 reverse high off ptr");ck(cudaMemcpyToSymbol(D_RBF_LOW_PITCH,&f.low_pitch,sizeof(f.low_pitch)),"inline8 reverse low pitch");ck(cudaMemcpyToSymbol(D_RBF_HIGH_PITCH,&f.high_pitch,sizeof(f.high_pitch)),"inline8 reverse high pitch");}
    void release(){if(low_off)cudaFree(low_off);if(high_off)cudaFree(high_off);low_off=high_off=nullptr;}
};
