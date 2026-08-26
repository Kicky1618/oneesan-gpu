#pragma once

#include "ramstream32_bucket_orbit_closure_rec8_hybrid18.cuh"
#include "ramstream32_bucket_reverse_split18_direct.hpp"

struct BucketForwardHybrid18Rec8Host {
    BucketForwardOrbitClosureAttach18Host attach;
    BucketForwardOnePassRec8Host rec8;
    size_t bytes() const { return attach.bytes() + rec8.bytes(); }
};
struct BucketReverseHybrid18Rec8Host {
    ReverseSplit18Host split;
    BucketReverseOnePassRec8Host rec8;
    size_t bytes() const { return split.bytes() + rec8.bytes(); }
};

static BucketForwardHybrid18Rec8Host build_bucket_forward_hybrid18_rec8(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,const BucketFusedHost&bf
){
    BucketForwardHybrid18Rec8Host out;
    out.attach=build_bucket_forward_orbit_closure_attach18(layout,bo,bf);
    out.rec8=build_bucket_forward_onepass_rec8(bf);
    return out;
}
static BucketReverseHybrid18Rec8Host build_bucket_reverse_hybrid18_rec8_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,const BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,const ReverseBucketFusedHost&rf
){
    BucketReverseHybrid18Rec8Host out;
    out.rec8=build_bucket_reverse_onepass_rec8(rf);
    out.split=build_reverse_split18_direct_checked(layout,bo,bf,rb,rf,true);
    return out;
}

struct BucketForwardHybrid18Rec8DeviceTables {
    BucketForwardOrbitClosureAttach18DeviceTables attach;
    BucketForwardOnePassRec8DeviceTables rec8;
    void install(const BucketForwardHybrid18Rec8Host&h){attach.install(h.attach);rec8.install(h.rec8);}
    void release(){rec8.release();attach.release();}
};
struct BucketReverseHybrid18Rec8DeviceTables {
    ReverseSplit18DeviceTables split;
    BucketReverseOnePassRec8DeviceTables rec8;
    void install(const BucketReverseHybrid18Rec8Host&h){split.install(h.split);rec8.install(h.rec8);}
    void release(){rec8.release();split.release();}
};

// Forward bucket tables still provide orbit streams, block views, ternary
// direct maps, code tables and fused offsets. One-pass rec8 has its own record
// and source arrays, so release the legacy 16-B destination records and the
// separate source arrays immediately after install.
struct BucketFusedHybrid18Rec8Tables {
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

// Reverse split18 owns all reverse orbit metadata; rec8 owns all reverse
// closure source records. Only per-step destination-block offsets are required
// from ReverseBucketFusedHost to reconstruct the attached record id.
struct ReverseBucketRec8OffsetTables {
    uint32_t *low_off=nullptr,*high_off=nullptr;
    static void cp(uint32_t*&d,const std::vector<uint32_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(uint32_t)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),w);}
    void install(const ReverseBucketAtomicHost&,const ReverseBucketFusedHost&f){
        cp(low_off,f.low_off,"rec8 reverse low off");cp(high_off,f.high_off,"rec8 reverse high off");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_OFF,&low_off,sizeof(low_off)),"rec8 reverse low off ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_OFF,&high_off,sizeof(high_off)),"rec8 reverse high off ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_PITCH,&f.low_pitch,sizeof(f.low_pitch)),"rec8 reverse low pitch");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_PITCH,&f.high_pitch,sizeof(f.high_pitch)),"rec8 reverse high pitch");
    }
    void release(){if(low_off)cudaFree(low_off);if(high_off)cudaFree(high_off);low_off=high_off=nullptr;}
};
