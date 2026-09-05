#pragma once

#include "ramstream32_bucket_orbit_closure_inverse_hybrid18.cuh"
#include "ramstream32_bucket_reverse_split18_direct.hpp"

struct BucketForwardHybrid18InverseHost{
    BucketForwardOrbitClosureAttach18Host attach;
    BucketForwardClosureInverseHost inverse;
    size_t bytes()const{return attach.bytes()+inverse.bytes();}
};
struct BucketReverseHybrid18InverseHost{
    ReverseSplit18Host split;
    BucketReverseClosureInverseHost inverse;
    size_t bytes()const{return split.bytes()+inverse.bytes();}
};

static void bucket_inverse_release_forward_legacy(BucketFusedHost&f){
    std::vector<BucketFusedDst>().swap(f.low_dst);std::vector<BucketFusedDst>().swap(f.high_dst);
    std::vector<uint32_t>().swap(f.low_local_src);std::vector<uint32_t>().swap(f.low_cross_op);
    std::vector<uint32_t>().swap(f.high_local_src);std::vector<uint32_t>().swap(f.high_cross_op);
}
static void bucket_inverse_release_reverse_legacy(ReverseBucketFusedHost&f){
    std::vector<BucketFusedDst>().swap(f.low_dst);std::vector<BucketFusedDst>().swap(f.high_dst);
    std::vector<uint32_t>().swap(f.low_local_src);std::vector<uint32_t>().swap(f.low_cross_op);
    std::vector<uint32_t>().swap(f.high_local_src);std::vector<uint32_t>().swap(f.high_cross_op);
}
static BucketForwardHybrid18InverseHost build_bucket_forward_hybrid18_inverse(
    const StorageLayout&layout,BucketOrbitStreamsHost&bo,BucketFusedHost&bf
){
    BucketForwardHybrid18InverseHost out;out.attach=build_bucket_forward_orbit_closure_attach18(layout,bo,bf);out.inverse=build_bucket_forward_closure_inverse(bf);return out;
}
static BucketReverseHybrid18InverseHost build_bucket_reverse_hybrid18_inverse_checked(
    const StorageLayout&layout,const BucketOrbitStreamsHost&bo,BucketFusedHost&bf,
    ReverseBucketAtomicHost&rb,ReverseBucketFusedHost&rf
){
    BucketReverseHybrid18InverseHost out;out.inverse=build_bucket_reverse_closure_inverse(rf);out.split=build_reverse_split18_direct_checked(layout,bo,bf,rb,rf,true);bucket_inverse_release_forward_legacy(bf);bucket_inverse_release_reverse_legacy(rf);return out;
}

struct BucketForwardHybrid18InverseDeviceTables{
    BucketForwardOrbitClosureAttach18DeviceTables attach;BucketForwardClosureInverseDeviceTables inverse;
    void install(const BucketForwardHybrid18InverseHost&h){attach.install(h.attach);inverse.install(h.inverse);}
    void release(){inverse.release();attach.release();}
};
struct BucketReverseHybrid18InverseDeviceTables{
    ReverseSplit18DeviceTables split;BucketReverseClosureInverseDeviceTables inverse;
    void install(const BucketReverseHybrid18InverseHost&h){split.install(h.split);inverse.install(h.inverse);}
    void release(){inverse.release();split.release();}
};

struct BucketFusedHybrid18InverseTables{
    BucketFusedDeviceTables base;
    void install_metadata(const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f){
        base.install_metadata(layout,o,f);
        if(base.low_dst)cudaFree(base.low_dst);base.low_dst=nullptr;if(base.high_dst)cudaFree(base.high_dst);base.high_dst=nullptr;
        if(base.low_local_src)cudaFree(base.low_local_src);base.low_local_src=nullptr;if(base.low_cross_op)cudaFree(base.low_cross_op);base.low_cross_op=nullptr;
        if(base.high_local_src)cudaFree(base.high_local_src);base.high_local_src=nullptr;if(base.high_cross_op)cudaFree(base.high_cross_op);base.high_cross_op=nullptr;
    }
    void bind_owner(uint32_t fixed,const BucketPhysicalLayoutHost&buckets,const std::array<Count*,BUCKET_NGPU>&slot){base.bind_owner(fixed,buckets,slot);}
    void release(){base.release();}
};

struct ReverseBucketInverseOffsetTables{
    uint32_t *low_off=nullptr,*high_off=nullptr;
    static void cp(uint32_t*&d,const std::vector<uint32_t>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(uint32_t)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),w);}
    void install(const ReverseBucketAtomicHost&,const ReverseBucketFusedHost&f){cp(low_off,f.low_off,"inverse reverse low off");cp(high_off,f.high_off,"inverse reverse high off");ck(cudaMemcpyToSymbol(D_RBF_LOW_OFF,&low_off,sizeof(low_off)),"inverse reverse low off ptr");ck(cudaMemcpyToSymbol(D_RBF_HIGH_OFF,&high_off,sizeof(high_off)),"inverse reverse high off ptr");ck(cudaMemcpyToSymbol(D_RBF_LOW_PITCH,&f.low_pitch,sizeof(f.low_pitch)),"inverse reverse low pitch");ck(cudaMemcpyToSymbol(D_RBF_HIGH_PITCH,&f.high_pitch,sizeof(f.high_pitch)),"inverse reverse high pitch");}
    void release(){if(low_off)cudaFree(low_off);if(high_off)cudaFree(high_off);low_off=high_off=nullptr;}
};
