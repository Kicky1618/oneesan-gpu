#pragma once

#include "ramstream32_bucket_reverse_fused.cuh"

// Reverse destination-gather tables without legacy reverse orbit metadata.
// Split18 provides its own NN/NR/NL orbit streams, so allocating D_RB_* orbit
// arrays would only create an avoidable HBM peak during setup.
struct ReverseBucketFusedClosureOnlyDirectTables {
    BucketFusedDst *low_dst=nullptr,*high_dst=nullptr;
    uint32_t *low_off=nullptr,*high_off=nullptr;
    uint32_t *low_local=nullptr,*low_cross=nullptr,*high_local=nullptr,*high_cross=nullptr;

    template<class T>
    static void cp(T*&d,const std::vector<T>&s,const char*w){
        if(s.empty())return;
        ck(cudaMalloc(&d,s.size()*sizeof(T)),w);
        ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);
    }

    void install(const ReverseBucketAtomicHost&,const ReverseBucketFusedHost&f){
        cp(low_dst,f.low_dst,"reverse closure-only low dst");
        cp(high_dst,f.high_dst,"reverse closure-only high dst");
        cp(low_off,f.low_off,"reverse closure-only low off");
        cp(high_off,f.high_off,"reverse closure-only high off");
        cp(low_local,f.low_local_src,"reverse closure-only low local");
        cp(low_cross,f.low_cross_op,"reverse closure-only low cross");
        cp(high_local,f.high_local_src,"reverse closure-only high local");
        cp(high_cross,f.high_cross_op,"reverse closure-only high cross");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_DST,&low_dst,sizeof(low_dst)),"reverse closure-only low dst ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_DST,&high_dst,sizeof(high_dst)),"reverse closure-only high dst ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_OFF,&low_off,sizeof(low_off)),"reverse closure-only low off ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_OFF,&high_off,sizeof(high_off)),"reverse closure-only high off ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_LOCAL_SRC,&low_local,sizeof(low_local)),"reverse closure-only low local ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_CROSS_OP,&low_cross,sizeof(low_cross)),"reverse closure-only low cross ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_LOCAL_SRC,&high_local,sizeof(high_local)),"reverse closure-only high local ptr");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_CROSS_OP,&high_cross,sizeof(high_cross)),"reverse closure-only high cross ptr");
        ck(cudaMemcpyToSymbol(D_RBF_LOW_PITCH,&f.low_pitch,sizeof(f.low_pitch)),"reverse closure-only low pitch");
        ck(cudaMemcpyToSymbol(D_RBF_HIGH_PITCH,&f.high_pitch,sizeof(f.high_pitch)),"reverse closure-only high pitch");
    }

    void release(){
        if(low_dst)cudaFree(low_dst);if(high_dst)cudaFree(high_dst);
        if(low_off)cudaFree(low_off);if(high_off)cudaFree(high_off);
        if(low_local)cudaFree(low_local);if(low_cross)cudaFree(low_cross);
        if(high_local)cudaFree(high_local);if(high_cross)cudaFree(high_cross);
        low_dst=high_dst=nullptr;low_off=high_off=nullptr;
        low_local=low_cross=high_local=high_cross=nullptr;
    }
};
