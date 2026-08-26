#pragma once

#include "ramstream32_bucket_reverse_split54.hpp"

__constant__ BucketOrbitOp *D_RS54_LOW_NN,*D_RS54_LOW_NR,*D_RS54_LOW_NL,*D_RS54_HIGH_NN,*D_RS54_HIGH_NR,*D_RS54_HIGH_NL;
__constant__ uint32_t *D_RS54_LOW_NN_OFF,*D_RS54_LOW_NR_OFF,*D_RS54_LOW_NL_OFF,*D_RS54_HIGH_NN_OFF,*D_RS54_HIGH_NR_OFF,*D_RS54_HIGH_NL_OFF;
__constant__ uint32_t D_RS54_PITCH;

struct ReverseSplit54DeviceTables{
    BucketOrbitOp *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nr=nullptr,*high_nl=nullptr;
    uint32_t *low_nn_off=nullptr,*low_nr_off=nullptr,*low_nl_off=nullptr,*high_nn_off=nullptr,*high_nr_off=nullptr,*high_nl_off=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const ReverseSplit54Host&h){cp(low_nn,h.low.nn,"rs54 low nn");cp(low_nr,h.low.nr,"rs54 low nr");cp(low_nl,h.low.nl,"rs54 low nl");cp(high_nn,h.high.nn,"rs54 high nn");cp(high_nr,h.high.nr,"rs54 high nr");cp(high_nl,h.high.nl,"rs54 high nl");cp(low_nn_off,h.low.nn_off,"rs54 low nn off");cp(low_nr_off,h.low.nr_off,"rs54 low nr off");cp(low_nl_off,h.low.nl_off,"rs54 low nl off");cp(high_nn_off,h.high.nn_off,"rs54 high nn off");cp(high_nr_off,h.high.nr_off,"rs54 high nr off");cp(high_nl_off,h.high.nl_off,"rs54 high nl off");
#define RS54_SET(sym,p,msg) ck(cudaMemcpyToSymbol(sym,&p,sizeof(p)),msg)
        RS54_SET(D_RS54_LOW_NN,low_nn,"rs54 low nn ptr");RS54_SET(D_RS54_LOW_NR,low_nr,"rs54 low nr ptr");RS54_SET(D_RS54_LOW_NL,low_nl,"rs54 low nl ptr");RS54_SET(D_RS54_HIGH_NN,high_nn,"rs54 high nn ptr");RS54_SET(D_RS54_HIGH_NR,high_nr,"rs54 high nr ptr");RS54_SET(D_RS54_HIGH_NL,high_nl,"rs54 high nl ptr");RS54_SET(D_RS54_LOW_NN_OFF,low_nn_off,"rs54 low nn off ptr");RS54_SET(D_RS54_LOW_NR_OFF,low_nr_off,"rs54 low nr off ptr");RS54_SET(D_RS54_LOW_NL_OFF,low_nl_off,"rs54 low nl off ptr");RS54_SET(D_RS54_HIGH_NN_OFF,high_nn_off,"rs54 high nn off ptr");RS54_SET(D_RS54_HIGH_NR_OFF,high_nr_off,"rs54 high nr off ptr");RS54_SET(D_RS54_HIGH_NL_OFF,high_nl_off,"rs54 high nl off ptr");
#undef RS54_SET
        uint32_t pitch=h.nblocks+1;ck(cudaMemcpyToSymbol(D_RS54_PITCH,&pitch,sizeof(pitch)),"rs54 pitch");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nr);cudaFree(high_nl);cudaFree(low_nn_off);cudaFree(low_nr_off);cudaFree(low_nl_off);cudaFree(high_nn_off);cudaFree(high_nr_off);cudaFree(high_nl_off);low_nn=low_nr=low_nl=high_nn=high_nr=high_nl=nullptr;low_nn_off=low_nr_off=low_nl_off=high_nn_off=high_nr_off=high_nl_off=nullptr;}
};
