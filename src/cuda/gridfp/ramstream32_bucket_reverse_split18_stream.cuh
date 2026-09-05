#pragma once
#include "ramstream32_bucket_reverse_split18.cuh"

static void bucket_enqueue_reverse_low_split18(const StorageLayout&layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=1;p<=LOW_LUT_K;++p){bucket_reverse_low_split18_kernel<<<grid,block,0,stream>>>(p);ck(cudaGetLastError(),"bucket reverse low split18 stream");}}
static void bucket_enqueue_reverse_high_split18(const StorageLayout&layout,cudaStream_t stream,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_split18_kernel<<<grid,block,0,stream>>>(p);ck(cudaGetLastError(),"bucket reverse high split18 stream");}}
