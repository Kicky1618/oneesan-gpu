#pragma once

// Corrected LOW orbit launcher layered on ramstream32_bucket_fused.cuh.
// At the center step p==LOW_LUT_K, cpu_sparse_jblock maps NN and NL to the
// center=L factor block and NR to center=R.  The first experimental bucket
// kernel incorrectly left NN in its source block.  Keep this v2 entry point
// separate until the W=10 exhaustive GPU test has validated the bucket backend.

__global__ void bucket_low_orbit_kernel_v2(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;
    uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_PITCH+bid;
    uint32_t na=D_BKF_LOW_NN_OFF[oi],nb=D_BKF_LOW_NN_OFF[oi+1];
    uint32_t ra=D_BKF_LOW_NR_OFF[oi],rb=D_BKF_LOW_NR_OFF[oi+1];
    uint32_t la=D_BKF_LOW_NL_OFF[oi],lb=D_BKF_LOW_NL_OFF[oi+1];
    uint32_t n0=nb-na,n1=rb-ra,total=n0+n1+(lb-la);if(!total)return;
    for(uint32_t k=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;k<total;k+=uint32_t(gridDim.x)*blockDim.x){
        uint32_t kind;BucketOrbitOp op;
        if(k<n0){kind=CPU_ORBIT_NN;op=D_BKF_LOW_NN[na+k];}
        else if(k<n0+n1){kind=CPU_ORBIT_NR;op=D_BKF_LOW_NR[ra+k-n0];}
        else{kind=CPU_ORBIT_NL;op=D_BKF_LOW_NL[la+k-n0-n1];}
        uint32_t sl=bucket_orbit_src(op),jl=bucket_orbit_partner(op),dl=bucket_orbit_drop(op);
        uint32_t ss=bucket_locator_owner(sl),js=bucket_locator_owner(jl),ds=bucket_locator_owner(dl);
        BucketPhysicalBlock xb=bkf_low_main(ss,bid);
        if(!xb.valid||!xb.rows||!xb.cols)continue;
        uint32_t jbid=bid;
        if(p==LOW_LUT_K){
            uint32_t center=(kind==CPU_ORBIT_NR)?uint32_t(R):uint32_t(::L);
            jbid=3u*uint32_t(xb.he)+center;
        }
        BucketPhysicalBlock jb=bkf_low_main(js,jbid),db=bkf_low_block(ds,uint32_t(xb.he));
        uint32_t sr=bucket_locator_rank(sl),jr=bucket_locator_rank(jl),dr=bucket_locator_rank(dl);
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){
            Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+sr);
            Count*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+jr);
            Count*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+dr);
            Count c=*ip,old=*dp;
            if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=0;}
            else{Count cc=*jp,all=gpu_direct_add(gpu_direct_add(c,cc),old);if(p==1){*ip=all;*jp=gpu_direct_add(c,cc);*dp=0;}else{*ip=all;*dp=c;}}
        }
    }
}

static void bucket_run_low_fused_v2(const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8){
    dim3 block(threads),grid(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K;p>=1;--p){
        bucket_low_orbit_kernel_v2<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit v2");
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 cg(grid_x,grid_y,nt);bucket_low_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket low closure v2");
    }
    ck(cudaDeviceSynchronize(),"bucket low v2 sync");
}
