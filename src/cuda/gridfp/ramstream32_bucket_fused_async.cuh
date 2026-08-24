#pragma once

// Non-synchronizing launchers for the multi-GPU bucket driver. Call one on
// every GPU first, then synchronize all GPUs once at the window boundary.

static void bucket_launch_low_fused(
    const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8
){
    dim3 block(threads),grid(grid_x,grid_y,unsigned(layout.main_blocks.size()));
    for(int p=LOW_LUT_K;p>=1;--p){
        bucket_low_orbit_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit async");
        unsigned nt=p==1?unsigned(layout.main_blocks.size()):unsigned(layout.block_blocks.size());
        dim3 cg(grid_x,grid_y,nt);bucket_low_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket low closure async");
    }
}

// Temporary source-compatibility alias for the first B300 driver revision.
static void bucket_launch_low_fused_v2(
    const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8
){
    bucket_launch_low_fused(layout,threads,grid_x,grid_y);
}

static void bucket_launch_high_fused(
    const StorageLayout&layout,int threads=256,int grid_x=16,int grid_y=8
){
    dim3 block(threads),og(grid_x,grid_y,unsigned(layout.main_blocks.size())),cg(grid_x,grid_y,unsigned(layout.block_blocks.size()));
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        bucket_high_orbit_kernel<<<og,block>>>(p);ck(cudaGetLastError(),"bucket high orbit async");
        bucket_high_fused_closure_kernel<<<cg,block>>>(p);ck(cudaGetLastError(),"bucket high closure async");
    }
}

static void bucket_sync_devices(int ngpu){
    for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"bucket sync set device");ck(cudaDeviceSynchronize(),"bucket window sync");}
}
