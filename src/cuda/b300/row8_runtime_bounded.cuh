#pragma once
// Requires solver kernels/types to be included first.
__global__ void bounded_fill_row2_automaton_runtime_kernel(Count* out,Code n,int width){
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank; int h=1;
        long long v0=1,v1=0,v2=0,v3=0,v4=0,v5=0;
        for(int pos=width-1;pos>=0;--pos){
            MateValue z=N; Code a=D_BOUND_DP[pos][h];
            if(r<a) z=N;
            else { r-=a; if(h>0){ a=D_BOUND_DP[pos][h-1]; if(r<a) z=R; else {r-=a; z=L;} } else z=L; }
            long long w0=0,w1=0,w2=0,w3=0,w4=0,w5=0;
            if(z==N){w1=v0+2*v1;w2=-v4-v5;w3=v3;w4=-v5;w5=v2+2*v4+3*v5;}
            else if(z==R){w0=v3;w2=v0;w4=v1;--h;}
            else {w0=v5;w1=v2+2*v4+2*v5;w3=v0+v1;++h;}
            v0=w0;v1=w1;v2=w2;v3=w3;v4=w4;v5=w5;
        }
        long long raw=v2+2*v4+2*v5;
        long long q=raw%(long long)D_MOD; if(q<0)q+=D_MOD;
        out[brank]=Count(q);
    }
}

static Count* build_row8_bounded_compact_runtime_width(int width,int K,Count mod,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2],int dev=0){
    if(K<2)throw std::runtime_error("row8 runtime bounded requires K>=2");
    cudaSetDevice(dev);ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"r8 runtime set mod");
    Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(2,dpOld);Code oldN=dpOld[width][1];Count*cur=nullptr;
    ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"r8 runtime base");
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"r8 runtime dp2");int cap2=2;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap2,sizeof(cap2)),"r8 runtime cap2");
    int bi=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_fill_row2_automaton_runtime_kernel<<<std::max(1,bi),threads>>>(cur,oldN,width);ck(cudaDeviceSynchronize(),"r8 runtime row2 init");
    for(int cap=3;cap<=K;++cap){
        build_bounded_dp(cap,dpNew);Code n=dpNew[width][1],dn=dpNew[width-1][1];Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;
        ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"r8 runtime a");ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"r8 runtime b");ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"r8 runtime d");ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"r8 runtime e");
        ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"r8 runtime zero a");ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"r8 runtime zero d");
        ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"r8 runtime old dp");ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"r8 runtime new dp");int oldcap=cap-1;ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"r8 runtime old cap");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"r8 runtime cap");
        int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads)),be=int(std::min<Code>(65535,(oldN+threads-1)/threads));
        bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,width);ck(cudaDeviceSynchronize(),"r8 runtime embed");cudaFree(cur);cur=a;Count*nxt=b,*dc=d,*de=e;
        for(int p=width-1;p>=1;--p){
            if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"r8 runtime clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,width,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,width,p);}
            else {ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"r8 runtime identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"r8 runtime clear d1");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,width,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,width,p);}
            ck(cudaDeviceSynchronize(),"r8 runtime step");std::swap(cur,nxt);std::swap(dc,de);
        }
        cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;
    }
    outN=oldN;std::memcpy(outDp,dpOld,sizeof(dpOld));return cur;
}
