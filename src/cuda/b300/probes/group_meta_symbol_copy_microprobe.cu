#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using Code=unsigned long long;
static constexpr int MAXW=28;

__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ std::uint32_t D_MAIN_FIXED,D_MAIN_OCC,D_BLOCK_FIXED,D_BLOCK_OCC;

struct DeviceGroupMeta{
    Code main_dp[MAXW+1][MAXW+2];
    Code block_dp[MAXW+1][MAXW+2];
    std::uint32_t main_fixed,main_occ,block_fixed,block_occ;
};
static_assert(sizeof(DeviceGroupMeta)==13936,"unexpected packed metadata size");
__constant__ DeviceGroupMeta D_GROUP_META;

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(1);}}

__global__ void checksum_unpacked(unsigned long long* out){
    if(threadIdx.x||blockIdx.x)return;
    unsigned long long x=0;
    x^=D_MAIN_DP[0][0];x^=D_MAIN_DP[MAXW][MAXW+1]<<1;
    x^=D_BLOCK_DP[1][2]<<2;x^=D_BLOCK_DP[MAXW][0]<<3;
    x^=unsigned long long(D_MAIN_FIXED)<<4;x^=unsigned long long(D_MAIN_OCC)<<5;
    x^=unsigned long long(D_BLOCK_FIXED)<<6;x^=unsigned long long(D_BLOCK_OCC)<<7;
    *out=x;
}
__global__ void checksum_packed(unsigned long long* out){
    if(threadIdx.x||blockIdx.x)return;
    unsigned long long x=0;
    x^=D_GROUP_META.main_dp[0][0];x^=D_GROUP_META.main_dp[MAXW][MAXW+1]<<1;
    x^=D_GROUP_META.block_dp[1][2]<<2;x^=D_GROUP_META.block_dp[MAXW][0]<<3;
    x^=unsigned long long(D_GROUP_META.main_fixed)<<4;x^=unsigned long long(D_GROUP_META.main_occ)<<5;
    x^=unsigned long long(D_GROUP_META.block_fixed)<<6;x^=unsigned long long(D_GROUP_META.block_occ)<<7;
    *out=x;
}

static void fill(DeviceGroupMeta& m){
    for(int i=0;i<=MAXW;++i)for(int j=0;j<=MAXW+1;++j){
        m.main_dp[i][j]=Code(i+1)*1000003ULL+Code(j)*97ULL+11ULL;
        m.block_dp[i][j]=Code(i+3)*1000033ULL+Code(j)*193ULL+29ULL;
    }
    m.main_fixed=0x15555555u;m.main_occ=0x0aaaaaaau;m.block_fixed=0x05555555u;m.block_occ=0x02aaaaaau;
}

static void copy_unpacked(const DeviceGroupMeta& m){
    ck(cudaMemcpyToSymbol(D_MAIN_DP,m.main_dp,sizeof(m.main_dp)),"copy main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP,m.block_dp,sizeof(m.block_dp)),"copy block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED,&m.main_fixed,sizeof(m.main_fixed)),"copy main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC,&m.main_occ,sizeof(m.main_occ)),"copy main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED,&m.block_fixed,sizeof(m.block_fixed)),"copy block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC,&m.block_occ,sizeof(m.block_occ)),"copy block occ");
}
static void copy_packed(const DeviceGroupMeta& m){ck(cudaMemcpyToSymbol(D_GROUP_META,&m,sizeof(m)),"copy packed meta");}

static double run(bool packed,DeviceGroupMeta m,int iters){
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<iters;++i){
        m.main_fixed^=std::uint32_t(0x9e3779b9u+std::uint32_t(i));
        m.block_occ^=std::uint32_t(0x7f4a7c15u+std::uint32_t(i*3));
        if(packed)copy_packed(m);else copy_unpacked(m);
    }
    ck(cudaDeviceSynchronize(),"copy sync");
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/double(iters);
}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());size_t n=x.size();return n&1?x[n/2]:0.5*(x[n/2-1]+x[n/2]);}

int main(int argc,char**argv){
    int iters=argc>1?std::atoi(argv[1]):4096,repeats=argc>2?std::atoi(argv[2]):9,dev=argc>3?std::atoi(argv[3]):0;
    if(iters<1||repeats<1)return 2;int visible=0;ck(cudaGetDeviceCount(&visible),"device count");if(dev<0||dev>=visible)return 2;ck(cudaSetDevice(dev),"set device");
    DeviceGroupMeta meta{};fill(meta);copy_unpacked(meta);copy_packed(meta);
    unsigned long long* d=nullptr;ck(cudaMalloc(&d,sizeof(*d)),"checksum alloc");unsigned long long a=0,b=0;
    checksum_unpacked<<<1,1>>>(d);ck(cudaGetLastError(),"checksum unpacked");ck(cudaMemcpy(&a,d,sizeof(a),cudaMemcpyDeviceToHost),"checksum unpacked copy");
    checksum_packed<<<1,1>>>(d);ck(cudaGetLastError(),"checksum packed");ck(cudaMemcpy(&b,d,sizeof(b),cudaMemcpyDeviceToHost),"checksum packed copy");
    if(a!=b){std::fprintf(stderr,"initial checksum mismatch %llu != %llu\n",a,b);return 3;}
    run(false,meta,128);run(true,meta,128);
    std::vector<double> u,p;u.reserve(repeats);p.reserve(repeats);
    for(int r=0;r<repeats;++r){if(r&1){p.push_back(run(true,meta,iters));u.push_back(run(false,meta,iters));}else{u.push_back(run(false,meta,iters));p.push_back(run(true,meta,iters));}}
    double us_u=median(u),us_p=median(p);
    copy_unpacked(meta);copy_packed(meta);checksum_unpacked<<<1,1>>>(d);ck(cudaMemcpy(&a,d,sizeof(a),cudaMemcpyDeviceToHost),"final unpacked");checksum_packed<<<1,1>>>(d);ck(cudaMemcpy(&b,d,sizeof(b),cudaMemcpyDeviceToHost),"final packed");cudaFree(d);
    if(a!=b){std::fprintf(stderr,"final checksum mismatch %llu != %llu\n",a,b);return 4;}
    std::printf("b300-group-meta-symbol-copy-microprobe OK device=%d meta_bytes=%zu iters=%d repeats=%d unpacked_calls=6 packed_calls=1 unpacked_us_per_group=%.6f packed_us_per_group=%.6f speedup=%.6f call_reduction=6x pageable_host=1 exact=OK\n",dev,sizeof(meta),iters,repeats,us_u,us_p,us_u/us_p);
    return 0;
}
