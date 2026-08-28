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

static unsigned long long checksum_host(const DeviceGroupMeta& m){
    unsigned long long x=0;
    x^=m.main_dp[0][0];x^=m.main_dp[MAXW][MAXW+1]<<1;
    x^=m.block_dp[1][2]<<2;x^=m.block_dp[MAXW][0]<<3;
    x^=static_cast<unsigned long long>(m.main_fixed)<<4;x^=static_cast<unsigned long long>(m.main_occ)<<5;
    x^=static_cast<unsigned long long>(m.block_fixed)<<6;x^=static_cast<unsigned long long>(m.block_occ)<<7;
    return x;
}
__global__ void checksum_unpacked(unsigned long long* out){
    if(threadIdx.x||blockIdx.x)return;
    unsigned long long x=0;
    x^=D_MAIN_DP[0][0];x^=D_MAIN_DP[MAXW][MAXW+1]<<1;
    x^=D_BLOCK_DP[1][2]<<2;x^=D_BLOCK_DP[MAXW][0]<<3;
    x^=static_cast<unsigned long long>(D_MAIN_FIXED)<<4;x^=static_cast<unsigned long long>(D_MAIN_OCC)<<5;
    x^=static_cast<unsigned long long>(D_BLOCK_FIXED)<<6;x^=static_cast<unsigned long long>(D_BLOCK_OCC)<<7;
    *out=x;
}
__global__ void checksum_packed(unsigned long long* out){
    if(threadIdx.x||blockIdx.x)return;
    unsigned long long x=0;
    x^=D_GROUP_META.main_dp[0][0];x^=D_GROUP_META.main_dp[MAXW][MAXW+1]<<1;
    x^=D_GROUP_META.block_dp[1][2]<<2;x^=D_GROUP_META.block_dp[MAXW][0]<<3;
    x^=static_cast<unsigned long long>(D_GROUP_META.main_fixed)<<4;x^=static_cast<unsigned long long>(D_GROUP_META.main_occ)<<5;
    x^=static_cast<unsigned long long>(D_GROUP_META.block_fixed)<<6;x^=static_cast<unsigned long long>(D_GROUP_META.block_occ)<<7;
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
static void copy_staged(const DeviceGroupMeta* d){ck(cudaMemcpyToSymbol(D_GROUP_META,d,sizeof(DeviceGroupMeta),0,cudaMemcpyDeviceToDevice),"copy staged meta D2D");}

static double run_host(bool packed,DeviceGroupMeta m,int iters){
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<iters;++i){
        m.main_fixed^=std::uint32_t(0x9e3779b9u+std::uint32_t(i));
        m.block_occ^=std::uint32_t(0x7f4a7c15u+std::uint32_t(i*3));
        if(packed)copy_packed(m);else copy_unpacked(m);
    }
    ck(cudaDeviceSynchronize(),"host copy sync");
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/double(iters);
}
static double run_staged(const DeviceGroupMeta* d,int slots,int iters){
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<iters;++i)copy_staged(d+(i%slots));
    ck(cudaDeviceSynchronize(),"staged copy sync");
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/double(iters);
}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());size_t n=x.size();return n&1?x[n/2]:0.5*(x[n/2-1]+x[n/2]);}

int main(int argc,char**argv){
    int iters=argc>1?std::atoi(argv[1]):4096,repeats=argc>2?std::atoi(argv[2]):9,dev=argc>3?std::atoi(argv[3]):0,slots=argc>4?std::atoi(argv[4]):1024;
    if(iters<1||repeats<1||slots<1)return 2;int visible=0;ck(cudaGetDeviceCount(&visible),"device count");if(dev<0||dev>=visible)return 2;ck(cudaSetDevice(dev),"set device");
    DeviceGroupMeta meta{};fill(meta);copy_unpacked(meta);copy_packed(meta);
    unsigned long long* dsum=nullptr;ck(cudaMalloc(&dsum,sizeof(*dsum)),"checksum alloc");unsigned long long a=0,b=0;
    checksum_unpacked<<<1,1>>>(dsum);ck(cudaGetLastError(),"checksum unpacked");ck(cudaMemcpy(&a,dsum,sizeof(a),cudaMemcpyDeviceToHost),"checksum unpacked copy");
    checksum_packed<<<1,1>>>(dsum);ck(cudaGetLastError(),"checksum packed");ck(cudaMemcpy(&b,dsum,sizeof(b),cudaMemcpyDeviceToHost),"checksum packed copy");
    if(a!=b||a!=checksum_host(meta)){std::fprintf(stderr,"initial checksum mismatch %llu != %llu\n",a,b);return 3;}

    std::vector<DeviceGroupMeta> hstage(static_cast<size_t>(slots));
    for(int i=0;i<slots;++i){hstage[static_cast<size_t>(i)]=meta;hstage[static_cast<size_t>(i)].main_fixed^=std::uint32_t(i*0x9e3779b9u);hstage[static_cast<size_t>(i)].block_occ^=std::uint32_t(i*0x7f4a7c15u);hstage[static_cast<size_t>(i)].main_dp[i%(MAXW+1)][i%(MAXW+2)]^=Code(i+1);}
    DeviceGroupMeta* dstage=nullptr;ck(cudaMalloc(&dstage,hstage.size()*sizeof(DeviceGroupMeta)),"staged alloc");ck(cudaMemcpy(dstage,hstage.data(),hstage.size()*sizeof(DeviceGroupMeta),cudaMemcpyHostToDevice),"staged setup H2D");
    int check_slot=(iters-1)%slots;copy_staged(dstage+check_slot);checksum_packed<<<1,1>>>(dsum);ck(cudaMemcpy(&b,dsum,sizeof(b),cudaMemcpyDeviceToHost),"staged checksum copy");if(b!=checksum_host(hstage[static_cast<size_t>(check_slot)])){std::fprintf(stderr,"staged checksum mismatch\n");return 4;}

    run_host(false,meta,128);run_host(true,meta,128);run_staged(dstage,slots,128);
    std::vector<double> u,p,s;u.reserve(repeats);p.reserve(repeats);s.reserve(repeats);
    for(int r=0;r<repeats;++r){
        switch(r%3){
            case 0:u.push_back(run_host(false,meta,iters));p.push_back(run_host(true,meta,iters));s.push_back(run_staged(dstage,slots,iters));break;
            case 1:p.push_back(run_host(true,meta,iters));s.push_back(run_staged(dstage,slots,iters));u.push_back(run_host(false,meta,iters));break;
            default:s.push_back(run_staged(dstage,slots,iters));u.push_back(run_host(false,meta,iters));p.push_back(run_host(true,meta,iters));break;
        }
    }
    double us_u=median(u),us_p=median(p),us_s=median(s);
    copy_unpacked(meta);copy_packed(meta);checksum_unpacked<<<1,1>>>(dsum);ck(cudaMemcpy(&a,dsum,sizeof(a),cudaMemcpyDeviceToHost),"final unpacked");checksum_packed<<<1,1>>>(dsum);ck(cudaMemcpy(&b,dsum,sizeof(b),cudaMemcpyDeviceToHost),"final packed");
    cudaFree(dstage);cudaFree(dsum);
    if(a!=b){std::fprintf(stderr,"final checksum mismatch %llu != %llu\n",a,b);return 5;}
    std::printf("b300-group-meta-symbol-copy-microprobe OK device=%d meta_bytes=%zu iters=%d repeats=%d staged_slots=%d unpacked_calls=6 packed_calls=1 staged_calls=1 unpacked_us_per_group=%.6f packed_us_per_group=%.6f staged_us_per_group=%.6f speedup=%.6f staged_speedup_vs_packed=%.6f call_reduction=6x pageable_host=1 staged_setup_timed=0 staged_copy_kind=D2D exact=OK\n",dev,sizeof(meta),iters,repeats,slots,us_u,us_p,us_s,us_u/us_p,us_p/us_s);
    return 0;
}
