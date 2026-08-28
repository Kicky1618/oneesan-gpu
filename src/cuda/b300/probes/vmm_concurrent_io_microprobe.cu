#include "../b300_vmm_contiguous_storage.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Count=std::uint32_t;
using Code=unsigned long long;

static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(1);}}
__host__ __device__ static inline Count value_for(Code g,Count salt){return Count((g*2654435761ULL+Code(salt)*2246822519ULL+17ULL)&0xffffffffu);}
__global__ void fill_range(Count*base,Code begin,Code end,Count salt){Code i=begin+Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<end;i+=stride)base[i]=value_for(i,salt);}
__global__ void read_copy(const Count*auth,Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)out[i]=auth[i];}
__global__ void write_copy(const Count*in,Count*auth,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)auth[i]=in[i];}
__global__ void verify_copy(const Count*out,Code n,Count salt,unsigned long long*err){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;unsigned long long e=0;for(;i<n;i+=stride)e+=out[i]!=value_for(i,salt);if(e)atomicAdd(err,e);}
static double median(std::vector<double>x){std::sort(x.begin(),x.end());size_t n=x.size();return n&1?x[n/2]:0.5*(x[n/2-1]+x[n/2]);}

static void fill_storage(b300_vmm::ContiguousStorage&s,Count*base,Code logical,Count salt){
    for(int d=0;d<s.ngpu;++d){Code lo=Code(s.offsets[size_t(d)]/sizeof(Count));Code hi=std::min<Code>(logical,Code(s.offsets[size_t(d)+1]/sizeof(Count)));if(lo>=hi)continue;ck(cudaSetDevice(d),"fill set device");fill_range<<<256,256>>>(base,lo,hi,salt);ck(cudaGetLastError(),"fill launch");}
    for(int d=0;d<s.ngpu;++d){ck(cudaSetDevice(d),"fill sync set");ck(cudaDeviceSynchronize(),"fill sync");}
}

int main(int argc,char**argv){
    int ng=argc>1?std::atoi(argv[1]):8,src=argc>2?std::atoi(argv[2]):0,repeats=argc>3?std::atoi(argv[3]):9;
    Code main_elems=argc>4?std::strtoull(argv[4],nullptr,10):(16ull<<20);Code block_elems=argc>5?std::strtoull(argv[5],nullptr,10):Code(double(main_elems)*0.35);
    if(ng<1||ng>8||src<0||src>=ng||repeats<1||main_elems<Code(ng)*1024||block_elems<Code(ng)*1024)return 2;
    int visible=0;ck(cudaGetDeviceCount(&visible),"device count");if(visible<ng)return 2;
    b300_vmm::ContiguousStorage main_store,block_store;main_store.create(main_elems,ng,0,"concurrent-main");block_store.create(block_elems,ng,int(main_store.mapped_units%size_t(ng)),"concurrent-block");
    if(main_store.granularity!=block_store.granularity)return 3;main_store.zero_local_segments();block_store.zero_local_segments();Count*main_base=main_store.base_as<Count>();Count*block_base=block_store.base_as<Count>();
    fill_storage(main_store,main_base,main_elems,11);fill_storage(block_store,block_base,block_elems,29);
    ck(cudaSetDevice(src),"set source");Count *outm=nullptr,*outb=nullptr;ck(cudaMalloc(&outm,size_t(main_elems)*sizeof(Count)),"out main");ck(cudaMalloc(&outb,size_t(block_elems)*sizeof(Count)),"out block");
    cudaStream_t sm{},sb{};ck(cudaStreamCreateWithFlags(&sm,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sb,cudaStreamNonBlocking),"stream block");
    int bm=int(std::min<Code>(65535,(main_elems+255)/256)),bb=int(std::min<Code>(65535,(block_elems+255)/256));
    auto read_serial=[&](){auto t=std::chrono::steady_clock::now();read_copy<<<bm,256,0,sm>>>(main_base,outm,main_elems);ck(cudaGetLastError(),"read serial main");ck(cudaStreamSynchronize(sm),"read serial main sync");read_copy<<<bb,256,0,sb>>>(block_base,outb,block_elems);ck(cudaGetLastError(),"read serial block");ck(cudaStreamSynchronize(sb),"read serial block sync");return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();};
    auto read_concurrent=[&](){auto t=std::chrono::steady_clock::now();read_copy<<<bm,256,0,sm>>>(main_base,outm,main_elems);read_copy<<<bb,256,0,sb>>>(block_base,outb,block_elems);ck(cudaGetLastError(),"read concurrent launch");ck(cudaStreamSynchronize(sm),"read concurrent main sync");ck(cudaStreamSynchronize(sb),"read concurrent block sync");return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();};
    auto write_serial=[&](){auto t=std::chrono::steady_clock::now();write_copy<<<bm,256,0,sm>>>(outm,main_base,main_elems);ck(cudaGetLastError(),"write serial main");ck(cudaStreamSynchronize(sm),"write serial main sync");write_copy<<<bb,256,0,sb>>>(outb,block_base,block_elems);ck(cudaGetLastError(),"write serial block");ck(cudaStreamSynchronize(sb),"write serial block sync");return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();};
    auto write_concurrent=[&](){auto t=std::chrono::steady_clock::now();write_copy<<<bm,256,0,sm>>>(outm,main_base,main_elems);write_copy<<<bb,256,0,sb>>>(outb,block_base,block_elems);ck(cudaGetLastError(),"write concurrent launch");ck(cudaStreamSynchronize(sm),"write concurrent main sync");ck(cudaStreamSynchronize(sb),"write concurrent block sync");return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();};

    read_serial();read_concurrent();write_serial();write_concurrent();
    unsigned long long*derr=nullptr;ck(cudaMalloc(&derr,sizeof(*derr)),"err alloc");ck(cudaMemset(derr,0,sizeof(*derr)),"err zero");verify_copy<<<bm,256>>>(outm,main_elems,11,derr);verify_copy<<<bb,256>>>(outb,block_elems,29,derr);verify_copy<<<bm,256>>>(main_base,main_elems,11,derr);verify_copy<<<bb,256>>>(block_base,block_elems,29,derr);ck(cudaGetLastError(),"verify launch");ck(cudaDeviceSynchronize(),"verify sync");unsigned long long errors=0;ck(cudaMemcpy(&errors,derr,sizeof(errors),cudaMemcpyDeviceToHost),"err copy");if(errors){std::fprintf(stderr,"verification errors=%llu\n",errors);return 4;}

    std::vector<double>rs,rc,ws,wc;rs.reserve(repeats);rc.reserve(repeats);ws.reserve(repeats);wc.reserve(repeats);
    for(int r=0;r<repeats;++r){
        switch(r&3){
            case 0:rs.push_back(read_serial());rc.push_back(read_concurrent());ws.push_back(write_serial());wc.push_back(write_concurrent());break;
            case 1:rc.push_back(read_concurrent());ws.push_back(write_serial());wc.push_back(write_concurrent());rs.push_back(read_serial());break;
            case 2:ws.push_back(write_serial());wc.push_back(write_concurrent());rs.push_back(read_serial());rc.push_back(read_concurrent());break;
            default:wc.push_back(write_concurrent());rs.push_back(read_serial());rc.push_back(read_concurrent());ws.push_back(write_serial());break;
        }
    }
    double rsm=median(rs),rcm=median(rc),wsm=median(ws),wcm=median(wc);double combined=(rsm+wsm)/(rcm+wcm);
    ck(cudaMemset(derr,0,sizeof(*derr)),"final err zero");verify_copy<<<bm,256>>>(main_base,main_elems,11,derr);verify_copy<<<bb,256>>>(block_base,block_elems,29,derr);ck(cudaDeviceSynchronize(),"final verify sync");ck(cudaMemcpy(&errors,derr,sizeof(errors),cudaMemcpyDeviceToHost),"final err copy");if(errors){std::fprintf(stderr,"final write verification errors=%llu\n",errors);return 5;}
    std::printf("b300-vmm-concurrent-io-microprobe OK gpus=%d src_gpu=%d repeats=%d main_elems=%llu block_elems=%llu block_main_ratio=%.6f main_mib=%.3f block_mib=%.3f read_serial_ms=%.6f read_concurrent_ms=%.6f read_speedup=%.6f write_serial_ms=%.6f write_concurrent_ms=%.6f write_speedup=%.6f combined_speedup=%.6f streams_serial=2 streams_concurrent=2 serial_dependency=1 concurrent_overlap=1 read_exact=OK write_exact=OK exact=OK\n",ng,src,repeats,(unsigned long long)main_elems,(unsigned long long)block_elems,double(block_elems)/double(main_elems),double(main_elems*sizeof(Count))/(1<<20),double(block_elems*sizeof(Count))/(1<<20),rsm,rcm,rsm/rcm,wsm,wcm,wsm/wcm,combined);
    cudaFree(derr);cudaStreamDestroy(sm);cudaStreamDestroy(sb);cudaFree(outm);cudaFree(outb);main_store.destroy();block_store.destroy();return 0;
}
