#include "../b300_vmm_contiguous_storage.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using Count=std::uint32_t;
using Code=unsigned long long;
static constexpr int MAXGPU=8;

__constant__ Count* D_HELPER_A_PTR[MAXGPU];
__constant__ Count* D_HELPER_B_PTR[MAXGPU];
__constant__ Code D_HELPER_A_CHUNK,D_HELPER_B_CHUNK;
__constant__ int D_HELPER_NGPU;

__device__ __forceinline__ Count expected(Code g){return Count((g*2654435761ULL+0x9e3779b9ULL)&0xffffffffu);}

__global__ void fill_range(Count* base,Code begin,Code end){Code i=begin+Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<end;i+=stride)base[i]=expected(i);}
__global__ void verify_all(const Count* base,Code n,unsigned long long* errors){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;unsigned long long local=0;for(;i<n;i+=stride)local+=base[i]!=expected(i);if(local)atomicAdd(errors,local);}

__device__ __forceinline__ Count logical_view_load_a(Code g){int o=int(g/D_HELPER_A_CHUNK);if(o>=D_HELPER_NGPU)o=D_HELPER_NGPU-1;return D_HELPER_A_PTR[o][g-Code(o)*D_HELPER_A_CHUNK];}
__device__ __forceinline__ Count logical_view_load_b(Code g){int o=int(g/D_HELPER_B_CHUNK);if(o>=D_HELPER_NGPU)o=D_HELPER_NGPU-1;return D_HELPER_B_PTR[o][g-Code(o)*D_HELPER_B_CHUNK];}
__global__ void verify_logical_views(Code na,Code nb,unsigned long long* errors){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;unsigned long long local=0;for(Code g=i;g<na;g+=stride)local+=logical_view_load_a(g)!=expected(g);for(Code g=i;g<nb;g+=stride)local+=logical_view_load_b(g)!=expected(g);if(local)atomicAdd(errors,local);}

static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(1);}}

static int physical_owner(const b300_vmm::ContiguousStorage& s,Code g){const size_t byte=size_t(g)*sizeof(Count);for(int d=0;d<s.ngpu;++d)if(byte>=s.offsets[size_t(d)]&&byte<s.offsets[size_t(d)+1])return d;return -1;}

static Code find_logical_physical_mismatch(const b300_vmm::ContiguousStorage& s,Code logical,Code chunk){
    for(int d=0;d<s.ngpu;++d){
        const Code lo=Code(d)*chunk;if(lo>=logical)break;const Code hi=std::min<Code>(logical,(Code(d)+1)*chunk);
        const Code cand[4]={lo,std::min<Code>(hi-1,lo+1),hi-1,lo+(hi-lo)/2};
        for(Code g:cand)if(g<logical&&physical_owner(s,g)!=d)return g;
    }
    for(int p=1;p<s.ngpu;++p){
        const Code g=Code(s.offsets[size_t(p)]/sizeof(Count));
        if(g<logical){const int lo=int(g/chunk);if(physical_owner(s,g)!=lo)return g;if(g&&physical_owner(s,g-1)!=int((g-1)/chunk))return g-1;}
    }
    return ~Code(0);
}

static Code find_remote_from_device0(const b300_vmm::ContiguousStorage& s,Code logical){
    for(int p=1;p<s.ngpu;++p){
        const Code begin=Code(s.offsets[size_t(p)]/sizeof(Count));
        const Code end=Code(s.offsets[size_t(p+1)]/sizeof(Count));
        if(begin>=logical)break;
        const Code hi=std::min<Code>(logical,end);
        if(begin<hi&&physical_owner(s,begin)==p)return begin;
    }
    return ~Code(0);
}

static void verify_device0_direct_memcpy(const char* tag,b300_vmm::ContiguousStorage& s,Count* base,Code logical){
    if(s.ngpu<2)return;
    const Code g=find_remote_from_device0(s,logical);
    if(g==~Code(0)){std::fprintf(stderr,"%s could not find a remote physical page for device0 direct memcpy\n",tag);std::exit(8);}
    const int phys=physical_owner(s,g);
    if(phys<=0){std::fprintf(stderr,"%s remote selection failed g=%llu physical=%d\n",tag,(unsigned long long)g,phys);std::exit(8);}
    ck(cudaSetDevice(0),"device0 direct memcpy set device");
    Count* p=base+g;
    const Count marker=0xbb67ae85u^Count(g);Count got=0;
    ck(cudaMemcpy(p,&marker,sizeof(marker),cudaMemcpyHostToDevice),"device0 direct H2D to remote physical VMM page");
    ck(cudaMemcpy(&got,p,sizeof(got),cudaMemcpyDeviceToHost),"device0 direct D2H from remote physical VMM page");
    if(got!=marker){std::fprintf(stderr,"%s device0 direct memcpy mismatch got=%u expected=%u\n",tag,got,marker);std::exit(8);}
    const Count restore=Count((g*2654435761ULL+0x9e3779b9ULL)&0xffffffffu);
    ck(cudaMemcpy(p,&restore,sizeof(restore),cudaMemcpyHostToDevice),"device0 direct VMM restore");
    Count check=0;ck(cudaMemcpy(&check,p,sizeof(check),cudaMemcpyDeviceToHost),"device0 direct VMM restore verify");
    if(check!=restore){std::fprintf(stderr,"%s device0 direct restore mismatch\n",tag);std::exit(8);}
    std::fprintf(stderr,"%s device0-direct memcpy: g=%llu physical_owner=%d remote_physical=1 H2D_D2H=OK\n",tag,(unsigned long long)g,phys);
}

static void verify_runtime_memcpy_view(const char* tag,b300_vmm::ContiguousStorage& s,Count* base,Count* const* ptrs,Code logical,Code chunk){
    const Code g=find_logical_physical_mismatch(s,logical,chunk);
    if(g==~Code(0)){std::fprintf(stderr,"%s could not find logical/physical shard mismatch for memcpy preflight\n",tag);std::exit(7);}
    int owner=int(g/chunk);if(owner>=s.ngpu)owner=s.ngpu-1;const Code local=g-Code(owner)*chunk;const int phys=physical_owner(s,g);
    if(phys<0||phys==owner){std::fprintf(stderr,"%s mismatch selection failed g=%llu logical=%d physical=%d\n",tag,(unsigned long long)g,owner,phys);std::exit(7);}
    Count* view=ptrs[owner]+local;
    ck(cudaSetDevice(owner),"memcpy logical-owner set device");
    const Count marker=0x6a09e667u^Count(g);Count got=0;
    ck(cudaMemcpy(view,&marker,sizeof(marker),cudaMemcpyHostToDevice),"memcpy host to remote VMM logical view");
    ck(cudaMemcpy(&got,view,sizeof(got),cudaMemcpyDeviceToHost),"memcpy remote VMM logical view to host");
    if(got!=marker){std::fprintf(stderr,"%s runtime memcpy mismatch got=%u expected=%u\n",tag,got,marker);std::exit(7);}
    const Count restore=Count((g*2654435761ULL+0x9e3779b9ULL)&0xffffffffu);
    ck(cudaMemcpy(view,&restore,sizeof(restore),cudaMemcpyHostToDevice),"restore remote VMM logical view");
    Count direct=0;ck(cudaMemcpy(&direct,base+g,sizeof(direct),cudaMemcpyDeviceToHost),"verify restored direct VMM value");
    if(direct!=restore){std::fprintf(stderr,"%s restore mismatch\n",tag);std::exit(7);}
    std::fprintf(stderr,"%s logical-view memcpy: g=%llu logical_owner=%d physical_owner=%d remote_physical=1 OK\n",tag,(unsigned long long)g,owner,phys);
}

int main(int argc,char**argv){
    const int ng=argc>1?std::atoi(argv[1]):8;
    const Code elems=argc>2?std::strtoull(argv[2],nullptr,10):8388731ULL;
    if(ng<1||ng>8||elems<Code(ng)*1024)return 2;
    int visible=0;ck(cudaGetDeviceCount(&visible),"device count");if(visible<ng){std::fprintf(stderr,"visible=%d requested=%d\n",visible,ng);return 2;}

    b300_vmm::ContiguousStorage a,b;
    a.create(elems,ng,0,"helper-a");
    const Code elems_b=std::max<Code>(Code(ng)*1024,elems/3+17);
    b.create(elems_b,ng,int(a.mapped_units%size_t(ng)),"helper-b");
    if(a.granularity!=b.granularity){std::fprintf(stderr,"helper granularity mismatch\n");return 3;}
    a.zero_local_segments();b.zero_local_segments();
    Count* abase=a.base_as<Count>();Count* bbase=b.base_as<Count>();

    auto fill=[&](b300_vmm::ContiguousStorage& s,Count* base,Code logical){
        for(int d=0;d<ng;++d){
            const size_t lo_bytes=s.offsets[size_t(d)],hi_bytes=s.offsets[size_t(d)+1];
            Code lo=Code(lo_bytes/sizeof(Count));Code hi=std::min<Code>(logical,Code(hi_bytes/sizeof(Count)));
            if(lo>=hi)continue;ck(cudaSetDevice(d),"fill set device");fill_range<<<256,256>>>(base,lo,hi);ck(cudaGetLastError(),"fill launch");
        }
        for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"fill sync set");ck(cudaDeviceSynchronize(),"fill sync");}
    };
    fill(a,abase,elems);fill(b,bbase,elems_b);

    const Code mc=(elems+Code(ng)-1)/Code(ng),bc=(elems_b+Code(ng)-1)/Code(ng);
    Count* ap[MAXGPU]{};Count* bp[MAXGPU]{};
    for(int d=0;d<ng;++d){ap[d]=abase+Code(d)*mc;bp[d]=bbase+Code(d)*bc;}
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"logical symbols set device");
        ck(cudaMemcpyToSymbol(D_HELPER_A_PTR,ap,sizeof(ap)),"copy A logical pointers");
        ck(cudaMemcpyToSymbol(D_HELPER_B_PTR,bp,sizeof(bp)),"copy B logical pointers");
        ck(cudaMemcpyToSymbol(D_HELPER_A_CHUNK,&mc,sizeof(mc)),"copy A logical chunk");
        ck(cudaMemcpyToSymbol(D_HELPER_B_CHUNK,&bc,sizeof(bc)),"copy B logical chunk");
        ck(cudaMemcpyToSymbol(D_HELPER_NGPU,&ng,sizeof(ng)),"copy logical ngpu");
    }

    verify_device0_direct_memcpy("helper-a",a,abase,elems);
    verify_device0_direct_memcpy("helper-b",b,bbase,elems_b);
    verify_runtime_memcpy_view("helper-a",a,abase,ap,elems,mc);
    verify_runtime_memcpy_view("helper-b",b,bbase,bp,elems_b,bc);

    unsigned long long total_errors=0;
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"verify set device");unsigned long long* err=nullptr;ck(cudaMalloc(&err,sizeof(*err)),"err alloc");ck(cudaMemset(err,0,sizeof(*err)),"err zero");
        verify_all<<<256,256>>>(abase,elems,err);verify_all<<<256,256>>>(bbase,elems_b,err);verify_logical_views<<<256,256>>>(elems,elems_b,err);ck(cudaGetLastError(),"verify launch");ck(cudaDeviceSynchronize(),"verify sync");
        unsigned long long e=0;ck(cudaMemcpy(&e,err,sizeof(e),cudaMemcpyDeviceToHost),"err copy");cudaFree(err);total_errors+=e;
    }

    size_t min_combined=~size_t(0),max_combined=0;
    for(int d=0;d<ng;++d){size_t x=a.segment_bytes[size_t(d)]+b.segment_bytes[size_t(d)];min_combined=std::min(min_combined,x);max_combined=std::max(max_combined,x);}
    if(max_combined-min_combined>a.granularity){std::fprintf(stderr,"combined physical imbalance too large\n");return 5;}
    if(total_errors){std::fprintf(stderr,"VMM helper verification errors=%llu\n",total_errors);return 6;}

    std::printf("gridfp-b300-vmm-storage-helper-microprobe OK gpus=%d elems_a=%llu elems_b=%llu granularity=%zu padding_a=%zu padding_b=%zu combined_imbalance=%zu direct_base_index=1 device0_direct_memcpy_remote_physical=OK device0_direct_H2D_D2H=OK logical_shard_views=1 logical_shard_gpu_access=OK runtime_memcpy_logical_view=OK logical_physical_mismatch_tested=1 physical_boundary_independent=1 all_gpu_read=OK exact=OK\n",ng,(unsigned long long)elems,(unsigned long long)elems_b,a.granularity,a.mapped_bytes-a.logical_bytes,b.mapped_bytes-b.logical_bytes,max_combined-min_combined);
    a.destroy();b.destroy();return 0;
}
