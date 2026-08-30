#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: gen-b300-adaptive-storage.py INPUT.cu OUTPUT.cu")

src_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
s = src_path.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"adaptive-storage: expected one {label} anchor, found {count}")
    s = s.replace(old, new, 1)


replace_once(
    "#include <cuda_runtime.h>\n",
    "#include <cuda_runtime.h>\n#include <cuda.h>\n",
    "CUDA include",
)
replace_once(
    "static constexpr int MAXW=28, MAXGPU=8;",
    "static constexpr int MAXW=28, MAXGPU=16;",
    "MAXGPU",
)

helper = r'''

struct AdaptiveStateStorage {
    Count* main_ptr=nullptr;
    Count* block_ptr=nullptr;
    bool managed=false;

    struct Mapping {
        CUdeviceptr addr=0;
        size_t bytes=0;
        CUmemGenericAllocationHandle handle=0;
        int device=0;
    };

    static const char* cu_error_name(CUresult e) {
        const char* name=nullptr;
        return cuGetErrorName(e,&name)==CUDA_SUCCESS && name ? name : "CUDA_ERROR_UNKNOWN";
    }

    static size_t align_up(size_t x,size_t a) {
        return ((x+a-1)/a)*a;
    }

    bool try_vmm(Code mainN,Code blockN,int ng,int reserve_mib) {
        CUresult ce=cuInit(0);
        if(ce!=CUDA_SUCCESS){std::cerr<<"adaptive VMM: cuInit failed: "<<cu_error_name(ce)<<"\n";return false;}

        size_t gran=0;
        for(int d=0;d<ng;++d){
            CUmemAllocationProp prop{};
            prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id=d;
            size_t g=0;
            ce=cuMemGetAllocationGranularity(&g,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            if(ce!=CUDA_SUCCESS){std::cerr<<"adaptive VMM: granularity query failed on gpu "<<d<<": "<<cu_error_name(ce)<<"\n";return false;}
            gran=std::max(gran,g);
        }
        if(!gran)return false;

        const size_t main_bytes=size_t(mainN)*sizeof(Count);
        const size_t block_bytes=size_t(blockN)*sizeof(Count);
        const size_t main_span=align_up(main_bytes,gran);
        const size_t block_span=align_up(block_bytes,gran);
        const size_t total_span=main_span+block_span;
        const size_t total_units=total_span/gran;
        const size_t reserve=size_t(std::max(0,reserve_mib))<<20;

        std::vector<size_t> cap_units(ng),assign_units(ng);
        size_t cap_sum=0;
        for(int d=0;d<ng;++d){
            ck(cudaSetDevice(d),"adaptive VMM set device");
            size_t free_bytes=0,total_bytes=0;
            ck(cudaMemGetInfo(&free_bytes,&total_bytes),"adaptive VMM meminfo");
            size_t usable=free_bytes>reserve?free_bytes-reserve:0;
            cap_units[d]=usable/gran;
            cap_sum+=cap_units[d];
        }
        if(cap_sum<total_units){
            std::cerr<<"adaptive VMM: aggregate post-LUT VRAM is short by "
                     <<double(total_units-cap_sum)*gran/(1<<20)<<" MiB; falling back to Managed Memory\n";
            return false;
        }

        std::vector<long double> exact(ng);
        size_t assigned=0;
        for(int d=0;d<ng;++d){
            exact[d]=static_cast<long double>(total_units)*cap_units[d]/cap_sum;
            assign_units[d]=std::min(cap_units[d],size_t(exact[d]));
            assigned+=assign_units[d];
        }
        while(assigned<total_units){
            int best=-1;
            long double best_deficit=-1e300L;
            for(int d=0;d<ng;++d){
                if(assign_units[d]>=cap_units[d])continue;
                long double deficit=exact[d]-assign_units[d];
                if(best<0||deficit>best_deficit){best=d;best_deficit=deficit;}
            }
            if(best<0)return false;
            ++assign_units[best];
            ++assigned;
        }

        CUdeviceptr base=0;
        ce=cuMemAddressReserve(&base,total_span,gran,0,0);
        if(ce!=CUDA_SUCCESS){std::cerr<<"adaptive VMM: address reserve failed: "<<cu_error_name(ce)<<"\n";return false;}

        std::vector<Mapping> mappings;
        auto cleanup=[&](){
            for(auto it=mappings.rbegin();it!=mappings.rend();++it){
                cuMemUnmap(it->addr,it->bytes);
                cuMemRelease(it->handle);
            }
            cuMemAddressFree(base,total_span);
        };

        const size_t max_chunk_bytes=size_t(2048)<<20;
        const size_t max_chunk_units=std::max<size_t>(1,max_chunk_bytes/gran);
        size_t offset=0;
        for(int d=0;d<ng;++d){
            size_t left=assign_units[d];
            while(left){
                size_t units=std::min(left,max_chunk_units);
                size_t bytes=units*gran;
                CUmemAllocationProp prop{};
                prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
                prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
                prop.location.id=d;
                CUmemGenericAllocationHandle handle=0;
                ce=cuMemCreate(&handle,bytes,&prop,0);
                if(ce!=CUDA_SUCCESS){
                    std::cerr<<"adaptive VMM: physical allocation failed on gpu "<<d<<": "<<cu_error_name(ce)<<"\n";
                    cleanup();
                    return false;
                }
                CUdeviceptr addr=base+offset;
                ce=cuMemMap(addr,bytes,0,handle,0);
                if(ce!=CUDA_SUCCESS){
                    std::cerr<<"adaptive VMM: map failed on gpu "<<d<<": "<<cu_error_name(ce)<<"\n";
                    cuMemRelease(handle);
                    cleanup();
                    return false;
                }
                mappings.push_back({addr,bytes,handle,d});
                offset+=bytes;
                left-=units;
            }
        }
        if(offset!=total_span){cleanup();return false;}

        std::vector<CUmemAccessDesc> access(ng);
        for(int d=0;d<ng;++d){
            access[d].location.type=CU_MEM_LOCATION_TYPE_DEVICE;
            access[d].location.id=d;
            access[d].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        ce=cuMemSetAccess(base,total_span,access.data(),access.size());
        if(ce!=CUDA_SUCCESS){
            std::cerr<<"adaptive VMM: peer access mapping failed: "<<cu_error_name(ce)<<"\n";
            cleanup();
            return false;
        }

        for(auto const& m:mappings){
            ck(cudaSetDevice(m.device),"adaptive VMM zero set device");
            ck(cudaMemset(reinterpret_cast<void*>(m.addr),0,m.bytes),"adaptive VMM zero");
        }
        for(auto const& m:mappings)cuMemRelease(m.handle);

        main_ptr=reinterpret_cast<Count*>(base);
        block_ptr=reinterpret_cast<Count*>(base+main_span);
        std::cerr<<"adaptive storage=device-vmm mapped_mib="<<(total_span>>20)<<" granularity_kib="<<(gran>>10)<<" weights_mib=";
        for(int d=0;d<ng;++d){if(d)std::cerr<<',';std::cerr<<double(assign_units[d]*gran)/(1<<20);}
        std::cerr<<"\n";
        return true;
    }

    void allocate_managed(Code mainN,Code blockN,int ng) {
        const size_t main_bytes=size_t(mainN)*sizeof(Count);
        const size_t block_bytes=size_t(blockN)*sizeof(Count);
        const size_t total_bytes=main_bytes+block_bytes;
        Count* base=nullptr;
        ck(cudaMallocManaged(reinterpret_cast<void**>(&base),total_bytes,cudaMemAttachGlobal),"adaptive managed state");
        auto advise=[&](cudaError_t e,const char* what){
            if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<" (continuing)\n";cudaGetLastError();}
        };
        advise(cudaMemAdvise(base,total_bytes,cudaMemAdviseSetPreferredLocation,cudaCpuDeviceId),"managed preferred CPU");
        for(int d=0;d<ng;++d)advise(cudaMemAdvise(base,total_bytes,cudaMemAdviseSetAccessedBy,d),"managed accessed-by GPU");
        std::memset(base,0,total_bytes);
        main_ptr=base;
        block_ptr=base+mainN;
        managed=true;
        std::cerr<<"adaptive storage=managed-host state_mib="<<double(total_bytes)/(1<<20)<<"\n";
    }

    void allocate(Code mainN,Code blockN,int ng,int reserve_mib,bool prefer_managed) {
        if(!prefer_managed && try_vmm(mainN,blockN,ng,reserve_mib))return;
        allocate_managed(mainN,blockN,ng);
    }
};
'''

replace_once("\nint main(int argc,char**argv){", helper + "\n\nint main(int argc,char**argv){", "main")

old_peer = r'''    if(ng>1&&peers!=ng*(ng-1)){std::cerr<<"HBM mode requires full P2P: "<<peers<<"/"<<ng*(ng-1)<<"\n";return 3;}
'''
new_peer = r'''    bool full_p2p=(ng<=1||peers==ng*(ng-1));
    const char* storage_env=std::getenv("ONEESAN_STORAGE");
    std::string storage=storage_env?storage_env:"device-vmm";
    if(storage!="device-vmm"&&storage!="managed-host"){std::cerr<<"ONEESAN_STORAGE must be device-vmm or managed-host; got "<<storage<<"\n";return 3;}
    bool prefer_managed=(storage=="managed-host");
    if(!full_p2p&&!prefer_managed){std::cerr<<"adaptive VMM needs full P2P; peer matrix is "<<peers<<"/"<<ng*(ng-1)<<", falling back to managed-host\n";prefer_managed=true;}
'''
replace_once(old_peer, new_peer, "P2P gate")

old_alloc = r'''    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));cudaSetDevice(d);if(ml[d]){ck(cudaMalloc(&mp[d],size_t(ml[d])*sizeof(Count)),"auth main");ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)),"zero main");}if(bl[d]){ck(cudaMalloc(&bp[d],size_t(bl[d])*sizeof(Count)),"auth block");ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)),"zero block");}}
'''
new_alloc = r'''    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    size_t adaptive_min_total=~size_t(0);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"adaptive prealloc set device");size_t f=0,t=0;ck(cudaMemGetInfo(&f,&t),"adaptive prealloc meminfo");adaptive_min_total=std::min(adaptive_min_total,t);}
    int adaptive_reserve_mib=std::min(8192,std::max(512,int((adaptive_min_total>>20)/32)));if(const char*e=std::getenv("GRIDFP_VRAM_RESERVE_MIB")){int v=std::atoi(e);if(v>=0)adaptive_reserve_mib=v;}
    AdaptiveStateStorage auth;auth.allocate(mainN,blockN,ng,adaptive_reserve_mib,prefer_managed);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));mp[d]=ml[d]?auth.main_ptr+Code(d)*mc:auth.main_ptr;bp[d]=bl[d]?auth.block_ptr+Code(d)*bc:auth.block_ptr;}
'''
replace_once(old_alloc, new_alloc, "authoritative allocation")

replace_once(
    'if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\\n";return 2;}',
    'if(ng<1||ng>MAXGPU){std::cerr<<"need 1..16 GPUs\\n";return 2;}',
    "GPU count diagnostic",
)

out_path.write_text(s)
print(f"generated {out_path} from {src_path}: adaptive_storage=1 max_gpu=16 vmm=weighted managed_fallback=1")
