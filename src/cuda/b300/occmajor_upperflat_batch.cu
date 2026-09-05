#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/atomic>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <functional>
#include <vector>
#include "../../common/gridfp_transition.hpp"
#include "row3_automaton_generated.hpp"
#include "row4_automaton_mod4294967291.hpp"
#include "row5_automaton_mod4294967291.hpp"
#include "row5_automaton_rational.hpp"
#include "row6_automaton_mod1000000007.hpp"
#include "row6_automaton_crt20_generated.hpp"

using Count = uint32_t;
using MateID = unsigned long long;
using Code = unsigned long long;
static constexpr int MAXW=28, MAXGPU=8;
#ifndef IO_TILE_ELEMS
#define IO_TILE_ELEMS 65536ULL
#endif
#ifndef TARGET_W
#define TARGET_W 28
#endif
#ifndef LOW_LUT_K
#define LOW_LUT_K 0
#endif
#ifndef HIGH_LUT_K
#define HIGH_LUT_K 0
#endif

enum MateValue:uint8_t{N=0,R=1,L=2,X=3};
enum MateValuePair:uint8_t{NN=0x0,NR=0x1,NL=0x2,NX=0x3,RN=0x4,RR=0x5,RL=0x6,RX=0x7,LN=0x8,LR=0x9,LL=0xa,LX=0xb,XN=0xc,XR=0xd,XL=0xe,XX=0xf};

static Code H_DP[MAXW+1][MAXW+2];
__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];
__constant__ Code D_BOUND_DP[MAXW+1][MAXW+2];
__constant__ Code D_BOUND_OLD_DP[MAXW+1][MAXW+2];
__constant__ int D_BOUND_CAP,D_BOUND_OLD_CAP;
__constant__ int D_NGPU;
__constant__ Count D_MOD;
__constant__ Count* D_R3_PREF;
__constant__ Count* D_R3_SUFF;
__constant__ Count D_R3_INV_SCALE;
__constant__ Count* D_R4_PREF;
__constant__ Count* D_R4_SUFF;
__constant__ Count* D_R5_PREF;
__constant__ Count* D_R5_SUFF;
__constant__ uint32_t* D_R5_PREF_IDX;
__constant__ uint32_t* D_R5_SUFF_IDX;
__constant__ Count* D_R6_PREF;
__constant__ Count* D_R6_SUFF;
__constant__ uint32_t* D_R6_PREF_IDX;
__constant__ uint32_t* D_R6_SUFF_IDX;
__constant__ uint32_t D_R6_GPU_MAT_OFF[3][7];
__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;
struct LowEntry{uint32_t mate,full_rank;};
__constant__ LowEntry* D_LOW_ENTRIES;
__constant__ uint32_t* D_LOW_OFFSETS;
__constant__ uint32_t* D_LOW_LOCAL_RANK;
struct HighEntry{uint32_t code,base;};
__constant__ HighEntry* D_HIGH_ENTRIES_MAIN;
__constant__ HighEntry* D_HIGH_ENTRIES_BLOCK;
__constant__ uint32_t* D_HIGH_OFFSETS_MAIN;
__constant__ uint32_t* D_HIGH_OFFSETS_BLOCK;

__constant__ uint32_t* D_F_LOW_ALL_CODES;
__constant__ uint32_t* D_F_LOW_MASK_CODES;
__constant__ uint32_t* D_F_LOW_MASK_OFF;
__constant__ uint32_t* D_F_LOW_OCC_BASE;
__constant__ uint32_t* D_F_LOW_DENSE_PACKED_RANK;
__constant__ uint32_t* D_F_HIGH_ALL_CODES;
__constant__ uint32_t* D_F_HIGH_MASK_CODES;
__constant__ uint32_t* D_F_HIGH_MASK_OFF;
__constant__ uint32_t* D_F_HIGH_OCC_BASE;
__constant__ uint32_t* D_F_HIGH_PACKED_RANK;
__constant__ Code* D_F_HIGH_MAIN_BASE;
__constant__ Code* D_F_HIGH_BLOCK_BASE;
__constant__ uint32_t D_F_LOW_ALL_OFF[MAXW+2];
__constant__ uint32_t D_F_HIGH_ALL_OFF[MAXW+2];
struct FBlock { Code off,end; uint32_t stride; uint8_t he,hs,c,pad; };
__constant__ FBlock D_F_FULL_MAIN_BLOCKS[64];
__constant__ FBlock D_F_FULL_BLOCK_BLOCKS[32];
struct GroupRuntimeCfg {
    Code main_dp[MAXW+1][MAXW+2];
    Code block_dp[MAXW+1][MAXW+2];
    FBlock f_main[64];
    FBlock f_block[32];
    uint32_t main_fixed,main_occ,block_fixed,block_occ,f_mask;
    int main_w,block_w,f_main_n,f_block_n,f_fix_low;
};
__constant__ GroupRuntimeCfg D_GRP;
#define D_MAIN_DP D_GRP.main_dp
#define D_BLOCK_DP D_GRP.block_dp
#define D_MAIN_FIXED D_GRP.main_fixed
#define D_MAIN_OCC D_GRP.main_occ
#define D_BLOCK_FIXED D_GRP.block_fixed
#define D_BLOCK_OCC D_GRP.block_occ
#define D_MAIN_W D_GRP.main_w
#define D_BLOCK_W D_GRP.block_w
#define D_F_MAIN_BLOCKS D_GRP.f_main
#define D_F_BLOCK_BLOCKS D_GRP.f_block
#define D_F_MAIN_NBLOCKS D_GRP.f_main_n
#define D_F_BLOCK_NBLOCKS D_GRP.f_block_n
#define D_F_MASK D_GRP.f_mask
#define D_F_FIX_LOW D_GRP.f_fix_low
struct SparseSel { uint32_t off,count; };
__constant__ uint32_t* D_SP_OWNER_LR;
__constant__ uint32_t* D_SP_CLOSURE_LR;
__constant__ unsigned long long* D_PR_OWNER_REC;
__constant__ unsigned long long* D_PR_OWNER_BLOCK_REC;
__constant__ unsigned long long* D_PR_CLOSURE_REC;
__constant__ unsigned long long* D_PR_CLOSURE_INV_REC;
__constant__ uint32_t* D_PR_CLOSURE_INV_SRC;
__constant__ unsigned long long* D_PR_HIGHRR_REC;
__constant__ unsigned long long* D_HP_OWNER_REC;
__constant__ unsigned long long* D_HP_OWNER_BLOCK_REC;
__constant__ unsigned long long* D_HP_CLOSURE_REC;
__constant__ unsigned long long* D_HP_CLOSURE_INV_REC;
__constant__ uint32_t* D_HP_CLOSURE_INV_SRC;
__constant__ unsigned long long* D_HP_CROSSLL_REC;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
__host__ __device__ static inline MateID mshrink(MateID m,int k){MateID mask=(1ULL<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}


static void ckcu(CUresult e,const char*w){if(e!=CUDA_SUCCESS){const char*n=nullptr,*d=nullptr;cuGetErrorName(e,&n);cuGetErrorString(e,&d);std::cerr<<w<<": "<<(n?n:"?")<<" "<<(d?d:"?")<<"\n";std::exit(1);}}
static size_t vmm_granularity_for_device(int dev){ckcu(cuInit(0),"cuInit");CUdevice cdev;ckcu(cuDeviceGet(&cdev,dev),"cuDeviceGet");int ok=0;ckcu(cuDeviceGetAttribute(&ok,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,cdev),"vmm attr");if(!ok)throw std::runtime_error("CUDA VMM unsupported");CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=cdev;size_t gran=0;ckcu(cuMemGetAllocationGranularity(&gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"vmm granularity");return gran;}
struct SparseVmmU32 {
    CUdeviceptr base=0; size_t bytes=0,gran=0; std::vector<CUmemGenericAllocationHandle> handles; std::vector<size_t> offsets,sizes;
    uint32_t* ptr() const { return reinterpret_cast<uint32_t*>(base); }
    void destroy(){if(!base)return;for(size_t i=0;i<handles.size();++i){ckcu(cuMemUnmap(base+offsets[i],sizes[i]),"vmm unmap");ckcu(cuMemRelease(handles[i]),"vmm release");}ckcu(cuMemAddressFree(base,bytes),"vmm address free");base=0;handles.clear();offsets.clear();sizes.clear();}
};
static SparseVmmU32 make_sparse_vmm_u32(int dev,size_t virtualElems,const std::vector<uint32_t>&codes,const std::vector<uint32_t>&values,const char*label){
    if(codes.size()!=values.size())throw std::runtime_error("VMM sparse code/value size mismatch");
    SparseVmmU32 z;ckcu(cuInit(0),"cuInit");CUdevice cdev;ckcu(cuDeviceGet(&cdev,dev),"cuDeviceGet");int ok=0;ckcu(cuDeviceGetAttribute(&ok,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,cdev),"vmm attr");if(!ok)throw std::runtime_error("CUDA VMM unsupported");
    CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;prop.location.id=cdev;
    ckcu(cuMemGetAllocationGranularity(&z.gran,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"vmm granularity");size_t raw=virtualElems*sizeof(uint32_t);z.bytes=(raw+z.gran-1)/z.gran*z.gran;ckcu(cuMemAddressReserve(&z.base,z.bytes,z.gran,0,0),"vmm reserve");
    if(z.gran%sizeof(uint32_t))throw std::runtime_error("VMM granularity is not u32 aligned");size_t elemsPer=z.gran/sizeof(uint32_t);std::vector<uint64_t> kv;kv.reserve(codes.size());for(size_t i=0;i<codes.size();++i){if(size_t(codes[i])>=virtualElems)throw std::runtime_error("VMM sparse code outside virtual rank table");kv.push_back((uint64_t(codes[i])<<32)|values[i]);}std::sort(kv.begin(),kv.end());for(size_t i=1;i<kv.size();++i)if(uint32_t(kv[i]>>32)==uint32_t(kv[i-1]>>32))throw std::runtime_error("duplicate VMM sparse rank code");
    CUmemAccessDesc ad{};ad.location.type=CU_MEM_LOCATION_TYPE_DEVICE;ad.location.id=cdev;ad.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    std::vector<size_t> pages;pages.reserve(kv.size());size_t prevPage=~size_t(0);for(uint64_t x:kv){size_t pg=size_t(uint32_t(x>>32))/elemsPer;if(pg!=prevPage){pages.push_back(pg);prevPage=pg;}}
    std::vector<uint32_t> page(elemsPer,0xffffffffu);size_t mapped=0,ki=0,maxRunPages=0;for(size_t ri=0;ri<pages.size();){size_t rj=ri+1;while(rj<pages.size()&&pages[rj]==pages[rj-1]+1)++rj;size_t startPg=pages[ri],runPages=rj-ri,off=startPg*z.gran,runBytes=runPages*z.gran;maxRunPages=std::max(maxRunPages,runPages);
        CUmemGenericAllocationHandle h;ckcu(cuMemCreate(&h,runBytes,&prop,0),"vmm create run");ckcu(cuMemMap(z.base+off,runBytes,0,h,0),"vmm map run");ckcu(cuMemSetAccess(z.base+off,runBytes,&ad,1),"vmm access run");z.handles.push_back(h);z.offsets.push_back(off);z.sizes.push_back(runBytes);mapped+=runBytes;
        for(size_t q=ri;q<rj;++q){size_t pg=pages[q];std::fill(page.begin(),page.end(),0xffffffffu);while(ki<kv.size()&&size_t(uint32_t(kv[ki]>>32))/elemsPer==pg){uint32_t c=uint32_t(kv[ki]>>32);page[size_t(c)%elemsPer]=uint32_t(kv[ki]);++ki;}auto ce=cudaMemcpy(reinterpret_cast<void*>(z.base+pg*z.gran),page.data(),z.gran,cudaMemcpyHostToDevice);if(ce!=cudaSuccess){std::cerr<<"vmm copy: "<<cudaGetErrorString(ce)<<"\n";std::exit(1);}}ri=rj;}
    if(ki!=kv.size())throw std::runtime_error("VMM sparse rank page fill mismatch");
    std::cerr<<"factor "<<label<<" VMM virtual_mib="<<(z.bytes>>20)<<" mapped_mib="<<(mapped>>20)<<" pages="<<pages.size()<<" runs="<<z.handles.size()<<" max_run_pages="<<maxRunPages<<" gran_kib="<<(z.gran>>10)<<" sparse_entries="<<codes.size()<<"\n";return z;
}
static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}
static void build_full_dp(){for(int h=0;h<=MAXW+1;++h)H_DP[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){Code x=H_DP[w-1][h];if(h>0)x+=H_DP[w-1][h-1];if(h<MAXW+1)x+=H_DP[w-1][h+1];H_DP[w][h]=x;}}
static void build_bounded_dp(int cap,Code out[MAXW+1][MAXW+2]){
    std::memset(out,0,sizeof(Code)*(MAXW+1)*(MAXW+2));
    out[0][0]=1;
    for(int w=1;w<=MAXW;++w)for(int h=0;h<=cap;++h){Code x=out[w-1][h];if(h>0)x+=out[w-1][h-1];if(h<cap)x+=out[w-1][h+1];out[w][h]=x;}
}


struct GroupSpec{int width=0;uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+2]{};Code size=0;};
static GroupSpec make_spec(int width,uint32_t fixed,uint32_t occ){GroupSpec s;s.width=width;s.fixed=fixed;s.occ=occ;for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);for(int w=1;w<=width;++w){int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;for(int h=0;h<=MAXW;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h>0)x+=s.dp[w-1][h-1];if(h<MAXW+1)x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}s.size=s.dp[width][1];return s;}

static bool allowed_host(uint32_t fixed,uint32_t occ,int pos,MateValue v){if(!((fixed>>pos)&1u))return v!=X;bool o=(occ>>pos)&1u;return o?(v==R||v==L):(v==N);}
static MateID unrank_suffix_host(Code rank,int width,int start_h,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID m=0;int h=start_h;
    for(int pos=width-1;pos>=0;--pos){
        if(allowed_host(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
        if(h>0&&allowed_host(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}
static Code rank_full_suffix_host(MateID m,int width,int start_h){Code rank=0;int h=start_h;for(int pos=width-1;pos>=0;--pos){auto v=mget(m,pos);if(v>N)rank+=H_DP[pos][h];if(v>R&&h>0)rank+=H_DP[pos][h-1];if(v==R)--h;else if(v==L)++h;}return rank;}
struct LowTablesHost{std::vector<LowEntry> entries;std::vector<uint32_t> offsets,local_rank;};
static LowTablesHost build_low_tables(){
    LowTablesHost t;
    if constexpr(LOW_LUT_K==0)return t;
    constexpr uint32_t NM=1u<<LOW_LUT_K;constexpr uint32_t NC=1u<<(2*LOW_LUT_K);
    t.offsets.resize(size_t(NM)*(LOW_LUT_K+2));t.local_rank.assign(NC,0xffffffffu);
    constexpr uint32_t FIX=NM-1;
    for(uint32_t mask=0;mask<NM;++mask){
        auto sp=make_spec(LOW_LUT_K,FIX,mask);
        for(int h=0;h<LOW_LUT_K+2;++h){
            t.offsets[size_t(mask)*(LOW_LUT_K+2)+h]=(uint32_t)t.entries.size();
            Code cnt=sp.dp[LOW_LUT_K][h];
            for(Code r=0;r<cnt;++r){MateID m=unrank_suffix_host(r,LOW_LUT_K,h,FIX,mask,sp.dp);Code fr=rank_full_suffix_host(m,LOW_LUT_K,h);if(fr>0xffffffffULL){std::cerr<<"low full rank overflow\n";std::exit(6);}uint32_t code=(uint32_t)m;t.entries.push_back({code,(uint32_t)fr});t.local_rank[code]=(uint32_t)r;}
        }
    }
    std::cerr<<"low_lut K="<<LOW_LUT_K<<" entries="<<t.entries.size()<<" entries_mib="<<double(t.entries.size()*sizeof(LowEntry))/(1<<20)<<" dense_rank_mib="<<double(t.local_rank.size()*sizeof(uint32_t))/(1<<20)<<"\n";
    return t;
}


static constexpr uint32_t POW3_7=2187u;
static uint32_t pow3_host(int n){uint32_t z=1;while(n--)z*=3u;return z;}

struct FactorTablesHost {
    static constexpr int STRIDE=MAXW+2;
    std::vector<uint32_t> low_all_codes,low_packed_values,low_mask_codes,low_mask_off,low_occ_base;
    std::vector<uint32_t> high_all_codes,high_packed_values,high_mask_codes,high_mask_off,high_occ_base;
    std::vector<Code> high_main_base,high_block_base;
    std::vector<uint64_t> low_code_rank,high_code_rank;
    std::array<uint32_t,MAXW+2> low_all_off{},high_all_off{};
};
static uint32_t seg_occ(uint32_t code,int len){uint32_t z=0;for(int p=0;p<len;++p)if((code>>(2*p))&3u)z|=1u<<p;return z;}
static FactorTablesHost build_factor_tables(){
    FactorTablesHost f;
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=FactorTablesHost::STRIDE;
    static_assert(L>0&&H>0,"factorized forced2 requires nonzero LOW/HIGH");
    const uint32_t LM=1u<<L, HM=1u<<H;
    std::vector<std::vector<uint32_t>> lm(size_t(LM)*S),hm(size_t(HM)*S);
    for(int h0=0;h0<=L+1;++h0){
        std::function<void(int,int,uint32_t)> rec=[&](int pos,int h,uint32_t code){
            if(pos<0){if(h==0)lm[size_t(seg_occ(code,L))*S+h0].push_back(code);return;}
            if(h<0||h>pos+1)return;
            rec(pos-1,h,code);
            if(h>0)rec(pos-1,h-1,code|(uint32_t(R)<<(2*pos)));
            rec(pos-1,h+1,code|(uint32_t(::L)<<(2*pos)));
        };rec(L-1,h0,0);
    }
    std::function<void(int,int,uint32_t)> rech=[&](int pos,int h,uint32_t code){
        if(pos<0){hm[size_t(seg_occ(code,H))*S+h].push_back(code);return;}
        rech(pos-1,h,code);
        if(h>0)rech(pos-1,h-1,code|(uint32_t(R)<<(2*pos)));
        rech(pos-1,h+1,code|(uint32_t(::L)<<(2*pos)));
    };rech(H-1,1,0);

    f.low_occ_base.resize(size_t(LM)*S);
    for(int h=0;h<=MAXW;++h){
        f.low_all_off[h]=f.low_all_codes.size();uint32_t ar=0;
        for(uint32_t m=0;m<LM;++m){size_t ix=size_t(m)*S+h;f.low_occ_base[ix]=ar;
            auto const&v=lm[ix];for(uint32_t lr=0;lr<v.size();++lr,++ar){
                if(uint64_t(ar)>=(uint64_t(1)<<(32-L)))throw std::runtime_error("low all rank exceeds packed field");
                uint32_t c=v[lr];f.low_all_codes.push_back(c);f.low_packed_values.push_back((ar<<L)|lr);f.low_code_rank.push_back((uint64_t(c)<<32)|ar);
            }
        }
    }
    f.low_all_off[MAXW+1]=f.low_all_codes.size();std::sort(f.low_code_rank.begin(),f.low_code_rank.end());
    f.low_mask_off.resize(size_t(LM)*S);
    for(uint32_t m=0;m<LM;++m)for(int h=0;h<S;++h){size_t ix=size_t(m)*S+h;f.low_mask_off[ix]=f.low_mask_codes.size();if(h<=MAXW)f.low_mask_codes.insert(f.low_mask_codes.end(),lm[ix].begin(),lm[ix].end());}

    f.high_occ_base.resize(size_t(HM)*S);
    for(int h=0;h<=MAXW;++h){
        f.high_all_off[h]=f.high_all_codes.size();uint32_t ar=0;
        for(uint32_t m=0;m<HM;++m){size_t ix=size_t(m)*S+h;f.high_occ_base[ix]=ar;
            auto const&v=hm[ix];for(uint32_t hr=0;hr<v.size();++hr,++ar){
                if(uint64_t(ar)>=(uint64_t(1)<<(32-H)))throw std::runtime_error("high all rank exceeds packed field");
                uint32_t c=v[hr];f.high_all_codes.push_back(c);f.high_packed_values.push_back((ar<<H)|hr);f.high_code_rank.push_back((uint64_t(c)<<32)|ar);
            }
        }
    }
    f.high_all_off[MAXW+1]=f.high_all_codes.size();std::sort(f.high_code_rank.begin(),f.high_code_rank.end());
    f.high_mask_off.resize(size_t(HM)*S);
    for(uint32_t m=0;m<HM;++m)for(int h=0;h<S;++h){size_t ix=size_t(m)*S+h;f.high_mask_off[ix]=f.high_mask_codes.size();if(h<=MAXW)f.high_mask_codes.insert(f.high_mask_codes.end(),hm[ix].begin(),hm[ix].end());}

    auto prefix_base=[&](uint32_t code,int offset){Code rank=0;int h=1;for(int p=H-1;p>=0;--p){MateValue v=MateValue((code>>(2*p))&3u);int fp=offset+p;if(v>N)rank+=H_DP[fp][h];if(v>R&&h>0)rank+=H_DP[fp][h-1];if(v==R)--h;else if(v==::L)++h;}return rank;};
    f.high_main_base.resize(f.high_all_codes.size());f.high_block_base.resize(f.high_all_codes.size());
    for(size_t i=0;i<f.high_all_codes.size();++i){f.high_main_base[i]=prefix_base(f.high_all_codes[i],L+1);f.high_block_base[i]=prefix_base(f.high_all_codes[i],L);}
    std::cerr<<"factor OCCMAJOR low_all="<<f.low_all_codes.size()<<" high_all="<<f.high_all_codes.size()<<" low_occ_base_mib="<<double(f.low_occ_base.size()*4)/(1<<20)<<" high_occ_base_mib="<<double(f.high_occ_base.size()*4)/(1<<20)<<"\n";
    return f;
}
static FactorTablesHost G_FACTOR;
struct SparseOrbitTablesHost {
    std::vector<uint32_t> owner_lr,closure_lr;
    std::vector<SparseSel> owner_sel,closure_sel;
};
static SparseOrbitTablesHost G_SPARSE;
static size_t sparse_key(int p,int hs,int c){return (size_t(p)*(MAXW+2)+size_t(hs))*3u+size_t(c);}
static SparseOrbitTablesHost build_sparse_orbit_tables(){
    constexpr int L=LOW_LUT_K;
    SparseOrbitTablesHost z;
    size_t keys=size_t(MAXW+1)*(MAXW+2)*3u;
    z.owner_sel.assign(keys,{0,0});z.closure_sel.assign(keys,{0,0});
    auto build_one=[&](int p,int hs,int c){
        uint32_t beg=G_FACTOR.low_all_off[hs],end=G_FACTOR.low_all_off[hs+1];
        SparseSel os{uint32_t(z.owner_lr.size()),0},cs{uint32_t(z.closure_lr.size()),0};
        for(uint32_t a=beg;a<end;++a){
            uint32_t lr=a-beg,lc=G_FACTOR.low_all_codes[a];
            oneesan::gridfp::MateID m=oneesan::gridfp::MateID(lc)|(oneesan::gridfp::MateID(c)<<(2*L));
            auto w=oneesan::gridfp::mpair(m,p);
            if(w==oneesan::gridfp::NN||w==oneesan::gridfp::NR||w==oneesan::gridfp::NL)z.owner_lr.push_back(lr);
            if(w==oneesan::gridfp::LL||w==oneesan::gridfp::RR||w==oneesan::gridfp::RL)z.closure_lr.push_back(lr);
        }
        os.count=uint32_t(z.owner_lr.size())-os.off;cs.count=uint32_t(z.closure_lr.size())-cs.off;
        z.owner_sel[sparse_key(p,hs,c)]=os;z.closure_sel[sparse_key(p,hs,c)]=cs;
        return std::pair<SparseSel,SparseSel>{os,cs};
    };
    if constexpr(L>=1)for(int hs=0;hs<=L+1;++hs){auto q=build_one(1,hs,0);for(int c=1;c<3;++c){z.owner_sel[sparse_key(1,hs,c)]=q.first;z.closure_sel[sparse_key(1,hs,c)]=q.second;}}
    double mib=double(z.owner_lr.size()+z.closure_lr.size())*4.0/(1<<20);
    std::cerr<<"sparse orbit LUT owner="<<z.owner_lr.size()<<" closure="<<z.closure_lr.size()<<" mib="<<mib<<"\n";
    return z;
}

struct PreRankOrbitTablesHost {
    std::vector<unsigned long long> owner_rec,owner_block_rec,closure_rec;
    std::vector<unsigned long long> highrr_rec;
    std::vector<unsigned long long> closure_inv_rec;
    std::vector<uint32_t> closure_inv_src;
    std::vector<SparseSel> owner_sel,closure_sel,highrr_sel,closure_inv_sel;
};
static PreRankOrbitTablesHost G_PRERANK;
static uint32_t low_all_rank_host(uint32_t code,int h){
    if(h<0||h>MAXW)throw std::runtime_error("low rank height");
    auto &v=G_FACTOR.low_code_rank;auto it=std::lower_bound(v.begin(),v.end(),uint64_t(code)<<32);
    if(it==v.end()||uint32_t(*it>>32)!=code)throw std::runtime_error("pre-rank low destination missing");
    uint32_t r=uint32_t(*it);if(r>=G_FACTOR.low_all_off[h+1]-G_FACTOR.low_all_off[h])throw std::runtime_error("pre-rank low height mismatch");return r;
}
static unsigned long long pack_owner_rec(uint32_t src,uint32_t main,uint32_t block,uint32_t type){
    constexpr uint32_t B=20, M=(1u<<B)-1u;
    if(src>M||main>M||block>M||type>2)throw std::runtime_error("pre-rank owner field overflow");
    return (unsigned long long)src | ((unsigned long long)main<<B) | ((unsigned long long)block<<(2*B)) | ((unsigned long long)type<<(3*B));
}
static unsigned long long pack_closure_rec(uint32_t src,uint32_t dst,uint32_t center){
    constexpr uint32_t B=20, M=(1u<<B)-1u;
    if(src>M||dst>M||center>3)throw std::runtime_error("pre-rank closure field overflow");
    return (unsigned long long)src | ((unsigned long long)dst<<B) | ((unsigned long long)center<<(2*B));
}
static unsigned long long pack_closure_inv_rec(uint32_t dst,unsigned long long payload,uint32_t count,uint32_t center){
    constexpr uint32_t B=18,M=(1u<<B)-1u,P=36;constexpr unsigned long long PM=(1ULL<<P)-1ULL;
    if(dst>M||payload>PM||count==0||count>7||center>3)throw std::runtime_error("pre-rank compact closure inverse field overflow");
    return (unsigned long long)dst | (payload<<B) | ((unsigned long long)count<<(B+P)) | ((unsigned long long)center<<(B+P+3));
}
static unsigned long long pack_highrr_rec(uint32_t src,uint32_t dstlow,uint32_t depth){
    constexpr uint32_t B=20, M=(1u<<B)-1u;
    if(src>M||dstlow>M||depth==0||depth>63)throw std::runtime_error("pre-rank high-RR field overflow");
    return (unsigned long long)src | ((unsigned long long)dstlow<<B) | ((unsigned long long)depth<<(2*B));
}
static PreRankOrbitTablesHost build_prerank_orbit_tables(){
    constexpr int L=LOW_LUT_K;
    PreRankOrbitTablesHost z;size_t keys=size_t(MAXW+1)*(MAXW+2)*3u;
    z.owner_sel.assign(keys,{0,0});z.closure_sel.assign(keys,{0,0});z.highrr_sel.assign(keys,{0,0});z.closure_inv_sel.assign(keys,{0,0});
    if constexpr(L<1)return z;
    const size_t lowTotal=G_FACTOR.low_all_codes.size();z.owner_block_rec.assign(size_t(L)*lowTotal,~0ULL);
    for(int p=1;p<=L;++p)for(int hs=0;hs<=L+1;++hs)for(int c=0;c<3;++c){
        auto key=sparse_key(p,hs,c);SparseSel os{uint32_t(z.owner_rec.size()),0},cs{uint32_t(z.closure_rec.size()),0},ss{uint32_t(z.highrr_rec.size()),0};
        int dc=(c==int(::L)?1:c==int(::R)?-1:0),he=hs-dc;
        if(he<0||he>HIGH_LUT_K+1){z.owner_sel[key]=os;z.closure_sel[key]=cs;z.highrr_sel[key]=ss;z.closure_inv_sel[key]={uint32_t(z.closure_inv_rec.size()),0};continue;}
        uint32_t beg=G_FACTOR.low_all_off[hs],end=G_FACTOR.low_all_off[hs+1];
        for(uint32_t a=beg;a<end;++a){
            uint32_t lr=a-beg,lc=G_FACTOR.low_all_codes[a];
            oneesan::gridfp::MateID m=oneesan::gridfp::MateID(lc)|(oneesan::gridfp::MateID(c)<<(2*L));
            auto w=oneesan::gridfp::mpair(m,p);
            if(w==oneesan::gridfp::NN||w==oneesan::gridfp::NR||w==oneesan::gridfp::NL){
                oneesan::gridfp::MateID t;
                uint32_t ty=0;
                if(w==oneesan::gridfp::NN){t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::LR);ty=0;}
                else if(w==oneesan::gridfp::NR){t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::RN);ty=1;}
                else {t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::LN);ty=2;}
                int tc=int(oneesan::gridfp::mget(t,L)),td=(tc==int(::L)?1:tc==int(::R)?-1:0),ths=he+td;
                uint32_t mainlr=low_all_rank_host(uint32_t(t)&((1u<<(2*L))-1u),ths);
                auto b=oneesan::gridfp::mshrink(m,p);
                uint32_t blr=low_all_rank_host(uint32_t(b)&((1u<<(2*L))-1u),he);
                z.owner_rec.push_back(pack_owner_rec(lr,mainlr,blr,ty));
                constexpr uint32_t OB=20,OM=(1u<<OB)-1u;if(lr>OM||mainlr>OM)throw std::runtime_error("low blocked-order owner rank overflow");
                size_t obi=size_t(p-1)*lowTotal+G_FACTOR.low_all_off[he]+blr;if(z.owner_block_rec[obi]!=~0ULL)throw std::runtime_error("low blocked-order owner duplicate");
                z.owner_block_rec[obi]=(unsigned long long)lr|((unsigned long long)mainlr<<OB)|((unsigned long long)c<<(2*OB))|((unsigned long long)ty<<(2*OB+2));
            }
            if(w==oneesan::gridfp::LL||w==oneesan::gridfp::RR||w==oneesan::gridfp::RL){
                auto iz=oneesan::gridfp::include_horizontal(m,L+1,p);
                if(iz.valid){
                    if(p==1){
                        if(iz.blocked)throw std::runtime_error("pre-rank p1 closure unexpectedly blocked");
                        int tc=int(oneesan::gridfp::mget(iz.mate,L)),td=(tc==int(::L)?1:tc==int(::R)?-1:0),ths=he+td;
                        uint32_t dlr=low_all_rank_host(uint32_t(iz.mate)&((1u<<(2*L))-1u),ths);
                        z.closure_rec.push_back(pack_closure_rec(lr,dlr,uint32_t(tc)));
                    }else{
                        if(!iz.blocked)throw std::runtime_error("pre-rank closure expected blocked");
                        uint32_t blr=low_all_rank_host(uint32_t(iz.mate)&((1u<<(2*L))-1u),he);
                        z.closure_rec.push_back(pack_closure_rec(lr,blr,3));
                    }
                }else if(w==oneesan::gridfp::RR){
                    // The matching L lies in HIGH.  Precompute everything on the LOW/center side:
                    // source low rank, destination low rank, and the unmatched stack depth entering HIGH.
                    auto t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::NN);
                    int depth=1;
                    for(int q=p+1;q<=L;++q){auto v=oneesan::gridfp::mget(t,q);if(v==oneesan::gridfp::L)--depth;else if(v==oneesan::gridfp::R)++depth;if(depth<=0)throw std::runtime_error("high-RR classification mismatch");}
                    int nhe=he-2;if(nhe<0)throw std::runtime_error("high-RR destination height underflow");
                    uint32_t dlr=0;
                    if(p==1){int tc=int(oneesan::gridfp::mget(t,L)),td=(tc==int(::L)?1:tc==int(::R)?-1:0),ths=nhe+td;dlr=low_all_rank_host(uint32_t(t)&((1u<<(2*L))-1u),ths);}
                    else{auto b=oneesan::gridfp::mshrink(t,p-1);dlr=low_all_rank_host(uint32_t(b)&((1u<<(2*L))-1u),nhe);}
                    z.highrr_rec.push_back(pack_highrr_rec(lr,dlr,uint32_t(depth)));
                }
            }
        }
        os.count=uint32_t(z.owner_rec.size())-os.off;cs.count=uint32_t(z.closure_rec.size())-cs.off;ss.count=uint32_t(z.highrr_rec.size())-ss.off;
        SparseSel is{uint32_t(z.closure_inv_rec.size()),0};
        { constexpr uint32_t B=20,M=(1u<<B)-1u; std::vector<std::pair<uint32_t,uint32_t>> a; a.reserve(cs.count);
          for(uint32_t q=0;q<cs.count;++q){auto r=z.closure_rec[cs.off+q];uint32_t src=uint32_t(r)&M,dst=uint32_t(r>>B)&M,dc=uint32_t(r>>(2*B))&3u;uint32_t ik=dst|(dc<<B);a.push_back({ik,src});}
          std::sort(a.begin(),a.end()); for(size_t q=0;q<a.size();){size_t e=q+1;while(e<a.size()&&a[e].first==a[q].first)++e;uint32_t cnt=uint32_t(e-q);unsigned long long payload=0;if(cnt<=2){for(uint32_t j=0;j<cnt;++j){uint32_t src=a[q+j].second;if(src>=(1u<<18))throw std::runtime_error("low inverse source exceeds 18 bits");payload|=(unsigned long long)src<<(18*j);}}else{payload=z.closure_inv_src.size();for(size_t j=q;j<e;++j)z.closure_inv_src.push_back(a[j].second);}uint32_t ik=a[q].first;z.closure_inv_rec.push_back(pack_closure_inv_rec(ik&M,payload,cnt,ik>>B));q=e;} }
        is.count=uint32_t(z.closure_inv_rec.size())-is.off;
        z.owner_sel[key]=os;z.closure_sel[key]=cs;z.highrr_sel[key]=ss;z.closure_inv_sel[key]=is;
    }
    for(auto r:z.owner_block_rec)if(r==~0ULL)throw std::runtime_error("low blocked-order owner missing");
    double mib=(double(z.owner_block_rec.size()+z.highrr_rec.size()+z.closure_inv_rec.size())*8.0+double(z.closure_inv_src.size())*4.0)/(1<<20);
    std::cerr<<"pre-rank orbit owner="<<z.owner_block_rec.size()<<" closure_fast="<<z.closure_inv_src.size()<<" closure_inv="<<z.closure_inv_rec.size()<<" closure_highrr="<<z.highrr_rec.size()<<" device_mib="<<mib<<"\n";
    z.owner_rec.clear();z.owner_rec.shrink_to_fit();z.closure_rec.clear();z.closure_rec.shrink_to_fit();
    return z;
}

struct HighPreRankTablesHost {
    std::vector<unsigned long long> owner_rec,owner_block_rec,closure_rec,crossll_rec;
    std::vector<unsigned long long> closure_inv_rec;
    std::vector<uint32_t> closure_inv_src;
    std::vector<SparseSel> owner_sel,closure_sel,crossll_sel,closure_inv_sel;
};
static HighPreRankTablesHost G_HPR;
static size_t high_pr_key(int p,int he,int c){return (size_t(p)*(MAXW+2)+size_t(he))*3u+size_t(c);}
static uint32_t high_all_rank_host(uint32_t code,int h){
    if(h<0||h>MAXW)throw std::runtime_error("high rank height");
    auto &v=G_FACTOR.high_code_rank;auto it=std::lower_bound(v.begin(),v.end(),uint64_t(code)<<32);
    if(it==v.end()||uint32_t(*it>>32)!=code)throw std::runtime_error("high pre-rank destination missing");
    uint32_t r=uint32_t(*it);if(r>=G_FACTOR.high_all_off[h+1]-G_FACTOR.high_all_off[h])throw std::runtime_error("high pre-rank height mismatch");return r;
}
static unsigned long long pack_high_owner_rec(uint32_t src,uint32_t main,uint32_t block,uint32_t type,uint32_t center){
    constexpr uint32_t B=20,M=(1u<<B)-1u;if(src>M||main>M||block>M||type>2||center>2)throw std::runtime_error("high owner field overflow");return (unsigned long long)src|((unsigned long long)main<<B)|((unsigned long long)block<<(2*B))|((unsigned long long)type<<(3*B))|((unsigned long long)center<<(3*B+2));
}
static unsigned long long pack_high_closure_rec(uint32_t src,uint32_t block){constexpr uint32_t B=20,M=(1u<<B)-1u;if(src>M||block>M)throw std::runtime_error("high closure field overflow");return (unsigned long long)src|((unsigned long long)block<<B);}
static unsigned long long pack_high_closure_inv_rec(uint32_t dst,unsigned long long payload,uint32_t count){constexpr uint32_t B=18,M=(1u<<B)-1u,P=36;constexpr unsigned long long PM=(1ULL<<P)-1ULL;if(dst>M||payload>PM||count==0||count>7)throw std::runtime_error("high compact closure inverse field overflow");return (unsigned long long)dst|(payload<<B)|((unsigned long long)count<<(B+P));}
static unsigned long long pack_high_crossll_rec(uint32_t src,uint32_t block,uint32_t depth){constexpr uint32_t B=20,M=(1u<<B)-1u;if(src>M||block>M||depth==0||depth>63)throw std::runtime_error("high cross-LL field overflow");return (unsigned long long)src|((unsigned long long)block<<B)|((unsigned long long)depth<<(2*B));}
static HighPreRankTablesHost build_high_prerank_tables(){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t HM=(1u<<(2*H))-1u;HighPreRankTablesHost z;size_t keys=size_t(MAXW+1)*(MAXW+2)*3u;const size_t highTotal=G_FACTOR.high_all_codes.size();z.owner_block_rec.assign(size_t(H)*highTotal,~0ULL);z.owner_sel.assign(keys,{0,0});z.closure_sel.assign(keys,{0,0});z.crossll_sel.assign(keys,{0,0});z.closure_inv_sel.assign(keys,{0,0});
    for(int p=L+1;p<=L+H;++p)for(int he=0;he<=H+1;++he)for(int c=0;c<3;++c){auto key=high_pr_key(p,he,c);SparseSel os{uint32_t(z.owner_rec.size()),0},cs{uint32_t(z.closure_rec.size()),0},xs{uint32_t(z.crossll_rec.size()),0};int dc=(c==int(::L)?1:c==int(::R)?-1:0),hs=he+dc;if(hs<0||hs>L+1){z.owner_sel[key]=os;z.closure_sel[key]=cs;z.crossll_sel[key]=xs;z.closure_inv_sel[key]={uint32_t(z.closure_inv_rec.size()),0};continue;}uint32_t beg=G_FACTOR.high_all_off[he],end=G_FACTOR.high_all_off[he+1];
        for(uint32_t a=beg;a<end;++a){uint32_t hr=a-beg,hc=G_FACTOR.high_all_codes[a];oneesan::gridfp::MateID m=(oneesan::gridfp::MateID(c)<<(2*L))|(oneesan::gridfp::MateID(hc)<<(2*(L+1)));auto w=oneesan::gridfp::mpair(m,p);
            if(w==oneesan::gridfp::NN||w==oneesan::gridfp::NR||w==oneesan::gridfp::NL){oneesan::gridfp::MateID t;uint32_t ty;if(w==oneesan::gridfp::NN){t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::LR);ty=0;}else if(w==oneesan::gridfp::NR){t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::RN);ty=1;}else{t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::LN);ty=2;}int tc=int(oneesan::gridfp::mget(t,L)),td=(tc==int(::L)?1:tc==int(::R)?-1:0),the=hs-td;uint32_t thc=uint32_t((t>>(2*(L+1)))&HM),mhr=high_all_rank_host(thc,the);auto b=oneesan::gridfp::mshrink(m,p);uint32_t bhc=uint32_t((b>>(2*L))&HM),bhr=high_all_rank_host(bhc,hs);z.owner_rec.push_back(pack_high_owner_rec(hr,mhr,bhr,ty,uint32_t(tc)));constexpr uint32_t OB=20,OM=(1u<<OB)-1u;if(hr>OM||mhr>OM)throw std::runtime_error("high blocked-order owner rank overflow");size_t obi=size_t(p-L-1)*highTotal+G_FACTOR.high_all_off[hs]+bhr;if(z.owner_block_rec[obi]!=~0ULL)throw std::runtime_error("high blocked-order owner duplicate");z.owner_block_rec[obi]=(unsigned long long)hr|((unsigned long long)mhr<<OB)|((unsigned long long)c<<(2*OB))|((unsigned long long)ty<<(2*OB+2))|((unsigned long long)tc<<(2*OB+4));}
            if(w==oneesan::gridfp::LL||w==oneesan::gridfp::RR||w==oneesan::gridfp::RL){auto iz=oneesan::gridfp::include_horizontal(m,L+1+H,p);if(iz.valid){if(!iz.blocked)throw std::runtime_error("upper closure expected blocked");uint32_t bhc=uint32_t((iz.mate>>(2*L))&HM),bhr=high_all_rank_host(bhc,hs);z.closure_rec.push_back(pack_high_closure_rec(hr,bhr));}
                else if(w==oneesan::gridfp::LL){int nh=hs-2;if(nh>=0){auto t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::NN);int depth=1;for(int q=p-2;q>=L;--q){auto v=oneesan::gridfp::mget(t,q);if(v==oneesan::gridfp::L)++depth;else if(v==oneesan::gridfp::R)--depth;if(depth<=0)throw std::runtime_error("cross-LL classification mismatch");}auto b=oneesan::gridfp::mshrink(t,p-1);uint32_t bhc=uint32_t((b>>(2*L))&HM),bhr=high_all_rank_host(bhc,nh);z.crossll_rec.push_back(pack_high_crossll_rec(hr,bhr,uint32_t(depth)));}}
            }
        }
        os.count=uint32_t(z.owner_rec.size())-os.off;cs.count=uint32_t(z.closure_rec.size())-cs.off;xs.count=uint32_t(z.crossll_rec.size())-xs.off;
        SparseSel is{uint32_t(z.closure_inv_rec.size()),0};
        { constexpr uint32_t B=20,M=(1u<<B)-1u; std::vector<std::pair<uint32_t,uint32_t>> a; a.reserve(cs.count);
          for(uint32_t q=0;q<cs.count;++q){auto r=z.closure_rec[cs.off+q];a.push_back({uint32_t(r>>B)&M,uint32_t(r)&M});}
          std::sort(a.begin(),a.end()); for(size_t q=0;q<a.size();){size_t e=q+1;while(e<a.size()&&a[e].first==a[q].first)++e;uint32_t cnt=uint32_t(e-q);unsigned long long payload=0;if(cnt<=2){for(uint32_t j=0;j<cnt;++j){uint32_t src=a[q+j].second;if(src>=(1u<<18))throw std::runtime_error("high inverse source exceeds 18 bits");payload|=(unsigned long long)src<<(18*j);}}else{payload=z.closure_inv_src.size();for(size_t j=q;j<e;++j)z.closure_inv_src.push_back(a[j].second);}z.closure_inv_rec.push_back(pack_high_closure_inv_rec(a[q].first,payload,cnt));q=e;} }
        is.count=uint32_t(z.closure_inv_rec.size())-is.off;z.owner_sel[key]=os;z.closure_sel[key]=cs;z.crossll_sel[key]=xs;z.closure_inv_sel[key]=is;
    }
    for(auto r:z.owner_block_rec)if(r==~0ULL)throw std::runtime_error("high blocked-order owner missing");double mib=(double(z.owner_block_rec.size()+z.crossll_rec.size()+z.closure_inv_rec.size())*8.0+double(z.closure_inv_src.size())*4.0)/(1<<20);std::cerr<<"high pre-rank owner="<<z.owner_block_rec.size()<<" closure_fast="<<z.closure_inv_src.size()<<" closure_inv="<<z.closure_inv_rec.size()<<" closure_crossll="<<z.crossll_rec.size()<<" device_mib="<<mib<<"\n";z.owner_rec.clear();z.owner_rec.shrink_to_fit();z.closure_rec.clear();z.closure_rec.shrink_to_fit();return z;
}
static uint32_t factor_count(const std::vector<uint32_t>&off,uint32_t mask,int h){constexpr int S=FactorTablesHost::STRIDE;size_t i=size_t(mask)*S+h;return off[i+1]-off[i];}
static std::vector<FBlock> make_factor_main_blocks(bool fix_low,uint32_t mask){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;std::vector<FBlock> v;v.reserve(3*(H+2));Code off=0;
    for(int he=0;he<=H+1;++he)for(int cv=0;cv<3;++cv){int hs=he+(cv==int(::L)?1:cv==int(::R)?-1:0);uint32_t hc=0,lc=0;if(hs>=0&&hs<=L+1){hc=fix_low?(G_FACTOR.high_all_off[he+1]-G_FACTOR.high_all_off[he]):factor_count(G_FACTOR.high_mask_off,mask,he);lc=fix_low?factor_count(G_FACTOR.low_mask_off,mask,hs):(G_FACTOR.low_all_off[hs+1]-G_FACTOR.low_all_off[hs]);}uint64_t n=uint64_t(hc)*lc;v.push_back({off,off+n,lc,(uint8_t)he,(uint8_t)std::max(0,hs),(uint8_t)cv,0});off+=n;}
    return v;
}
static std::vector<FBlock> make_factor_block_blocks(bool fix_low,uint32_t mask){
    constexpr int H=HIGH_LUT_K;std::vector<FBlock> v;v.reserve(H+2);Code off=0;
    for(int h=0;h<=H+1;++h){uint32_t hc=fix_low?(G_FACTOR.high_all_off[h+1]-G_FACTOR.high_all_off[h]):factor_count(G_FACTOR.high_mask_off,mask,h);uint32_t lc=fix_low?factor_count(G_FACTOR.low_mask_off,mask,h):(G_FACTOR.low_all_off[h+1]-G_FACTOR.low_all_off[h]);uint64_t n=uint64_t(hc)*lc;v.push_back({off,off+n,lc,(uint8_t)h,(uint8_t)h,0,0});off+=n;}return v;
}
static std::vector<FBlock> G_F_FULL_MAIN_BLOCKS,G_F_FULL_BLOCK_BLOCKS;
static std::vector<FBlock> make_factor_full_main_blocks(){
    constexpr int H=HIGH_LUT_K;std::vector<FBlock> v;v.reserve(3*(H+2));Code off=0;
    for(int he=0;he<=H+1;++he)for(int cv=0;cv<3;++cv){int hs=he+(cv==int(::L)?1:cv==int(::R)?-1:0);uint32_t hc=0,lc=0;if(hs>=0&&hs<=LOW_LUT_K+1){hc=G_FACTOR.high_all_off[he+1]-G_FACTOR.high_all_off[he];lc=G_FACTOR.low_all_off[hs+1]-G_FACTOR.low_all_off[hs];}uint64_t n=uint64_t(hc)*lc;v.push_back({off,off+n,lc,(uint8_t)he,(uint8_t)std::max(0,hs),(uint8_t)cv,0});off+=n;}return v;
}
static std::vector<FBlock> make_factor_full_block_blocks(){
    constexpr int H=HIGH_LUT_K;std::vector<FBlock> v;v.reserve(H+2);Code off=0;for(int h=0;h<=H+1;++h){uint32_t hc=G_FACTOR.high_all_off[h+1]-G_FACTOR.high_all_off[h],lc=G_FACTOR.low_all_off[h+1]-G_FACTOR.low_all_off[h];uint64_t n=uint64_t(hc)*lc;v.push_back({off,off+n,lc,(uint8_t)h,(uint8_t)h,0,0});off+=n;}return v;
}
struct Interval{Code global,local,len;};
struct HighTablesHost{std::vector<HighEntry> entries;std::vector<uint32_t> offsets;};
static void high_enum_rec(int pos,int h,uint32_t occ,uint32_t code,int lowlen,uint64_t&base,std::vector<HighEntry>&entries){
    if(pos<0){Code cnt=H_DP[lowlen][h];if(!cnt)return;if(base>0xffffffffULL||base+cnt>0x100000000ULL){std::cerr<<"high base overflow\n";std::exit(7);}entries.push_back({code,(uint32_t)base});base+=cnt;return;}
    bool o=(occ>>pos)&1u;
    if(!o){high_enum_rec(pos-1,h,occ,code,lowlen,base,entries);return;}
    if(h>0)high_enum_rec(pos-1,h-1,occ,code|(uint32_t(R)<<(2*pos)),lowlen,base,entries);
    high_enum_rec(pos-1,h+1,occ,code|(uint32_t(::L)<<(2*pos)),lowlen,base,entries);
}
static HighTablesHost build_high_tables(int width){
    HighTablesHost t;if constexpr(HIGH_LUT_K==0)return t;
    constexpr uint32_t NM=1u<<HIGH_LUT_K;int lowlen=width-HIGH_LUT_K;
    t.offsets.resize(NM+1);
    for(uint32_t mask=0;mask<NM;++mask){t.offsets[mask]=(uint32_t)t.entries.size();uint64_t base=0;high_enum_rec(HIGH_LUT_K-1,1,mask,0,lowlen,base,t.entries);}t.offsets[NM]=(uint32_t)t.entries.size();
    std::cerr<<"high_lut width="<<width<<" K="<<HIGH_LUT_K<<" entries="<<t.entries.size()<<" entries_mib="<<double(t.entries.size()*sizeof(HighEntry))/(1<<20)<<" offsets_mib="<<double(t.offsets.size()*sizeof(uint32_t))/(1<<20)<<"\n";return t;
}

static void add_interval(std::vector<Interval>&out,Code g,Code l,Code n){if(!n)return;if(!out.empty()&&out.back().global+out.back().len==g&&out.back().local+out.back().len==l)out.back().len+=n;else out.push_back({g,l,n});}
static void intervals_rec(const GroupSpec&s,int pos,int h,Code gbase,Code lbase,std::vector<Interval>&out){
    if(pos<0){if(h==0)add_interval(out,gbase,lbase,1);return;}
    uint32_t lower=(pos==31)?0xffffffffu:((1u<<(pos+1))-1u);
    if((s.fixed&lower)==0){add_interval(out,gbase,lbase,H_DP[pos+1][h]);return;}
    bool f=(s.fixed>>pos)&1u,o=(s.occ>>pos)&1u;
    Code gsz=H_DP[pos][h];
    if(!f||!o){Code lsz=s.dp[pos][h];intervals_rec(s,pos-1,h,gbase,lbase,out);lbase+=lsz;}gbase+=gsz;
    if(h>0){gsz=H_DP[pos][h-1];if(!f||o){Code lsz=s.dp[pos][h-1];intervals_rec(s,pos-1,h-1,gbase,lbase,out);lbase+=lsz;}gbase+=gsz;}
    if(h<MAXW+1&&(!f||o))intervals_rec(s,pos-1,h+1,gbase,lbase,out);
}
static std::vector<Interval> make_intervals(const GroupSpec&s){std::vector<Interval>v;v.reserve(1024);intervals_rec(s,s.width-1,1,0,0,v);Code sum=0;for(auto const&i:v)sum+=i.len;if(sum!=s.size){std::cerr<<"interval size mismatch "<<sum<<" != "<<s.size<<"\n";std::exit(2);}return v;}

static Code interval_leaf_upper_rec(const GroupSpec& s,int pos,int h,Code memo[MAXW+1][MAXW+2],bool seen[MAXW+1][MAXW+2]){
    if(pos<0)return h==0?1:0;
    if(seen[pos][h])return memo[pos][h];
    seen[pos][h]=true;
    uint32_t lower=(pos==31)?0xffffffffu:((1u<<(pos+1))-1u);
    if((s.fixed&lower)==0)return memo[pos][h]=H_DP[pos+1][h]?1:0;
    bool f=(s.fixed>>pos)&1u,o=(s.occ>>pos)&1u;
    Code z=0;
    if(!f||!o)z+=interval_leaf_upper_rec(s,pos-1,h,memo,seen);
    if(!f||o){
        if(h>0)z+=interval_leaf_upper_rec(s,pos-1,h-1,memo,seen);
        if(h<MAXW+1)z+=interval_leaf_upper_rec(s,pos-1,h+1,memo,seen);
    }
    return memo[pos][h]=z;
}
static Code interval_leaf_upper(const GroupSpec& s){
    Code memo[MAXW+1][MAXW+2]{};bool seen[MAXW+1][MAXW+2]{};
    return interval_leaf_upper_rec(s,s.width-1,1,memo,seen);
}

struct PeerInterval{Code remote,local,len;uint32_t owner,pad;};
static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,Code chunk,int ng,bool& use_interval){
    constexpr Code MIN_AVG_INTERVAL_ELEMS = 65536;
    Code est=interval_leaf_upper(s);
    // Every shard boundary can split at most one globally ordered interval.
    Code est_peer=est+Code(std::max(0,ng-1));
    use_interval = est_peer==0 || s.size >= est_peer*MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval)return {};
    auto base=make_intervals(s);std::vector<PeerInterval>out;out.reserve(base.size()+ng);
    for(auto const&x:base){Code g=x.global,l=x.local,left=x.len;while(left){int owner=int(g/chunk);if(owner>=ng)owner=ng-1;Code shard0=Code(owner)*chunk;Code shard_end=(owner+1<ng)?shard0+chunk:~Code(0);Code take=left;if(shard_end!=~Code(0)&&g+take>shard_end)take=shard_end-g;Code remote=g-shard0;
            if(!out.empty()&&out.back().owner==(uint32_t)owner&&out.back().remote+out.back().len==remote&&out.back().local+out.back().len==l)out.back().len+=take;else out.push_back({remote,l,take,(uint32_t)owner,0});
            g+=take;l+=take;left-=take;}}
    // Interval I/O wins only when the canonical layout contains reasonably long
    // contiguous runs.  Low fixed bits can fragment a group into hundreds of
    // thousands of tiny runs; in that case rank/unrank gather/scatter is faster.
    use_interval = out.empty() || s.size >= Code(out.size()) * MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval) return {};

    std::vector<PeerInterval> tiled;
    size_t nt=0;
    for(auto const& x:out) nt += size_t((x.len + IO_TILE_ELEMS - 1) / IO_TILE_ELEMS);
    tiled.reserve(nt);
    for(auto const& x:out){
        Code off=0;
        while(off<x.len){
            Code take=std::min<Code>(x.len-off,Code(IO_TILE_ELEMS));
            tiled.push_back({x.remote+off,x.local+off,take,x.owner,0});
            off+=take;
        }
    }
    return tiled;
}

static std::vector<int> window_candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
static void window_masks(int W,int hi,int lo,const std::vector<int>&fp,uint32_t group,uint32_t&mf,uint32_t&mo,uint32_t&bf,uint32_t&bo){mf=mo=bf=bo=0;for(size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=(q<lo-1)?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
struct WindowPlan{int p_hi=0,p_lo=0;std::vector<int>fixed_pos;size_t max_bytes=0;Code max_main=0,max_block=0;};
static WindowPlan plan_window(int W,int hi,int lo,size_t target,int maxbits=20){WindowPlan best;best.p_hi=hi;best.p_lo=lo;auto cand=window_candidates(W,hi,lo);int klim=std::min<int>(cand.size(),maxbits);for(int k=0;k<=klim;++k){std::vector<int>fp(cand.begin(),cand.begin()+k);uint64_t ng=1ull<<k;size_t mx=0;Code mm=0,md=0;for(uint64_t g=0;g<ng;++g){uint32_t mf,mo,bf,bo;window_masks(W,hi,lo,fp,(uint32_t)g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count);if(b>mx){mx=b;mm=ms.size;md=ds.size;}if(mx>target&&k<klim)break;}if(mx<=target||k==klim){best.fixed_pos=std::move(fp);best.max_bytes=mx;best.max_main=mm;best.max_block=md;return best;}}return best;}

__device__ __forceinline__ bool allowed(uint32_t fixed,uint32_t occ,int pos,MateValue v){if(!((fixed>>pos)&1u))return v!=X;bool o=(occ>>pos)&1u;return o?(v==R||v==L):(v==N);}
__device__ __forceinline__ MateID unrank_group(Code rank,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){MateID m=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}m|=MateID(L)<<(2*pos);++h;}return m;}
__device__ __forceinline__ Code rank_group(MateID m,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){Code rank=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;MateValue s=mget(m,pos);if(s>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(s>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}
__device__ __forceinline__ Code rank_full_dev(MateID m,int width){Code rank=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;MateValue s=mget(m,pos);if(s>N)rank+=D_FULL_DP[pos][h];if(s>R&&h>0)rank+=D_FULL_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}

template<int WIDTH> __device__ __forceinline__ Code rank_full_t(MateID m);

template<int WIDTH>
__device__ __forceinline__ bool high_lut_applicable(uint32_t fixed){
    if constexpr(HIGH_LUT_K==0)return false;
    else {
        constexpr int LOW=WIDTH-HIGH_LUT_K;
        constexpr uint32_t HM=((1u<<HIGH_LUT_K)-1u)<<LOW;
        constexpr uint32_t WM=(WIDTH==32)?0xffffffffu:((1u<<WIDTH)-1u);
        return LOW>=0 && (fixed&HM)==HM && (fixed&(WM^HM))==0;
    }
}

template<int WIDTH>
__device__ __forceinline__ int high_final_height(uint32_t code){
    int h=1;
#pragma unroll
    for(int p=HIGH_LUT_K-1;p>=0;--p){
        MateValue v=MateValue((code>>(2*p))&3u);
        if(v==R)--h; else if(v==L)++h;
    }
    return h;
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_free_suffix(Code rank,int width,int h){
    MateID m=0;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(pos>=width)continue;
        Code z=D_FULL_DP[pos][h];
        if(rank<z)continue;
        rank-=z;
        if(h>0){z=D_FULL_DP[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}

template<int WIDTH>
__device__ __forceinline__ Code rank_free_suffix(MateID m,int width,int h){
    Code rank=0;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(pos>=width)continue;
        MateValue v=mget(m,pos);
        if(v>N)rank+=D_FULL_DP[pos][h];
        if(v>R&&h>0)rank+=D_FULL_DP[pos][h-1];
        if(v==R)--h;else if(v==L)++h;
    }
    return rank;
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_high_lut(Code rank,uint32_t occ){
    constexpr int LOW=WIDTH-HIGH_LUT_K;
    constexpr uint32_t OM=(1u<<HIGH_LUT_K)-1u;
    uint32_t mask=(occ>>LOW)&OM;
    HighEntry* entries = WIDTH==TARGET_W ? D_HIGH_ENTRIES_MAIN : D_HIGH_ENTRIES_BLOCK;
    uint32_t* offsets = WIDTH==TARGET_W ? D_HIGH_OFFSETS_MAIN : D_HIGH_OFFSETS_BLOCK;
    uint32_t a=offsets[mask], b=offsets[mask+1];
    // Find the last prefix whose base is <= rank.
    uint32_t lo=a,hi=b;
    while(lo+1<hi){uint32_t mid=(lo+hi)>>1;if(Code(entries[mid].base)<=rank)lo=mid;else hi=mid;}
    HighEntry e=entries[lo];
    int h=high_final_height<WIDTH>(e.code);
    MateID low=unrank_free_suffix<WIDTH>(rank-Code(e.base),LOW,h);
    return (MateID(e.code)<<(2*LOW))|low;
}

template<int WIDTH>
__device__ __forceinline__ Code rank_high_lut(MateID m,uint32_t occ){
    constexpr int LOW=WIDTH-HIGH_LUT_K;
    constexpr MateID CM=(MateID(1)<<(2*HIGH_LUT_K))-1;
    constexpr uint32_t OM=(1u<<HIGH_LUT_K)-1u;
    uint32_t code=(uint32_t)((m>>(2*LOW))&CM);
    uint32_t mask=(occ>>LOW)&OM;
    HighEntry* entries = WIDTH==TARGET_W ? D_HIGH_ENTRIES_MAIN : D_HIGH_ENTRIES_BLOCK;
    uint32_t* offsets = WIDTH==TARGET_W ? D_HIGH_OFFSETS_MAIN : D_HIGH_OFFSETS_BLOCK;
    uint32_t lo=offsets[mask],hi=offsets[mask+1];
    while(lo<hi){uint32_t mid=(lo+hi)>>1;if(entries[mid].code<code)lo=mid+1;else hi=mid;}
    HighEntry e=entries[lo];
    int h=high_final_height<WIDTH>(code);
    constexpr MateID LM=(MateID(1)<<(2*LOW))-1;
    return Code(e.base)+rank_free_suffix<WIDTH>(m&LM,LOW,h);
}
template<int WIDTH>
__device__ __forceinline__ bool low_lut_applicable(uint32_t fixed){
    if constexpr(LOW_LUT_K==0)return false;
    else return (fixed&((1u<<LOW_LUT_K)-1u))==((1u<<LOW_LUT_K)-1u);
}
template<int WIDTH>
__device__ __forceinline__ MateID unrank_group_t(Code rank,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    if constexpr(HIGH_LUT_K>0){if(high_lut_applicable<WIDTH>(fixed))return unrank_high_lut<WIDTH>(rank,occ);}
    MateID m=0;int h=1;
    if constexpr(LOW_LUT_K>0){if(low_lut_applicable<WIDTH>(fixed)){
#pragma unroll
        for(int pos=WIDTH-1;pos>=LOW_LUT_K;--pos){
            if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
            if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
            m|=MateID(L)<<(2*pos);++h;
        }
        uint32_t mask=occ&((1u<<LOW_LUT_K)-1u);uint32_t base=D_LOW_OFFSETS[size_t(mask)*(LOW_LUT_K+2)+h];m|=D_LOW_ENTRIES[base+(uint32_t)rank].mate;return m;
    }}
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
        if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_group_t(MateID m,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    if constexpr(HIGH_LUT_K>0){if(high_lut_applicable<WIDTH>(fixed))return rank_high_lut<WIDTH>(m,occ);}
    Code rank=0;int h=1;
    if constexpr(LOW_LUT_K>0){if(low_lut_applicable<WIDTH>(fixed)){
#pragma unroll
        for(int pos=WIDTH-1;pos>=LOW_LUT_K;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
        constexpr MateID CM=(MateID(1)<<(2*LOW_LUT_K))-1;uint32_t lr=D_LOW_LOCAL_RANK[(uint32_t)(m&CM)];return rank+lr;
    }}
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
template<int WIDTH>
__device__ __forceinline__ int height_before_rank_pos(MateID m,int hi){
    constexpr MateID EVEN=0x5555555555555555ULL;
    constexpr MateID WM=(MateID(1)<<(2*WIDTH))-1ULL;
    MateID lower=(MateID(1)<<(2*(hi+1)))-1ULL;
    MateID pm=EVEN&WM&~lower;
    int nr=__popcll(m&pm), nl=__popcll((m>>1)&pm);
    return 1+nl-nr;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_slice_t(MateID m,int hi,int lo,int h,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    Code rank=0;
    for(int pos=hi;pos>=lo;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_same_t(Code src_rank,MateID src,MateID dst,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID diff=src^dst;
    int hi=(63-__clzll(diff))/2;
    int lo=(__ffsll((long long)diff)-1)/2;
    int h=height_before_rank_pos<WIDTH>(src,hi);
    Code a=rank_slice_t<WIDTH>(src,hi,lo,h,fixed,occ,dp);
    Code b=rank_slice_t<WIDTH>(dst,hi,lo,h,fixed,occ,dp);
    return b>=a?src_rank+(b-a):src_rank-(a-b);
}
template<int WIDTH>
__device__ __forceinline__ Code rank_known_t(Code src_rank,MateID src,MateID dst,int hi,int lo,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    int h=height_before_rank_pos<WIDTH>(src,hi);
    Code a=rank_slice_t<WIDTH>(src,hi,lo,h,fixed,occ,dp);
    Code b=rank_slice_t<WIDTH>(dst,hi,lo,h,fixed,occ,dp);
    return b>=a?src_rank+(b-a):src_rank-(a-b);
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_group_global_t(Code rank,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2],Code& global_rank){
    if constexpr(LOW_LUT_K>0){if(low_lut_applicable<WIDTH>(fixed)){
        MateID m=0;int h=1;global_rank=0;
#pragma unroll
        for(int pos=WIDTH-1;pos>=LOW_LUT_K;--pos){
            bool chose=false;
            if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z){chose=true;}else rank-=z;}
            if(chose)continue;
            if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){global_rank+=D_FULL_DP[pos][h];m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
            global_rank+=D_FULL_DP[pos][h]+(h>0?D_FULL_DP[pos][h-1]:0);m|=MateID(L)<<(2*pos);++h;
        }
        uint32_t mask=occ&((1u<<LOW_LUT_K)-1u);uint32_t base=D_LOW_OFFSETS[size_t(mask)*(LOW_LUT_K+2)+h];LowEntry e=D_LOW_ENTRIES[base+(uint32_t)rank];m|=e.mate;global_rank+=e.full_rank;return m;
    }}
    MateID m=unrank_group_t<WIDTH>(rank,fixed,occ,dp);global_rank=rank_full_t<WIDTH>(m);return m;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_full_t(MateID m){
    Code rank=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){MateValue v=mget(m,pos);if(v>N)rank+=D_FULL_DP[pos][h];if(v>R&&h>0)rank+=D_FULL_DP[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
__device__ __forceinline__ Count global_load_main(Code g){if(D_NGPU==1)return D_MAIN_PTR[0][g];int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){if(D_NGPU==1)return D_BLOCK_PTR[0][g];int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){if(D_NGPU==1){D_MAIN_PTR[0][g]=v;return;}int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){if(D_NGPU==1){D_BLOCK_PTR[0][g]=v;return;}int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}

__device__ __forceinline__ MateID bounded_unrank(Code rank,int width,const Code dp[MAXW+1][MAXW+2],int cap){
    MateID m=0;int h=1;
    for(int pos=width-1;pos>=0;--pos){Code z=dp[pos][h];if(rank<z)continue;rank-=z;if(h>0){z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}if(h<cap){m|=MateID(L)<<(2*pos);++h;}}
    return m;
}
__device__ __forceinline__ Code bounded_rank(MateID m,int width,const Code dp[MAXW+1][MAXW+2],int cap){
    Code rank=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto v=mget(m,pos);if(v>N)rank+=dp[pos][h];if(v>R&&h>0)rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}return rank;
}
__global__ void bounded_fill_one_kernel(Count* a,Code n){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st)a[i]=1;}
__global__ void bounded_fill_row2_automaton_kernel(Count* out,Code n,int width){
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank; int h=1;
        long long v0=1,v1=0,v2=0,v3=0,v4=0,v5=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){
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
__device__ __forceinline__ Count r3_add_mod(Count a,Count b){Count m=D_MOD;return a>=m-b?a-(m-b):a+b;}
__device__ __forceinline__ Count r3_mul_mod(Count a,Count b){
    unsigned long long z=(unsigned long long)a*b,m=D_MOD,d=(1ULL<<32)-m;
    if(d<65536ULL){unsigned long long t=(uint32_t)z+(z>>32)*d;t=(uint32_t)t+(t>>32)*d;if(t>=m)t-=m;if(t>=m)t-=m;return (Count)t;}
    return (Count)(z%m);
}
__global__ void bounded_fill_row3_lut_kernel(Count*out,Code n,int width){
    constexpr int DIM=oneesan::row3auto::DIM; constexpr int LO=TARGET_W/2;
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank;int h=1;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;--h;}else{r-=a;dig=2;++h;}}else{dig=2;++h;}}if(pos>=LO)ch=ch*3u+dig;else cl=cl*3u+dig;}
        const Count*pv=D_R3_PREF+(size_t)ch*DIM;const Count*sv=D_R3_SUFF+(size_t)cl*DIM;Count acc=0;
#pragma unroll
        for(int i=0;i<DIM;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        out[brank]=r3_mul_mod(acc,D_R3_INV_SCALE);
    }
}

__device__ __forceinline__ int r4_block_size(int h){return h==0?19:h==1?20:h==2?12:h==3?4:h==4?1:0;}
__global__ void bounded_fill_row4_lut_kernel(Count*out,Code n,int width){
    constexpr int STRIDE=20,LO=TARGET_W/2;
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;--h;}else{r-=a;dig=2;++h;}}else{dig=2;++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
        const Count*pv=D_R4_PREF+(size_t)ch*STRIDE;const Count*sv=D_R4_SUFF+(size_t)cl*STRIDE;Count acc=0;int dim=r4_block_size(hs);
#pragma unroll 4
        for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));out[brank]=acc;
    }
}

__device__ __forceinline__ int r5_block_size(int h){return h==0?51:h==1?61:h==2?40:h==3?18:h==4?5:h==5?1:0;}
__global__ void bounded_fill_row5_full_kernel(Count*fullMain,Code n,int width){
    constexpr int STRIDE=61,LO=TARGET_W/2;
    for(Code brank=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<n;brank+=st){
        Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){
            int dig=0;Code a=D_BOUND_DP[pos][h];
            if(r<a)dig=0;
            else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}
            if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;
        }
        const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;int dim=r5_block_size(hs);
#pragma unroll 4
        for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        fullMain[grank]=acc;
    }
}

__global__ void bounded_fill_row5_full_warp_kernel(Count*fullMain,Code n,int width){
    constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&31;Code warp=(Code(blockIdx.x)*blockDim.x+threadIdx.x)>>5,step=(Code(gridDim.x)*blockDim.x)>>5;
    for(Code brank=warp;brank<n;brank+=step){
        unsigned long long meta=0;Code grank=0;
        if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
            for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(0xffffffffu,meta,0);grank=__shfl_sync(0xffffffffu,grank,0);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=32)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        for(int off=16;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(0xffffffffu,acc,off));if(lane==0)fullMain[grank]=acc;
    }
}

__global__ void bounded_fill_row5_full_lane8_kernel(Count*fullMain,Code n,int width){
    constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&7,warpLane=threadIdx.x&31,leader=warpLane&~7;unsigned mask=0xffu<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)>>3,step=(Code(gridDim.x)*blockDim.x)>>3;
    for(Code brank=group;brank<n;brank+=step){
        unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
            for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=8)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));
        for(int off=4;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,8));if(lane==0)fullMain[grank]=acc;
    }
}

template<int LANES> __global__ void bounded_fill_row5_full_group_kernel(Count*fullMain,Code n,int width){
    static_assert(LANES==4||LANES==8||LANES==16);constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=group;brank<n;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank]=acc;
    }
}


__global__ void bounded_fill_row5_full_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    constexpr int STRIDE=61,LO=TARGET_W/2;
    for(Code brank=blo+Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<bhi;brank+=st){Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
        const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0,dim=r5_block_size(hs);for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));fullMain[grank-fullBase]=acc;}
}
template<int LANES> __global__ void bounded_fill_row5_full_group_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    static_assert(LANES==4||LANES==8||LANES==16);constexpr int STRIDE=61,LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=blo+group;brank<bhi;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r5_block_size(hs);const Count*pv=D_R5_PREF+D_R5_PREF_IDX[ch];const Count*sv=D_R5_SUFF+D_R5_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank-fullBase]=acc;}
}


__host__ __device__ __forceinline__ int r6_block_size(int h){return h==0?141:h==1?182:h==2?135:h==3?68:h==4?25:h==5?6:h==6?1:0;}
__device__ __forceinline__ int r6_gpu_delta(int sym){return sym==0?0:(sym==1?-1:1);}
__global__ void r6_prefix_level_kernel(Code parents,const Count*cur,const int8_t*hp,Count*nxt,int8_t*hn,const Count*mat){
    Code children=parents*3;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<children;q+=st){Code p=q/3;int sym=int(q-p*3),h=hp[p];if(h<0){hn[q]=-1;continue;}int h2=h+r6_gpu_delta(sym);if(h2<0||h2>6){hn[q]=-1;continue;}hn[q]=int8_t(h2);int ns=r6_block_size(h),nd=r6_block_size(h2);const Count*M=mat+D_R6_GPU_MAT_OFF[sym][h];for(int d=0;d<nd;++d){Count acc=0;for(int a=0;a<ns;++a){Count u=cur[Code(a)*parents+p],c=M[a*nd+d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(u,c));}nxt[Code(d)*children+q]=acc;}}
}
__global__ void r6_suffix_level_kernel(Code parents,const Count*cur,const int8_t*hp,Count*nxt,int8_t*hn,const Count*mat){
    Code children=parents*3;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<children;q+=st){int sym=int(q/parents);Code p=q-Code(sym)*parents;int h=hp[p];if(h<0){hn[q]=-1;continue;}int hs=h-r6_gpu_delta(sym);if(hs<0||hs>6){hn[q]=-1;continue;}hn[q]=int8_t(hs);int ns=r6_block_size(hs),nd=r6_block_size(h);const Count*M=mat+D_R6_GPU_MAT_OFF[sym][hs];for(int a=0;a<ns;++a){Count acc=0;const Count*row=M+a*nd;for(int d=0;d<nd;++d){Count u=cur[Code(d)*parents+p],c=row[d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(c,u));}nxt[Code(a)*children+q]=acc;}}
}
__global__ void r6_pack_level_kernel(Code codes,const Count*dense,const int8_t*height,const uint32_t*idx,Count*packed){for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<codes;q+=st){int h=height[q];if(h<0)continue;uint32_t o=idx[q];int n=r6_block_size(h);for(int j=0;j<n;++j)packed[Code(o)+j]=dense[Code(j)*codes+q];}}
__global__ void r6_prefix_level_wide_kernel(Code parents,const Count*cur,const int8_t*hp,Count*nxt,int8_t*hn,const Count*mat){
    Code children=parents*3;for(Code q=blockIdx.x;q<children;q+=gridDim.x){Code p=q/3;int sym=int(q-p*3),h=hp[p];if(threadIdx.x==0){if(h<0)hn[q]=-1;else{int h2=h+r6_gpu_delta(sym);hn[q]=(h2>=0&&h2<=6)?int8_t(h2):int8_t(-1);}}if(h<0)continue;int h2=h+r6_gpu_delta(sym);if(h2<0||h2>6)continue;int ns=r6_block_size(h),nd=r6_block_size(h2);const Count*M=mat+D_R6_GPU_MAT_OFF[sym][h];for(int d=threadIdx.x;d<nd;d+=blockDim.x){Count acc=0;for(int a=0;a<ns;++a){Count u=cur[Code(a)*parents+p],c=M[a*nd+d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(u,c));}nxt[Code(d)*children+q]=acc;}}
}
__global__ void r6_suffix_level_wide_kernel(Code parents,const Count*cur,const int8_t*hp,Count*nxt,int8_t*hn,const Count*mat){
    Code children=parents*3;for(Code q=blockIdx.x;q<children;q+=gridDim.x){int sym=int(q/parents);Code p=q-Code(sym)*parents;int h=hp[p];if(threadIdx.x==0){if(h<0)hn[q]=-1;else{int hs=h-r6_gpu_delta(sym);hn[q]=(hs>=0&&hs<=6)?int8_t(hs):int8_t(-1);}}if(h<0)continue;int hs=h-r6_gpu_delta(sym);if(hs<0||hs>6)continue;int ns=r6_block_size(hs),nd=r6_block_size(h);for(int a=threadIdx.x;a<ns;a+=blockDim.x){const Count*row=mat+D_R6_GPU_MAT_OFF[sym][hs]+a*nd;Count acc=0;for(int d=0;d<nd;++d){Count u=cur[Code(d)*parents+p],c=row[d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(c,u));}nxt[Code(a)*children+q]=acc;}}
}
__global__ void r6_prefix_level_packed_kernel(Code parents,const Count*cur,const int8_t*hp,const uint32_t*pidx,Count*nxt,const int8_t*hn,const uint32_t*nidx,const Count*mat){
    Code children=parents*3;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<children;q+=st){Code p=q/3;int h=hp[p],h2=hn[q];if(h<0||h2<0)continue;int ns=r6_block_size(h),nd=r6_block_size(h2);const Count*in=cur+pidx[p];Count*out=nxt+nidx[q];const Count*M=mat+D_R6_GPU_MAT_OFF[int(q-p*3)][h];for(int d=0;d<nd;++d){Count acc=0;for(int a=0;a<ns;++a){Count u=in[a],c=M[a*nd+d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(u,c));}out[d]=acc;}}
}
__global__ void r6_suffix_level_packed_kernel(Code parents,const Count*cur,const int8_t*hp,const uint32_t*pidx,Count*nxt,const int8_t*hn,const uint32_t*nidx,const Count*mat){
    Code children=parents*3;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<children;q+=st){int sym=int(q/parents);Code p=q-Code(sym)*parents;int h=hp[p],hs=hn[q];if(h<0||hs<0)continue;int ns=r6_block_size(hs),nd=r6_block_size(h);const Count*in=cur+pidx[p];Count*out=nxt+nidx[q];const Count*M=mat+D_R6_GPU_MAT_OFF[sym][hs];for(int a=0;a<ns;++a){Count acc=0;const Count*row=M+a*nd;for(int d=0;d<nd;++d){Count u=in[d],c=row[d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(c,u));}out[a]=acc;}}
}
__global__ void r6_prefix_level_packed_wide_kernel(Code parents,const Count*cur,const int8_t*hp,const uint32_t*pidx,Count*nxt,const int8_t*hn,const uint32_t*nidx,const Count*mat){
    Code children=parents*3;for(Code q=blockIdx.x;q<children;q+=gridDim.x){Code p=q/3;int h=hp[p],h2=hn[q];if(h<0||h2<0)continue;int ns=r6_block_size(h),nd=r6_block_size(h2);const Count*in=cur+pidx[p];Count*out=nxt+nidx[q];const Count*M=mat+D_R6_GPU_MAT_OFF[int(q-p*3)][h];for(int d=threadIdx.x;d<nd;d+=blockDim.x){Count acc=0;for(int a=0;a<ns;++a){Count u=in[a],c=M[a*nd+d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(u,c));}out[d]=acc;}}
}
__global__ void r6_suffix_level_packed_wide_kernel(Code parents,const Count*cur,const int8_t*hp,const uint32_t*pidx,Count*nxt,const int8_t*hn,const uint32_t*nidx,const Count*mat){
    Code children=parents*3;for(Code q=blockIdx.x;q<children;q+=gridDim.x){int sym=int(q/parents);Code p=q-Code(sym)*parents;int h=hp[p],hs=hn[q];if(h<0||hs<0)continue;int ns=r6_block_size(hs),nd=r6_block_size(h);const Count*in=cur+pidx[p];Count*out=nxt+nidx[q];const Count*M=mat+D_R6_GPU_MAT_OFF[sym][hs];for(int a=threadIdx.x;a<ns;a+=blockDim.x){Count acc=0;const Count*row=M+a*nd;for(int d=0;d<nd;++d){Count u=in[d],c=row[d];if(u&&c)acc=r3_add_mod(acc,r3_mul_mod(c,u));}out[a]=acc;}}
}
template<int LANES> __global__ void bounded_fill_row6_dense_group_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase,const Count*pref,Code pn,const Count*suff,Code sn){
    static_assert(LANES==2||LANES==4||LANES==8||LANES==16);constexpr int LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=blo+group;brank<bhi;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r6_block_size(hs);Count acc=0;for(int j=lane;j<dim;j+=LANES){Count a=pref[Code(j)*pn+ch],b=suff[Code(j)*sn+cl];if(a&&b)acc=r3_add_mod(acc,r3_mul_mod(a,b));}for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank-fullBase]=acc;}
}
__global__ void bounded_fill_row6_dense_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase,const Count*pref,Code pn,const Count*suff,Code sn){constexpr int LO=TARGET_W/2;for(Code brank=blo+Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<bhi;brank+=st){Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}Count acc=0,dim=r6_block_size(hs);for(int j=0;j<dim;++j){Count a=pref[Code(j)*pn+ch],b=suff[Code(j)*sn+cl];if(a&&b)acc=r3_add_mod(acc,r3_mul_mod(a,b));}fullMain[grank-fullBase]=acc;}}
__global__ void bounded_fill_row6_tiled4_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    constexpr int BT=256,LANES=4,GROUPS=BT/LANES,LO=TARGET_W/2;
    __shared__ unsigned long long SMETA[BT];
    __shared__ Code SGRANK[BT];
    int tid=threadIdx.x;
    if(blockDim.x!=BT)return;
    for(Code base=blo+Code(blockIdx.x)*BT,step=Code(gridDim.x)*BT;base<bhi;base+=step){
        Code brank=base+tid,grank=0;unsigned long long meta=0;
        if(brank<bhi){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
            for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
            meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);
        }
        SMETA[tid]=meta;SGRANK[tid]=grank;__syncthreads();
        int lane=tid&(LANES-1),group=tid/LANES,warpLane=tid&31,leader=warpLane&~(LANES-1);unsigned mask=0xfu<<leader;
#pragma unroll
        for(int wave=0;wave<LANES;++wave){int idx=wave*GROUPS+group;if(base+idx<bhi){auto m=SMETA[idx];Code g=SGRANK[idx];uint32_t ch=uint32_t(m&0xffffffu),cl=uint32_t((m>>24)&0xffffffu);int hs=int((m>>48)&0xffu),dim=r6_block_size(hs);const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[g-fullBase]=acc;}}
        __syncthreads();
    }
}
struct Row6CartTask{uint32_t ip,slo,shi;};
template<int LANES> __global__ void row6_cartesian_task_kernel(Count*fullMain,Code fullBase,const uint32_t*pc,const Code*pr,const uint32_t*sc,const Code*sr,const Row6CartTask*tasks,Code ntasks,int dim){
    static_assert(LANES==1||LANES==2||LANES==4||LANES==8||LANES==16);Code ti=blockIdx.x;if(ti>=ntasks)return;Row6CartTask t=tasks[ti];uint32_t ch=pc[t.ip];Code rp=pr[t.ip];int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1),group=threadIdx.x/LANES,groups=blockDim.x/LANES;unsigned mask=((1u<<LANES)-1u)<<leader;const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];for(uint32_t is=t.slo+group;is<t.shi;is+=groups){uint32_t cl=0;Code gr=0;if(lane==0){cl=sc[is];gr=rp+sr[is];}cl=__shfl_sync(mask,cl,leader);gr=__shfl_sync(mask,gr,leader);const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[gr-fullBase]=acc;}
}
template<int LANES> __global__ void row6_cartesian_task_factor_kernel(Count*fullMain,Code fullBase,const uint32_t*pc,const Code*fp,const uint32_t*sc,const Code*sr,const Row6CartTask*tasks,Code ntasks,int dim){
    static_assert(LANES==1||LANES==2||LANES==4||LANES==8||LANES==16);Code ti=blockIdx.x;if(ti>=ntasks)return;Row6CartTask t=tasks[ti];uint32_t ch=pc[t.ip];Code rb=fp[t.ip];int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1),group=threadIdx.x/LANES,groups=blockDim.x/LANES;unsigned mask=((1u<<LANES)-1u)<<leader;const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];for(uint32_t is=t.slo+group;is<t.shi;is+=groups){uint32_t cl=0;Code gr=0;if(lane==0){cl=sc[is];gr=rb+sr[is];}cl=__shfl_sync(mask,cl,leader);gr=__shfl_sync(mask,gr,leader);const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[gr-fullBase]=acc;}
}

template<int TILE=8,int LANES=4> __global__ void row6_cartesian_gemm_kernel(
    Count*fullMain,Code fullBase,Code fullEnd,
    const uint32_t*pc,const Code*pr,Code np,
    const uint32_t*sc,const Code*sr,Code ns,int dim){
    static_assert(TILE*TILE*LANES==256);
    __shared__ Count AP[TILE][182];
    __shared__ Count BS[TILE][182];
    int tid=threadIdx.x;
    Code ip0=Code(blockIdx.x)*TILE,is0=Code(blockIdx.y)*TILE;
    for(int q=tid;q<TILE*dim;q+=256){int i=q/dim,k=q-i*dim;Code ip=ip0+i;if(ip<np){uint32_t c=pc[ip];AP[i][k]=D_R6_PREF[D_R6_PREF_IDX[c]+k];}else AP[i][k]=0;}
    for(int q=tid;q<TILE*dim;q+=256){int j=q/dim,k=q-j*dim;Code is=is0+j;if(is<ns){uint32_t c=sc[is];BS[j][k]=D_R6_SUFF[D_R6_SUFF_IDX[c]+k];}else BS[j][k]=0;}
    __syncthreads();
    int lane=tid&(LANES-1),g=tid/LANES,i=g/TILE,j=g-i*TILE;
    Code ip=ip0+i,is=is0+j;
    if(ip>=np||is>=ns)return;
    Count acc=0;
    for(int k=lane;k<dim;k+=LANES)acc=r3_add_mod(acc,r3_mul_mod(AP[i][k],BS[j][k]));
    unsigned mask=0xfu<<((threadIdx.x&31)&~3);
    for(int off=2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,4));
    if(lane==0){Code gr=pr[ip]+sr[is];if(gr>=fullBase&&gr<fullEnd)fullMain[gr-fullBase]=acc;}
}
template<int LANES> __global__ void row6_cartesian_group_kernel(Count*fullMain,Code fullBase,const uint32_t*pc,const Code*pr,Code np,const uint32_t*sc,const Code*sr,Code ns,int dim){
    static_assert(LANES==2||LANES==4||LANES==8||LANES==16);int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code total=np*ns,group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code q=group;q<total;q+=step){uint32_t ch=0,cl=0;Code gr=0;if(lane==0){Code ip=q/ns,is=q-ip*ns;ch=pc[ip];cl=sc[is];gr=pr[ip]+sr[is];}ch=__shfl_sync(mask,ch,leader);cl=__shfl_sync(mask,cl,leader);gr=__shfl_sync(mask,gr,leader);const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[gr-fullBase]=acc;}
}
__global__ void bounded_fill_row6_full_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    constexpr int LO=TARGET_W/2;
    for(Code brank=blo+Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;brank<bhi;brank+=st){Code r=brank,grank=0;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}
        const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;int dim=r6_block_size(hs);for(int i=0;i<dim;++i)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));fullMain[grank-fullBase]=acc;}
}
template<int LANES> __global__ void bounded_fill_row6_full_group_range_kernel(Count*fullMain,Code blo,Code bhi,Code fullBase){
    static_assert(LANES==2||LANES==4||LANES==8||LANES==16);constexpr int LO=TARGET_W/2;int lane=threadIdx.x&(LANES-1),warpLane=threadIdx.x&31,leader=warpLane&~(LANES-1);unsigned mask=((1u<<LANES)-1u)<<leader;Code group=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/LANES,step=(Code(gridDim.x)*blockDim.x)/LANES;
    for(Code brank=blo+group;brank<bhi;brank+=step){unsigned long long meta=0;Code grank=0;if(lane==0){Code r=brank;int h=1,hs=0;uint32_t ch=0,cl=0;
#pragma unroll
        for(int pos=TARGET_W-1;pos>=0;--pos){int dig=0;Code a=D_BOUND_DP[pos][h];if(r<a)dig=0;else{r-=a;if(h>0){a=D_BOUND_DP[pos][h-1];if(r<a){dig=1;grank+=D_FULL_DP[pos][h];--h;}else{r-=a;dig=2;grank+=D_FULL_DP[pos][h]+D_FULL_DP[pos][h-1];++h;}}else{dig=2;grank+=D_FULL_DP[pos][h];++h;}}if(pos>=LO){ch=ch*3u+dig;if(pos==LO)hs=h;}else cl=cl*3u+dig;}meta=uint64_t(ch)|(uint64_t(cl)<<24)|(uint64_t(hs)<<48);}
        meta=__shfl_sync(mask,meta,leader);grank=__shfl_sync(mask,grank,leader);uint32_t ch=uint32_t(meta&0xffffffu),cl=uint32_t((meta>>24)&0xffffffu);int hs=int((meta>>48)&0xffu),dim=r6_block_size(hs);const Count*pv=D_R6_PREF+D_R6_PREF_IDX[ch];const Count*sv=D_R6_SUFF+D_R6_SUFF_IDX[cl];Count acc=0;for(int i=lane;i<dim;i+=LANES)acc=r3_add_mod(acc,r3_mul_mod(pv[i],sv[i]));for(int off=LANES/2;off;off>>=1)acc=r3_add_mod(acc,__shfl_down_sync(mask,acc,off,LANES));if(lane==0)fullMain[grank-fullBase]=acc;}
}

__global__ void bounded_embed_kernel(const Count*old,Code oldN,Count*neu,int width){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<oldN;i+=st){Count c=old[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_OLD_DP,D_BOUND_OLD_CAP);Code j=bounded_rank(m,width,D_BOUND_DP,D_BOUND_CAP);neu[j]=c;}}
__global__ void bounded_scatter_full_kernel(const Count*in,Code n,int width){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP);global_store_main(rank_full_t<TARGET_W>(m),c);}}



__global__ void bounded_inplace_expand_kernel(const Count*tmp,Code lo,Code n,Count*fullMain,int width){
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<n;q+=st){Count c=tmp[q];if(!c)continue;Code br=lo+q;MateID m=bounded_unrank(br,width,D_BOUND_DP,D_BOUND_CAP);Code j=rank_full_t<TARGET_W>(m);fullMain[j]=c;}
}
static void bounded_expand_inplace_single_gpu(Count*fullMain,Code compactN,int W,int threads){
    const Code chunk=Code(1)<<22;Count*tmp=nullptr;ck(cudaMalloc(&tmp,size_t(chunk)*sizeof(Count)),"inplace tmp");auto t0=std::chrono::steady_clock::now();
    for(Code hi=compactN;hi;){Code lo=hi>chunk?hi-chunk:0,n=hi-lo;ck(cudaMemcpy(tmp,fullMain+lo,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"inplace copy");ck(cudaMemset(fullMain+lo,0,size_t(n)*sizeof(Count)),"inplace clear");int b=int(std::min<Code>(65535,(n+threads-1)/threads));bounded_inplace_expand_kernel<<<std::max(1,b),threads>>>(tmp,lo,n,fullMain,W);ck(cudaDeviceSynchronize(),"inplace expand");hi=lo;}
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();cudaFree(tmp);std::cerr<<"bounded inplace expand states="<<compactN<<" sec="<<sec<<"\n";
}

__device__ __forceinline__ int f_find_main(Code i){int lo=0,hi=D_F_MAIN_NBLOCKS;while(lo<hi){int m=(lo+hi)>>1;if(i<D_F_MAIN_BLOCKS[m].end)hi=m;else lo=m+1;}return lo;}
__device__ __forceinline__ int f_find_block(Code i){int lo=0,hi=D_F_BLOCK_NBLOCKS;while(lo<hi){int m=(lo+hi)>>1;if(i<D_F_BLOCK_BLOCKS[m].end)hi=m;else lo=m+1;}return lo;}
__device__ __forceinline__ MateID factor_unrank_main(Code i){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=MAXW+2;int b=f_find_main(i);FBlock x=D_F_MAIN_BLOCKS[b];Code r=i-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;uint32_t hc,lc;
    if(D_F_FIX_LOW){hc=D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he]+hr];lc=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];}
    else{hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];lc=D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs]+lr];}
    return MateID(lc)|(MateID(x.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
}
__device__ __forceinline__ MateID factor_unrank_block(Code i){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=MAXW+2;int b=f_find_block(i);FBlock x=D_F_BLOCK_BLOCKS[b];Code r=i-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;uint32_t hc,lc;
    if(D_F_FIX_LOW){hc=D_F_HIGH_ALL_CODES[D_F_HIGH_ALL_OFF[x.he]+hr];lc=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];}
    else{hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];lc=D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs]+lr];}
    return MateID(lc)|(MateID(hc)<<(2*L));
}
__device__ __forceinline__ int seg_end_height(uint32_t code,int len){constexpr uint64_t E=0x5555555555555555ULL;uint64_t m=(len==16?0xffffffffULL:((1ULL<<(2*len))-1));uint64_t e=E&m;int nr=__popcll(uint64_t(code)&e),nl=__popcll((uint64_t(code)>>1)&e);return 1+nl-nr;}
__device__ __forceinline__ Code factor_rank_main(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*(L+1)))&HM);int he=seg_end_height(hc,H);int cv=int(mget(m,L));int bid=3*he+cv;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lp=D_F_LOW_DENSE_PACKED_RANK[lc],hp=D_F_HIGH_PACKED_RANK[hc];uint32_t lr=D_F_FIX_LOW?(lp&((1u<<L)-1u)):(lp>>L);uint32_t hr=D_F_FIX_LOW?(hp>>H):(hp&((1u<<H)-1u));return x.off+Code(hr)*x.stride+lr;
}
__device__ __forceinline__ Code factor_rank_block(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*L))&HM);int h=seg_end_height(hc,H);FBlock x=D_F_BLOCK_BLOCKS[h];uint32_t lp=D_F_LOW_DENSE_PACKED_RANK[lc],hp=D_F_HIGH_PACKED_RANK[hc];uint32_t lr=D_F_FIX_LOW?(lp&((1u<<L)-1u)):(lp>>L);uint32_t hr=D_F_FIX_LOW?(hp>>H):(hp&((1u<<H)-1u));return x.off+Code(hr)*x.stride+lr;
}
__device__ __forceinline__ Code factor_global_rank_main(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*(L+1)))&HM);int he=seg_end_height(hc,H);int cv=int(mget(m,L));uint32_t hp=D_F_HIGH_PACKED_RANK[hc],lp=D_F_LOW_DENSE_PACKED_RANK[lc];uint32_t har=hp>>H,lar=lp>>L;FBlock x=D_F_FULL_MAIN_BLOCKS[3*he+cv];return x.off+Code(har)*x.stride+lar;
}
__device__ __forceinline__ Code factor_global_rank_block(MateID m){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;constexpr uint32_t LM=(1u<<(2*L))-1u,HM=(1u<<(2*H))-1u;uint32_t lc=uint32_t(m)&LM,hc=uint32_t((m>>(2*L))&HM);int he=seg_end_height(hc,H);uint32_t hp=D_F_HIGH_PACKED_RANK[hc],lp=D_F_LOW_DENSE_PACKED_RANK[lc];uint32_t har=hp>>H,lar=lp>>L;FBlock x=D_F_FULL_BLOCK_BLOCKS[he];return x.off+Code(har)*x.stride+lar;
}
__global__ void init_after_first_row_kernel(int shard,int ng){
    constexpr Code total=Code(1)<<(TARGET_W-1);
    Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    Code stride=Code(gridDim.x)*blockDim.x;
    for(Code mask=Code(shard)+tid*Code(ng);mask<total;mask+=stride*Code(ng)){
        MateID m=MateID(oneesan::gridfp::R)<<(2*(TARGET_W-1));
#pragma unroll
        for(int p=TARGET_W-1;p>=1;--p){
            if((mask>>(p-1))&1ULL){
                auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);
                m=z.mate;
            }
        }
        global_store_main(factor_global_rank_main(m),Count(1));
    }
}

__device__ __forceinline__ void atomic_add_mod(Count*p,Count v);
__global__ void sum_first_row_states_kernel(int shard,int ng,Count* out){
    extern __shared__ unsigned long long sh[];
    constexpr Code total=Code(1)<<(TARGET_W-1);
    Code tid=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    Code stride=Code(gridDim.x)*blockDim.x;
    unsigned long long acc=0;
    for(Code mask=Code(shard)+tid*Code(ng);mask<total;mask+=stride*Code(ng)){
        MateID m=MateID(oneesan::gridfp::R)<<(2*(TARGET_W-1));
#pragma unroll
        for(int p=TARGET_W-1;p>=1;--p){
            if((mask>>(p-1))&1ULL){auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);m=z.mate;}
        }
        acc += global_load_main(factor_global_rank_main(m));
    }
    sh[threadIdx.x]=acc; __syncthreads();
    for(unsigned s=blockDim.x/2;s;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0){Count v=Count(sh[0]%D_MOD);atomic_add_mod(out,v);}
}

__device__ __forceinline__ Code factor_global_rank_main_index(Code i){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=MAXW+2;int b=f_find_main(i);FBlock x=D_F_MAIN_BLOCKS[b];Code r=i-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;uint32_t har,lar;
    if(D_F_FIX_LOW){har=hr;uint32_t lc=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];lar=D_F_LOW_DENSE_PACKED_RANK[lc]>>L;}
    else{uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];har=D_F_HIGH_PACKED_RANK[hc]>>H;lar=lr;}
    FBlock gx=D_F_FULL_MAIN_BLOCKS[3*int(x.he)+int(x.c)];return gx.off+Code(har)*gx.stride+lar;
}
__device__ __forceinline__ Code factor_global_rank_block_index(Code i){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K,S=MAXW+2;int b=f_find_block(i);FBlock x=D_F_BLOCK_BLOCKS[b];Code r=i-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;uint32_t har,lar;
    if(D_F_FIX_LOW){har=hr;uint32_t lc=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];lar=D_F_LOW_DENSE_PACKED_RANK[lc]>>L;}
    else{uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];har=D_F_HIGH_PACKED_RANK[hc]>>H;lar=lr;}
    FBlock gx=D_F_FULL_BLOCK_BLOCKS[x.he];return gx.off+Code(har)*gx.stride+lar;
}
__global__ void gather_main_kernel(Count*out,MateID*mates,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g=factor_global_rank_main_index(i);out[i]=global_load_main(g);if(mates)mates[i]=factor_unrank_main(i);}}
__global__ void gather_block_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)out[i]=global_load_block(factor_global_rank_block_index(i));}
__global__ void scatter_main_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)global_store_main(factor_global_rank_main_index(i),in[i]);}
__global__ void scatter_block_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)global_store_block(factor_global_rank_block_index(i),in[i]);}

// Lower-window fast I/O for factorized authoritative layout.  Here HIGH is the
// masked half and LOW is the all-state half, so every masked HIGH rank owns one
// contiguous LOW-all row in both local scratch and authoritative storage.
template<bool BLOCK,bool SCATTER>
__global__ void factor_lower_row_io_kernel(Count*buf,Code nrows){
    constexpr int H=HIGH_LUT_K,S=MAXW+2;
    __shared__ Code slocal,sglobal,slen;
    for(Code row=Code(blockIdx.y);row<nrows;row+=Code(gridDim.y)){
        if(threadIdx.x==0){
            Code base=0;int bid=-1;uint32_t hr=0;FBlock x{};int nb=BLOCK?D_F_BLOCK_NBLOCKS:D_F_MAIN_NBLOCKS;
            for(int b=0;b<nb;++b){FBlock y=BLOCK?D_F_BLOCK_BLOCKS[b]:D_F_MAIN_BLOCKS[b];Code nr=y.stride?(y.end-y.off)/y.stride:0;if(row<base+nr){bid=b;hr=uint32_t(row-base);x=y;break;}base+=nr;}
            if(bid<0){slen=0;}else{uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];uint32_t har=D_F_HIGH_PACKED_RANK[hc]>>H;FBlock gx=BLOCK?D_F_FULL_BLOCK_BLOCKS[x.he]:D_F_FULL_MAIN_BLOCKS[3*int(x.he)+int(x.c)];slocal=x.off+Code(hr)*x.stride;sglobal=gx.off+Code(har)*gx.stride;slen=x.stride;}
        }
        __syncthreads();
        Code off=Code(blockIdx.x)*blockDim.x+threadIdx.x;if(off<slen){if constexpr(SCATTER){if constexpr(BLOCK)global_store_block(sglobal+off,buf[slocal+off]);else global_store_main(sglobal+off,buf[slocal+off]);}else{if constexpr(BLOCK)buf[slocal+off]=global_load_block(sglobal+off);else buf[slocal+off]=global_load_main(sglobal+off);}}
        __syncthreads();
    }
}




// Occupancy-major upper-window I/O. LOW occupancy is fixed, hence the masked
// LOW ranks form one contiguous authoritative slice inside every HIGH-all row.
template<bool BLOCK,bool SCATTER>
__global__ void factor_upper_flat_io_kernel(Count*buf,Code n){
    constexpr int L=LOW_LUT_K,S=MAXW+2;
    __shared__ Code sg; __shared__ int scontig;
    Code chunk0=Code(blockIdx.x)*blockDim.x,step=Code(gridDim.x)*blockDim.x;
    for(Code base=chunk0;base<n;base+=step){
        Code remain=n-base,count=remain<Code(blockDim.x)?remain:Code(blockDim.x);
        if(threadIdx.x==0){
            int bid=BLOCK?f_find_block(base):f_find_main(base);FBlock x=BLOCK?D_F_BLOCK_BLOCKS[bid]:D_F_MAIN_BLOCKS[bid];
            Code r=base-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;
            scontig=(x.stride&&Code(lr)+count<=x.stride);
            if(scontig){FBlock gx=BLOCK?D_F_FULL_BLOCK_BLOCKS[x.he]:D_F_FULL_MAIN_BLOCKS[3*int(x.he)+int(x.c)];uint32_t lar=D_F_LOW_OCC_BASE[size_t(D_F_MASK)*S+x.hs]+lr;sg=gx.off+Code(hr)*gx.stride+lar;}
        }
        __syncthreads();
        Code i=base+threadIdx.x;if(i<base+count){Code g=scontig?sg+threadIdx.x:(BLOCK?factor_global_rank_block_index(i):factor_global_rank_main_index(i));if constexpr(SCATTER){if constexpr(BLOCK)global_store_block(g,buf[i]);else global_store_main(g,buf[i]);}else{if constexpr(BLOCK)buf[i]=global_load_block(g);else buf[i]=global_load_main(g);}}
        __syncthreads();
    }
}

template<bool BLOCK,bool SCATTER>
__global__ void factor_lower_flat_io_kernel(Count*buf,Code n){
    constexpr int H=HIGH_LUT_K,S=MAXW+2;
    __shared__ Code sg;__shared__ int scontig;
    Code chunk0=Code(blockIdx.x)*blockDim.x,step=Code(gridDim.x)*blockDim.x;
    for(Code base=chunk0;base<n;base+=step){
        Code remain=n-base;Code count=remain<Code(blockDim.x)?remain:Code(blockDim.x);
        if(threadIdx.x==0){int bid=BLOCK?f_find_block(base):f_find_main(base);FBlock x=BLOCK?D_F_BLOCK_BLOCKS[bid]:D_F_MAIN_BLOCKS[bid];Code r=base-x.off;uint32_t hr=x.stride?uint32_t(r/x.stride):0;uint32_t lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;scontig=(x.stride&&Code(lr)+count<=x.stride);if(scontig){uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];uint32_t har=D_F_HIGH_PACKED_RANK[hc]>>H;FBlock gx=BLOCK?D_F_FULL_BLOCK_BLOCKS[x.he]:D_F_FULL_MAIN_BLOCKS[3*int(x.he)+int(x.c)];sg=gx.off+Code(har)*gx.stride+lr;}}
        __syncthreads();
        Code i=base+threadIdx.x;if(i<base+count){Code g=scontig?sg+threadIdx.x:(BLOCK?factor_global_rank_block_index(i):factor_global_rank_main_index(i));if constexpr(SCATTER){if constexpr(BLOCK)global_store_block(g,buf[i]);else global_store_main(g,buf[i]);}else{if constexpr(BLOCK)buf[i]=global_load_block(g);else buf[i]=global_load_main(g);}}
        __syncthreads();
    }
}

template<bool BLOCK,bool SCATTER>
__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv){
    for(size_t k=blockIdx.x;k<niv;k+=gridDim.x){
        PeerInterval x=iv[k];
        Count*peer=(BLOCK?D_BLOCK_PTR[x.owner]:D_MAIN_PTR[x.owner])+x.remote;
        for(Code off=threadIdx.x;off<x.len;off+=blockDim.x){
            if constexpr(SCATTER) peer[off]=buf[x.local+off];
            else buf[x.local+off]=peer[off];
        }
    }
}
static int interval_blocks(size_t niv,int){return int(std::min<size_t>(65535,std::max<size_t>(1,niv)));}

__device__ __forceinline__ void atomic_add_mod(Count*p,Count v){if(!v)return;Count mod=D_MOD;cuda::atomic_ref<Count,cuda::thread_scope_device> a(*p);Count old=a.load(cuda::memory_order_relaxed);for(;;){Count neu=(old>=mod-v)?old-(mod-v):old+v;if(a.compare_exchange_weak(old,neu,cuda::memory_order_relaxed,cuda::memory_order_relaxed))return;}}
__device__ __forceinline__ Count add_mod_plain(Count a,Count b);

__global__ void factor_main_closure_inplace_kernel(Count*mainv,Count*blockv,const MateID*mates,Code n,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        Count c=mainv[i];if(!c)continue;
        MateID m=mates?mates[i]:factor_unrank_main(i);
        auto w=oneesan::gridfp::mpair(m,p);
        if(w!=oneesan::gridfp::LL && w!=oneesan::gridfp::RR && w!=oneesan::gridfp::RL)continue;
        auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);
        if(!z.valid)continue;
        if(z.blocked)atomic_add_mod(blockv+factor_rank_block(z.mate),c);
        else atomic_add_mod(mainv+factor_rank_main(z.mate),c);
    }
}


struct SparseOrbitMap {
    Code end[64];
    uint32_t sel_off[64],sel_count[64];
    int nblocks;
};
static SparseOrbitMap make_sparse_orbit_map(const std::vector<FBlock>&fb,int p,bool closure){
    SparseOrbitMap m{};m.nblocks=int(fb.size());Code sum=0;
    for(int b=0;b<m.nblocks;++b){auto const&x=fb[b];SparseSel q=(closure?G_SPARSE.closure_sel:G_SPARSE.owner_sel)[sparse_key(p,x.hs,x.c)];
        Code hc=(x.stride&&x.end>x.off)?((x.end-x.off)/x.stride):0;sum+=hc*Code(q.count);m.end[b]=sum;m.sel_off[b]=q.off;m.sel_count[b]=q.count;}
    return m;
}
__device__ __forceinline__ MateID sparse_unrank_main_known(int bid,uint32_t hr,uint32_t lr){
    constexpr int L=LOW_LUT_K,S=MAXW+2;FBlock x=D_F_MAIN_BLOCKS[bid];
    uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];
    uint32_t lc=D_F_LOW_ALL_CODES[D_F_LOW_ALL_OFF[x.hs]+lr];
    return MateID(lc)|(MateID(x.c)<<(2*L))|(MateID(hc)<<(2*(L+1)));
}
__device__ __forceinline__ int sparse_find_block(Code q,const SparseOrbitMap&m){int lo=0,hi=m.nblocks;while(lo<hi){int md=(lo+hi)>>1;if(q<m.end[md])hi=md;else lo=md+1;}return lo;}
template<bool CLOSURE>
__global__ void factor_sparse_orbit_kernel(Count*mainv,Count*blockv,const MateID*mates,SparseOrbitMap map,int p){
    Code total=map.nblocks?map.end[map.nblocks-1]:0;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        int bid=sparse_find_block(q,map);Code base=bid?map.end[bid-1]:0,loc=q-base;uint32_t sc=map.sel_count[bid];
        uint32_t hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);uint32_t lr=(CLOSURE?D_SP_CLOSURE_LR:D_SP_OWNER_LR)[map.sel_off[bid]+k];
        FBlock x=D_F_MAIN_BLOCKS[bid];Code i=x.off+Code(hr)*x.stride+lr;Count c=mainv[i];
        MateID m=mates?mates[i]:sparse_unrank_main_known(bid,hr,lr);
        if constexpr(CLOSURE){
            if(!c)continue;auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);if(!z.valid)continue;
            if(z.blocked)atomic_add_mod(blockv+factor_rank_block(z.mate),c);else atomic_add_mod(mainv+factor_rank_main(z.mate),c);
        }else{
            auto w=oneesan::gridfp::mpair(m,p);MateID b=oneesan::gridfp::mshrink(m,p);Code dj=factor_rank_block(b);Count d=blockv[dj];
            if(w==oneesan::gridfp::NN){MateID t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::LR);Code j=factor_rank_main(t);mainv[j]=add_mod_plain(mainv[j],c);mainv[i]=add_mod_plain(c,d);blockv[dj]=0;}
            else {MateID t=oneesan::gridfp::msetpair(m,p,w==oneesan::gridfp::NR?oneesan::gridfp::RN:oneesan::gridfp::LN);Code j=factor_rank_main(t);Count cc=mainv[j];Count all=add_mod_plain(add_mod_plain(c,cc),d);if(p==1){mainv[i]=all;mainv[j]=add_mod_plain(c,cc);blockv[dj]=0;}else{mainv[i]=all;blockv[dj]=c;}}
        }
    }
}


struct PreRankOrbitMap {
    Code end[64];uint32_t sel_off[64],sel_count[64];int nblocks;
};
static PreRankOrbitMap make_prerank_orbit_map(const std::vector<FBlock>&fb,int p,int kind){
    PreRankOrbitMap m{};m.nblocks=int(fb.size());Code sum=0;
    for(int b=0;b<m.nblocks;++b){auto const&x=fb[b];auto key=sparse_key(p,x.hs,x.c);SparseSel q=kind==0?G_PRERANK.owner_sel[key]:(kind==1?G_PRERANK.closure_sel[key]:(kind==2?G_PRERANK.highrr_sel[key]:G_PRERANK.closure_inv_sel[key]));Code hc=(x.stride&&x.end>x.off)?((x.end-x.off)/x.stride):0;sum+=hc*Code(q.count);m.end[b]=sum;m.sel_off[b]=q.off;m.sel_count[b]=q.count;}
    return m;
}
__device__ __forceinline__ int prerank_find_block(Code q,const PreRankOrbitMap&m){int lo=0,hi=m.nblocks;while(lo<hi){int md=(lo+hi)>>1;if(q<m.end[md])hi=md;else lo=md+1;}return lo;}
__global__ void factor_prerank_owner_kernel(Count*mainv,Count*blockv,PreRankOrbitMap map,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;Code total=map.nblocks?map.end[map.nblocks-1]:0;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        int bid=prerank_find_block(q,map);Code base=bid?map.end[bid-1]:0,loc=q-base;uint32_t sc=map.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long r=D_PR_OWNER_REC[map.sel_off[bid]+k];uint32_t lr=uint32_t(r)&M,mlr=uint32_t(r>>B)&M,blr=uint32_t(r>>(2*B))&M,ty=uint32_t(r>>(3*B))&3u;FBlock x=D_F_MAIN_BLOCKS[bid],bx=D_F_BLOCK_BLOCKS[x.he],dx=x;if(p==LOW_LUT_K){int dc=(ty==1?int(R):int(::L));dx=D_F_MAIN_BLOCKS[3*x.he+dc];}Code i=x.off+Code(hr)*x.stride+lr,j=dx.off+Code(hr)*dx.stride+mlr,dj=bx.off+Code(hr)*bx.stride+blr;Count c=mainv[i],d=blockv[dj];
        if(ty==0){mainv[j]=add_mod_plain(mainv[j],c);mainv[i]=add_mod_plain(c,d);blockv[dj]=0;}
        else{Count cc=mainv[j],pair=add_mod_plain(c,cc);mainv[i]=add_mod_plain(pair,d);if(p==1){mainv[j]=pair;blockv[dj]=0;}else blockv[dj]=c;}
    }
}
__global__ void factor_prerank_owner_blockorder_kernel(Count*mainv,Count*blockv,Code n,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;Code lowTotal=D_F_LOW_ALL_OFF[MAXW+1];
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<n;q+=st){
        int bb=f_find_block(q);FBlock bx=D_F_BLOCK_BLOCKS[bb];Code loc=q-bx.off;uint32_t hr=uint32_t(loc/bx.stride),blr=uint32_t(loc-Code(hr)*bx.stride);
        unsigned long long r=D_PR_OWNER_BLOCK_REC[Code(p-1)*lowTotal+D_F_LOW_ALL_OFF[bx.he]+blr];uint32_t lr=uint32_t(r)&M,mlr=uint32_t(r>>B)&M,c=uint32_t(r>>(2*B))&3u,ty=uint32_t(r>>(2*B+2))&3u;
        FBlock x=D_F_MAIN_BLOCKS[3*int(bx.he)+int(c)],dx=x;if(p==LOW_LUT_K){int dc=(ty==1?int(R):int(::L));dx=D_F_MAIN_BLOCKS[3*int(bx.he)+dc];}
        Code i=x.off+Code(hr)*x.stride+lr,j=dx.off+Code(hr)*dx.stride+mlr;Count cv=mainv[i],d=blockv[q];
        if(ty==0){mainv[j]=add_mod_plain(mainv[j],cv);mainv[i]=add_mod_plain(cv,d);blockv[q]=0;}
        else{Count cc=mainv[j],pair=add_mod_plain(cv,cc);mainv[i]=add_mod_plain(pair,d);if(p==1){mainv[j]=pair;blockv[q]=0;}else blockv[q]=cv;}
    }
}
__global__ void factor_prerank_closure_kernel(Count*mainv,Count*blockv,PreRankOrbitMap map,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;Code total=map.nblocks?map.end[map.nblocks-1]:0;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        int bid=prerank_find_block(q,map);Code base=bid?map.end[bid-1]:0,loc=q-base;uint32_t sc=map.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long r=D_PR_CLOSURE_REC[map.sel_off[bid]+k];uint32_t lr=uint32_t(r)&M,dlr=uint32_t(r>>B)&M,dc=uint32_t(r>>(2*B))&3u;FBlock x=D_F_MAIN_BLOCKS[bid];Count c=mainv[x.off+Code(hr)*x.stride+lr];if(!c)continue;if(p==1){FBlock dx=D_F_MAIN_BLOCKS[3*x.he+int(dc)];atomic_add_mod(mainv+dx.off+Code(hr)*dx.stride+dlr,c);}else{FBlock bx=D_F_BLOCK_BLOCKS[x.he];atomic_add_mod(blockv+bx.off+Code(hr)*bx.stride+dlr,c);}
    }
}
__global__ void factor_prerank_highrr_kernel(Count*mainv,Count*blockv,PreRankOrbitMap map,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;constexpr int H=HIGH_LUT_K,S=MAXW+2;constexpr uint32_t HM=(1u<<H)-1u;
    Code total=map.nblocks?map.end[map.nblocks-1]:0;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        int bid=prerank_find_block(q,map);Code base=bid?map.end[bid-1]:0,loc=q-base;uint32_t sc=map.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long rec=D_PR_HIGHRR_REC[map.sel_off[bid]+k];uint32_t lr=uint32_t(rec)&M,dlr=uint32_t(rec>>B)&M;int depth=int((rec>>(2*B))&63u);FBlock x=D_F_MAIN_BLOCKS[bid];Code i=x.off+Code(hr)*x.stride+lr;Count c=mainv[i];if(!c)continue;
        uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];int match=-1,s=depth;
#pragma unroll
        for(int z=0;z<H;++z){auto v=MateValue((hc>>(2*z))&3u);if(v==L){if(--s==0){match=z;break;}}else if(v==R)++s;}
        if(match<0)continue;uint32_t nhc=(hc&~(3u<<(2*match)))|(uint32_t(R)<<(2*match));uint32_t hp=D_F_HIGH_PACKED_RANK[nhc],nhr=hp&HM;int nhe=int(x.he)-2;
        if(p==1){FBlock dx=D_F_MAIN_BLOCKS[3*nhe+int(x.c)];atomic_add_mod(mainv+dx.off+Code(nhr)*dx.stride+dlr,c);}
        else{FBlock bx=D_F_BLOCK_BLOCKS[nhe];atomic_add_mod(blockv+bx.off+Code(nhr)*bx.stride+dlr,c);}
    }
}

__global__ void factor_prerank_closure_inv_combined_kernel(Count*mainv,Count*blockv,PreRankOrbitMap invMap,PreRankOrbitMap highMap,int p){
    constexpr uint32_t IB=18,IM=(1u<<IB)-1u,P=36;constexpr unsigned long long PM=(1ULL<<P)-1ULL;constexpr uint32_t B=20,M=(1u<<B)-1u;constexpr int H=HIGH_LUT_K,S=MAXW+2;constexpr uint32_t HM=(1u<<H)-1u;
    Code ni=invMap.nblocks?invMap.end[invMap.nblocks-1]:0,nh=highMap.nblocks?highMap.end[highMap.nblocks-1]:0,total=ni+nh;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        if(q<ni){int bid=prerank_find_block(q,invMap);Code base=bid?invMap.end[bid-1]:0,loc=q-base;uint32_t sc=invMap.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long r=D_PR_CLOSURE_INV_REC[invMap.sel_off[bid]+k];uint32_t dlr=uint32_t(r)&IM;unsigned long long payload=(r>>IB)&PM;uint32_t cnt=uint32_t(r>>(IB+P))&7u,dc=uint32_t(r>>(IB+P+3))&3u;FBlock x=D_F_MAIN_BLOCKS[bid];Count acc=0;for(uint32_t j=0;j<cnt;++j){uint32_t lr=cnt<=2?uint32_t(payload>>(18*j))&IM:D_PR_CLOSURE_INV_SRC[uint32_t(payload)+j];Count c=mainv[x.off+Code(hr)*x.stride+lr];if(c)acc=add_mod_plain(acc,c);}if(!acc)continue;if(p==1){FBlock dx=D_F_MAIN_BLOCKS[3*x.he+int(dc)];atomic_add_mod(mainv+dx.off+Code(hr)*dx.stride+dlr,acc);}else{FBlock bx=D_F_BLOCK_BLOCKS[x.he];atomic_add_mod(blockv+bx.off+Code(hr)*bx.stride+dlr,acc);}}
        else{Code zq=q-ni;int bid=prerank_find_block(zq,highMap);Code base=bid?highMap.end[bid-1]:0,loc=zq-base;uint32_t sc=highMap.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long rec=D_PR_HIGHRR_REC[highMap.sel_off[bid]+k];uint32_t lr=uint32_t(rec)&M,dlr=uint32_t(rec>>B)&M;int depth=int((rec>>(2*B))&63u);FBlock x=D_F_MAIN_BLOCKS[bid];Code i=x.off+Code(hr)*x.stride+lr;Count c=mainv[i];if(!c)continue;uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr];int match=-1,s=depth;
#pragma unroll
            for(int z=0;z<H;++z){auto v=MateValue((hc>>(2*z))&3u);if(v==L){if(--s==0){match=z;break;}}else if(v==R)++s;}if(match<0)continue;uint32_t nhc=(hc&~(3u<<(2*match)))|(uint32_t(R)<<(2*match));uint32_t hp=D_F_HIGH_PACKED_RANK[nhc],nhr=hp&HM;int nhe=int(x.he)-2;if(p==1){FBlock dx=D_F_MAIN_BLOCKS[3*nhe+int(x.c)];atomic_add_mod(mainv+dx.off+Code(nhr)*dx.stride+dlr,c);}else{FBlock bx=D_F_BLOCK_BLOCKS[nhe];atomic_add_mod(blockv+bx.off+Code(nhr)*bx.stride+dlr,c);}}
    }
}


struct HighPreRankMap { Code end[64];uint32_t sel_off[64],sel_count[64];int nblocks; };
static HighPreRankMap make_high_prerank_map(const std::vector<FBlock>&fb,int p,int kind){HighPreRankMap m{};m.nblocks=int(fb.size());Code sum=0;for(int b=0;b<m.nblocks;++b){auto const&x=fb[b];auto key=high_pr_key(p,x.he,x.c);SparseSel q=kind==0?G_HPR.owner_sel[key]:(kind==1?G_HPR.closure_sel[key]:(kind==2?G_HPR.crossll_sel[key]:G_HPR.closure_inv_sel[key]));Code lc=x.stride;sum+=lc*Code(q.count);m.end[b]=sum;m.sel_off[b]=q.off;m.sel_count[b]=q.count;}return m;}
__device__ __forceinline__ int high_pr_find_block(Code q,const HighPreRankMap&m){int lo=0,hi=m.nblocks;while(lo<hi){int md=(lo+hi)>>1;if(q<m.end[md])hi=md;else lo=md+1;}return lo;}
__global__ void factor_high_prerank_owner_kernel(Count*mainv,Count*blockv,HighPreRankMap map){constexpr uint32_t B=20,M=(1u<<B)-1u;Code total=map.nblocks?map.end[map.nblocks-1]:0;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){int bid=high_pr_find_block(q,map);Code base=bid?map.end[bid-1]:0,loc=q-base;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lc=x.stride,k=uint32_t(loc/lc),lr=uint32_t(loc-Code(k)*lc);unsigned long long r=D_HP_OWNER_REC[map.sel_off[bid]+k];uint32_t hr=uint32_t(r)&M,mhr=uint32_t(r>>B)&M,bhr=uint32_t(r>>(2*B))&M,ty=uint32_t(r>>(3*B))&3u,tc=uint32_t(r>>(3*B+2))&3u;int td=(tc==uint32_t(::L)?1:tc==uint32_t(R)?-1:0),the=int(x.hs)-td;FBlock dx=D_F_MAIN_BLOCKS[3*the+int(tc)],bx=D_F_BLOCK_BLOCKS[x.hs];Code i=x.off+Code(hr)*x.stride+lr,j=dx.off+Code(mhr)*dx.stride+lr,dj=bx.off+Code(bhr)*bx.stride+lr;Count c=mainv[i],d=blockv[dj];if(ty==0){mainv[j]=add_mod_plain(mainv[j],c);mainv[i]=add_mod_plain(c,d);blockv[dj]=0;}else{Count cc=mainv[j];mainv[i]=add_mod_plain(add_mod_plain(c,cc),d);blockv[dj]=c;}}}
__global__ void factor_high_prerank_owner_blockorder_kernel(Count*mainv,Count*blockv,Code n,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;Code highTotal=D_F_HIGH_ALL_OFF[MAXW+1];
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<n;q+=st){
        int bb=f_find_block(q);FBlock bx=D_F_BLOCK_BLOCKS[bb];Code loc=q-bx.off;uint32_t bhr=uint32_t(loc/bx.stride),lr=uint32_t(loc-Code(bhr)*bx.stride);
        unsigned long long r=D_HP_OWNER_BLOCK_REC[Code(p-LOW_LUT_K-1)*highTotal+D_F_HIGH_ALL_OFF[bx.he]+bhr];uint32_t hr=uint32_t(r)&M,mhr=uint32_t(r>>B)&M,c=uint32_t(r>>(2*B))&3u,ty=uint32_t(r>>(2*B+2))&3u,tc=uint32_t(r>>(2*B+4))&3u;
        int sd=(c==uint32_t(::L)?1:c==uint32_t(R)?-1:0),he=int(bx.he)-sd;FBlock x=D_F_MAIN_BLOCKS[3*he+int(c)];int td=(tc==uint32_t(::L)?1:tc==uint32_t(R)?-1:0),the=int(bx.he)-td;FBlock dx=D_F_MAIN_BLOCKS[3*the+int(tc)];
        Code i=x.off+Code(hr)*x.stride+lr,j=dx.off+Code(mhr)*dx.stride+lr;Count cv=mainv[i],d=blockv[q];if(ty==0){mainv[j]=add_mod_plain(mainv[j],cv);mainv[i]=add_mod_plain(cv,d);blockv[q]=0;}else{Count cc=mainv[j];mainv[i]=add_mod_plain(add_mod_plain(cv,cc),d);blockv[q]=cv;}
    }
}
__global__ void factor_high_prerank_closure_inv_combined_kernel(Count*mainv,Count*blockv,HighPreRankMap invMap,HighPreRankMap crossMap){constexpr uint32_t IB=18,IM=(1u<<IB)-1u,P=36;constexpr unsigned long long PM=(1ULL<<P)-1ULL;constexpr uint32_t B=20,M=(1u<<B)-1u;constexpr int LOWK=LOW_LUT_K,S=MAXW+2;constexpr uint32_t LM=(1u<<LOWK)-1u;Code ni=invMap.nblocks?invMap.end[invMap.nblocks-1]:0,nx=crossMap.nblocks?crossMap.end[crossMap.nblocks-1]:0,total=ni+nx;for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){if(q<ni){int bid=high_pr_find_block(q,invMap);Code base=bid?invMap.end[bid-1]:0,loc=q-base;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lc=x.stride,k=uint32_t(loc/lc),lr=uint32_t(loc-Code(k)*lc);unsigned long long r=D_HP_CLOSURE_INV_REC[invMap.sel_off[bid]+k];uint32_t bhr=uint32_t(r)&IM;unsigned long long payload=(r>>IB)&PM;uint32_t cnt=uint32_t(r>>(IB+P))&7u;Count acc=0;for(uint32_t j=0;j<cnt;++j){uint32_t hr=cnt<=2?uint32_t(payload>>(18*j))&IM:D_HP_CLOSURE_INV_SRC[uint32_t(payload)+j];Count c=mainv[x.off+Code(hr)*x.stride+lr];if(c)acc=add_mod_plain(acc,c);}if(acc){FBlock bx=D_F_BLOCK_BLOCKS[x.hs];atomic_add_mod(blockv+bx.off+Code(bhr)*bx.stride+lr,acc);}}
        else{Code zq=q-ni;int bid=high_pr_find_block(zq,crossMap);Code base=bid?crossMap.end[bid-1]:0,loc=zq-base;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lc=x.stride,k=uint32_t(loc/lc),lr=uint32_t(loc-Code(k)*lc);unsigned long long r=D_HP_CROSSLL_REC[crossMap.sel_off[bid]+k];uint32_t hr=uint32_t(r)&M,bhr=uint32_t(r>>B)&M;int depth=int((r>>(2*B))&63u);Count c=mainv[x.off+Code(hr)*x.stride+lr];if(!c)continue;uint32_t code=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr];int match=-1,s=depth;
#pragma unroll
            for(int z=LOWK-1;z>=0;--z){auto v=MateValue((code>>(2*z))&3u);if(v==R){if(--s==0){match=z;break;}}else if(v==oneesan::gridfp::L)++s;}if(match<0)continue;uint32_t nc=(code&~(3u<<(2*match)))|(uint32_t(::L)<<(2*match));uint32_t nlr=D_F_LOW_DENSE_PACKED_RANK[nc]&LM;int nh=int(x.hs)-2;FBlock bx=D_F_BLOCK_BLOCKS[nh];atomic_add_mod(blockv+bx.off+Code(bhr)*bx.stride+nlr,c);}}
}

__device__ __forceinline__ Count* global_main_ptr(Code g){if(D_NGPU==1)return D_MAIN_PTR[0]+g;int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o]+g-Code(o)*D_MAIN_CHUNK;}
__device__ __forceinline__ Count* global_block_ptr(Code g){if(D_NGPU==1)return D_BLOCK_PTR[0]+g;int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o]+g-Code(o)*D_BLOCK_CHUNK;}
__device__ __forceinline__ void global_atomic_add_main(Code g,Count v){atomic_add_mod(global_main_ptr(g),v);}
__device__ __forceinline__ void global_atomic_add_block(Code g,Count v){atomic_add_mod(global_block_ptr(g),v);}
__device__ __forceinline__ uint32_t direct_low_all_rank(int h,uint32_t lr){constexpr int S=MAXW+2;return D_F_LOW_OCC_BASE[size_t(D_F_MASK)*S+h]+lr;}
__device__ __forceinline__ uint32_t direct_high_all_rank(int h,uint32_t hr){constexpr int S=MAXW+2;return D_F_HIGH_OCC_BASE[size_t(D_F_MASK)*S+h]+hr;}

// Zero-copy lower-window owner update. HIGH occupancy is fixed, so one masked
// HIGH rank maps to one authoritative HIGH-all rank and all LOW ranks are
// already the authoritative LOW-all ranks.
__global__ void direct_prerank_owner_blockorder_kernel(Code n,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;Code lowTotal=D_F_LOW_ALL_OFF[MAXW+1];
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<n;q+=st){
        int bb=f_find_block(q);FBlock bx=D_F_BLOCK_BLOCKS[bb];Code loc=q-bx.off;uint32_t hr=uint32_t(loc/bx.stride),blr=uint32_t(loc-Code(hr)*bx.stride);
        uint32_t har=direct_high_all_rank(bx.he,hr);
        unsigned long long r=D_PR_OWNER_BLOCK_REC[Code(p-1)*lowTotal+D_F_LOW_ALL_OFF[bx.he]+blr];uint32_t lr=uint32_t(r)&M,mlr=uint32_t(r>>B)&M,c=uint32_t(r>>(2*B))&3u,ty=uint32_t(r>>(2*B+2))&3u;
        FBlock x=D_F_FULL_MAIN_BLOCKS[3*int(bx.he)+int(c)],dx=x;if(p==LOW_LUT_K){int dc=(ty==1?int(R):int(::L));dx=D_F_FULL_MAIN_BLOCKS[3*int(bx.he)+dc];}
        FBlock gbx=D_F_FULL_BLOCK_BLOCKS[bx.he];Code i=x.off+Code(har)*x.stride+lr,j=dx.off+Code(har)*dx.stride+mlr,dj=gbx.off+Code(har)*gbx.stride+blr;
        Count cv=global_load_main(i),d=global_load_block(dj);
        if(ty==0){global_store_main(j,add_mod_plain(global_load_main(j),cv));global_store_main(i,add_mod_plain(cv,d));global_store_block(dj,0);}
        else{Count cc=global_load_main(j),pair=add_mod_plain(cv,cc);global_store_main(i,add_mod_plain(pair,d));if(p==1){global_store_main(j,pair);global_store_block(dj,0);}else global_store_block(dj,cv);}
    }
}

__global__ void direct_prerank_closure_inv_combined_kernel(PreRankOrbitMap invMap,PreRankOrbitMap highMap,int p){
    constexpr uint32_t IB=18,IM=(1u<<IB)-1u,P=36;constexpr unsigned long long PM=(1ULL<<P)-1ULL;constexpr uint32_t B=20,M=(1u<<B)-1u;constexpr int H=HIGH_LUT_K,S=MAXW+2;
    Code ni=invMap.nblocks?invMap.end[invMap.nblocks-1]:0,nh=highMap.nblocks?highMap.end[highMap.nblocks-1]:0,total=ni+nh;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        if(q<ni){
            int bid=prerank_find_block(q,invMap);Code base=bid?invMap.end[bid-1]:0,loc=q-base;uint32_t sc=invMap.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long r=D_PR_CLOSURE_INV_REC[invMap.sel_off[bid]+k];uint32_t dlr=uint32_t(r)&IM;unsigned long long payload=(r>>IB)&PM;uint32_t cnt=uint32_t(r>>(IB+P))&7u,dc=uint32_t(r>>(IB+P+3))&3u;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t har=direct_high_all_rank(x.he,hr);FBlock gx=D_F_FULL_MAIN_BLOCKS[bid];Count acc=0;
            for(uint32_t j=0;j<cnt;++j){uint32_t lr=cnt<=2?uint32_t(payload>>(18*j))&IM:D_PR_CLOSURE_INV_SRC[uint32_t(payload)+j];Count c=global_load_main(gx.off+Code(har)*gx.stride+lr);if(c)acc=add_mod_plain(acc,c);}if(!acc)continue;
            if(p==1){FBlock dx=D_F_FULL_MAIN_BLOCKS[3*x.he+int(dc)];global_atomic_add_main(dx.off+Code(har)*dx.stride+dlr,acc);}else{FBlock bx=D_F_FULL_BLOCK_BLOCKS[x.he];global_atomic_add_block(bx.off+Code(har)*bx.stride+dlr,acc);}
        }else{
            Code zq=q-ni;int bid=prerank_find_block(zq,highMap);Code base=bid?highMap.end[bid-1]:0,loc=zq-base;uint32_t sc=highMap.sel_count[bid],hr=uint32_t(loc/sc),k=uint32_t(loc-Code(hr)*sc);unsigned long long rec=D_PR_HIGHRR_REC[highMap.sel_off[bid]+k];uint32_t lr=uint32_t(rec)&M,dlr=uint32_t(rec>>B)&M;int depth=int((rec>>(2*B))&63u);FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t hc=D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(D_F_MASK)*S+x.he]+hr],hp0=D_F_HIGH_PACKED_RANK[hc],har=hp0>>H;FBlock gx=D_F_FULL_MAIN_BLOCKS[bid];Count c=global_load_main(gx.off+Code(har)*gx.stride+lr);if(!c)continue;int match=-1,ss=depth;
#pragma unroll
            for(int z=0;z<H;++z){auto v=MateValue((hc>>(2*z))&3u);if(v==L){if(--ss==0){match=z;break;}}else if(v==R)++ss;}if(match<0)continue;uint32_t nhc=(hc&~(3u<<(2*match)))|(uint32_t(R)<<(2*match));uint32_t nhar=D_F_HIGH_PACKED_RANK[nhc]>>H;int nhe=int(x.he)-2;
            if(p==1){FBlock dx=D_F_FULL_MAIN_BLOCKS[3*nhe+int(x.c)];global_atomic_add_main(dx.off+Code(nhar)*dx.stride+dlr,c);}else{FBlock bx=D_F_FULL_BLOCK_BLOCKS[nhe];global_atomic_add_block(bx.off+Code(nhar)*bx.stride+dlr,c);}
        }
    }
}

// Zero-copy upper-window owner update. LOW occupancy is fixed, so masked LOW
// rank is converted once to its authoritative LOW-all rank.
__global__ void direct_high_prerank_owner_blockorder_kernel(Code n,int p){
    constexpr uint32_t B=20,M=(1u<<B)-1u;Code highTotal=D_F_HIGH_ALL_OFF[MAXW+1];
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<n;q+=st){
        int bb=f_find_block(q);FBlock bx=D_F_BLOCK_BLOCKS[bb];Code loc=q-bx.off;uint32_t bhr=uint32_t(loc/bx.stride),lr=uint32_t(loc-Code(bhr)*bx.stride),lar=direct_low_all_rank(bx.he,lr);
        unsigned long long r=D_HP_OWNER_BLOCK_REC[Code(p-LOW_LUT_K-1)*highTotal+D_F_HIGH_ALL_OFF[bx.he]+bhr];uint32_t hr=uint32_t(r)&M,mhr=uint32_t(r>>B)&M,c=uint32_t(r>>(2*B))&3u,ty=uint32_t(r>>(2*B+2))&3u,tc=uint32_t(r>>(2*B+4))&3u;
        int sd=(c==uint32_t(::L)?1:c==uint32_t(R)?-1:0),he=int(bx.he)-sd;FBlock x=D_F_FULL_MAIN_BLOCKS[3*he+int(c)];int td=(tc==uint32_t(::L)?1:tc==uint32_t(R)?-1:0),the=int(bx.he)-td;FBlock dx=D_F_FULL_MAIN_BLOCKS[3*the+int(tc)],gbx=D_F_FULL_BLOCK_BLOCKS[bx.he];
        Code i=x.off+Code(hr)*x.stride+lar,j=dx.off+Code(mhr)*dx.stride+lar,dj=gbx.off+Code(bhr)*gbx.stride+lar;Count cv=global_load_main(i),d=global_load_block(dj);
        if(ty==0){global_store_main(j,add_mod_plain(global_load_main(j),cv));global_store_main(i,add_mod_plain(cv,d));global_store_block(dj,0);}else{Count cc=global_load_main(j);global_store_main(i,add_mod_plain(add_mod_plain(cv,cc),d));global_store_block(dj,cv);}
    }
}

__global__ void direct_high_prerank_closure_inv_combined_kernel(HighPreRankMap invMap,HighPreRankMap crossMap){
    constexpr uint32_t IB=18,IM=(1u<<IB)-1u,P=36;constexpr unsigned long long PM=(1ULL<<P)-1ULL;constexpr uint32_t B=20,M=(1u<<B)-1u;constexpr int LOWK=LOW_LUT_K,S=MAXW+2;
    Code ni=invMap.nblocks?invMap.end[invMap.nblocks-1]:0,nx=crossMap.nblocks?crossMap.end[crossMap.nblocks-1]:0,total=ni+nx;
    for(Code q=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;q<total;q+=st){
        if(q<ni){
            int bid=high_pr_find_block(q,invMap);Code base=bid?invMap.end[bid-1]:0,loc=q-base;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lc=x.stride,k=uint32_t(loc/lc),lr=uint32_t(loc-Code(k)*lc),lar=direct_low_all_rank(x.hs,lr);unsigned long long r=D_HP_CLOSURE_INV_REC[invMap.sel_off[bid]+k];uint32_t bhr=uint32_t(r)&IM;unsigned long long payload=(r>>IB)&PM;uint32_t cnt=uint32_t(r>>(IB+P))&7u;FBlock gx=D_F_FULL_MAIN_BLOCKS[bid];Count acc=0;
            for(uint32_t j=0;j<cnt;++j){uint32_t hr=cnt<=2?uint32_t(payload>>(18*j))&IM:D_HP_CLOSURE_INV_SRC[uint32_t(payload)+j];Count c=global_load_main(gx.off+Code(hr)*gx.stride+lar);if(c)acc=add_mod_plain(acc,c);}if(acc){FBlock bx=D_F_FULL_BLOCK_BLOCKS[x.hs];global_atomic_add_block(bx.off+Code(bhr)*bx.stride+lar,acc);}
        }else{
            Code zq=q-ni;int bid=high_pr_find_block(zq,crossMap);Code base=bid?crossMap.end[bid-1]:0,loc=zq-base;FBlock x=D_F_MAIN_BLOCKS[bid];uint32_t lc=x.stride,k=uint32_t(loc/lc),lr=uint32_t(loc-Code(k)*lc);unsigned long long r=D_HP_CROSSLL_REC[crossMap.sel_off[bid]+k];uint32_t hr=uint32_t(r)&M,bhr=uint32_t(r>>B)&M;int depth=int((r>>(2*B))&63u);uint32_t code=D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(D_F_MASK)*S+x.hs]+lr],lar=D_F_LOW_DENSE_PACKED_RANK[code]>>LOWK;FBlock gx=D_F_FULL_MAIN_BLOCKS[bid];Count c=global_load_main(gx.off+Code(hr)*gx.stride+lar);if(!c)continue;int match=-1,ss=depth;
#pragma unroll
            for(int z=LOWK-1;z>=0;--z){auto v=MateValue((code>>(2*z))&3u);if(v==R){if(--ss==0){match=z;break;}}else if(v==oneesan::gridfp::L)++ss;}if(match<0)continue;uint32_t nc=(code&~(3u<<(2*match)))|(uint32_t(::L)<<(2*match));uint32_t nlar=D_F_LOW_DENSE_PACKED_RANK[nc]>>LOWK;int nh=int(x.hs)-2;FBlock bx=D_F_FULL_BLOCK_BLOCKS[nh];global_atomic_add_block(bx.off+Code(bhr)*bx.stride+nlar,c);
        }
    }
}
__global__ void bounded_main_kernel(const Count*in,Code n,Count*outM,Count*outD,int width,int p){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP);auto z=oneesan::gridfp::include_horizontal(m,width,p);if(!z.valid)continue;if(z.blocked)atomic_add_mod(outD+bounded_rank(z.mate,width-1,D_BOUND_DP,D_BOUND_CAP),c);else atomic_add_mod(outM+bounded_rank(z.mate,width,D_BOUND_DP,D_BOUND_CAP),c);}}
__global__ void bounded_block_kernel(const Count*in,Code n,Count*outM,int width,int p){for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID b=bounded_unrank(i,width-1,D_BOUND_DP,D_BOUND_CAP);MateID t=oneesan::gridfp::blocked_exclude(b,p);atomic_add_mod(outM+bounded_rank(t,width,D_BOUND_DP,D_BOUND_CAP),c);}}

__global__ void bounded_reverse_main_kernel(const Count*in,const Count*din,Code n,Count*out,int width,int p){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){
        MateID t=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP); Count acc=in[i]; MateID pred=0; bool have=false;
        switch(oneesan::gridfp::mpair(t,p)){
        case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);have=true;break;
        case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);have=true;break;
        case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);have=true;break;
        default: break;
        }
        if(have){Count c=in[bounded_rank(pred,width,D_BOUND_DP,D_BOUND_CAP)];if(c)acc=(acc>=D_MOD-c)?acc-(D_MOD-c):acc+c;}
        if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){MateID b=oneesan::gridfp::mshrink(t,p);Count c=din[bounded_rank(b,width-1,D_BOUND_DP,D_BOUND_CAP)];if(c)acc=(acc>=D_MOD-c)?acc-(D_MOD-c):acc+c;}
        out[i]=acc;
    }
}
__global__ void bounded_forward_block_kernel(const Count*in,Code n,Count*outD,int width,int p){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,st=Code(gridDim.x)*blockDim.x;i<n;i+=st){Count c=in[i];if(!c)continue;MateID m=bounded_unrank(i,width,D_BOUND_DP,D_BOUND_CAP);auto z=oneesan::gridfp::include_horizontal(m,width,p);if(z.valid&&z.blocked)atomic_add_mod(outD+bounded_rank(z.mate,width-1,D_BOUND_DP,D_BOUND_CAP),c);}
}

template<int WIDTH>
__device__ __forceinline__ Code rank_drop_n_t(Code src_rank,MateID m,int p){Code a=0,b=0;int h=1;
#pragma unroll
for(int pos=WIDTH-1;pos>p;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,N))a+=D_MAIN_DP[pos][h];if(v>R&&h>0&&allowed(D_MAIN_FIXED,D_MAIN_OCC,pos,R))a+=D_MAIN_DP[pos][h-1];int q=pos-1;if(v>N&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,N))b+=D_BLOCK_DP[q][h];if(v>R&&h>0&&allowed(D_BLOCK_FIXED,D_BLOCK_OCC,q,R))b+=D_BLOCK_DP[q][h-1];if(v==R)--h;else if(v==L)++h;}return b>=a?src_rank+(b-a):src_rank-(a-b);}
__global__ void blocked_group_kernel(const Count*in,Code n,Count*out_main,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID sm=factor_unrank_block(i);MateID t=oneesan::gridfp::blocked_exclude(sm,p);Code j=factor_rank_main(t);atomic_add_mod(out_main+j,c);}}
__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m=mates?mates[i]:factor_unrank_main(i);auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);if(!z.valid)continue;if(z.blocked)atomic_add_mod(out_block+factor_rank_block(z.mate),c);else atomic_add_mod(out_main+factor_rank_main(z.mate),c);}}

__device__ __forceinline__ Count add_mod_plain(Count a,Count b){Count mod=D_MOD;return a>=mod-b?a-(mod-b):a+b;}


// Re-rank a state that differs only at the adjacent pair (p,p-1). Away from
// the LOW/center/HIGH boundaries only one factor half changes.
__device__ __forceinline__ Code orbit_rank_main_pair_from(Code ti,MateID pred,int p){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    int bid=f_find_main(ti);FBlock x=D_F_MAIN_BLOCKS[bid];Code r=ti-x.off;
    uint32_t hr=x.stride?uint32_t(r/x.stride):0,lr=x.stride?uint32_t(r-Code(hr)*x.stride):0;
    if(p<L){
        constexpr uint32_t LM=(1u<<(2*L))-1u;uint32_t lc=uint32_t(pred)&LM;
        uint32_t lp=D_F_LOW_DENSE_PACKED_RANK[lc],nlr=D_F_FIX_LOW?(lp&((1u<<L)-1u)):(lp>>L);
        return ti-Code(lr)+Code(nlr);
    }
    if(p>=L+2){
        constexpr uint32_t HM=(1u<<(2*H))-1u;uint32_t hc=uint32_t((pred>>(2*(L+1)))&HM);
        uint32_t hp=D_F_HIGH_PACKED_RANK[hc],nhr=D_F_FIX_LOW?(hp>>H):(hp&((1u<<H)-1u));
        return x.off+Code(nhr)*x.stride+lr;
    }
    return factor_rank_main(pred);
}

// Rank mshrink(t,p) by reusing the factor-half rank that deletion leaves
// unchanged. The forced two-window schedule guarantees p>L in FIX_LOW mode
// and p<=L in the complementary mode.
__device__ __forceinline__ Code orbit_rank_block_from_main(Code ti,MateID t,int p){
    constexpr int L=LOW_LUT_K,H=HIGH_LUT_K;
    constexpr uint32_t HM=(1u<<(2*H))-1u;
    uint32_t thc=uint32_t((t>>(2*(L+1)))&HM);int the=seg_end_height(thc,H);int cv=int(mget(t,L));
    FBlock mx=D_F_MAIN_BLOCKS[3*the+cv];Code r=ti-mx.off;
    uint32_t hr=mx.stride?uint32_t(r/mx.stride):0,lr=mx.stride?uint32_t(r-Code(hr)*mx.stride):0;
    if(D_F_FIX_LOW){
        MateID b=oneesan::gridfp::mshrink(t,p);uint32_t hc=uint32_t((b>>(2*L))&HM);int h=seg_end_height(hc,H);
        uint32_t hp=D_F_HIGH_PACKED_RANK[hc],nhr=hp>>H;FBlock bx=D_F_BLOCK_BLOCKS[h];
        return bx.off+Code(nhr)*bx.stride+lr;
    }
    FBlock bx=D_F_BLOCK_BLOCKS[the];
    if(p==L)return bx.off+Code(hr)*bx.stride+lr;
    MateID b=oneesan::gridfp::mshrink(t,p);constexpr uint32_t LM=(1u<<(2*L))-1u;uint32_t lc=uint32_t(b)&LM;
    uint32_t lp=D_F_LOW_DENSE_PACKED_RANK[lc],nlr=lp>>L;
    return bx.off+Code(hr)*bx.stride+nlr;
}


// In-place orbit update. The NN/NR/NL owners cover the non-closure transition
// orbits together with the old blocked contribution. LL/RR/RL are disjoint
// source classes and are added in a second pass, so they still hold their old
// values when the closure pass reads them.
__global__ void factor_main_block_orbit_kernel(Count*mainv,Count*blockv,const MateID*mates,Code n,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID m=mates?mates[i]:factor_unrank_main(i);
        auto w=oneesan::gridfp::mpair(m,p);
        if(w!=oneesan::gridfp::NN && w!=oneesan::gridfp::NR && w!=oneesan::gridfp::NL)continue;
        Count c=mainv[i];
        MateID b=oneesan::gridfp::mshrink(m,p);
        Code dj=factor_rank_block(b);
        Count d=blockv[dj];
        if(w==oneesan::gridfp::NN){
            MateID t=oneesan::gridfp::msetpair(m,p,oneesan::gridfp::LR);
            Code j=factor_rank_main(t);
            mainv[j]=add_mod_plain(mainv[j],c); // NN -> LR is injective.
            mainv[i]=add_mod_plain(c,d);        // identity + blocked exclude.
            blockv[dj]=0;
        }else{
            MateID t=oneesan::gridfp::msetpair(m,p,w==oneesan::gridfp::NR?oneesan::gridfp::RN:oneesan::gridfp::LN);
            Code j=factor_rank_main(t);
            Count cc=mainv[j];
            Count all=add_mod_plain(add_mod_plain(c,cc),d);
            if(p==1){
                mainv[i]=all;
                mainv[j]=add_mod_plain(c,cc);
                blockv[dj]=0;
            }else{
                mainv[i]=all;
                // RN/LN identity remains in mainv[j].
                blockv[dj]=c; // NR/NL -> blocked after consuming old blocked.
            }
        }
    }
}


// Reverse (target-gather) form of one Grid-FP update for p>1.  For legal
// targets the main predecessors below are always legal states in the same
// transition-closed occupancy group, so no rank/unrank membership roundtrip
// is required.
__global__ void reverse_main_group_kernel(const Count*in,const Count*din,const MateID*mates,
        Code n,Code dn,Count*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID t=mates?mates[i]:factor_unrank_main(i); Count acc=in[i];
        MateID pred=0; bool have=false;
        switch(oneesan::gridfp::mpair(t,p)){
        case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);have=true;break;
        case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);have=true;break;
        case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);have=true;break;
        default: break;
        }
        if(have){Count c=in[factor_rank_main(pred)];if(c)acc=add_mod_plain(acc,c);}
        if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){
            MateID b=oneesan::gridfp::mshrink(t,p);Count c=din[factor_rank_block(b)];if(c)acc=add_mod_plain(acc,c);
        }
        out[i]=acc;
    }
}


__global__ void forward_block_only_kernel(const Count*in,const MateID*mates,Code n,Count*out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        Count c=in[i]; if(!c)continue;
        MateID m=mates?mates[i]:factor_unrank_main(i);
        auto z=oneesan::gridfp::include_horizontal(m,TARGET_W,p);
        if(z.valid&&z.blocked)atomic_add_mod(out_block+factor_rank_block(z.mate),c);
    }
}
__device__ __forceinline__ void reverse_add_main(Count&acc,const Count*in,MateID m){
    Count c=in[factor_rank_main(m)];if(c)acc=add_mod_plain(acc,c);
}
__global__ void reverse_block_group_kernel(const Count*in,Code n,Count*out,Code dn,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<dn;i+=stride){
        MateID b=factor_unrank_block(i); Count acc=0; auto low=oneesan::gridfp::mget(b,p-1);
        if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
            // inverse of NR/NL -> shrink(p)
            reverse_add_main(acc,in,oneesan::gridfp::minsert(b,p,oneesan::gridfp::N));
        }else if(low==oneesan::gridfp::N){
            MateID u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N);
            // RL is legal iff the R at the high member of the pair does not
            // cross height zero.
            if(height_before_rank_pos<TARGET_W>(u,p)>0)
                reverse_add_main(acc,in,oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL));
            // Inverse LL closure.  A candidate occurs whenever the forward
            // closure stack is at depth one and the target carries L there.
            int s=1;
            for(int q=p-2;q>=0&&s>0;--q){
                auto v=oneesan::gridfp::mget(u,q);
                if(v==oneesan::gridfp::L&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::R);reverse_add_main(acc,in,c);}
                if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;
            }
            // Inverse RR closure, symmetrically to the right.
            s=1;
            for(int q=p+1;q<TARGET_W&&s>0;++q){
                auto v=oneesan::gridfp::mget(u,q);
                if(v==oneesan::gridfp::R&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::L);reverse_add_main(acc,in,c);}
                if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;
            }
        }
        out[i]=acc;
    }
}

// Evaluate one reverse Grid-FP step at an arbitrary main target without
// materializing the intermediate vector.  This is used to compose two
// consecutive cell updates.
__device__ __forceinline__ Count reverse_eval_main_step(const Count*in,const Count*din,MateID t,int p){
    Count acc=in[factor_rank_main(t)];
    MateID pred=0; bool have=false;
    switch(oneesan::gridfp::mpair(t,p)){
    case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);have=true;break;
    case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);have=true;break;
    case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);have=true;break;
    default: break;
    }
    if(have){Count c=in[factor_rank_main(pred)];if(c)acc=add_mod_plain(acc,c);}
    if(oneesan::gridfp::mget(t,p)==oneesan::gridfp::N){
        MateID b=oneesan::gridfp::mshrink(t,p);Count c=din[factor_rank_block(b)];if(c)acc=add_mod_plain(acc,c);
    }
    return acc;
}

__device__ __forceinline__ Count reverse_eval_block_step(const Count*in,MateID b,int p){
    Count acc=0; auto low=oneesan::gridfp::mget(b,p-1);
    auto addp=[&](MateID m){Count c=in[factor_rank_main(m)];if(c)acc=add_mod_plain(acc,c);};
    if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
        addp(oneesan::gridfp::minsert(b,p,oneesan::gridfp::N));
    }else if(low==oneesan::gridfp::N){
        MateID u=oneesan::gridfp::minsert(b,p-1,oneesan::gridfp::N);
        if(height_before_rank_pos<TARGET_W>(u,p)>0)addp(oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RL));
        int s=1;
        for(int q=p-2;q>=0&&s>0;--q){
            auto v=oneesan::gridfp::mget(u,q);
            if(v==oneesan::gridfp::L&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::R);addp(c);}
            if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;
        }
        s=1;
        for(int q=p+1;q<TARGET_W&&s>0;++q){
            auto v=oneesan::gridfp::mget(u,q);
            if(v==oneesan::gridfp::R&&s==1){MateID c=oneesan::gridfp::msetpair(u,p,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,q,oneesan::gridfp::L);addp(c);}
            if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;
        }
    }
    return acc;
}

// Apply update p and then q=p-1 in one target-gather kernel.  The first
// update is evaluated recursively from the old vectors, so its full M/D
// vectors never touch HBM.
__global__ void reverse2_main_group_kernel(const Count*in,const Count*din,const MateID*mates,
        Code n,Count*out,int p){
    int q=p-1;
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        MateID t=mates?mates[i]:factor_unrank_main(i);
        Count acc=reverse_eval_main_step(in,din,t,p);
        MateID pred=0; bool have=false;
        switch(oneesan::gridfp::mpair(t,q)){
        case oneesan::gridfp::LR: pred=oneesan::gridfp::msetpair(t,q,oneesan::gridfp::NN);have=true;break;
        case oneesan::gridfp::NR: pred=oneesan::gridfp::msetpair(t,q,oneesan::gridfp::RN);have=true;break;
        case oneesan::gridfp::NL: pred=oneesan::gridfp::msetpair(t,q,oneesan::gridfp::LN);have=true;break;
        default: break;
        }
        if(have){Count c=reverse_eval_main_step(in,din,pred,p);if(c)acc=add_mod_plain(acc,c);}
        if(oneesan::gridfp::mget(t,q)==oneesan::gridfp::N){
            MateID b=oneesan::gridfp::mshrink(t,q);Count c=reverse_eval_block_step(in,b,p);if(c)acc=add_mod_plain(acc,c);
        }
        out[i]=acc;
    }
}

__device__ __forceinline__ void reverse2_add_main(Count&acc,const Count*in,const Count*din,MateID m,int p){
    Count c=reverse_eval_main_step(in,din,m,p);if(c)acc=add_mod_plain(acc,c);
}

__global__ void reverse2_block_group_kernel(const Count*in,const Count*din,const MateID*mates,
        Count*out,Code dn,int p){
    int q=p-1;
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<dn;i+=stride){
        MateID b=mates?mates[i]:factor_unrank_block(i); Count acc=0; auto low=oneesan::gridfp::mget(b,q-1);
        if(low==oneesan::gridfp::R||low==oneesan::gridfp::L){
            reverse2_add_main(acc,in,din,oneesan::gridfp::minsert(b,q,oneesan::gridfp::N),p);
        }else if(low==oneesan::gridfp::N){
            MateID u=oneesan::gridfp::minsert(b,q-1,oneesan::gridfp::N);
            if(height_before_rank_pos<TARGET_W>(u,q)>0)
                reverse2_add_main(acc,in,din,oneesan::gridfp::msetpair(u,q,oneesan::gridfp::RL),p);
            int s=1;
            for(int r=q-2;r>=0&&s>0;--r){
                auto v=oneesan::gridfp::mget(u,r);
                if(v==oneesan::gridfp::L&&s==1){MateID c=oneesan::gridfp::msetpair(u,q,oneesan::gridfp::LL);c=oneesan::gridfp::mset(c,r,oneesan::gridfp::R);reverse2_add_main(acc,in,din,c,p);}
                if(v==oneesan::gridfp::L)++s;else if(v==oneesan::gridfp::R)--s;
            }
            s=1;
            for(int r=q+1;r<TARGET_W&&s>0;++r){
                auto v=oneesan::gridfp::mget(u,r);
                if(v==oneesan::gridfp::R&&s==1){MateID c=oneesan::gridfp::msetpair(u,q,oneesan::gridfp::RR);c=oneesan::gridfp::mset(c,r,oneesan::gridfp::L);reverse2_add_main(acc,in,din,c,p);}
                if(v==oneesan::gridfp::L)--s;else if(v==oneesan::gridfp::R)++s;
            }
        }
        out[i]=acc;
    }
}


static Code rank_full(MateID m,int width){Code r=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto s=mget(m,pos);if(s>N)r+=H_DP[pos][h];if(s>R&&h>0)r+=H_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return r;}

struct DeviceCtx{
    int dev=-1;uint8_t*arena=nullptr;size_t capArena=0;Count*dA=nullptr,*dB=nullptr,*dD=nullptr,*dE=nullptr;MateID*dMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;size_t capIM=0,capID=0,maxIntervals=0;double active=0,gather_s=0,transition_s=0,scatter_s=0;uint64_t groups=0;cudaStream_t sMain=nullptr,sBlock=nullptr;cudaEvent_t copyDone=nullptr,clearDone=nullptr,mainDone=nullptr,blockDone=nullptr;
    void init(int d,Count mod,Count**mp,Count**bp,Code mc,Code bc,int ng){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"main ptrs");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"block ptrs");ck(cudaMemcpyToSymbol(D_MAIN_CHUNK,&mc,sizeof(mc)),"main chunk");ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK,&bc,sizeof(bc)),"block chunk");ck(cudaMemcpyToSymbol(D_NGPU,&ng,sizeof(ng)),"ngpu");ck(cudaStreamCreateWithFlags(&sMain,cudaStreamNonBlocking),"stream main");ck(cudaStreamCreateWithFlags(&sBlock,cudaStreamNonBlocking),"stream block");ck(cudaEventCreateWithFlags(&copyDone,cudaEventDisableTiming),"event copy");ck(cudaEventCreateWithFlags(&clearDone,cudaEventDisableTiming),"event clear");ck(cudaEventCreateWithFlags(&mainDone,cudaEventDisableTiming),"event main");ck(cudaEventCreateWithFlags(&blockDone,cudaEventDisableTiming),"event block");}
    void ensure(Code m,Code b,bool useMate,size_t im,size_t id,bool orbit){ck(cudaSetDevice(dev),"set ensure");auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,need=(orbit?ab+db:2*ab+2*db)+mb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}size_t off=0;dA=(Count*)(arena+off);off+=ab;if(orbit){dB=nullptr;dD=(Count*)(arena+off);off+=db;dE=nullptr;}else{dB=(Count*)(arena+off);off+=ab;dD=(Count*)(arena+off);off+=db;dE=(Count*)(arena+off);off+=db;}dMate=useMate?(MateID*)(arena+off):nullptr;if(im>capIM){if(dIM)cudaFree(dIM);capIM=im;ck(cudaMalloc(&dIM,capIM*sizeof(PeerInterval)),"interval main");}if(id>capID){if(dID)cudaFree(dID);capID=id;ck(cudaMalloc(&dID,capID*sizeof(PeerInterval)),"interval block");}maxIntervals=std::max({maxIntervals,im,id});}
    void destroy(){if(dev<0)return;cudaSetDevice(dev);if(arena)cudaFree(arena);if(dIM)cudaFree(dIM);if(dID)cudaFree(dID);if(copyDone)cudaEventDestroy(copyDone);if(clearDone)cudaEventDestroy(clearDone);if(mainDone)cudaEventDestroy(mainDone);if(blockDone)cudaEventDestroy(blockDone);if(sMain)cudaStreamDestroy(sMain);if(sBlock)cudaStreamDestroy(sBlock);}
};

struct PreparedGroup{
    int g=0;
    uint32_t mf=0,mo=0,bf=0,bo=0;
    GroupSpec ms,ds;
    bool use_mi=false,use_di=false;
    std::vector<PeerInterval> mi,di;
    Code work=0;
};
struct PreparedWindow{
    WindowPlan wp;
    std::vector<PreparedGroup> groups;
};

static PreparedGroup prepare_group(int W,const WindowPlan&wp,int g,Code mc,Code bc,int ng){
    PreparedGroup pg;pg.g=g;
    window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,pg.mf,pg.mo,pg.bf,pg.bo);
    pg.ms=make_spec(W,pg.mf,pg.mo);pg.ds=make_spec(W-1,pg.bf,pg.bo);
    pg.work=2*pg.ms.size+pg.ds.size;
    pg.use_mi=false; pg.use_di=false;
    pg.mi.clear(); pg.di.clear();
    return pg;
}

static void process_group(DeviceCtx&c,int W,const WindowPlan&wp,const PreparedGroup&pg,int threads,size_t target){
    auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");
    auto const&ms=pg.ms;auto const&ds=pg.ds;if(!ms.size&&!ds.size)return;
    bool fixLow=wp.p_hi>LOW_LUT_K;uint32_t fmask=fixLow?(pg.mo&((1u<<LOW_LUT_K)-1u)):((pg.mo>>(LOW_LUT_K+1))&((1u<<HIGH_LUT_K)-1u));
    auto fmb=make_factor_main_blocks(fixLow,fmask);auto fdb=make_factor_block_blocks(fixLow,fmask);
    if(fmb.back().end!=ms.size||fdb.back().end!=ds.size){std::cerr<<"factor size mismatch main="<<fmb.back().end<<"/"<<ms.size<<" block="<<fdb.back().end<<"/"<<ds.size<<" fixLow="<<fixLow<<" mask="<<fmask<<"\n";std::exit(20);}
    Code mainRows=0,blockRows=0;uint32_t mainMaxStride=0,blockMaxStride=0;if(!fixLow){for(auto const&x:fmb)if(x.stride){mainRows+=(x.end-x.off)/x.stride;mainMaxStride=std::max(mainMaxStride,x.stride);}for(auto const&x:fdb)if(x.stride){blockRows+=(x.end-x.off)/x.stride;blockMaxStride=std::max(blockMaxStride,x.stride);}}
    bool lowerFlatIo=!fixLow;if(const char*e=std::getenv("GRIDFP_FACTOR_FLAT_IO"))lowerFlatIo=!fixLow&&std::atoi(e)!=0;bool upperFlatIo=fixLow;if(const char*e=std::getenv("GRIDFP_UPPER_FLAT_IO"))upperFlatIo=fixLow&&std::atoi(e)!=0;bool lowerRowIo=false;if(const char*e=std::getenv("GRIDFP_FACTOR_ROW_IO"))lowerRowIo=!fixLow&&std::atoi(e)!=0;
    int fm=(int)fmb.size(),fd=(int)fdb.size(),fl=fixLow?1:0;
    bool useOrbit=true;size_t countBytes=size_t(ms.size+ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=false;
    int directMode=1;if(const char*e=std::getenv("GRIDFP_DIRECT_AUTH"))directMode=std::atoi(e);bool directAuth=(directMode==1)||(directMode==2&&fixLow)||(directMode==3&&!fixLow);
    if(!directAuth){
        c.ensure(ms.size,ds.size,useMate,pg.mi.size(),pg.di.size(),useOrbit);
        if(!pg.mi.empty())ck(cudaMemcpy(c.dIM,pg.mi.data(),pg.mi.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"copy main intervals");
        if(!pg.di.empty())ck(cudaMemcpy(c.dID,pg.di.data(),pg.di.size()*sizeof(PeerInterval),cudaMemcpyHostToDevice),"copy block intervals");
    }
    GroupRuntimeCfg cfg{};
    std::memcpy(cfg.main_dp,ms.dp,sizeof(ms.dp));std::memcpy(cfg.block_dp,ds.dp,sizeof(ds.dp));
    std::memcpy(cfg.f_main,fmb.data(),fmb.size()*sizeof(FBlock));std::memcpy(cfg.f_block,fdb.data(),fdb.size()*sizeof(FBlock));
    cfg.main_fixed=pg.mf;cfg.main_occ=pg.mo;cfg.block_fixed=pg.bf;cfg.block_occ=pg.bo;cfg.f_mask=fmask;
    cfg.main_w=W;cfg.block_w=W-1;cfg.f_main_n=fm;cfg.f_block_n=fd;cfg.f_fix_low=fl;
    ck(cudaMemcpyToSymbol(D_GRP,&cfg,sizeof(cfg)),"group runtime config");
    ck(cudaEventRecord(c.copyDone,0),"group config ready event");
    ck(cudaStreamWaitEvent(c.sMain,c.copyDone,0),"main wait group config");
    ck(cudaStreamWaitEvent(c.sBlock,c.copyDone,0),"block wait group config");
    int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads)),bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));
    if(directAuth){
        auto tGather=std::chrono::steady_clock::now();
        for(int p=wp.p_hi;p>=wp.p_lo;--p){
            auto launchN=[&](Code n,auto fn,const char*w){if(!n)return;int nb=int(std::min<Code>(65535,(n+threads-1)/threads));fn(nb);ck(cudaGetLastError(),w);};
            if(fixLow){
                auto im=make_high_prerank_map(fmb,p,3),xm=make_high_prerank_map(fmb,p,2);Code on=ds.size,in=im.nblocks?im.end[im.nblocks-1]:0,xn=xm.nblocks?xm.end[xm.nblocks-1]:0;
                launchN(on,[&](int nb){direct_high_prerank_owner_blockorder_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(on,p);},"direct high owner");
                launchN(in+xn,[&](int nb){direct_high_prerank_closure_inv_combined_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(im,xm);},"direct high closure");
            }else{
                Code on=ds.size;launchN(on,[&](int nb){direct_prerank_owner_blockorder_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(on,p);},"direct low owner");
                auto im=make_prerank_orbit_map(fmb,p,3),sm=make_prerank_orbit_map(fmb,p,2);Code in=im.nblocks?im.end[im.nblocks-1]:0,sn=sm.nblocks?sm.end[sm.nblocks-1]:0;
                launchN(in+sn,[&](int nb){direct_prerank_closure_inv_combined_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(im,sm,p);},"direct low closure");
            }
        }
        ck(cudaStreamSynchronize(c.sMain),"direct authoritative transition sync");
        auto tTransition=std::chrono::steady_clock::now();
        c.groups++;c.gather_s+=std::chrono::duration<double>(tGather-t0).count();c.transition_s+=std::chrono::duration<double>(tTransition-tGather).count();c.active+=std::chrono::duration<double>(tTransition-t0).count();
        return;
    }
    if(ms.size){if(upperFlatIo){factor_upper_flat_io_kernel<false,false><<<bm,threads,0,c.sMain>>>(c.dA,ms.size);}else if(lowerFlatIo){factor_lower_flat_io_kernel<false,false><<<bm,threads,0,c.sMain>>>(c.dA,ms.size);}else if(lowerRowIo){dim3 gd(std::max(1u,(mainMaxStride+unsigned(threads)-1)/unsigned(threads)),unsigned(std::max<Code>(1,std::min<Code>(65535,mainRows))));factor_lower_row_io_kernel<false,false><<<gd,threads,0,c.sMain>>>(c.dA,mainRows);}else if(pg.use_mi)interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>(c.dA,c.dIM,pg.mi.size());else gather_main_kernel<<<bm,threads,0,c.sMain>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}
    if(ds.size){if(upperFlatIo){factor_upper_flat_io_kernel<true,false><<<bd,threads,0,c.sBlock>>>(c.dD,ds.size);}else if(lowerFlatIo){factor_lower_flat_io_kernel<true,false><<<bd,threads,0,c.sBlock>>>(c.dD,ds.size);}else if(lowerRowIo){dim3 gd(std::max(1u,(blockMaxStride+unsigned(threads)-1)/unsigned(threads)),unsigned(std::max<Code>(1,std::min<Code>(65535,blockRows))));factor_lower_row_io_kernel<true,false><<<gd,threads,0,c.sBlock>>>(c.dD,blockRows);}else if(pg.use_di)interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>(c.dD,c.dID,pg.di.size());else gather_block_kernel<<<bd,threads,0,c.sBlock>>>(c.dD,ds.size);}
    ck(cudaGetLastError(),"doubleD gather");ck(cudaStreamSynchronize(c.sMain),"main gather sync");ck(cudaStreamSynchronize(c.sBlock),"block gather sync");auto tGather=std::chrono::steady_clock::now();
    Count*cur=c.dA,*dcur=c.dD;
    if(useOrbit){
        for(int p=wp.p_hi;p>=wp.p_lo;--p){auto launchN=[&](Code n,auto fn,const char*w){if(!n)return;int nb=int(std::min<Code>(65535,(n+threads-1)/threads));fn(nb);ck(cudaGetLastError(),w);};
            if(fixLow){auto im=make_high_prerank_map(fmb,p,3),xm=make_high_prerank_map(fmb,p,2);Code on=ds.size,in=im.nblocks?im.end[im.nblocks-1]:0,xn=xm.nblocks?xm.end[xm.nblocks-1]:0;launchN(on,[&](int nb){factor_high_prerank_owner_blockorder_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(cur,dcur,on,p);},"factor high prerank owner blockorder");launchN(in+xn,[&](int nb){factor_high_prerank_closure_inv_combined_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(cur,dcur,im,xm);},"factor high prerank inverse closure");}
            else{Code on=ds.size;launchN(on,[&](int nb){factor_prerank_owner_blockorder_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(cur,dcur,on,p);},"factor prerank owner blockorder");auto im=make_prerank_orbit_map(fmb,p,3),sm=make_prerank_orbit_map(fmb,p,2);Code in=im.nblocks?im.end[im.nblocks-1]:0,sn=sm.nblocks?sm.end[sm.nblocks-1]:0;launchN(in+sn,[&](int nb){factor_prerank_closure_inv_combined_kernel<<<std::max(1,nb),threads,0,c.sMain>>>(cur,dcur,im,sm,p);},"factor prerank inverse closure");}
        }
        ck(cudaStreamSynchronize(c.sMain),"full prerank orbit transition sync");
    }else{
        Count*nxt=c.dB,*dnext=c.dE;
        for(int p=wp.p_hi;p>=wp.p_lo;){
            if(p>2 && p-1>=wp.p_lo){
                if(ms.size)reverse2_main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,dcur,useMate?c.dMate:nullptr,ms.size,nxt,p);
                if(ds.size)reverse2_block_group_kernel<<<bd,threads,0,c.sBlock>>>(cur,dcur,nullptr,dnext,ds.size,p);
                p-=2;
            }else{
                if(p>1){
                    if(ms.size)reverse_main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,dcur,useMate?c.dMate:nullptr,ms.size,ds.size,nxt,p);
                    if(ds.size)reverse_block_group_kernel<<<bd,threads,0,c.sBlock>>>(cur,ms.size,dnext,ds.size,p);
                }else{
                    if(ms.size)ck(cudaMemcpyAsync(nxt,cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToDevice,c.sMain),"p1 identity");
                    if(ds.size)ck(cudaMemsetAsync(dnext,0,size_t(ds.size)*sizeof(Count),c.sBlock),"p1 clear next D");
                    ck(cudaEventRecord(c.copyDone,c.sMain),"record p1 copy");ck(cudaEventRecord(c.clearDone,c.sBlock),"record p1 clear");
                    ck(cudaStreamWaitEvent(c.sMain,c.clearDone,0),"p1 main wait clear");ck(cudaStreamWaitEvent(c.sBlock,c.copyDone,0),"p1 block wait copy");
                    if(ms.size)main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p);
                    if(ds.size)blocked_group_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size,nxt,p);
                }
                --p;
            }
            ck(cudaGetLastError(),"hybrid reverse2 transition");
            ck(cudaEventRecord(c.mainDone,c.sMain),"record main");ck(cudaEventRecord(c.blockDone,c.sBlock),"record block");
            ck(cudaStreamWaitEvent(c.sMain,c.blockDone,0),"main wait block");ck(cudaStreamWaitEvent(c.sBlock,c.mainDone,0),"block wait main");
            std::swap(cur,nxt);std::swap(dcur,dnext);
        }
        ck(cudaStreamSynchronize(c.sMain),"hybrid reverse2 main sync");ck(cudaStreamSynchronize(c.sBlock),"hybrid reverse2 block sync");
    }
    auto tTransition=std::chrono::steady_clock::now();
    if(ms.size){if(upperFlatIo){factor_upper_flat_io_kernel<false,true><<<bm,threads,0,c.sMain>>>(cur,ms.size);}else if(lowerFlatIo){factor_lower_flat_io_kernel<false,true><<<bm,threads,0,c.sMain>>>(cur,ms.size);}else if(lowerRowIo){dim3 gd(std::max(1u,(mainMaxStride+unsigned(threads)-1)/unsigned(threads)),unsigned(std::max<Code>(1,std::min<Code>(65535,mainRows))));factor_lower_row_io_kernel<false,true><<<gd,threads,0,c.sMain>>>(cur,mainRows);}else if(pg.use_mi)interval_io_kernel<false,true><<<interval_blocks(pg.mi.size(),threads),threads,0,c.sMain>>>(cur,c.dIM,pg.mi.size());else scatter_main_kernel<<<bm,threads,0,c.sMain>>>(cur,ms.size);}
    if(ds.size){if(upperFlatIo){factor_upper_flat_io_kernel<true,true><<<bd,threads,0,c.sBlock>>>(dcur,ds.size);}else if(lowerFlatIo){factor_lower_flat_io_kernel<true,true><<<bd,threads,0,c.sBlock>>>(dcur,ds.size);}else if(lowerRowIo){dim3 gd(std::max(1u,(blockMaxStride+unsigned(threads)-1)/unsigned(threads)),unsigned(std::max<Code>(1,std::min<Code>(65535,blockRows))));factor_lower_row_io_kernel<true,true><<<gd,threads,0,c.sBlock>>>(dcur,blockRows);}else if(pg.use_di)interval_io_kernel<true,true><<<interval_blocks(pg.di.size(),threads),threads,0,c.sBlock>>>(dcur,c.dID,pg.di.size());else scatter_block_kernel<<<bd,threads,0,c.sBlock>>>(dcur,ds.size);}
    ck(cudaGetLastError(),"orbit scatter");ck(cudaStreamSynchronize(c.sMain),"main scatter sync");ck(cudaStreamSynchronize(c.sBlock),"block scatter sync");auto tScatter=std::chrono::steady_clock::now();c.groups++;c.gather_s+=std::chrono::duration<double>(tGather-t0).count();c.transition_s+=std::chrono::duration<double>(tTransition-tGather).count();c.scatter_s+=std::chrono::duration<double>(tScatter-tTransition).count();c.active+=std::chrono::duration<double>(tScatter-t0).count();
}


static void run_bounded_prefix_single_gpu(int W,int K,Count* fullMain,int threads){
    if(K<1)return;cudaSetDevice(0);Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(1,dpOld);Code oldN=dpOld[W][1];Count*cur=nullptr;ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"bounded cap1");int bo=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_fill_one_kernel<<<std::max(1,bo),threads>>>(cur,oldN);ck(cudaDeviceSynchronize(),"bounded fill1");
    for(int cap=2;cap<=K;++cap){build_bounded_dp(cap,dpNew);Code n=dpNew[W][1],dn=dpNew[W-1][1];Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"bounded a");ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"bounded b");ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"bounded d");ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"bounded e");ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"bounded zero a");ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"bounded zero d");ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"bounded old dp");ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"bounded new dp");int oldcap=cap-1;ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"bounded old cap");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"bounded cap");int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads));int be=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,W);ck(cudaDeviceSynchronize(),"bounded embed");cudaFree(cur);cur=a;Count*nxt=b,*dc=d,*de=e;
        for(int p=W-1;p>=1;--p){
            if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"bounded clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,W,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,W,p);}
            else {ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"bounded identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"bounded clear d");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,W,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,W,p);}
            ck(cudaDeviceSynchronize(),"bounded step");std::swap(cur,nxt);std::swap(dc,de);}cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;std::cerr<<"bounded prefix cap="<<cap<<" states="<<n<<" block="<<dn<<"\n";
    }
    if(K==1){ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"bounded final dp1");int cap=1;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"bounded final cap1");}
    else {ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"bounded final dp");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&K,sizeof(K)),"bounded final cap");}
    int bs=int(std::min<Code>(65535,(oldN+threads-1)/threads));bounded_scatter_full_kernel<<<std::max(1,bs),threads>>>(cur,oldN,W);ck(cudaDeviceSynchronize(),"bounded scatter full");cudaFree(cur);
}


static Count host_pow_mod(Count a,unsigned long long e,Count mod){unsigned long long x=a,r=1;while(e){if(e&1)r=(__uint128_t)r*x%mod;x=(__uint128_t)x*x%mod;e>>=1;}return Count(r);}
static Count host_add_signed(Count acc,Count x,int coeff,Count mod){long long c=coeff;unsigned long long mag=(unsigned long long)(c<0?-c:c);Count z=Count((__uint128_t)x*mag%mod);if(c<0)return acc>=z?acc-z:acc+(mod-z);return acc>=mod-z?acc-(mod-z):acc+z;}
static void row3_apply_row(const Count*in,Count*out,int sym,Count mod){using namespace oneesan::row3auto;std::fill(out,out+DIM,0);const uint16_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)if(in[i])for(int q=off[i];q<off[i+1];++q)out[tr[q].dst]=host_add_signed(out[tr[q].dst],in[i],tr[q].coeff,mod);}
static void row3_apply_col(int sym,const Count*in,Count*out,Count mod){using namespace oneesan::row3auto;std::fill(out,out+DIM,0);const uint16_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)for(int q=off[i];q<off[i+1];++q)if(in[tr[q].dst])out[i]=host_add_signed(out[i],in[tr[q].dst],tr[q].coeff,mod);}
static void row4_apply_row(const Count*in,Count*out,int sym,Count mod){using namespace oneesan::row4mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)if(in[i])for(uint32_t q=off[i];q<off[i+1];++q)out[tr[q].dst]=Count((out[tr[q].dst]+(__uint128_t)in[i]*tr[q].coeff)%mod);}
static void row4_apply_col(int sym,const Count*in,Count*out,Count mod){using namespace oneesan::row4mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)for(uint32_t q=off[i];q<off[i+1];++q)if(in[tr[q].dst])out[i]=Count((out[i]+(__uint128_t)tr[q].coeff*in[tr[q].dst])%mod);}
static std::vector<Count> make_row4_prefix_lut(int len,Count mod){using namespace oneesan::row4mod;std::vector<Count>cur(DIM);cur[0]=1;size_t cnt=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(size_t c=0;c<cnt;++c)for(int a=0;a<3;++a)row4_apply_row(&cur[c*DIM],&nxt[(c*3+a)*DIM],a,mod);cur.swap(nxt);cnt*=3;}return cur;}
static std::vector<Count> make_row4_suffix_lut(int len,Count mod){using namespace oneesan::row4mod;std::vector<Count>cur(DIM);for(int i=0;i<DIM;++i)cur[i]=BETA[i];size_t cnt=1,pow3=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(int a=0;a<3;++a)for(size_t c=0;c<cnt;++c)row4_apply_col(a,&cur[c*DIM],&nxt[(size_t(a)*pow3+c)*DIM],mod);cur.swap(nxt);cnt*=3;pow3*=3;}return cur;}
static constexpr int R4_BSZ[5]={19,20,12,4,1};
static constexpr int R4_BIDX[5][20]={
 {2,5,7,13,15,20,23,29,35,37,41,44,48,49,51,52,53,54,55,-1},
 {0,1,4,8,10,12,16,18,21,22,26,28,34,38,40,42,43,46,47,50},
 {3,6,9,14,17,24,25,30,32,36,39,45,-1,-1,-1,-1,-1,-1,-1,-1},
 {11,19,27,31,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {33,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};
static int ternary_prefix_height(size_t code,int len){std::vector<int>d(len);for(int i=len-1;i>=0;--i){d[i]=code%3;code/=3;}int h=1;for(int x:d){if(x==1)--h;else if(x==2)++h;}return h;}
static int ternary_suffix_required_height(size_t code,int len){int delta=0;for(int i=0;i<len;++i){int x=code%3;code/=3;if(x==1)--delta;else if(x==2)++delta;}return -delta;}
static std::vector<Count> compact_row4_lut(const std::vector<Count>&full,int len,bool prefix){using namespace oneesan::row4mod;size_t cnt=full.size()/DIM;std::vector<Count>z(cnt*20);for(size_t c=0;c<cnt;++c){int h=prefix?ternary_prefix_height(c,len):ternary_suffix_required_height(c,len);if(h<0||h>4)continue;for(int q=0;q<R4_BSZ[h];++q)z[c*20+q]=full[c*DIM+R4_BIDX[h][q]];}return z;}

static void row5_apply_row(const Count*in,Count*out,int sym,Count mod){using namespace oneesan::row5mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)if(in[i])for(uint32_t q=off[i];q<off[i+1];++q)out[tr[q].dst]=Count((out[tr[q].dst]+(__uint128_t)in[i]*tr[q].coeff)%mod);}
static void row5_apply_col(int sym,const Count*in,Count*out,Count mod){using namespace oneesan::row5mod;std::fill(out,out+DIM,0);const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;for(int i=0;i<DIM;++i)for(uint32_t q=off[i];q<off[i+1];++q)if(in[tr[q].dst])out[i]=Count((out[i]+(__uint128_t)tr[q].coeff*in[tr[q].dst])%mod);}
static std::vector<Count> make_row5_prefix_lut(int len,Count mod){using namespace oneesan::row5mod;std::vector<Count>cur(DIM);cur[0]=1;size_t cnt=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(size_t c=0;c<cnt;++c)for(int a=0;a<3;++a)row5_apply_row(&cur[c*DIM],&nxt[(c*3+a)*DIM],a,mod);cur.swap(nxt);cnt*=3;}return cur;}
static std::vector<Count> make_row5_suffix_lut(int len,Count mod){using namespace oneesan::row5mod;std::vector<Count>cur(DIM);for(int i=0;i<DIM;++i)cur[i]=BETA[i];size_t cnt=1,pow3=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(int a=0;a<3;++a)for(size_t c=0;c<cnt;++c)row5_apply_col(a,&cur[c*DIM],&nxt[(size_t(a)*pow3+c)*DIM],mod);cur.swap(nxt);cnt*=3;pow3*=3;}return cur;}
struct Row5RuntimeTr{uint16_t dst;Count coeff;};
static std::array<std::vector<Row5RuntimeTr>,3> G_R5_TR;
static Count G_R5_BETA[176]{};
static Count G_R5_MOD=0;
static uint64_t row5_big_mod(oneesan::row5rat::BigNum z,Count mod){
    uint64_t r64=uint64_t(((__uint128_t(1)<<64)%mod));
    uint64_t v=z.x2%mod;v=(__uint128_t(v)*r64+z.x1)%mod;v=(__uint128_t(v)*r64+z.x0)%mod;return v;
}
static uint64_t row5_pow_mod(uint64_t a,uint64_t e,Count mod){uint64_t r=1;while(e){if(e&1)r=(__uint128_t)r*a%mod;a=(__uint128_t)a*a%mod;e>>=1;}return r;}
static void prepare_row5_runtime(Count mod){
    using namespace oneesan::row5rat;if(G_R5_MOD==mod)return;
    std::vector<Count> dinv(NDEN);for(int i=0;i<NDEN;++i){Count d=Count(row5_big_mod(DEN[i],mod));if(!d)throw std::runtime_error("row5 rational denominator is zero modulo selected prime");dinv[i]=Count(row5_pow_mod(d,uint64_t(mod)-2,mod));}
    const Tr* src[3]={TR_N,TR_R,TR_L};const uint32_t* off[3]={OFF_N,OFF_R,OFF_L};
    for(int a=0;a<3;++a){G_R5_TR[a].resize(off[a][DIM]);for(uint32_t q=0;q<off[a][DIM];++q){auto const&t=src[a][q];uint64_t n=row5_big_mod(t.coeff.num,mod);uint64_t v=(__uint128_t)n*dinv[t.coeff.den]%mod;if(t.coeff.sign<0&&v)v=mod-v;G_R5_TR[a][q]={t.dst,Count(v)};}}
    for(int i=0;i<DIM;++i){int64_t b=BETA[i];uint64_t v=uint64_t(b<0?-b:b)%mod;G_R5_BETA[i]=Count(b<0&&v?mod-v:v);}G_R5_MOD=mod;
}
static constexpr int R5_BSZ[6]={51,61,40,18,5,1};
static constexpr int R5_BIDX[6][61]={
 {2,5,7,13,15,20,23,29,35,37,42,45,51,56,59,62,64,73,79,81,96,98,103,106,112,116,119,122,124,132,136,138,144,147,149,150,154,155,158,161,164,166,167,168,169,170,171,172,173,174,175,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {0,1,4,8,10,12,16,18,21,22,26,28,34,38,40,43,44,48,50,57,58,61,65,67,70,72,78,82,84,90,95,99,101,104,105,109,111,117,118,121,125,127,130,131,135,139,141,143,145,146,148,151,152,153,156,157,159,160,162,163,165},
 {3,6,9,14,17,24,25,30,32,36,39,46,47,52,54,60,63,66,69,74,76,80,83,87,89,97,100,107,108,113,115,120,123,126,129,133,134,137,140,142,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {11,19,27,31,41,49,53,68,71,75,85,86,91,93,102,110,114,128,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {33,55,77,88,92,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
 {94,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};
static std::vector<Count> compact_row5_lut(const std::vector<Count>&full,int len,bool prefix){using namespace oneesan::row5mod;size_t cnt=full.size()/DIM;std::vector<Count>z(cnt*61);for(size_t c=0;c<cnt;++c){int h=prefix?ternary_prefix_height(c,len):ternary_suffix_required_height(c,len);if(h<0||h>5)continue;for(int q=0;q<R5_BSZ[h];++q)z[c*61+q]=full[c*DIM+R5_BIDX[h][q]];}return z;}

static inline Count row5_mul_mod_host(Count a,Count b){
    if(G_R5_MOD==4294967291u){uint64_t z=uint64_t(a)*b;uint64_t t=uint32_t(z)+(z>>32)*5ULL;t=uint32_t(t)+(t>>32)*5ULL;if(t>=4294967291ULL)t-=4294967291ULL;if(t>=4294967291ULL)t-=4294967291ULL;return Count(t);}
    return Count(uint64_t(a)*b%G_R5_MOD);
}
static inline Count row5_add_mod_host(Count a,Count b){return a>=G_R5_MOD-b?a-(G_R5_MOD-b):a+b;}
static const std::array<int,176>& row5_local_index(){
    static const std::array<int,176> idx=[](){std::array<int,176>x{};x.fill(-1);for(int h=0;h<6;++h)for(int q=0;q<R5_BSZ[h];++q)x[R5_BIDX[h][q]]=q;return x;}();
    return idx;
}
static inline int row5_delta(int sym){return sym==0?0:sym==1?-1:1;}
static void row5_apply_row_compact(const Count*in,Count*out,int h,int sym){
    int h2=h+row5_delta(sym);if(h<0||h>5||h2<0||h2>5)return;using namespace oneesan::row5rat;
    const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;auto const&tr=G_R5_TR[sym];const auto&li=row5_local_index();
    for(int q=0;q<R5_BSZ[h];++q){Count v=in[q];if(!v)continue;int i=R5_BIDX[h][q];for(uint32_t k=off[i];k<off[i+1];++k){int d=li[tr[k].dst];Count z=row5_mul_mod_host(v,tr[k].coeff);out[d]=row5_add_mod_host(out[d],z);}}
}
static void row5_apply_col_compact(const Count*in,Count*out,int hdst,int sym){
    int hsrc=hdst-row5_delta(sym);if(hdst<0||hdst>5||hsrc<0||hsrc>5)return;using namespace oneesan::row5rat;
    const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;auto const&tr=G_R5_TR[sym];const auto&li=row5_local_index();
    for(int q=0;q<R5_BSZ[hsrc];++q){int i=R5_BIDX[hsrc][q];Count acc=0;for(uint32_t k=off[i];k<off[i+1];++k){Count v=in[li[tr[k].dst]];if(v)acc=row5_add_mod_host(acc,row5_mul_mod_host(tr[k].coeff,v));}out[q]=acc;}
}
template<class F> static void row5_parallel_for(size_t n,F&&fn){
    unsigned hc=std::thread::hardware_concurrency();int nt=hc?int(hc/2):4;if(const char*e=std::getenv("GRIDFP_ROW5_HOST_THREADS"))nt=std::max(1,std::atoi(e));
    if(n<2048||nt<=1){for(size_t i=0;i<n;++i)fn(i);return;}nt=std::min<int>(nt,int(n));std::vector<std::thread>ts;ts.reserve(nt);
    for(int t=0;t<nt;++t){size_t lo=n*size_t(t)/nt,hi=n*size_t(t+1)/nt;ts.emplace_back([&,lo,hi]{for(size_t i=lo;i<hi;++i)fn(i);});}for(auto&t:ts)t.join();
}
static std::vector<Count> make_row5_prefix_lut_compact_direct(int len){
    constexpr int S=61;std::vector<Count>cur(S);const auto&li=row5_local_index();cur[li[0]]=1;size_t cnt=1;
    std::vector<int8_t> height(1,1);
    for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t> nh(cnt*3,-127);row5_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>5)return;for(int a=0;a<3;++a){int h2=h+row5_delta(a);nh[c*3+a]=(h2>=0&&h2<=5)?int8_t(h2):int8_t(-127);row5_apply_row_compact(&cur[c*S],&nxt[(c*3+a)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;
}
static std::vector<Count> make_row5_suffix_lut_compact_direct(int len){
    constexpr int S=61;std::vector<Count>cur(S);const auto&li=row5_local_index();for(int i=0;i<176;++i)if(G_R5_BETA[i])cur[li[i]]=G_R5_BETA[i];size_t cnt=1;
    std::vector<int8_t> height(1,0);
    for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t> nh(cnt*3,-127);row5_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>5)return;for(int a=0;a<3;++a){int h2=h-row5_delta(a);nh[size_t(a)*cnt+c]=(h2>=0&&h2<=5)?int8_t(h2):int8_t(-127);row5_apply_col_compact(&cur[c*S],&nxt[(size_t(a)*cnt+c)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;
}

static Code row5_bounded_to_full_rank_host(Code brank,int W,const Code dp[MAXW+1][MAXW+2]){
    Code r=brank,g=0;int h=1;for(int pos=W-1;pos>=0;--pos){Code a=dp[pos][h];if(r<a)continue;r-=a;if(h>0){a=dp[pos][h-1];if(r<a){g+=H_DP[pos][h];--h;continue;}r-=a;g+=H_DP[pos][h]+H_DP[pos][h-1];++h;}else{g+=H_DP[pos][h];++h;}}return g;
}
static Code row5_lower_bound_full_rank(Code n,Code target,int W,const Code dp[MAXW+1][MAXW+2]){Code lo=0,hi=n;while(lo<hi){Code m=lo+(hi-lo)/2;if(row5_bounded_to_full_rank_host(m,W,dp)<target)lo=m+1;else hi=m;}return lo;}


struct PackedRow5Lut { std::vector<Count> data; std::vector<uint32_t> idx; };
static int row5_prefix_code_height(size_t code,int len){
    int dig[32];for(int i=len-1;i>=0;--i){dig[i]=int(code%3);code/=3;}int h=1;
    for(int i=0;i<len;++i){h+=row5_delta(dig[i]);if(h<0||h>5)return -1;}return h;
}
static int row5_suffix_code_height(size_t code,int len){
    int h=0;for(int i=0;i<len;++i){int a=int(code%3);code/=3;h-=row5_delta(a);if(h<0||h>5)return -1;}return h;
}
static PackedRow5Lut pack_row5_dense_lut(const std::vector<Count>&dense,int len,bool prefix){
    constexpr int S=61;size_t cnt=dense.size()/S;PackedRow5Lut z;z.idx.assign(cnt,0xffffffffu);
    std::vector<uint8_t> hs(cnt,255);std::vector<uint32_t> off(cnt+1);uint64_t total=0;
    for(size_t c=0;c<cnt;++c){int h=prefix?row5_prefix_code_height(c,len):row5_suffix_code_height(c,len);if(h>=0){hs[c]=uint8_t(h);z.idx[c]=uint32_t(total);total+=R5_BSZ[h];if(total>0xffffffffULL)throw std::runtime_error("row5 packed LUT exceeds uint32 offset");}off[c+1]=uint32_t(total);}
    z.data.resize(size_t(total));
    row5_parallel_for(cnt,[&](size_t c){if(hs[c]==255)return;int h=hs[c],n=R5_BSZ[h];std::copy_n(dense.data()+c*S,n,z.data.data()+z.idx[c]);});
    return z;
}

static void init_direct_row5_lut_full_multi_gpu(int W,Count mod,int threads,Count**fullMain,const std::vector<Code>&mainLen,Code mainChunk,int ng){
    prepare_row5_runtime(mod);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(5,dp);Code n=dp[W][1];constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;
    auto t0=std::chrono::steady_clock::now();std::vector<Count>prefDense,suffDense;std::thread tp([&]{prefDense=make_row5_prefix_lut_compact_direct(hi);});std::thread ts([&]{suffDense=make_row5_suffix_lut_compact_direct(lo);});tp.join();ts.join();auto pref=pack_row5_dense_lut(prefDense,hi,true);auto suff=pack_row5_dense_lut(suffDense,lo,false);prefDense.clear();prefDense.shrink_to_fit();suffDense.clear();suffDense.shrink_to_fit();double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::vector<Code> cut(ng+1);cut[0]=0;cut[ng]=n;for(int d=1;d<ng;++d)cut[d]=row5_lower_bound_full_rank(n,Code(d)*mainChunk,W,dp);
    Code mn=~Code(0),mx=0;for(int d=0;d<ng;++d){Code z=cut[d+1]-cut[d];mn=std::min(mn,z);mx=std::max(mx,z);Code fb=Code(d)*mainChunk,first=z?row5_bounded_to_full_rank_host(cut[d],W,dp):fb,last=z?row5_bounded_to_full_rank_host(cut[d+1]-1,W,dp):fb;std::cerr<<"row5 shard gpu="<<d<<" bounded=["<<cut[d]<<","<<cut[d+1]<<") states="<<z<<" full=["<<first<<","<<last<<"] local_cap="<<mainLen[d]<<"\n";}
    int lanes=4;if(const char*e=std::getenv("GRIDFP_ROW5_LANES"))lanes=std::atoi(e);else if(const char*e=std::getenv("GRIDFP_ROW5_WARP"))lanes=std::atoi(e)?32:0;
    std::vector<double> gpu_s(ng);std::vector<std::thread> workers;workers.reserve(ng);auto g0=std::chrono::steady_clock::now();
    for(int d=0;d<ng;++d)workers.emplace_back([&,d]{ck(cudaSetDevice(d),"row5 multi set");ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row5 lut dp");int cap=5;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row5 lut cap");Count *dpref=nullptr,*dsuff=nullptr;uint32_t *dpidx=nullptr,*dsidx=nullptr;
        ck(cudaMalloc(&dpref,pref.data.size()*sizeof(Count)),"row5 pref");ck(cudaMalloc(&dsuff,suff.data.size()*sizeof(Count)),"row5 suff");
        ck(cudaMalloc(&dpidx,pref.idx.size()*sizeof(uint32_t)),"row5 pref idx");ck(cudaMalloc(&dsidx,suff.idx.size()*sizeof(uint32_t)),"row5 suff idx");
        ck(cudaMemcpy(dpref,pref.data.data(),pref.data.size()*sizeof(Count),cudaMemcpyHostToDevice),"row5 pref copy");ck(cudaMemcpy(dsuff,suff.data.data(),suff.data.size()*sizeof(Count),cudaMemcpyHostToDevice),"row5 suff copy");
        ck(cudaMemcpy(dpidx,pref.idx.data(),pref.idx.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"row5 pref idx copy");ck(cudaMemcpy(dsidx,suff.idx.data(),suff.idx.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),"row5 suff idx copy");
        ck(cudaMemcpyToSymbol(D_R5_PREF,&dpref,sizeof(dpref)),"row5 pref ptr");ck(cudaMemcpyToSymbol(D_R5_SUFF,&dsuff,sizeof(dsuff)),"row5 suff ptr");ck(cudaMemcpyToSymbol(D_R5_PREF_IDX,&dpidx,sizeof(dpidx)),"row5 pref idx ptr");ck(cudaMemcpyToSymbol(D_R5_SUFF_IDX,&dsidx,sizeof(dsidx)),"row5 suff idx ptr");Code cnt=cut[d+1]-cut[d],base=Code(d)*mainChunk;int b=lanes?65535:int(std::min<Code>(65535,(cnt+threads-1)/threads));auto q0=std::chrono::steady_clock::now();if(cnt){if(lanes==32&&ng==1)bounded_fill_row5_full_warp_kernel<<<std::max(1,b),threads>>>(fullMain[d],n,W);else if(lanes==16)bounded_fill_row5_full_group_range_kernel<16><<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);else if(lanes==8)bounded_fill_row5_full_group_range_kernel<8><<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);else if(lanes==4)bounded_fill_row5_full_group_range_kernel<4><<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);else bounded_fill_row5_full_range_kernel<<<std::max(1,b),threads>>>(fullMain[d],cut[d],cut[d+1],base);}ck(cudaDeviceSynchronize(),"row5 lut init");gpu_s[d]=std::chrono::duration<double>(std::chrono::steady_clock::now()-q0).count();cudaFree(dpref);cudaFree(dsuff);cudaFree(dpidx);cudaFree(dsidx);});
    for(auto&t:workers)t.join();double gpu_wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count(),gmax=*std::max_element(gpu_s.begin(),gpu_s.end());
    std::cerr<<"direct row5 full states="<<n<<" lut_host_s="<<host_s<<" kernel_max_s="<<gmax<<" gpu_phase_wall_s="<<gpu_wall<<" pref_mib="<<(pref.data.size()*sizeof(Count)>>20)<<" suff_mib="<<(suff.data.size()*sizeof(Count)>>20)<<" pref_idx_mib="<<(pref.idx.size()*sizeof(uint32_t)>>20)<<" suff_idx_mib="<<(suff.idx.size()*sizeof(uint32_t)>>20)<<" shard_min="<<mn<<" shard_max="<<mx<<"\n";
}


struct Row6Layout { std::array<std::array<int,182>,7> bidx{}; std::array<int,558> li{}; Row6Layout(){for(auto&a:bidx)a.fill(-1);li.fill(-1);int n[7]{};for(int i=0;i<558;++i){int h=oneesan::row6mod::HEIGHT[i];li[i]=n[h];bidx[h][n[h]++]=i;}const int w[7]={141,182,135,68,25,6,1};for(int h=0;h<7;++h)if(n[h]!=w[h])throw std::runtime_error("row6 layout mismatch");}};
static const Row6Layout& row6_layout(){static Row6Layout z;return z;}
static inline int row6_delta(int sym){return sym==0?0:sym==1?-1:1;}
static Count G_R6_MOD=1000000007u;
static inline Count row6_add_host(Count a,Count b){Count m=G_R6_MOD;return a>=m-b?a-(m-b):a+b;}
static inline Count row6_mul_host(Count a,Count b){return Count(uint64_t(a)*b%G_R6_MOD);}
static void row6_apply_row_compact(const Count*in,Count*out,int h,int sym){
    using namespace oneesan::row6mod;int h2=h+row6_delta(sym);if(h<0||h>6||h2<0||h2>6)return;const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;auto const&lay=row6_layout();
    for(int q=0;q<r6_block_size(h);++q){Count v=in[q];if(!v)continue;int i=lay.bidx[h][q];for(uint32_t k=off[i];k<off[i+1];++k){int d=lay.li[tr[k].dst];out[d]=row6_add_host(out[d],row6_mul_host(v,tr[k].coeff));}}
}
static void row6_apply_col_compact(const Count*in,Count*out,int hdst,int sym){
    using namespace oneesan::row6mod;int hsrc=hdst-row6_delta(sym);if(hdst<0||hdst>6||hsrc<0||hsrc>6)return;const uint32_t*off=sym==0?OFF_N:sym==1?OFF_R:OFF_L;const Tr*tr=sym==0?TR_N:sym==1?TR_R:TR_L;auto const&lay=row6_layout();
    for(int q=0;q<r6_block_size(hsrc);++q){int i=lay.bidx[hsrc][q];Count acc=0;for(uint32_t k=off[i];k<off[i+1];++k){Count v=in[lay.li[tr[k].dst]];if(v)acc=row6_add_host(acc,row6_mul_host(tr[k].coeff,v));}out[q]=acc;}
}
template<class F> static void row6_parallel_for(size_t n,F&&fn){unsigned hc=std::thread::hardware_concurrency();int nt=hc?int(hc/2):4;if(const char*e=std::getenv("GRIDFP_ROW6_HOST_THREADS"))nt=std::max(1,std::atoi(e));if(n<2048||nt<=1){for(size_t i=0;i<n;++i)fn(i);return;}nt=std::min<int>(nt,int(n));std::vector<std::thread>ts;for(int t=0;t<nt;++t){size_t lo=n*size_t(t)/nt,hi=n*size_t(t+1)/nt;ts.emplace_back([&,lo,hi]{for(size_t i=lo;i<hi;++i)fn(i);});}for(auto&t:ts)t.join();}
static std::vector<Count> make_row6_prefix_lut_dense(int len){constexpr int S=182;auto const&lay=row6_layout();std::vector<Count>cur(S);cur[lay.li[0]]=1;size_t cnt=1;std::vector<int8_t> height(1,1);for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t>nh(cnt*3,-127);row6_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>6)return;for(int a=0;a<3;++a){int h2=h+row6_delta(a);nh[c*3+a]=(h2>=0&&h2<=6)?int8_t(h2):int8_t(-127);row6_apply_row_compact(&cur[c*S],&nxt[(c*3+a)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;}
static std::vector<Count> make_row6_suffix_lut_dense(int len){using namespace oneesan::row6mod;constexpr int S=182;auto const&lay=row6_layout();std::vector<Count>cur(S);for(int i=0;i<DIM;++i)if(BETA[i]){if(HEIGHT[i]!=0)throw std::runtime_error("row6 beta height");cur[lay.li[i]]=BETA[i];}size_t cnt=1;std::vector<int8_t>height(1,0);for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*S);std::vector<int8_t>nh(cnt*3,-127);row6_parallel_for(cnt,[&](size_t c){int h=height[c];if(h<0||h>6)return;for(int a=0;a<3;++a){int h2=h-row6_delta(a);nh[size_t(a)*cnt+c]=(h2>=0&&h2<=6)?int8_t(h2):int8_t(-127);row6_apply_col_compact(&cur[c*S],&nxt[(size_t(a)*cnt+c)*S],h,a);}});cur.swap(nxt);height.swap(nh);cnt*=3;}return cur;}
struct PackedRow6Lut { std::vector<Count> data; std::vector<uint32_t> idx; };
static int row6_prefix_code_height(size_t code,int len){int d[32];for(int i=len-1;i>=0;--i){d[i]=code%3;code/=3;}int h=1;for(int i=0;i<len;++i){h+=row6_delta(d[i]);if(h<0||h>6)return -1;}return h;}
static int row6_suffix_code_height(size_t code,int len){int h=0;for(int i=0;i<len;++i){int a=code%3;code/=3;h-=row6_delta(a);if(h<0||h>6)return -1;}return h;}
static PackedRow6Lut pack_row6_dense(const std::vector<Count>&dense,int len,bool prefix){constexpr int S=182;size_t cnt=dense.size()/S;PackedRow6Lut z;z.idx.assign(cnt,0xffffffffu);std::vector<uint8_t>hs(cnt,255);uint64_t total=0;for(size_t c=0;c<cnt;++c){int h=prefix?row6_prefix_code_height(c,len):row6_suffix_code_height(c,len);if(h>=0){hs[c]=h;z.idx[c]=uint32_t(total);total+=r6_block_size(h);if(total>0xffffffffULL)throw std::runtime_error("row6 packed offset overflow");}}z.data.resize(total);row6_parallel_for(cnt,[&](size_t c){if(hs[c]==255)return;std::copy_n(dense.data()+c*S,r6_block_size(hs[c]),z.data.data()+z.idx[c]);});return z;}
struct Row6CartLists{std::array<std::vector<uint32_t>,7> pc,sc;std::array<std::vector<Code>,7> pr,sr,fp;};
static Row6CartLists make_row6_cart_lists(int W){
    constexpr int lo=TARGET_W/2,hi=TARGET_W-lo,LK=LOW_LUT_K,HK=HIGH_LUT_K;static_assert(lo==LK&&hi==HK+1,"row6/factor split mismatch");Row6CartLists z;size_t pn=1,sn=1;for(int i=0;i<hi;++i)pn*=3;for(int i=0;i<lo;++i)sn*=3;
    auto digits=[](size_t code,int len,int*d){for(int i=len-1;i>=0;--i){d[i]=int(code%3);code/=3;}};
    for(size_t c=0;c<pn;++c){int d[32];digits(c,hi,d);int h=1;Code gr=0;bool ok=true;for(int k=0;k<hi;++k){int pos=W-1-k,a=d[k];if(a>0)gr+=H_DP[pos][h];if(a>1&&h>0)gr+=H_DP[pos][h-1];if(a==1)--h;else if(a==2)++h;if(h<0||h>6){ok=false;break;}}if(ok){int he=1;uint32_t hc=0;for(int k=0;k<HK;++k){int a=d[k];hc|=uint32_t(a)<<(2*(HK-1-k));if(a==1)--he;else if(a==2)++he;}int cv=d[HK],hs=he+(cv==1?-1:cv==2?1:0);if(hs!=h)throw std::runtime_error("row6 factor prefix height mismatch");uint32_t har=high_all_rank_host(hc,he);FBlock gx=G_F_FULL_MAIN_BLOCKS[3*he+cv];z.pc[h].push_back(uint32_t(c));z.pr[h].push_back(gr);z.fp[h].push_back(gx.off+Code(har)*gx.stride);}}
    for(size_t c=0;c<sn;++c){int d[32];digits(c,lo,d);int h=0;bool ok=true;for(int k=lo-1;k>=0;--k){int a=d[k];if(a==1)++h;else if(a==2)--h;if(h<0||h>6){ok=false;break;}}if(!ok)continue;int hs=h;Code gr=0;h=hs;for(int k=0;k<lo;++k){int pos=lo-1-k,a=d[k];if(a>0)gr+=H_DP[pos][h];if(a>1&&h>0)gr+=H_DP[pos][h-1];if(a==1)--h;else if(a==2)++h;if(h<0||h>6){ok=false;break;}}if(ok&&h==0){uint32_t lc=0;for(int k=0;k<LK;++k)lc|=uint32_t(d[k])<<(2*(LK-1-k));uint32_t lar=low_all_rank_host(lc,hs);z.sc[hs].push_back(uint32_t(c));z.sr[hs].push_back(Code(lar));}}
    // The suffix automaton enumerates ternary codes, not LOW-all ranks.  Reorder each
    // fixed-height suffix list once so suffix index == LOW-all rank.  This makes every
    // factorized row6 prefix own one contiguous authoritative interval [base,base+ns).
    for(int hs=0;hs<=6;++hs){size_t n=z.sc[hs].size();std::vector<size_t> ord(n);std::iota(ord.begin(),ord.end(),0);std::sort(ord.begin(),ord.end(),[&](size_t a,size_t b){return z.sr[hs][a]<z.sr[hs][b];});std::vector<uint32_t> sc(n);std::vector<Code> sr(n);for(size_t j=0;j<n;++j){sc[j]=z.sc[hs][ord[j]];sr[j]=z.sr[hs][ord[j]];}z.sc[hs].swap(sc);z.sr[hs].swap(sr);}
    return z;
}
struct Row6PackedLevelMeta { std::vector<uint32_t> idx; std::vector<int8_t> height; uint64_t elems=0; };
static std::vector<Row6PackedLevelMeta> make_row6_packed_level_meta(int len,bool prefix){
    std::vector<Row6PackedLevelMeta> z(len+1);z[0].idx={0};z[0].height={int8_t(prefix?1:0)};z[0].elems=r6_block_size(prefix?1:0);size_t parents=1;
    for(int lev=0;lev<len;++lev){size_t children=parents*3;auto&cur=z[lev];auto&nxt=z[lev+1];nxt.idx.assign(children,0xffffffffu);nxt.height.assign(children,int8_t(-1));uint64_t total=0;
        if(prefix){for(size_t p=0;p<parents;++p){int h=cur.height[p];if(h<0)continue;for(int sym=0;sym<3;++sym){int h2=h+row6_delta(sym);if(h2<0||h2>6)continue;size_t q=p*3+sym;nxt.height[q]=int8_t(h2);nxt.idx[q]=uint32_t(total);total+=r6_block_size(h2);}}}
        else{for(int sym=0;sym<3;++sym)for(size_t p=0;p<parents;++p){int h=cur.height[p];if(h<0)continue;int hs=h-row6_delta(sym);if(hs<0||hs>6)continue;size_t q=size_t(sym)*parents+p;nxt.height[q]=int8_t(hs);nxt.idx[q]=uint32_t(total);total+=r6_block_size(hs);}}
        if(total>0xffffffffULL)throw std::runtime_error("row6 true-packed level offset overflow");nxt.elems=total;parents=children;
    }return z;
}
static void init_direct_row6_lut_full_multi_gpu(int W,Count mod,int threads,Count**fullMain,const std::vector<Code>&mainLen,Code mainChunk,int ng){
    using namespace oneesan::row6mod;int r6pi=oneesan::row6crt::prime_index(mod);if(r6pi<0)throw std::runtime_error("row6 CRT20 modulus not in production prime set");G_R6_MOD=mod;Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(6,dp);Code n=dp[W][1];constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto const&lay=row6_layout();
    std::vector<Count>mat,beta(r6_block_size(0));uint32_t moff[3][7]{};const uint32_t*offs[3]={OFF_N,OFF_R,OFF_L};const Tr*trs[3]={TR_N,TR_R,TR_L};const uint32_t*coef[3]={oneesan::row6crt::CO_N[r6pi],oneesan::row6crt::CO_R[r6pi],oneesan::row6crt::CO_L[r6pi]};for(int sym=0;sym<3;++sym)for(int h=0;h<=6;++h){int h2=h+row6_delta(sym);if(h2<0||h2>6)continue;int ns=r6_block_size(h),nd=r6_block_size(h2);moff[sym][h]=uint32_t(mat.size());mat.resize(mat.size()+size_t(ns)*nd,0);Count*M=mat.data()+moff[sym][h];for(int a=0;a<ns;++a){int gi=lay.bidx[h][a];for(uint32_t q=offs[sym][gi];q<offs[sym][gi+1];++q){int gd=trs[sym][q].dst;if(HEIGHT[gd]!=h2)throw std::runtime_error("row6 transition violates height grading");int d=lay.li[gd];M[size_t(a)*nd+d]=row6_add_host(M[size_t(a)*nd+d],coef[sym][q]);}}}for(int i=0;i<DIM;++i)if(BETA[i]){if(HEIGHT[i]!=0)throw std::runtime_error("row6 beta height");beta[lay.li[i]]=row6_add_host(beta[lay.li[i]],Count(BETA[i]%mod));}int initLocal=lay.li[0];
    auto make_idx=[&](int len,bool prefix){size_t codes=1;for(int i=0;i<len;++i)codes*=3;std::vector<uint32_t>idx(codes,0xffffffffu);uint64_t total=0;for(size_t c=0;c<codes;++c){int h=prefix?row6_prefix_code_height(c,len):row6_suffix_code_height(c,len);if(h>=0){if(total>0xffffffffULL)throw std::runtime_error("row6 packed offset overflow");idx[c]=uint32_t(total);total+=r6_block_size(h);}}if(total>0xffffffffULL)throw std::runtime_error("row6 packed table too large");return std::pair<std::vector<uint32_t>,uint64_t>(std::move(idx),total);};std::pair<std::vector<uint32_t>,uint64_t>pi,si;std::vector<Row6PackedLevelMeta> plev,slev;std::thread ixp([&]{pi=make_idx(hi,true);plev=make_row6_packed_level_meta(hi,true);}),ixs([&]{si=make_idx(lo,false);slev=make_row6_packed_level_meta(lo,false);});ixp.join();ixs.join();if(plev.back().idx!=pi.first||plev.back().elems!=pi.second||slev.back().idx!=si.first||slev.back().elems!=si.second)throw std::runtime_error("row6 true-packed metadata mismatch");double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    Row6CartLists cart=make_row6_cart_lists(W);unsigned __int128 cartPairs=0;for(int h=0;h<=6;++h){if(!std::is_sorted(cart.pr[h].begin(),cart.pr[h].end())||!std::is_sorted(cart.sr[h].begin(),cart.sr[h].end()))throw std::runtime_error("row6 cart rank list not sorted");cartPairs+=(unsigned __int128)cart.pr[h].size()*cart.sr[h].size();}if(cartPairs!=n)throw std::runtime_error("row6 Cartesian pair count mismatch");uint32_t taskChunk=(TARGET_W>=28?65536u:1024u);if(const char*e=std::getenv("GRIDFP_ROW6_TASK_CHUNK"))taskChunk=std::max(64,std::atoi(e));
    std::vector<Code>cut(ng+1);cut[0]=0;cut[ng]=n;for(int d=1;d<ng;++d)cut[d]=row5_lower_bound_full_rank(n,Code(d)*mainChunk,W,dp);int lanes=4;if(const char*e=std::getenv("GRIDFP_ROW6_LANES"))lanes=std::atoi(e);int rt=128;if(const char*e=std::getenv("GRIDFP_ROW6_LEVEL_THREADS"))rt=std::max(32,std::atoi(e));std::vector<double>table_s(ng),init_s(ng);std::vector<uint64_t>cartTaskCount(ng),cartPairCount(ng);std::vector<std::thread>workers;auto g0=std::chrono::steady_clock::now();
    for(int dev=0;dev<ng;++dev)workers.emplace_back([&,dev]{ck(cudaSetDevice(dev),"r6 levelpack set");ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"r6 levelpack dp");int cap=6;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"r6 levelpack cap");ck(cudaMemcpyToSymbol(D_R6_GPU_MAT_OFF,moff,sizeof(moff)),"r6 levelpack mat off");Count*dmat=nullptr,*dbeta=nullptr;uint32_t*dpidx=nullptr,*dsidx=nullptr;ck(cudaMalloc(&dmat,mat.size()*4),"r6 levelpack mat");ck(cudaMalloc(&dbeta,beta.size()*4),"r6 levelpack beta");ck(cudaMemcpy(dmat,mat.data(),mat.size()*4,cudaMemcpyHostToDevice),"r6 levelpack mat cp");ck(cudaMemcpy(dbeta,beta.data(),beta.size()*4,cudaMemcpyHostToDevice),"r6 levelpack beta cp");auto tb=std::chrono::steady_clock::now();
        auto build_pref=[&](){Count*cur=nullptr;int8_t*hp=nullptr;uint32_t*ip=nullptr;Code parents=1;ck(cudaMalloc(&cur,size_t(plev[0].elems)*4),"r6 tp pref base");ck(cudaMemset(cur,0,size_t(plev[0].elems)*4),"r6 tp pref zero");Count one=1;ck(cudaMemcpy(cur+initLocal,&one,4,cudaMemcpyHostToDevice),"r6 tp pref one");ck(cudaMalloc(&hp,1),"r6 tp pref h");ck(cudaMalloc(&ip,4),"r6 tp pref idx");ck(cudaMemcpy(hp,plev[0].height.data(),1,cudaMemcpyHostToDevice),"r6 tp pref h cp");ck(cudaMemcpy(ip,plev[0].idx.data(),4,cudaMemcpyHostToDevice),"r6 tp pref idx cp");for(int lev=0;lev<hi;++lev){Code children=parents*3;Count*nxt=nullptr;int8_t*hn=nullptr;uint32_t*in=nullptr;ck(cudaMalloc(&nxt,size_t(plev[lev+1].elems)*4),"r6 tp pref next");ck(cudaMalloc(&hn,size_t(children)),"r6 tp pref hn");ck(cudaMalloc(&in,size_t(children)*4),"r6 tp pref in");ck(cudaMemcpy(hn,plev[lev+1].height.data(),size_t(children),cudaMemcpyHostToDevice),"r6 tp pref hn cp");ck(cudaMemcpy(in,plev[lev+1].idx.data(),size_t(children)*4,cudaMemcpyHostToDevice),"r6 tp pref in cp");bool wide=true;if(const char*e=std::getenv("GRIDFP_ROW6_LEVEL_WIDE"))wide=std::atoi(e)!=0;int wt=128;if(const char*e=std::getenv("GRIDFP_ROW6_WIDE_THREADS"))wt=std::max(32,std::min(256,std::atoi(e)));Code wideMin=243;if(const char*e=std::getenv("GRIDFP_ROW6_WIDE_MIN"))wideMin=std::max<Code>(1,std::strtoull(e,nullptr,10));if(wide&&children>=wideMin){int gb=int(std::min<Code>(children,2147483647));r6_prefix_level_packed_wide_kernel<<<std::max(1,gb),wt>>>(parents,cur,hp,ip,nxt,hn,in,dmat);}else{int bl=int(std::min<Code>(65535,(children+rt-1)/rt));r6_prefix_level_packed_kernel<<<std::max(1,bl),rt>>>(parents,cur,hp,ip,nxt,hn,in,dmat);}ck(cudaGetLastError(),"r6 tp pref launch");cudaFree(cur);cudaFree(hp);cudaFree(ip);cur=nxt;hp=hn;ip=in;parents=children;}cudaFree(hp);return std::pair<Count*,uint32_t*>(cur,ip);};
        auto build_suff=[&](){Count*cur=nullptr;int8_t*hp=nullptr;uint32_t*ip=nullptr;Code parents=1;ck(cudaMalloc(&cur,size_t(slev[0].elems)*4),"r6 tp suff base");ck(cudaMemcpy(cur,dbeta,size_t(slev[0].elems)*4,cudaMemcpyDeviceToDevice),"r6 tp suff beta");ck(cudaMalloc(&hp,1),"r6 tp suff h");ck(cudaMalloc(&ip,4),"r6 tp suff idx");ck(cudaMemcpy(hp,slev[0].height.data(),1,cudaMemcpyHostToDevice),"r6 tp suff h cp");ck(cudaMemcpy(ip,slev[0].idx.data(),4,cudaMemcpyHostToDevice),"r6 tp suff idx cp");for(int lev=0;lev<lo;++lev){Code children=parents*3;Count*nxt=nullptr;int8_t*hn=nullptr;uint32_t*in=nullptr;ck(cudaMalloc(&nxt,size_t(slev[lev+1].elems)*4),"r6 tp suff next");ck(cudaMalloc(&hn,size_t(children)),"r6 tp suff hn");ck(cudaMalloc(&in,size_t(children)*4),"r6 tp suff in");ck(cudaMemcpy(hn,slev[lev+1].height.data(),size_t(children),cudaMemcpyHostToDevice),"r6 tp suff hn cp");ck(cudaMemcpy(in,slev[lev+1].idx.data(),size_t(children)*4,cudaMemcpyHostToDevice),"r6 tp suff in cp");bool wide=true;if(const char*e=std::getenv("GRIDFP_ROW6_LEVEL_WIDE"))wide=std::atoi(e)!=0;int wt=128;if(const char*e=std::getenv("GRIDFP_ROW6_WIDE_THREADS"))wt=std::max(32,std::min(256,std::atoi(e)));Code wideMin=243;if(const char*e=std::getenv("GRIDFP_ROW6_WIDE_MIN"))wideMin=std::max<Code>(1,std::strtoull(e,nullptr,10));if(wide&&children>=wideMin){int gb=int(std::min<Code>(children,2147483647));r6_suffix_level_packed_wide_kernel<<<std::max(1,gb),wt>>>(parents,cur,hp,ip,nxt,hn,in,dmat);}else{int bl=int(std::min<Code>(65535,(children+rt-1)/rt));r6_suffix_level_packed_kernel<<<std::max(1,bl),rt>>>(parents,cur,hp,ip,nxt,hn,in,dmat);}ck(cudaGetLastError(),"r6 tp suff launch");cudaFree(cur);cudaFree(hp);cudaFree(ip);cur=nxt;hp=hn;ip=in;parents=children;}cudaFree(hp);return std::pair<Count*,uint32_t*>(cur,ip);};
        auto PP=build_pref();Count*P=PP.first;dpidx=PP.second;auto SS=build_suff();Count*S=SS.first;dsidx=SS.second;ck(cudaDeviceSynchronize(),"r6 levelpack tables");table_s[dev]=std::chrono::duration<double>(std::chrono::steady_clock::now()-tb).count();ck(cudaMemcpyToSymbol(D_R6_PREF,&P,sizeof(P)),"r6 levelpack pref ptr");ck(cudaMemcpyToSymbol(D_R6_SUFF,&S,sizeof(S)),"r6 levelpack suff ptr");ck(cudaMemcpyToSymbol(D_R6_PREF_IDX,&dpidx,sizeof(dpidx)),"r6 levelpack pidx ptr");ck(cudaMemcpyToSymbol(D_R6_SUFF_IDX,&dsidx,sizeof(dsidx)),"r6 levelpack sidx ptr");Code cnt=mainLen[dev],base=Code(dev)*mainChunk;int blocks=lanes?65535:int(std::min<Code>(65535,(cnt+threads-1)/threads));auto ib=std::chrono::steady_clock::now();bool cartesian=true;if(const char*e=std::getenv("GRIDFP_ROW6_CARTESIAN"))cartesian=std::atoi(e)!=0;if(cnt&&cartesian){
            for(int h=0;h<=6;++h){Code np=cart.pc[h].size(),ns=cart.sc[h].size();if(!np||!ns)continue;uint32_t*dpc=nullptr,*dsc=nullptr;Code*dfp=nullptr,*dsr=nullptr;ck(cudaMalloc(&dpc,np*4),"r6 cart pc");ck(cudaMalloc(&dsc,ns*4),"r6 cart sc");ck(cudaMalloc(&dfp,np*sizeof(Code)),"r6 cart factor prefix");ck(cudaMalloc(&dsr,ns*sizeof(Code)),"r6 cart factor suffix rank");ck(cudaMemcpy(dpc,cart.pc[h].data(),np*4,cudaMemcpyHostToDevice),"r6 cart pc cp");ck(cudaMemcpy(dsc,cart.sc[h].data(),ns*4,cudaMemcpyHostToDevice),"r6 cart sc cp");ck(cudaMemcpy(dfp,cart.fp[h].data(),np*sizeof(Code),cudaMemcpyHostToDevice),"r6 cart factor prefix cp");ck(cudaMemcpy(dsr,cart.sr[h].data(),ns*sizeof(Code),cudaMemcpyHostToDevice),"r6 cart factor suffix cp");bool useTasks=true;if(const char*e=std::getenv("GRIDFP_ROW6_TASKS"))useTasks=std::atoi(e)!=0;if(!useTasks)throw std::runtime_error("factor-authoritative row6 requires task mode");std::vector<Row6CartTask> ht;ht.reserve(np);Code fend=base+mainLen[dev];auto const&sv=cart.sr[h];for(uint32_t ip=0;ip<np;++ip){Code rb=cart.fp[h][ip];auto lb=(base<=rb)?sv.begin():std::lower_bound(sv.begin(),sv.end(),base-rb);auto ub=(fend<=rb)?sv.begin():std::lower_bound(sv.begin(),sv.end(),fend-rb);uint32_t a=uint32_t(lb-sv.begin()),b=uint32_t(ub-sv.begin());if(a<b){cartPairCount[dev]+=uint64_t(b-a);for(uint32_t x=a;x<b;x+=taskChunk){ht.push_back({ip,x,std::min<uint32_t>(b,x+taskChunk)});++cartTaskCount[dev];}}}Row6CartTask*dt=nullptr;if(!ht.empty()){ck(cudaMalloc(&dt,ht.size()*sizeof(Row6CartTask)),"r6 cart tasks");ck(cudaMemcpy(dt,ht.data(),ht.size()*sizeof(Row6CartTask),cudaMemcpyHostToDevice),"r6 cart tasks cp");int gb=int(ht.size());int hlanes=lanes;{char en[64];std::snprintf(en,sizeof(en),"GRIDFP_ROW6_LANES_H%d",h);if(const char*e=std::getenv(en))hlanes=std::atoi(e);}if(hlanes==16)row6_cartesian_task_factor_kernel<16><<<gb,threads>>>(fullMain[dev],base,dpc,dfp,dsc,dsr,dt,ht.size(),r6_block_size(h));else if(hlanes==8)row6_cartesian_task_factor_kernel<8><<<gb,threads>>>(fullMain[dev],base,dpc,dfp,dsc,dsr,dt,ht.size(),r6_block_size(h));else if(hlanes==2)row6_cartesian_task_factor_kernel<2><<<gb,threads>>>(fullMain[dev],base,dpc,dfp,dsc,dsr,dt,ht.size(),r6_block_size(h));else if(hlanes==1)row6_cartesian_task_factor_kernel<1><<<gb,threads>>>(fullMain[dev],base,dpc,dfp,dsc,dsr,dt,ht.size(),r6_block_size(h));else row6_cartesian_task_factor_kernel<4><<<gb,threads>>>(fullMain[dev],base,dpc,dfp,dsc,dsr,dt,ht.size(),r6_block_size(h));ck(cudaGetLastError(),"r6 factor task launch");cudaFree(dt);}cudaFree(dpc);cudaFree(dsc);cudaFree(dfp);cudaFree(dsr);}
        }else if(cnt){bool tiled4=(lanes==4&&threads==256);if(const char*e=std::getenv("GRIDFP_ROW6_TILED4"))tiled4=std::atoi(e)!=0;if(tiled4)bounded_fill_row6_tiled4_range_kernel<<<std::max(1,blocks),256>>>(fullMain[dev],cut[dev],cut[dev+1],base);else if(lanes==16)bounded_fill_row6_full_group_range_kernel<16><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else if(lanes==8)bounded_fill_row6_full_group_range_kernel<8><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else if(lanes==4)bounded_fill_row6_full_group_range_kernel<4><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else if(lanes==2)bounded_fill_row6_full_group_range_kernel<2><<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);else bounded_fill_row6_full_range_kernel<<<std::max(1,blocks),threads>>>(fullMain[dev],cut[dev],cut[dev+1],base);}ck(cudaDeviceSynchronize(),"r6 levelpack init");init_s[dev]=std::chrono::duration<double>(std::chrono::steady_clock::now()-ib).count();cudaFree(P);cudaFree(S);cudaFree(dpidx);cudaFree(dsidx);cudaFree(dmat);cudaFree(dbeta);});for(auto&t:workers)t.join();double gw=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count(),tm=*std::max_element(table_s.begin(),table_s.end()),im=*std::max_element(init_s.begin(),init_s.end());uint64_t pairSum=0,taskSum=0;for(int d=0;d<ng;++d){pairSum+=cartPairCount[d];taskSum+=cartTaskCount[d];}bool useCart=true;if(const char*e=std::getenv("GRIDFP_ROW6_CARTESIAN"))useCart=std::atoi(e)!=0;if(useCart&&pairSum!=n)throw std::runtime_error("row6 Cartesian shard pair sum mismatch");std::cerr<<"direct row6 GPU-level-packed states="<<n<<" lut_host_s="<<host_s<<" table_kernel_max_s="<<tm<<" init_kernel_max_s="<<im<<" gpu_phase_wall_s="<<gw<<" pref_mib="<<(pi.second*4>>20)<<" suff_mib="<<(si.second*4>>20)<<" pidx_mib="<<(pi.first.size()*4>>20)<<" sidx_mib="<<(si.first.size()*4>>20)<<" mat_kib="<<(mat.size()*4>>10)<<" level_threads="<<rt<<" cart_task_chunk="<<taskChunk<<" cart_pairs="<<pairSum<<" cart_tasks="<<taskSum<<"\n";
}

static Count* build_direct_row4_lut_compact_single_gpu(int W,Count mod,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2]){
    using namespace oneesan::row4mod;if(mod!=MOD){std::cerr<<"row4 LUT prototype only supports modulus "<<MOD<<"\n";return nullptr;}cudaSetDevice(0);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(4,dp);outN=dp[W][1];std::memcpy(outDp,dp,sizeof(dp));ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row4 lut dp");int cap=4;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row4 lut cap");constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto prefFull=make_row4_prefix_lut(hi,mod);auto suffFull=make_row4_suffix_lut(lo,mod);auto pref=compact_row4_lut(prefFull,hi,true);auto suff=compact_row4_lut(suffFull,lo,false);prefFull.clear();prefFull.shrink_to_fit();suffFull.clear();suffFull.shrink_to_fit();double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();Count *dpref=nullptr,*dsuff=nullptr;ck(cudaMalloc(&dpref,pref.size()*sizeof(Count)),"row4 pref");ck(cudaMalloc(&dsuff,suff.size()*sizeof(Count)),"row4 suff");ck(cudaMemcpy(dpref,pref.data(),pref.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 pref copy");ck(cudaMemcpy(dsuff,suff.data(),suff.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 suff copy");ck(cudaMemcpyToSymbol(D_R4_PREF,&dpref,sizeof(dpref)),"row4 pref ptr");ck(cudaMemcpyToSymbol(D_R4_SUFF,&dsuff,sizeof(dsuff)),"row4 suff ptr");Count*cur=nullptr;ck(cudaMalloc(&cur,size_t(outN)*sizeof(Count)),"row4 compact out");int b=int(std::min<Code>(65535,(outN+threads-1)/threads));auto g0=std::chrono::steady_clock::now();bounded_fill_row4_lut_kernel<<<std::max(1,b),threads>>>(cur,outN,W);ck(cudaDeviceSynchronize(),"row4 lut init");double gpu_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();cudaFree(dpref);cudaFree(dsuff);std::cerr<<"direct row4 lut states="<<outN<<" lut_host_s="<<host_s<<" kernel_s="<<gpu_s<<" pref_mib="<<(pref.size()*sizeof(Count)>>20)<<" suff_mib="<<(suff.size()*sizeof(Count)>>20)<<"\n";return cur;
}

static std::vector<Count> make_row3_prefix_lut(int len,Count mod){using namespace oneesan::row3auto;std::vector<Count>cur(DIM);cur[0]=1;size_t cnt=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(size_t c=0;c<cnt;++c)for(int a=0;a<3;++a)row3_apply_row(&cur[c*DIM],&nxt[(c*3+a)*DIM],a,mod);cur.swap(nxt);cnt*=3;}return cur;}
static std::vector<Count> make_row3_suffix_lut(int len,Count mod){using namespace oneesan::row3auto;std::vector<Count>cur(DIM);for(int i=0;i<DIM;++i)cur[i]=Count(BETA[i]%mod);size_t cnt=1,pow3=1;for(int d=0;d<len;++d){std::vector<Count>nxt(cnt*3*DIM);for(int a=0;a<3;++a)for(size_t c=0;c<cnt;++c)row3_apply_col(a,&cur[c*DIM],&nxt[(size_t(a)*pow3+c)*DIM],mod);cur.swap(nxt);cnt*=3;pow3*=3;}return cur;}
static Count* build_direct_row3_lut_compact_single_gpu(int W,Count mod,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2]){
    using namespace oneesan::row3auto;cudaSetDevice(0);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(3,dp);outN=dp[W][1];std::memcpy(outDp,dp,sizeof(dp));ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row3 lut dp");int cap=3;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row3 lut cap");
    constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto pref=make_row3_prefix_lut(hi,mod);auto suff=make_row3_suffix_lut(lo,mod);double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    Count *dpref=nullptr,*dsuff=nullptr;ck(cudaMalloc(&dpref,pref.size()*sizeof(Count)),"row3 pref");ck(cudaMalloc(&dsuff,suff.size()*sizeof(Count)),"row3 suff");ck(cudaMemcpy(dpref,pref.data(),pref.size()*sizeof(Count),cudaMemcpyHostToDevice),"row3 pref copy");ck(cudaMemcpy(dsuff,suff.data(),suff.size()*sizeof(Count),cudaMemcpyHostToDevice),"row3 suff copy");ck(cudaMemcpyToSymbol(D_R3_PREF,&dpref,sizeof(dpref)),"row3 pref ptr");ck(cudaMemcpyToSymbol(D_R3_SUFF,&dsuff,sizeof(dsuff)),"row3 suff ptr");Count scale=host_pow_mod(6,W,mod),inv=host_pow_mod(scale,(unsigned long long)mod-2,mod);ck(cudaMemcpyToSymbol(D_R3_INV_SCALE,&inv,sizeof(inv)),"row3 inv scale");
    Count*cur=nullptr;ck(cudaMalloc(&cur,size_t(outN)*sizeof(Count)),"row3 compact out");int b=int(std::min<Code>(65535,(outN+threads-1)/threads));auto g0=std::chrono::steady_clock::now();bounded_fill_row3_lut_kernel<<<std::max(1,b),threads>>>(cur,outN,W);ck(cudaDeviceSynchronize(),"row3 lut init");double gpu_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();cudaFree(dpref);cudaFree(dsuff);std::cerr<<"direct row3 lut states="<<outN<<" lut_host_s="<<host_s<<" kernel_s="<<gpu_s<<" pref_mib="<<(pref.size()*sizeof(Count)>>20)<<" suff_mib="<<(suff.size()*sizeof(Count)>>20)<<"\n";return cur;
}

static void init_direct_row4_lut_inplace_single_gpu(int W,Count mod,int threads,Count*fullMain){
    using namespace oneesan::row4mod;cudaSetDevice(0);Code dp[MAXW+1][MAXW+2]{};build_bounded_dp(4,dp);Code n=dp[W][1];ck(cudaMemcpyToSymbol(D_BOUND_DP,dp,sizeof(dp)),"row4 inplace dp");int cap=4;ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row4 inplace cap");constexpr int lo=TARGET_W/2,hi=TARGET_W-lo;auto t0=std::chrono::steady_clock::now();auto prefFull=make_row4_prefix_lut(hi,mod);auto suffFull=make_row4_suffix_lut(lo,mod);auto pref=compact_row4_lut(prefFull,hi,true);auto suff=compact_row4_lut(suffFull,lo,false);double host_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();Count *dpref=nullptr,*dsuff=nullptr;ck(cudaMalloc(&dpref,pref.size()*sizeof(Count)),"row4 inplace pref");ck(cudaMalloc(&dsuff,suff.size()*sizeof(Count)),"row4 inplace suff");ck(cudaMemcpy(dpref,pref.data(),pref.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 inplace pref copy");ck(cudaMemcpy(dsuff,suff.data(),suff.size()*sizeof(Count),cudaMemcpyHostToDevice),"row4 inplace suff copy");ck(cudaMemcpyToSymbol(D_R4_PREF,&dpref,sizeof(dpref)),"row4 inplace pref ptr");ck(cudaMemcpyToSymbol(D_R4_SUFF,&dsuff,sizeof(dsuff)),"row4 inplace suff ptr");auto g0=std::chrono::steady_clock::now();int b=int(std::min<Code>(65535,(n+threads-1)/threads));bounded_fill_row4_lut_kernel<<<std::max(1,b),threads>>>(fullMain,n,W);ck(cudaDeviceSynchronize(),"row4 inplace init");double gpu_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();cudaFree(dpref);cudaFree(dsuff);std::cerr<<"direct row4 inplace states="<<n<<" lut_host_s="<<host_s<<" kernel_s="<<gpu_s<<"\n";bounded_expand_inplace_single_gpu(fullMain,n,W,threads);
}

static Count* build_direct_row2_bounded_compact_single_gpu(int W,int K,int threads,Code&outN,Code outDp[MAXW+1][MAXW+2]){
    if(K<2)return nullptr;
    cudaSetDevice(0); Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(2,dpOld); Code oldN=dpOld[W][1]; Count*cur=nullptr;
    ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"row2 compact base");
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"row2 compact dp"); int cap2=2; ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap2,sizeof(cap2)),"row2 compact cap");
    int bi=int(std::min<Code>(65535,(oldN+threads-1)/threads)); bounded_fill_row2_automaton_kernel<<<std::max(1,bi),threads>>>(cur,oldN,W); ck(cudaDeviceSynchronize(),"row2 compact init");
    std::cerr<<"direct row2 compact states="<<oldN<<"\n";
    for(int cap=3;cap<=K;++cap){
        build_bounded_dp(cap,dpNew);Code n=dpNew[W][1],dn=dpNew[W-1][1];Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;
        ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"delay bounded a");ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"delay bounded b");ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"delay bounded d");ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"delay bounded e");
        ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"delay bounded zero a");ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"delay bounded zero d");
        ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"delay bounded old dp");ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"delay bounded new dp");int oldcap=cap-1;ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"delay bounded old cap");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"delay bounded cap");
        int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads)),be=int(std::min<Code>(65535,(oldN+threads-1)/threads));
        bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,W);ck(cudaDeviceSynchronize(),"delay bounded embed");cudaFree(cur);cur=a;Count*nxt=b,*dc=d,*de=e;
        for(int p=W-1;p>=1;--p){if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"delay bounded clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,W,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,W,p);}else{ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"delay bounded identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"delay bounded clear d");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,W,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,W,p);}ck(cudaDeviceSynchronize(),"delay bounded step");std::swap(cur,nxt);std::swap(dc,de);}
        cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;std::cerr<<"delay bounded cap="<<cap<<" states="<<n<<" block="<<dn<<"\n";
    }
    outN=oldN;std::memcpy(outDp,dpOld,sizeof(dpOld));return cur;
}

static void run_direct_row2_bounded_prefix_single_gpu(int W,int K,Count* fullMain,int threads){
    if(K<2){run_bounded_prefix_single_gpu(W,K,fullMain,threads);return;}
    cudaSetDevice(0);
    Code dpOld[MAXW+1][MAXW+2]{},dpNew[MAXW+1][MAXW+2]{};
    build_bounded_dp(2,dpOld); Code oldN=dpOld[W][1];
    Count*cur=nullptr; ck(cudaMalloc(&cur,size_t(oldN)*sizeof(Count)),"row2 auto compact");
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"row2 auto dp"); int cap2=2; ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap2,sizeof(cap2)),"row2 auto cap");
    int bi=int(std::min<Code>(65535,(oldN+threads-1)/threads));
    bounded_fill_row2_automaton_kernel<<<std::max(1,bi),threads>>>(cur,oldN,W);
    ck(cudaDeviceSynchronize(),"row2 auto init");
    std::cerr<<"direct row2 compact states="<<oldN<<"\n";
    for(int cap=3;cap<=K;++cap){
        build_bounded_dp(cap,dpNew); Code n=dpNew[W][1],dn=dpNew[W-1][1];
        Count *a=nullptr,*b=nullptr,*d=nullptr,*e=nullptr;
        ck(cudaMalloc(&a,size_t(n)*sizeof(Count)),"row2bound a"); ck(cudaMalloc(&b,size_t(n)*sizeof(Count)),"row2bound b");
        ck(cudaMalloc(&d,size_t(dn)*sizeof(Count)),"row2bound d"); ck(cudaMalloc(&e,size_t(dn)*sizeof(Count)),"row2bound e");
        ck(cudaMemset(a,0,size_t(n)*sizeof(Count)),"row2bound zero a"); ck(cudaMemset(d,0,size_t(dn)*sizeof(Count)),"row2bound zero d");
        ck(cudaMemcpyToSymbol(D_BOUND_OLD_DP,dpOld,sizeof(dpOld)),"row2bound old dp"); ck(cudaMemcpyToSymbol(D_BOUND_DP,dpNew,sizeof(dpNew)),"row2bound new dp");
        int oldcap=cap-1; ck(cudaMemcpyToSymbol(D_BOUND_OLD_CAP,&oldcap,sizeof(oldcap)),"row2bound old cap"); ck(cudaMemcpyToSymbol(D_BOUND_CAP,&cap,sizeof(cap)),"row2bound cap");
        int bm=int(std::min<Code>(65535,(n+threads-1)/threads)),bd=int(std::min<Code>(65535,(dn+threads-1)/threads)),be=int(std::min<Code>(65535,(oldN+threads-1)/threads));
        bounded_embed_kernel<<<std::max(1,be),threads>>>(cur,oldN,a,W); ck(cudaDeviceSynchronize(),"row2bound embed"); cudaFree(cur); cur=a;
        Count*nxt=b,*dc=d,*de=e;
        for(int p=W-1;p>=1;--p){
            if(p>1){ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"row2bound clear d");bounded_reverse_main_kernel<<<std::max(1,bm),threads>>>(cur,dc,n,nxt,W,p);bounded_forward_block_kernel<<<std::max(1,bm),threads>>>(cur,n,de,W,p);}
            else {ck(cudaMemcpy(nxt,cur,size_t(n)*sizeof(Count),cudaMemcpyDeviceToDevice),"row2bound identity");ck(cudaMemset(de,0,size_t(dn)*sizeof(Count)),"row2bound clear d");bounded_main_kernel<<<std::max(1,bm),threads>>>(cur,n,nxt,de,W,p);bounded_block_kernel<<<std::max(1,bd),threads>>>(dc,dn,nxt,W,p);}
            ck(cudaDeviceSynchronize(),"row2bound step"); std::swap(cur,nxt); std::swap(dc,de);
        }
        cudaFree(nxt);cudaFree(dc);cudaFree(de);std::memcpy(dpOld,dpNew,sizeof(dpOld));oldN=n;
        std::cerr<<"direct row2 bounded cap="<<cap<<" states="<<n<<" block="<<dn<<"\n";
    }
    ck(cudaMemcpyToSymbol(D_BOUND_DP,dpOld,sizeof(dpOld)),"row2bound final dp"); ck(cudaMemcpyToSymbol(D_BOUND_CAP,&K,sizeof(K)),"row2bound final cap");
    int bs=int(std::min<Code>(65535,(oldN+threads-1)/threads)); bounded_scatter_full_kernel<<<std::max(1,bs),threads>>>(cur,oldN,W); ck(cudaDeviceSynchronize(),"row2bound scatter full"); cudaFree(cur);
}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):16;
    int target_mib=argc>2?std::atoi(argv[2]):16384;
    int max_window=argc>3?std::atoi(argv[3]):14;
    int requested=argc>4?std::atoi(argv[4]):0;
    std::vector<Count> mods;
    for(int i=5;i<argc;++i){
        unsigned long long raw=std::strtoull(argv[i],nullptr,10);
        if(raw<2||raw>0xffffffffULL){std::cerr<<"HBM32 modulus must be in [2, 4294967295], got "<<raw<<"\n";return 1;}
        mods.push_back((Count)raw);
    }
    if(mods.empty())mods.push_back(4294967291u);
    int W=n+1;
    if(n<2||W>MAXW){std::cerr<<"n=2..27\n";return 1;}
    if(W!=TARGET_W){std::cerr<<"specialized for width "<<TARGET_W<<" (n="<<(TARGET_W-1)<<")\n";return 1;}
    build_full_dp();
    for(Count mod:mods){
        if(oneesan::row6crt::prime_index(mod)<0){
            std::cerr<<"unsupported row6 CRT modulus: "<<mod<<"; rebuild row6_automaton_crt20_generated.hpp for the solver prime set\n";
            return 34;
        }
    }
    G_FACTOR=build_factor_tables();G_F_FULL_MAIN_BLOCKS=make_factor_full_main_blocks();G_F_FULL_BLOCK_BLOCKS=make_factor_full_block_blocks();
    if(G_F_FULL_MAIN_BLOCKS.empty()||G_F_FULL_MAIN_BLOCKS.back().end!=H_DP[W][1]||G_F_FULL_BLOCK_BLOCKS.empty()||G_F_FULL_BLOCK_BLOCKS.back().end!=H_DP[W-1][1])throw std::runtime_error("factor authoritative cardinality mismatch");
    G_PRERANK=build_prerank_orbit_tables();G_HPR=build_high_prerank_tables();
    if(const char*e=std::getenv("GRIDFP_PLAN_ONLY");e&&std::atoi(e)!=0){
        int png=requested>0?requested:8;if(png<1||png>MAXGPU){std::cerr<<"plan requires 1..8 GPUs\n";return 2;}
        Code pmain=H_DP[W][1],pblock=H_DP[W-1][1],pch=(pmain+png-1)/png;
        Code bdp[MAXW+1][MAXW+2]{};build_bounded_dp(6,bdp);Code r6n=bdp[W][1];
        Row6CartLists cart=make_row6_cart_lists(W);uint64_t pairs=0,pvals=0,svals=0;for(int h=0;h<=6;++h){pairs+=uint64_t(cart.pc[h].size())*cart.sc[h].size();pvals+=uint64_t(cart.pc[h].size())*r6_block_size(h);svals+=uint64_t(cart.sc[h].size())*r6_block_size(h);}
        if(pairs!=r6n){std::cerr<<"plan row6 Cartesian mismatch\n";return 33;}
        uint32_t tch=(TARGET_W>=28?65536u:16384u);if(const char*x=std::getenv("GRIDFP_ROW6_TASK_CHUNK"))tch=std::max(64,std::atoi(x));
        auto vb4=[](auto const&v)->uint64_t{return uint64_t(v.size())*4;};auto vb8=[](auto const&v)->uint64_t{return uint64_t(v.size())*8;};
        uint64_t vmmGran=0;if(const char*x=std::getenv("GRIDFP_VMM_GRAN_KIB")){uint64_t k=std::strtoull(x,nullptr,10);if(k)vmmGran=k<<10;}if(!vmmGran){int dc=0;if(cuInit(0)==CUDA_SUCCESS&&cuDeviceGetCount(&dc)==CUDA_SUCCESS&&dc>0)vmmGran=vmm_granularity_for_device(0);else vmmGran=2ull<<20;}
        auto vmmMapped=[&](size_t denseElems,std::vector<uint32_t> const&codes)->uint64_t{uint64_t raw=uint64_t(denseElems)*4,elemsPer=vmmGran/4,np=(raw+vmmGran-1)/vmmGran;std::vector<uint8_t> used(np,0);for(uint32_t c:codes)used[uint64_t(c)/elemsPer]=1;uint64_t n=0;for(uint8_t x:used)n+=x!=0;return n*vmmGran;};
        uint64_t lowVmmBytes=vmmMapped(size_t(1)<<(2*LOW_LUT_K),G_FACTOR.low_all_codes),highVmmBytes=vmmMapped(size_t(1)<<(2*HIGH_LUT_K),G_FACTOR.high_all_codes);
        uint64_t sparseBytes=0;uint64_t preRankBytes=uint64_t(G_PRERANK.owner_block_rec.size()+G_PRERANK.highrr_rec.size()+G_PRERANK.closure_inv_rec.size()+G_HPR.owner_block_rec.size()+G_HPR.crossll_rec.size()+G_HPR.closure_inv_rec.size())*8+uint64_t(G_PRERANK.closure_inv_src.size()+G_HPR.closure_inv_src.size())*4;uint64_t factorBytes=vb4(G_FACTOR.low_all_codes)+vb4(G_FACTOR.low_mask_codes)+vb4(G_FACTOR.low_mask_off)+lowVmmBytes+vb4(G_FACTOR.high_all_codes)+vb4(G_FACTOR.high_mask_codes)+vb4(G_FACTOR.high_mask_off)+highVmmBytes+vb8(G_FACTOR.high_main_base)+vb8(G_FACTOR.high_block_base)+sparseBytes+preRankBytes;
        Code pbch=(pblock+png-1)/png;uint64_t authMaxBytes=uint64_t(pch+pbch)*4;
        constexpr int r6lo=TARGET_W/2,r6hi=TARGET_W-r6lo;
        uint64_t matElems=0;for(int h=0;h<=6;++h){uint64_t a=r6_block_size(h);matElems+=a*a;if(h<6)matElems+=2*a*uint64_t(r6_block_size(h+1));}
        uint64_t r6Base=matElems*4+uint64_t(r6_block_size(0))*4;
        auto ppl=make_row6_packed_level_meta(r6hi,true),spl=make_row6_packed_level_meta(r6lo,false);
        auto liveLevel=[](Row6PackedLevelMeta const&m)->uint64_t{return m.elems*4+uint64_t(m.idx.size())*4+uint64_t(m.height.size());};
        auto retained=[](Row6PackedLevelMeta const&m)->uint64_t{return m.elems*4+uint64_t(m.idx.size())*4;};
        uint64_t prefBuild=liveLevel(ppl[0]);for(int i=0;i<r6hi;++i)prefBuild=std::max(prefBuild,liveLevel(ppl[i])+liveLevel(ppl[i+1]));
        uint64_t prefKeep=retained(ppl.back());
        uint64_t suffBuild=prefKeep+liveLevel(spl[0]);for(int i=0;i<r6lo;++i)suffBuild=std::max(suffBuild,prefKeep+liveLevel(spl[i])+liveLevel(spl[i+1]));
        uint64_t suffKeep=retained(spl.back());
        uint64_t row6Extra=r6Base+std::max({prefBuild,suffBuild,prefKeep+suffKeep});
        uint64_t scratchLimit=uint64_t(std::max(1,target_mib))<<20,row6Peak=authMaxBytes+factorBytes+row6Extra;
        uint64_t minScratchBytes=0,actualScratchBytes=0,totalGroups=0,mateGroups=0,windowScratch[2]={0,0};{const int rr[2][2]={{W-1,LOW_LUT_K+1},{LOW_LUT_K,1}};int wi=0;for(auto const&r:rr){auto fp=window_candidates(W,r[0],r[1]);int nj=1<<int(fp.size());for(int g=0;g<nj;++g){uint32_t mf,mo,bf,bo;window_masks(W,r[0],r[1],fp,uint32_t(g),mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);bool orbit=true;uint64_t countBytes=uint64_t(ms.size+ds.size)*sizeof(Count),mateBytes=uint64_t(ms.size)*sizeof(MateID);windowScratch[wi]=std::max(windowScratch[wi],countBytes);bool directLower=false;if(const char*e=std::getenv("GRIDFP_DIRECT_AUTH")){int dm=std::atoi(e);directLower=(dm==1||dm==3)&&wi==1;}if(!directLower){minScratchBytes=std::max(minScratchBytes,countBytes);bool useMate=!orbit&&countBytes+mateBytes<=scratchLimit;actualScratchBytes=std::max(actualScratchBytes,countBytes+(useMate?mateBytes:0));mateGroups+=useMate;}totalGroups++;}++wi;}}
        std::cerr<<"PLAN window_scratch upper_gib="<<double(windowScratch[0])/(1ull<<30)<<" lower_gib="<<double(windowScratch[1])/(1ull<<30)<<" effective_gib="<<double(actualScratchBytes)/(1ull<<30)<<"\n";
        bool scratchFits=minScratchBytes<=scratchLimit;uint64_t dpPeak=authMaxBytes+factorBytes+actualScratchBytes;
        std::cerr<<"PLAN n="<<n<<" W="<<W<<" gpus="<<png<<" main="<<pmain<<" blocked="<<pblock<<" auth_gib="<<double(pmain+pblock)*4.0/(1ull<<30)<<" auth_per_gpu_gib="<<double(authMaxBytes)/(1ull<<30)<<" factor_gpu_gib="<<double(factorBytes)/(1ull<<30)<<" sparse_orbit_mib="<<(sparseBytes>>20)<<" prerank_mib="<<(preRankBytes>>20)<<" vmm_gran_kib="<<(vmmGran>>10)<<" low_vmm_mib="<<(lowVmmBytes>>20)<<" high_vmm_mib="<<(highVmmBytes>>20)<<" row6_extra_gib="<<double(row6Extra)/(1ull<<30)<<" row6_peak_gib="<<double(row6Peak)/(1ull<<30)<<" scratch_limit_mib="<<target_mib<<" scratch_fits="<<(scratchFits?1:0)<<" count_scratch_gib="<<double(minScratchBytes)/(1ull<<30)<<" actual_scratch_gib="<<double(actualScratchBytes)/(1ull<<30)<<" mate_groups="<<mateGroups<<"/"<<totalGroups<<" dp_peak_gib="<<double(dpPeak)/(1ull<<30)<<" min_dp_peak_gib="<<double(authMaxBytes+factorBytes+minScratchBytes)/(1ull<<30)<<" row6_states="<<r6n<<" pref_mib="<<(pvals*4>>20)<<" suff_mib="<<(svals*4>>20)<<" task_chunk="<<tch<<"\n";
        uint64_t sumPairs=0,sumTasks=0;for(int d=0;d<png;++d){Code base=Code(d)*pch,end=std::min<Code>(pmain,base+pch);uint64_t npair=0,ntask=0,work=0;for(int h=0;h<=6;++h){auto const&sv=cart.sr[h];for(size_t ip=0;ip<cart.fp[h].size();++ip){Code rb=cart.fp[h][ip];auto lb=(base<=rb)?sv.begin():std::lower_bound(sv.begin(),sv.end(),base-rb);auto ub=(end<=rb)?sv.begin():std::lower_bound(sv.begin(),sv.end(),end-rb);uint64_t z=uint64_t(ub-lb);if(z){npair+=z;work+=z*uint64_t(r6_block_size(h));ntask+=(z+tch-1)/tch;}}}sumPairs+=npair;sumTasks+=ntask;std::cerr<<"PLAN gpu="<<d<<" pairs="<<npair<<" work="<<work<<" tasks="<<ntask<<" owner=["<<base<<","<<end<<")\n";}
        std::cerr<<"PLAN pair_sum="<<sumPairs<<" task_sum="<<sumTasks<<" crt_moduli="<<mods.size();for(Count m:mods)std::cerr<<" "<<m<<":"<<(oneesan::row6crt::prime_index(m)>=0?"ok":"BAD");std::cerr<<"\n";return 0;
    }
    int visible=0;ck(cudaGetDeviceCount(&visible),"count");
    int ng=requested<=0?visible:std::min(requested,visible);
    if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\n";return 2;}
    int peers=0;
    for(int a=0;a<ng;++a)for(int b=0;b<ng;++b)if(a!=b){int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"can peer");if(can){cudaSetDevice(a);auto e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer");peers++;}}
    if(ng>1&&peers!=ng*(ng-1)){std::cerr<<"HBM mode requires full P2P: "<<peers<<"/"<<ng*(ng-1)<<"\n";return 3;}

    SparseVmmU32 fLDv[MAXGPU]{},fHRv[MAXGPU]{};uint32_t *fLA[MAXGPU]{},*fLM[MAXGPU]{},*fLO[MAXGPU]{},*fLOB[MAXGPU]{},*fLD[MAXGPU]{},*fHA[MAXGPU]{},*fHM[MAXGPU]{},*fHO[MAXGPU]{},*fHOB[MAXGPU]{},*fHR[MAXGPU]{};unsigned long long *prOB[MAXGPU]{},*prCI[MAXGPU]{},*prH[MAXGPU]{},*hpOB[MAXGPU]{},*hpCI[MAXGPU]{},*hpX[MAXGPU]{};uint32_t *prCIS[MAXGPU]{},*hpCIS[MAXGPU]{};Code *fHMB[MAXGPU]{},*fHBB[MAXGPU]{};
    for(int d=0;d<ng;++d){cudaSetDevice(d);ck(cudaMemcpyToSymbol(D_F_FULL_MAIN_BLOCKS,G_F_FULL_MAIN_BLOCKS.data(),G_F_FULL_MAIN_BLOCKS.size()*sizeof(FBlock)),"full factor main blocks");ck(cudaMemcpyToSymbol(D_F_FULL_BLOCK_BLOCKS,G_F_FULL_BLOCK_BLOCKS.data(),G_F_FULL_BLOCK_BLOCKS.size()*sizeof(FBlock)),"full factor block blocks");auto cp=[&](uint32_t**dst,const std::vector<uint32_t>&v,const char*w){if(v.empty())return;ck(cudaMalloc(dst,v.size()*sizeof(uint32_t)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(uint32_t),cudaMemcpyHostToDevice),w);};cp(&fLA[d],G_FACTOR.low_all_codes,"f low all");cp(&fLM[d],G_FACTOR.low_mask_codes,"f low mask");cp(&fLO[d],G_FACTOR.low_mask_off,"f low off");cp(&fLOB[d],G_FACTOR.low_occ_base,"f low occ base");fLDv[d]=make_sparse_vmm_u32(d,size_t(1)<<(2*LOW_LUT_K),G_FACTOR.low_all_codes,G_FACTOR.low_packed_values,"LOW");fLD[d]=fLDv[d].ptr();cp(&fHA[d],G_FACTOR.high_all_codes,"f high all");cp(&fHM[d],G_FACTOR.high_mask_codes,"f high mask");cp(&fHO[d],G_FACTOR.high_mask_off,"f high off");cp(&fHOB[d],G_FACTOR.high_occ_base,"f high occ base");fHRv[d]=make_sparse_vmm_u32(d,size_t(1)<<(2*HIGH_LUT_K),G_FACTOR.high_all_codes,G_FACTOR.high_packed_values,"HIGH");fHR[d]=fHRv[d].ptr();auto cpc=[&](Code**dst,const std::vector<Code>&v,const char*w){ck(cudaMalloc(dst,v.size()*sizeof(Code)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(Code),cudaMemcpyHostToDevice),w);};cpc(&fHMB[d],G_FACTOR.high_main_base,"f high main base");cpc(&fHBB[d],G_FACTOR.high_block_base,"f high block base");ck(cudaMemcpyToSymbol(D_F_LOW_ALL_CODES,&fLA[d],sizeof(fLA[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_MASK_CODES,&fLM[d],sizeof(fLM[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_MASK_OFF,&fLO[d],sizeof(fLO[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_OCC_BASE,&fLOB[d],sizeof(fLOB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_DENSE_PACKED_RANK,&fLD[d],sizeof(fLD[d])),"f dense ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_CODES,&fHA[d],sizeof(fHA[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_CODES,&fHM[d],sizeof(fHM[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_OFF,&fHO[d],sizeof(fHO[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_OCC_BASE,&fHOB[d],sizeof(fHOB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_PACKED_RANK,&fHR[d],sizeof(fHR[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_MAIN_BASE,&fHMB[d],sizeof(fHMB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_HIGH_BLOCK_BASE,&fHBB[d],sizeof(fHBB[d])),"f ptr");ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"f low all off");ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF,G_FACTOR.high_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"f high all off");auto cp8=[&](unsigned long long**dst,const std::vector<unsigned long long>&v,const char*w){if(v.empty())return;ck(cudaMalloc(dst,v.size()*sizeof(unsigned long long)),w);ck(cudaMemcpy(*dst,v.data(),v.size()*sizeof(unsigned long long),cudaMemcpyHostToDevice),w);};cp8(&prOB[d],G_PRERANK.owner_block_rec,"pr owner blockorder");cp8(&prCI[d],G_PRERANK.closure_inv_rec,"pr closure inv");cp(&prCIS[d],G_PRERANK.closure_inv_src,"pr closure inv src");cp8(&prH[d],G_PRERANK.highrr_rec,"pr highrr");ck(cudaMemcpyToSymbol(D_PR_OWNER_BLOCK_REC,&prOB[d],sizeof(prOB[d])),"pr owner block ptr");ck(cudaMemcpyToSymbol(D_PR_CLOSURE_INV_REC,&prCI[d],sizeof(prCI[d])),"pr inv ptr");ck(cudaMemcpyToSymbol(D_PR_CLOSURE_INV_SRC,&prCIS[d],sizeof(prCIS[d])),"pr inv src ptr");ck(cudaMemcpyToSymbol(D_PR_HIGHRR_REC,&prH[d],sizeof(prH[d])),"pr ptr");cp8(&hpOB[d],G_HPR.owner_block_rec,"hp owner blockorder");cp8(&hpCI[d],G_HPR.closure_inv_rec,"hp closure inv");cp(&hpCIS[d],G_HPR.closure_inv_src,"hp closure inv src");cp8(&hpX[d],G_HPR.crossll_rec,"hp crossll");ck(cudaMemcpyToSymbol(D_HP_OWNER_BLOCK_REC,&hpOB[d],sizeof(hpOB[d])),"hp owner block ptr");ck(cudaMemcpyToSymbol(D_HP_CLOSURE_INV_REC,&hpCI[d],sizeof(hpCI[d])),"hp inv ptr");ck(cudaMemcpyToSymbol(D_HP_CLOSURE_INV_SRC,&hpCIS[d],sizeof(hpCIS[d])),"hp inv src ptr");ck(cudaMemcpyToSymbol(D_HP_CROSSLL_REC,&hpX[d],sizeof(hpX[d])),"hp ptr");}

    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));cudaSetDevice(d);if(ml[d])ck(cudaMalloc(&mp[d],size_t(ml[d])*sizeof(Count)),"auth main");if(bl[d])ck(cudaMalloc(&bp[d],size_t(bl[d])*sizeof(Count)),"auth block");}
    std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d)ctx[d].init(d,mods[0],mp,bp,mc,bc,ng);
    size_t min_free=~size_t(0),min_total=~size_t(0);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set meminfo");size_t f=0,t=0;ck(cudaMemGetInfo(&f,&t),"cudaMemGetInfo");min_free=std::min(min_free,f);min_total=std::min(min_total,t);}
    int reserve_mib=std::min(8192,std::max(256,int((min_total>>20)/32)));if(const char*e=std::getenv("GRIDFP_VRAM_RESERVE_MIB")){int v=std::atoi(e);if(v>=0)reserve_mib=v;}
    size_t requested_target=size_t(std::max(1,target_mib))<<20;size_t reserve=size_t(reserve_mib)<<20;if(min_free<=reserve+(64ull<<20)){std::cerr<<"insufficient HBM after authoritative state: min_free_mib="<<(min_free>>20)<<" reserve_mib="<<reserve_mib<<"\n";return 5;}
    size_t target=std::min(requested_target,min_free-reserve);int effective_target_mib=int(target>>20);
    std::cerr<<"HBM32 batch memory: auth_gib="<<double(mainN+blockN)*sizeof(Count)/(1ull<<30)<<" auth_per_gpu_gib="<<double(mainN+blockN)*sizeof(Count)/ng/(1ull<<30)<<" min_total_gib="<<double(min_total)/(1ull<<30)<<" min_free_after_auth_gib="<<double(min_free)/(1ull<<30)<<" requested_scratch_mib="<<target_mib<<" effective_scratch_mib="<<effective_target_mib<<" reserve_mib="<<reserve_mib<<" moduli="<<mods.size()<<"\n";

    if constexpr(LOW_LUT_K+HIGH_LUT_K != TARGET_W-1){std::cerr<<"forced2 requires LOW+HIGH=W-1\n";return 4;}
    int threads=256;if(const char*e=std::getenv("GRIDFP_THREADS")){int v=std::atoi(e);if(v==64||v==128||v==256||v==512||v==1024)threads=v;else throw std::runtime_error("GRIDFP_THREADS must be 64,128,256,512,1024");}int maxgroups=0;auto prep0=std::chrono::steady_clock::now();std::vector<PreparedWindow> schedule;
    {
        const int ranges[2][2]={{W-1,LOW_LUT_K+1},{LOW_LUT_K,1}};
        for(auto const& r:ranges){
            int hi=r[0],lo=r[1];WindowPlan wp;wp.p_hi=hi;wp.p_lo=lo;wp.fixed_pos=window_candidates(W,hi,lo);
            int k=(int)wp.fixed_pos.size();int nj=1<<k;PreparedWindow pw;pw.wp=wp;pw.groups.reserve(nj);size_t mx=0;
            for(int g=0;g<nj;++g){auto pg=prepare_group(W,pw.wp,g,mc,bc,ng);size_t b=size_t(2*pg.ms.size+2*pg.ds.size)*sizeof(Count);mx=std::max(mx,b);pw.groups.push_back(std::move(pg));}
            if(mx>target){std::cerr<<"forced window does not fit p="<<hi<<".."<<lo<<" max_bytes="<<mx<<" target="<<target<<"\n";return 4;}
            maxgroups=std::max(maxgroups,nj);std::sort(pw.groups.begin(),pw.groups.end(),[](auto const&a,auto const&b){return a.work>b.work;});
            std::cerr<<"forced window p="<<hi<<".."<<lo<<" fixed="<<k<<" groups="<<nj<<" max_mib="<<(mx>>20)<<"\n";schedule.push_back(std::move(pw));
        }
    }
    double prepare_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-prep0).count();std::cerr<<"prepared windows="<<schedule.size()<<" max_groups="<<maxgroups<<" prepare_s="<<prepare_s<<"\n";

    MateID init=MateID(R)<<(2*(W-1));Code ig=rank_full(init,W);int io=int(ig/mc);Code fg=rank_full(MateID(R),W);int fo=int(fg/mc);Count one=1;
    for(size_t ri=0;ri<mods.size();++ri){Count mod=mods[ri];
        for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set residue reset");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"set modulus");if(ml[d])ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)),"zero main");if(bl[d])ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)),"zero block");ck(cudaDeviceSynchronize(),"zero sync");ctx[d].active=0;ctx[d].gather_s=ctx[d].transition_s=ctx[d].scatter_s=0;ctx[d].groups=0;}
        auto wall0=std::chrono::steady_clock::now();
        int prefixK=(W>=28?6:5);if(const char*e=std::getenv("GRIDFP_BOUNDED_PREFIX_K"))prefixK=std::max(1,std::min(W-2,std::atoi(e)));
        bool requestDirect5=(prefixK==5);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW5_LUT"))requestDirect5=std::atoi(e)!=0;
        bool requestDirect6=(prefixK==6);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW6_LUT"))requestDirect6=std::atoi(e)!=0;
        if(ng!=1&&prefixK>1&&!((requestDirect5&&prefixK==5)||(requestDirect6&&prefixK==6))){std::cerr<<"bounded prefix prototype currently requires NGPU=1 except direct row5/6\n";return 31;}
        if(prefixK==1){
            for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set row1 init device");Code total=Code(1)<<(W-1);Code mine=(total+ng-1-d)/ng;int blocks=int(std::min<Code>(65535,(mine+threads-1)/threads));if(mine)init_after_first_row_kernel<<<std::max(1,blocks),threads>>>(d,ng);}for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"row1 init sync set");ck(cudaDeviceSynchronize(),"row1 init sync");}
        }else {
            bool direct2=true;if(const char*e=std::getenv("GRIDFP_DIRECT_ROW2"))direct2=std::atoi(e)!=0;
            bool direct5=requestDirect5;
            bool direct6=requestDirect6;
            bool inplace4=(prefixK==4);if(const char*e=std::getenv("GRIDFP_INPLACE_ROW4"))inplace4=std::atoi(e)!=0;
            bool delayAuth=false;if(const char*e=std::getenv("GRIDFP_DELAY_AUTH"))delayAuth=std::atoi(e)!=0;
            if(direct6&&prefixK==6){
                init_direct_row6_lut_full_multi_gpu(W,mod,threads,mp,ml,mc,ng);
            }else if(direct5&&prefixK==5){
                init_direct_row5_lut_full_multi_gpu(W,mod,threads,mp,ml,mc,ng);
            }else if(inplace4&&prefixK==4&&ng==1){
                init_direct_row4_lut_inplace_single_gpu(W,mod,threads,mp[0]);
            }else if(delayAuth&&direct2&&ng==1){
                ck(cudaSetDevice(0),"delay auth set");if(mp[0]){cudaFree(mp[0]);mp[0]=nullptr;}if(bp[0]){cudaFree(bp[0]);bp[0]=nullptr;}
                Code compactN=0,compactDp[MAXW+1][MAXW+2]{};bool direct4=(prefixK==4);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW4_LUT"))direct4=std::atoi(e)!=0;bool direct3=(prefixK==3);if(const char*e=std::getenv("GRIDFP_DIRECT_ROW3_LUT"))direct3=std::atoi(e)!=0;Count*compact=direct4&&prefixK==4?build_direct_row4_lut_compact_single_gpu(W,mod,threads,compactN,compactDp):(direct3&&prefixK==3?build_direct_row3_lut_compact_single_gpu(W,mod,threads,compactN,compactDp):build_direct_row2_bounded_compact_single_gpu(W,prefixK,threads,compactN,compactDp));
                ck(cudaMalloc(&mp[0],size_t(ml[0])*sizeof(Count)),"delay auth main alloc");ck(cudaMemset(mp[0],0,size_t(ml[0])*sizeof(Count)),"delay auth main zero");
                ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"delay auth main ptr");ck(cudaMemcpyToSymbol(D_BOUND_DP,compactDp,sizeof(compactDp)),"delay auth scatter dp");ck(cudaMemcpyToSymbol(D_BOUND_CAP,&prefixK,sizeof(prefixK)),"delay auth scatter cap");
                int bs=int(std::min<Code>(65535,(compactN+threads-1)/threads));bounded_scatter_full_kernel<<<std::max(1,bs),threads>>>(compact,compactN,W);ck(cudaDeviceSynchronize(),"delay auth scatter");cudaFree(compact);
                ck(cudaMalloc(&bp[0],size_t(bl[0])*sizeof(Count)),"delay auth block alloc");ck(cudaMemset(bp[0],0,size_t(bl[0])*sizeof(Count)),"delay auth block zero");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"delay auth block ptr");
                size_t ff=0,tt=0;ck(cudaMemGetInfo(&ff,&tt),"delay auth meminfo");std::cerr<<"delay auth restored full arrays free_mib="<<(ff>>20)<<"\n";
            }else if(direct2)run_direct_row2_bounded_prefix_single_gpu(W,prefixK,mp[0],threads);else run_bounded_prefix_single_gpu(W,prefixK,mp[0],threads);
        }
        if(const char*e=std::getenv("GRIDFP_ROW6_INIT_ONLY")){if(std::atoi(e)!=0)return 0;}
        int done_windows=0;
        for(int row=prefixK;row<W-1;++row){for(auto const&pw:schedule){int nj=(int)pw.groups.size();std::atomic<int>next{0};std::vector<std::thread>ths;ths.reserve(ng);for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(;;){int q=next.fetch_add(1,std::memory_order_relaxed);if(q>=nj)break;process_group(ctx[d],W,pw.wp,pw.groups[q],threads,target);}});for(auto&t:ths)t.join();++done_windows;}std::cerr<<"mod "<<(ri+1)<<"/"<<mods.size()<<" p="<<mod<<" row "<<row+1<<"/"<<W<<"\n";}
        Count ans=0;
        for(int d=0;d<ng;++d){
            ck(cudaSetDevice(d),"final sum set");Count* da=nullptr;ck(cudaMalloc(&da,sizeof(Count)),"final sum malloc");ck(cudaMemset(da,0,sizeof(Count)),"final sum zero");
            Code total=Code(1)<<(W-1);Code mine=(total+ng-1-d)/ng;int blocks=int(std::min<Code>(65535,(mine+threads-1)/threads));
            if(mine)sum_first_row_states_kernel<<<std::max(1,blocks),threads,threads*sizeof(unsigned long long)>>>(d,ng,da);
            Count x=0;ck(cudaMemcpy(&x,da,sizeof(x),cudaMemcpyDeviceToHost),"final sum copy");cudaFree(da);ans=(ans>=mod-x)?ans-(mod-x):ans+x;
        }
        double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();double mx=0,sum=0,gs=0,ts=0,ss=0;size_t maxIntervals=0;for(auto&c:ctx){mx=std::max(mx,c.active);sum+=c.active;gs+=c.gather_s;ts+=c.transition_s;ss+=c.scatter_s;maxIntervals=std::max(maxIntervals,c.maxIntervals);}std::cout<<"backend=gridfp-b300-hbm32-occmajor-upperflat-direct-fullprerank-orbit-batch n="<<n<<" residue="<<ans<<" modulus="<<mod<<" residue_index="<<ri<<" residues_total="<<mods.size()<<" gpus="<<ng<<" peers="<<peers<<" main_states="<<mainN<<" blocked_states="<<blockN<<" scratch_target_mib="<<effective_target_mib<<" windows="<<done_windows<<" max_groups="<<maxgroups<<" max_intervals="<<maxIntervals<<" active_max_s="<<mx<<" active_sum_s="<<sum<<" gather_sum_s="<<gs<<" transition_sum_s="<<ts<<" scatter_sum_s="<<ss<<" prepare_s="<<prepare_s<<" wall_s="<<wall<<std::endl;
    }

    for(auto&c:ctx)c.destroy();for(int d=0;d<ng;++d){cudaSetDevice(d);if(mp[d])cudaFree(mp[d]);if(bp[d])cudaFree(bp[d]);if(fLA[d])cudaFree(fLA[d]);if(fLM[d])cudaFree(fLM[d]);if(fLO[d])cudaFree(fLO[d]);if(fLOB[d])cudaFree(fLOB[d]);fLDv[d].destroy();if(fHA[d])cudaFree(fHA[d]);if(fHM[d])cudaFree(fHM[d]);if(fHO[d])cudaFree(fHO[d]);if(fHOB[d])cudaFree(fHOB[d]);fHRv[d].destroy();if(fHMB[d])cudaFree(fHMB[d]);if(fHBB[d])cudaFree(fHBB[d]);if(prOB[d])cudaFree(prOB[d]);if(prCI[d])cudaFree(prCI[d]);if(prCIS[d])cudaFree(prCIS[d]);if(prH[d])cudaFree(prH[d]);if(hpOB[d])cudaFree(hpOB[d]);if(hpCI[d])cudaFree(hpCI[d]);if(hpCIS[d])cudaFree(hpCIS[d]);if(hpX[d])cudaFree(hpX[d]);}
}
