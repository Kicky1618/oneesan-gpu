#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

static constexpr Code MAIN_CHUNK_U32ADDR = 48214938328ULL;
static constexpr std::uint32_t MAIN_CHUNK_LO_U32ADDR = 970298072u;
static constexpr std::uint32_t MAIN_CHUNK_HI_U32ADDR = 11u;

__device__ __forceinline__ std::uint32_t seed_main_u32addr(std::uint32_t h){
    return ((h<<8)+(h<<6)+(h<<5)+(h<<3)+(h<<2)+h)>>12;
}

__device__ __forceinline__ Code masked_base64_u32addr(std::uint32_t owner){
    const Code u=Code(owner);
    return ((Code(0)-(u&1ULL))&MAIN_CHUNK_U32ADDR)+
           ((Code(0)-((u>>1)&1ULL))&(MAIN_CHUNK_U32ADDR<<1))+
           ((Code(0)-((u>>2)&1ULL))&(MAIN_CHUNK_U32ADDR<<2));
}

__device__ __forceinline__ ShardAddress8 hi32_u64_address_probe(Code g){
    std::uint32_t owner=seed_main_u32addr(std::uint32_t(g>>32));
    Code local=g-masked_base64_u32addr(owner);
    const std::uint32_t corr=std::uint32_t(local>=MAIN_CHUNK_U32ADDR);
    owner+=corr;
    local-=(Code(0)-Code(corr))&MAIN_CHUNK_U32ADDR;
    return{int(owner),local};
}

struct Pair32U32Addr{std::uint32_t lo,hi;};

__device__ __forceinline__ void add_masked_pair_u32addr(Pair32U32Addr& p,std::uint32_t lo,std::uint32_t hi,std::uint32_t bit){
    const std::uint32_t mask=0u-bit;
    const std::uint32_t xlo=lo&mask,xhi=hi&mask,old=p.lo;
    p.lo+=xlo;
    p.hi+=xhi+std::uint32_t(p.lo<old);
}

__device__ __forceinline__ Pair32U32Addr base_u32_probe(std::uint32_t owner){
    Pair32U32Addr p{0u,0u};
    add_masked_pair_u32addr(p,970298072u,11u,owner&1u);
    add_masked_pair_u32addr(p,1940596144u,22u,(owner>>1)&1u);
    add_masked_pair_u32addr(p,3881192288u,44u,(owner>>2)&1u);
    return p;
}

__device__ __forceinline__ ShardAddress8 hi32_u32_address_probe(Code g){
    const std::uint32_t glo=std::uint32_t(g),ghi=std::uint32_t(g>>32);
    std::uint32_t owner=seed_main_u32addr(ghi);
    const Pair32U32Addr b=base_u32_probe(owner);
    std::uint32_t rlo=glo-b.lo;
    const std::uint32_t borrow=std::uint32_t(glo<b.lo);
    std::uint32_t rhi=ghi-b.hi-borrow;
    const std::uint32_t corr=std::uint32_t(rhi>MAIN_CHUNK_HI_U32ADDR||(rhi==MAIN_CHUNK_HI_U32ADDR&&rlo>=MAIN_CHUNK_LO_U32ADDR));
    const std::uint32_t mask=0u-corr,sublo=MAIN_CHUNK_LO_U32ADDR&mask,subhi=MAIN_CHUNK_HI_U32ADDR&mask,old=rlo;
    rlo-=sublo;
    rhi-=subhi+std::uint32_t(old<sublo);
    owner+=corr;
    return{int(owner),(Code(rhi)<<32)|Code(rlo)};
}

extern "C" __global__ void b300_hi32_u32addr_compare_probe(const Code* g,int* owner,Code* local){
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);const auto a=shard_address8(g[i],MAIN_CHUNK_U32ADDR);owner[i]=a.owner;local[i]=a.local;
}
extern "C" __global__ void b300_hi32_u32addr_u64corr_probe(const Code* g,int* owner,Code* local){
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);const auto a=hi32_u64_address_probe(g[i]);owner[i]=a.owner;local[i]=a.local;
}
extern "C" __global__ void b300_hi32_u32addr_full_probe(const Code* g,int* owner,Code* local){
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);const auto a=hi32_u32_address_probe(g[i]);owner[i]=a.owner;local[i]=a.local;
}
