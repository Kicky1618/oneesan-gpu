#pragma once

#include "ramstream32_bucket_low_rankformula_nometa_direct64.cuh"
#include "ramstream32_bucket_closure_cross5_rankformula_nometa4_abstract.cuh"

// 16-byte, depth-major plan. Adjacent warp lanes read adjacent uint4 entries.
// x: src0|src1, y: src2|src3, z: src4|src5, w: src6|count.
__constant__ uint4* D_P10DC_LOW_RANKFORMULA_DIRECTPLAN16;
__constant__ uint32_t D_P10DC_LOW_RANKFORMULA_DIRECTPLAN16_STRIDE;

__device__ __forceinline__ uint32_t p10dc_directplan16_src(const uint4& p, uint32_t i) {
    const uint32_t w = i < 2u ? p.x : i < 4u ? p.y : i < 6u ? p.z : p.w;
    return (w >> (16u * (i & 1u))) & 0xffffu;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_directplan16_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row) {
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return BkczCrossAccum(0);
    const uint32_t flat = D_P10DC_LOW_PREKEY_HOFF[h] + rank;
    const uint32_t stride = D_P10DC_LOW_RANKFORMULA_DIRECTPLAN16_STRIDE;
    const uint4 p = __ldg(D_P10DC_LOW_RANKFORMULA_DIRECTPLAN16 +
                          (depth - 1u) * stride + flat);
    const uint32_t count = p.w >> 16;
    BkczCrossAccum sum = 0;
#pragma unroll
    for (uint32_t i = 0; i < 7u; ++i) {
        if (i < count)
            sum = bkcz_cross_add(sum, source_row[p10dc_directplan16_src(p, i)]);
    }
    return sum;
}

struct BucketFusedDirectHighRowsRankFormulaDirectPlan16Tables
    : BucketFusedDirectHighRowsRankFormulaNometa4Direct64Tables {
    uint4* low_rankformula_directplan16 = nullptr;
    size_t low_rankformula_directplan16_count = 0;
    size_t low_rankformula_directplan16_capacity = 0;

    static uint4 pack_plan(const std::array<uint16_t,7>& src, uint32_t count) {
        if (count > 7u) std::exit(792);
        return make_uint4(
            uint32_t(src[0]) | (uint32_t(src[1]) << 16),
            uint32_t(src[2]) | (uint32_t(src[3]) << 16),
            uint32_t(src[4]) | (uint32_t(src[5]) << 16),
            uint32_t(src[6]) | (count << 16));
    }

    void bind_owner(uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
                    const std::array<Count*, BUCKET_NGPU>& slot) {
        BucketFusedDirectHighRowsRankFormulaNometa4Direct64Tables::bind_owner(
            fixed, buckets, slot);
        if (!host_fused) std::exit(793);
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        struct G { uint32_t mask=0,start=0,end=0; };
        std::array<std::vector<G>, P10DC_RANKFORMULA_NOMETA4_HEIGHTS> gh;
        std::vector<int32_t> absbase(
            size_t(P10DC_RANKFORMULA_NOMETA4_HEIGHTS) *
                P10DC_RANKFORMULA_NOMETA4_MASKS, -1);
        auto bref=[&](uint32_t h,uint32_t mask)->int32_t&{
            return absbase[size_t(h)*P10DC_RANKFORMULA_NOMETA4_MASKS+mask];
        };
        std::array<uint32_t,MAXW+2> hoff{};
        uint32_t flat_count=0;
        for(uint32_t h=0;h<uint32_t(MAXW+2);++h){
            hoff[h]=flat_count;
            if(h>=P10DC_RANKFORMULA_NOMETA4_HEIGHTS)continue;
            const uint32_t a=f.low_code_off[owner_base+h];
            const uint32_t b=h+1u<uint32_t(MAXW+2)?f.low_code_off[owner_base+h+1u]:owner_end;
            uint32_t prev=0;bool have=false;
            for(uint32_t i=a;i<b;++i){
                const uint32_t mask=code_mask(f.low_codes[i]);
                if(have&&mask<prev)std::exit(794);
                if(!have||mask!=prev){const uint32_t start=i-a;if(!gh[h].empty())gh[h].back().end=start;gh[h].push_back(G{mask,start,0});bref(h,mask)=int32_t(start);prev=mask;have=true;}
            }
            if(!gh[h].empty())gh[h].back().end=b-a;
            flat_count+=b-a;
        }
        if(flat_count!=low_prekey_count||flat_count!=low_rankformula_direct64_count){
            std::cerr<<"p10dc directplan16 stride mismatch owner="<<fixed
                     <<" flat="<<flat_count<<" prekey="<<low_prekey_count
                     <<" direct64="<<low_rankformula_direct64_count<<'\n';std::exit(795);
        }

        const size_t total=size_t(flat_count)*P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;
        std::vector<uint4> plan(total,make_uint4(0,0,0,0));
        uint32_t max_count=0;uint64_t source_refs=0;
        for(uint32_t h=0;h<P10DC_RANKFORMULA_NOMETA4_HEIGHTS;++h){
            for(const G& z:gh[h]){
                const uint32_t n=uint32_t(__builtin_popcount(z.mask));
                if(n<h||((n-h)&1u))std::exit(796);
                const uint32_t lcount=(n-h)>>1;
                uint32_t source_base=0;
                if(lcount){if(h+2u>=P10DC_RANKFORMULA_NOMETA4_HEIGHTS||bref(h+2u,z.mask)<0)std::exit(797);source_base=uint32_t(bref(h+2u,z.mask));}
                for(uint32_t rank=z.start;rank<z.end;++rank){
                    const uint32_t local=rank-z.start;
                    const uint32_t lp=p10dc_rankformula_abstract_lpattern_host(int(n),int(h),local);
                    if(lp==0xffffffffu)std::exit(798);
                    for(uint32_t depth=1;depth<=P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;++depth){
                        std::array<uint16_t,7> src{};uint32_t count=0,state=depth;
                        for(uint32_t ord=0;ord<n;++ord){
                            if((lp>>ord)&1u){
                                if(state==1u){
                                    if(count>=7u)std::exit(799);
                                    const uint32_t sr=p10dc_rankformula_abstract_rank_host(int(n),int(h+2u),lp&~(1u<<ord));
                                    if(sr==0xffffffffu||source_base+sr>=65536u)std::exit(800);
                                    src[count++]=uint16_t(source_base+sr);
                                }
                                ++state;
                            }else{
                                if(state==1u)break;
                                --state;
                            }
                        }
                        const size_t ix=size_t(depth-1u)*flat_count+hoff[h]+rank;
                        plan[ix]=pack_plan(src,count);max_count=std::max(max_count,count);source_refs+=count;
                    }
                }
            }
        }

        low_rankformula_directplan16_count=plan.size();
        if(low_rankformula_directplan16_count>low_rankformula_directplan16_capacity){
            if(low_rankformula_directplan16)cudaFree(low_rankformula_directplan16);
            low_rankformula_directplan16=nullptr;low_rankformula_directplan16_capacity=low_rankformula_directplan16_count;
            if(low_rankformula_directplan16_capacity)ck(cudaMalloc(&low_rankformula_directplan16,low_rankformula_directplan16_capacity*sizeof(uint4)),"p10dc directplan16 alloc");
        }
        if(!plan.empty())ck(cudaMemcpy(low_rankformula_directplan16,plan.data(),plan.size()*sizeof(uint4),cudaMemcpyHostToDevice),"p10dc directplan16 H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_DIRECTPLAN16,&low_rankformula_directplan16,sizeof(low_rankformula_directplan16)),"p10dc directplan16 ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKFORMULA_DIRECTPLAN16_STRIDE,&flat_count,sizeof(flat_count)),"p10dc directplan16 stride");
        std::cerr<<"p10dc_low_rankformula_directplan16 fixed_owner="<<fixed
                 <<" rank_stride="<<flat_count<<" entries="<<plan.size()
                 <<" bytes="<<plan.size()*sizeof(uint4)
                 <<" mib="<<double(plan.size()*sizeof(uint4))/double(1<<20)
                 <<" max_sources="<<max_count<<" source_refs="<<source_refs
                 <<" hot_group_resolve=0 hot_depth_decode=0 hot_ffs=0 plan_load_bytes=16\n";
    }

    void release(){
        if(low_rankformula_directplan16)cudaFree(low_rankformula_directplan16);
        low_rankformula_directplan16=nullptr;low_rankformula_directplan16_count=0;low_rankformula_directplan16_capacity=0;
        BucketFusedDirectHighRowsRankFormulaNometa4Direct64Tables::release();
    }
};
