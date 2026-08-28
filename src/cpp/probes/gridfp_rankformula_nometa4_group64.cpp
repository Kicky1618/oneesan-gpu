#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

static uint64_t pack64(uint32_t start,uint32_t n,int delta,uint32_t count,uint32_t gi){
    if(start>=(1u<<16)||n>uint32_t(L)||n>15u||delta<-(1<<14)||delta>=(1<<14)||!count||count>=(1u<<10)||gi>=(1u<<14))return ~0ull;
    return uint64_t(start)|(uint64_t(n)<<16)|(uint64_t(uint32_t(delta)&0x7fffu)<<20)|(uint64_t(count)<<35)|(uint64_t(gi)<<45);
}
static uint32_t st(uint64_t x){return uint32_t(x)&0xffffu;}
static uint32_t nn(uint64_t x){return uint32_t(x>>16)&0x0fu;}
static int dl(uint64_t x){uint32_t z=uint32_t(x>>20)&0x7fffu;if(z&0x4000u)z|=0xffff8000u;return int(int32_t(z));}
static uint32_t ct(uint64_t x){return uint32_t(x>>35)&0x03ffu;}
static uint32_t ix(uint64_t x){return uint32_t(x>>45)&0x3fffu;}

int main(){
    Factors f=build_factors(); const auto owner=low_mask_owners(f); const uint32_t LM=1u<<L;
    uint64_t codes=0,groups=0,blocks=0,exact=0,steps_sum=0;
    uint32_t max_steps=0,max_count=0,max_owner_groups=0,max_n=0,max_group_index=0,max_start=0;
    int min_delta=32767,max_delta=-32768;
    for(int g=0;g<NG;++g){
        uint32_t owner_groups=0;
        std::vector<int32_t> base(size_t(S)*LM,-1);
        auto br=[&](int h,uint32_t m)->int32_t&{return base[size_t(h)*LM+m];};
        for(int h=0;h<=L+1;++h){uint32_t r=0;for(uint32_t m=0;m<LM;++m){if(owner[m]!=g)continue;auto const&v=f.low_mask_h[size_t(m)*S+h];if(!v.empty())br(h,m)=int32_t(r);r+=uint32_t(v.size());}}
        for(int h=0;h<=L+1;++h){
            struct G{uint32_t mask,start,count;int delta;uint32_t gi;}; std::vector<G> v; uint32_t rank=0;
            for(uint32_t m=0;m<LM;++m){
                if(owner[m]!=g)continue;uint32_t n=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(!n)continue;
                int d=0;if(h+2<=L+1&&br(h,m)>=0&&br(h+2,m)>=0)d=br(h+2,m)-br(h,m);
                v.push_back(G{m,rank,n,d,owner_groups++});rank+=n;
            }
            if(v.empty())continue; codes+=rank; groups+=v.size();
            std::vector<uint64_t> p; p.reserve(v.size());
            for(auto z:v){
                const uint32_t n=uint32_t(__builtin_popcount(z.mask));
                uint64_t x=pack64(z.start,n,z.delta,z.count,z.gi);
                if(x==~0ull||st(x)!=z.start||nn(x)!=n||dl(x)!=z.delta||ct(x)!=z.count||ix(x)!=z.gi||(x>>59)!=0u)return 2;
                p.push_back(x);max_count=std::max(max_count,z.count);max_n=std::max(max_n,n);max_group_index=std::max(max_group_index,z.gi);max_start=std::max(max_start,z.start);min_delta=std::min(min_delta,z.delta);max_delta=std::max(max_delta,z.delta);
            }
            uint32_t gi=0,nb=(rank+3u)/4u;blocks+=nb;
            for(uint32_t b=0;b<nb;++b){
                uint32_t lo=b*4u,hi=std::min(rank,lo+4u);while(gi+1<v.size()&&lo>=v[gi+1].start)++gi;
                for(uint32_t r=lo;r<hi;++r){
                    uint32_t q=gi,s=0;uint64_t e=p[q];
                    for(int k=0;k<3;++k){if(r<st(e)+ct(e))break;const uint32_t next=ix(e)+1u;++q;++s;if(q>=p.size()||next!=v[q].gi)return 3;e=p[q];}
                    if(r<st(e)||r>=st(e)+ct(e)||nn(e)!=uint32_t(__builtin_popcount(v[q].mask))||ix(e)!=v[q].gi)return 4;
                    ++exact;steps_sum+=s;max_steps=std::max(max_steps,s);
                }
            }
        }
        max_owner_groups=std::max(max_owner_groups,owner_groups);
    }
    if(codes!=1201917ull||groups!=69632ull||blocks!=300524ull||exact!=codes||max_steps!=3u||max_count!=1001u||max_owner_groups!=8709u||max_group_index!=8708u||max_n!=14u||max_start!=29113u||min_delta!=-12969||max_delta!=14873)return 5;
    const uint64_t gb=groups*8ull,bb=blocks*2ull;
    std::cout<<"gridfp-rankformula-nometa4-group64 OK codes="<<codes<<" groups="<<groups<<" blocks="<<blocks
             <<" group_bytes="<<gb<<" block_bytes="<<bb<<" total_bytes="<<(gb+bb)
             <<" max_start="<<max_start<<" max_group_count="<<max_count<<" max_owner_groups="<<max_owner_groups
             <<" max_group_index="<<max_group_index<<" max_support_count="<<max_n
             <<" min_base_delta="<<min_delta<<" max_base_delta="<<max_delta
             <<" max_locator_steps="<<max_steps<<" avg_locator_steps="<<double(steps_sum)/double(codes)
             <<" avg_group_loads_model="<<(1.0+double(steps_sum)/double(codes))
             <<" packed_bits=59 spare_bits=5 self_index_exact=1 count_pack_exact=1 support_count_pack_exact=1 support_position_bits=0 group_n_bits=4 sentinel_entries=0\n";
    return 0;
}
