#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

static uint64_t pack64(uint32_t start,uint32_t mask,int delta,uint32_t count){
    if(start>=65536u||mask>=(1u<<L)||delta<-32768||delta>32767||!count||count>=65536u)return ~0ull;
    return uint64_t(uint16_t(start))|(uint64_t(mask)<<16)|(uint64_t(uint16_t(int16_t(delta)))<<32)|(uint64_t(uint16_t(count))<<48);
}
static uint32_t st(uint64_t x){return uint32_t(x)&0xffffu;}
static uint32_t mk(uint64_t x){return uint32_t(x>>16)&((1u<<L)-1u);}
static int dl(uint64_t x){return int(int16_t(uint16_t(x>>32)));}
static uint32_t ct(uint64_t x){return uint32_t(uint16_t(x>>48));}

int main(){
    Factors f=build_factors(); const auto owner=low_mask_owners(f); const uint32_t LM=1u<<L;
    uint64_t codes=0,groups=0,blocks=0,exact=0,steps_sum=0;
    uint32_t max_steps=0,max_count=0,max_owner_groups=0;
    for(int g=0;g<NG;++g){
        uint32_t owner_groups=0;
        std::vector<int32_t> base(size_t(S)*LM,-1);
        auto br=[&](int h,uint32_t m)->int32_t&{return base[size_t(h)*LM+m];};
        for(int h=0;h<=L+1;++h){uint32_t r=0;for(uint32_t m=0;m<LM;++m){if(owner[m]!=g)continue;auto const&v=f.low_mask_h[size_t(m)*S+h];if(!v.empty())br(h,m)=int32_t(r);r+=uint32_t(v.size());}}
        for(int h=0;h<=L+1;++h){
            struct G{uint32_t mask,start,count;int delta;}; std::vector<G> v; uint32_t rank=0;
            for(uint32_t m=0;m<LM;++m){if(owner[m]!=g)continue;uint32_t n=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(!n)continue;int d=0;if(h+2<=L+1&&br(h,m)>=0&&br(h+2,m)>=0)d=br(h+2,m)-br(h,m);v.push_back(G{m,rank,n,d});rank+=n;}
            if(v.empty())continue; codes+=rank; groups+=v.size(); owner_groups+=uint32_t(v.size());
            std::vector<uint64_t> p; p.reserve(v.size());
            for(auto z:v){uint64_t x=pack64(z.start,z.mask,z.delta,z.count);if(x==~0ull||st(x)!=z.start||mk(x)!=z.mask||dl(x)!=z.delta||ct(x)!=z.count)return 2;p.push_back(x);max_count=std::max(max_count,z.count);}
            uint32_t gi=0,nb=(rank+3u)/4u;blocks+=nb;
            for(uint32_t b=0;b<nb;++b){uint32_t lo=b*4u,hi=std::min(rank,lo+4u);while(gi+1<v.size()&&lo>=v[gi+1].start)++gi;for(uint32_t r=lo;r<hi;++r){uint32_t q=gi,s=0;uint64_t e=p[q];for(int k=0;k<3;++k){if(r<st(e)+ct(e))break;++q;++s;if(q>=p.size())return 3;e=p[q];}if(r<st(e)||r>=st(e)+ct(e)||mk(e)!=v[q].mask)return 4;++exact;steps_sum+=s;max_steps=std::max(max_steps,s);}}
        }
        max_owner_groups=std::max(max_owner_groups,owner_groups);
    }
    if(codes!=1201917ull||groups!=69632ull||blocks!=300524ull||exact!=codes||max_steps!=3u||max_count!=1001u||max_owner_groups!=8709u)return 5;
    const uint64_t gb=groups*8ull,bb=blocks*2ull;
    std::cout<<"gridfp-rankformula-nometa4-group64 OK codes="<<codes<<" groups="<<groups<<" blocks="<<blocks
             <<" group_bytes="<<gb<<" block_bytes="<<bb<<" total_bytes="<<(gb+bb)
             <<" max_group_count="<<max_count<<" max_owner_groups="<<max_owner_groups
             <<" max_locator_steps="<<max_steps<<" avg_locator_steps="<<double(steps_sum)/double(codes)
             <<" avg_group_loads_model="<<(1.0+double(steps_sum)/double(codes))
             <<" count_pack_exact=1 sentinel_entries=0\n";
    return 0;
}
