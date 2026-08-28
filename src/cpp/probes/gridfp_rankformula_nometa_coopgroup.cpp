#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

struct CoopStats {
    uint64_t codes=0,blocks=0,per_lane_steps=0,unique_steps=0;
    uint32_t max_groups=0;
};

static CoopStats measure_coop(const Factors& f,const std::vector<uint8_t>& owner,int B){
    CoopStats z;
    for(int g=0;g<NG;++g){
        for(int h=0;h<=L+1;++h){
            struct G{uint32_t start,count;};std::vector<G> v;uint32_t rank=0;
            for(uint32_t m=0;m<(1u<<L);++m){
                if(owner[m]!=g)continue;uint32_t n=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(!n)continue;
                v.push_back(G{rank,n});rank+=n;
            }
            if(v.empty())continue;z.codes+=rank;
            uint32_t gi=0,nb=(rank+uint32_t(B)-1u)/uint32_t(B);z.blocks+=nb;
            for(uint32_t b=0;b<nb;++b){
                uint32_t lo=b*uint32_t(B),hi=std::min(rank,lo+uint32_t(B));
                while(gi+1<v.size()&&lo>=v[gi].start+v[gi].count)++gi;
                uint32_t last=gi;
                while(last+1<v.size()&&v[last+1].start<hi)++last;
                uint32_t gb=last-gi+1u;z.max_groups=std::max(z.max_groups,gb);z.unique_steps+=gb-1u;
                for(uint32_t r=lo;r<hi;++r){uint32_t q=gi;while(q+1<v.size()&&r>=v[q].start+v[q].count){++q;++z.per_lane_steps;}if(r<v[q].start||r>=v[q].start+v[q].count)return CoopStats{};}
            }
        }
    }
    return z;
}

static void prove_subgroups(int B){
    for(uint32_t active=1;active<=32;++active){
        for(uint32_t lane=0;lane<active;++lane){
            const uint32_t src=lane&~uint32_t(B-1);
            const uint32_t full=((1u<<B)-1u)<<src;
            const uint32_t sub=full&((active==32)?0xffffffffu:((1u<<active)-1u));
            if(src>=active||((sub>>lane)&1u)==0u||((sub>>src)&1u)==0u)return std::exit(20);
        }
    }
}

int main(){
    Factors f=build_factors();const auto owner=low_mask_owners(f);
    const auto b4=measure_coop(f,owner,4),b8=measure_coop(f,owner,8),b16=measure_coop(f,owner,16);
    prove_subgroups(4);prove_subgroups(8);prove_subgroups(16);
    const uint64_t l4=2*b4.blocks+b4.unique_steps,l8=2*b8.blocks+b8.unique_steps,l16=2*b16.blocks+b16.unique_steps;
    if(b4.codes!=1201917ull||b8.codes!=b4.codes||b16.codes!=b4.codes||
       b4.per_lane_steps!=104346ull||b8.per_lane_steps!=243417ull||b16.per_lane_steps!=521034ull||
       b4.unique_steps!=52183ull||b8.unique_steps!=60845ull||b16.unique_steps!=65159ull||
       l4!=653231ull||l8!=361431ull||l16!=215509ull||
       b4.max_groups!=4u||b8.max_groups!=8u||b16.max_groups!=16u)return 2;
    auto emit=[](int B,const CoopStats&z,uint64_t loads){
        std::cout<<"block="<<B<<" codes="<<z.codes<<" blocks="<<z.blocks
                 <<" per_lane_successor_loads="<<z.per_lane_steps
                 <<" cooperative_successor_loads="<<z.unique_steps
                 <<" cooperative_table_loads="<<loads
                 <<" cooperative_loads_per_code="<<double(loads)/double(z.codes)
                 <<" max_groups_per_block="<<z.max_groups<<'\n';
    };
    emit(4,b4,l4);emit(8,b8,l8);emit(16,b16,l16);
    std::cout<<"gridfp-rankformula-nometa-coopgroup OK"
             <<" block8_old_warpshare_loads=544003"
             <<" block8_cooperative_loads=361431"
             <<" block8_successor_reduction_fraction="<<1.0-double(b8.unique_steps)/double(b8.per_lane_steps)
             <<" fixed_rounds_block8=7 subgroup_prefix_safe=1\n";
    return 0;
}
