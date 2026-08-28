#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

struct CoopStats {
    uint64_t codes=0,blocks=0,per_lane_steps=0,unique_steps=0,cooperative_exact=0;
    uint32_t max_groups=0;
};
struct EarlyStats { uint64_t warps=0,transition_rounds=0,ballots=0; uint32_t max_rounds=0,max_round_warps=0; };

static CoopStats measure_coop(const Factors& f,const std::vector<uint8_t>& owner,int B){
    CoopStats z;
    for(int g=0;g<NG;++g)for(int h=0;h<=L+1;++h){
        struct G{uint32_t start,count;};std::vector<G> v;uint32_t rank=0;
        for(uint32_t m=0;m<(1u<<L);++m){if(owner[m]!=g)continue;uint32_t n=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(!n)continue;v.push_back(G{rank,n});rank+=n;}
        if(v.empty())continue;z.codes+=rank;uint32_t gi=0,nb=(rank+uint32_t(B)-1u)/uint32_t(B);z.blocks+=nb;
        for(uint32_t b=0;b<nb;++b){
            const uint32_t lo=b*uint32_t(B),hi=std::min(rank,lo+uint32_t(B));while(gi+1<v.size()&&lo>=v[gi].start+v[gi].count)++gi;
            uint32_t last=gi;while(last+1<v.size()&&v[last+1].start<hi)++last;const uint32_t gb=last-gi+1u;z.max_groups=std::max(z.max_groups,gb);z.unique_steps+=gb-1u;
            std::vector<uint32_t> expected(hi-lo),got(hi-lo,0xffffffffu);for(uint32_t r=lo;r<hi;++r){uint32_t q=gi;while(q+1<v.size()&&r>=v[q].start+v[q].count){++q;++z.per_lane_steps;}if(r<v[q].start||r>=v[q].start+v[q].count)return CoopStats{};expected[r-lo]=q;}
            uint32_t cursor=gi;std::vector<uint8_t> resolved(hi-lo,0);for(int round=0;round<B;++round){bool any_need=false;for(uint32_t r=lo;r<hi;++r){const bool need=r>=v[cursor].start+v[cursor].count;if(!resolved[r-lo]&&!need){got[r-lo]=cursor;resolved[r-lo]=1;}any_need|=need;}if(round+1<B&&any_need)++cursor;}
            for(uint32_t i=0;i<hi-lo;++i){if(!resolved[i]||got[i]!=expected[i])return CoopStats{};++z.cooperative_exact;}
        }
    }
    return z;
}

static EarlyStats measure_early(const Factors& f,const std::vector<uint8_t>& owner,int B){
    EarlyStats z;
    for(int g=0;g<NG;++g)for(int h=0;h<=L+1;++h){
        struct G{uint32_t start,count;};std::vector<G> v;uint32_t rank=0;
        for(uint32_t m=0;m<(1u<<L);++m){if(owner[m]!=g)continue;uint32_t n=uint32_t(f.low_mask_h[size_t(m)*S+h].size());if(!n)continue;v.push_back(G{rank,n});rank+=n;}
        if(v.empty())continue;
        std::vector<uint32_t> trans((rank+uint32_t(B)-1u)/uint32_t(B));uint32_t gi=0;
        for(uint32_t b=0;b<trans.size();++b){uint32_t lo=b*uint32_t(B),hi=std::min(rank,lo+uint32_t(B));while(gi+1<v.size()&&lo>=v[gi].start+v[gi].count)++gi;uint32_t last=gi;while(last+1<v.size()&&v[last+1].start<hi)++last;trans[b]=last-gi;}
        for(uint32_t w=0;w<rank;w+=32u){const uint32_t b0=w/uint32_t(B),b1=std::min<uint32_t>(trans.size(),(w+32u+uint32_t(B)-1u)/uint32_t(B));uint32_t rounds=0;for(uint32_t b=b0;b<b1;++b)rounds=std::max(rounds,trans[b]);++z.warps;z.transition_rounds+=rounds;z.max_rounds=std::max(z.max_rounds,rounds);if(rounds==uint32_t(B-1))++z.max_round_warps;z.ballots+=rounds+(rounds<uint32_t(B-1)?1u:0u);}
    }
    return z;
}

static void prove_subgroups(int B){for(uint32_t active=1;active<=32;++active)for(uint32_t lane=0;lane<active;++lane){const uint32_t src=lane&~uint32_t(B-1);const uint32_t full=((1u<<B)-1u)<<src;const uint32_t sub=full&((active==32)?0xffffffffu:((1u<<active)-1u));if(src>=active||((sub>>lane)&1u)==0u||((sub>>src)&1u)==0u)std::exit(20);}}

int main(){
    Factors f=build_factors();const auto owner=low_mask_owners(f);const auto b4=measure_coop(f,owner,4),b8=measure_coop(f,owner,8),b16=measure_coop(f,owner,16);const auto e4=measure_early(f,owner,4),e8=measure_early(f,owner,8),e16=measure_early(f,owner,16);prove_subgroups(4);prove_subgroups(8);prove_subgroups(16);
    const uint64_t l4=2*b4.blocks+b4.unique_steps,l8=2*b8.blocks+b8.unique_steps,l16=2*b16.blocks+b16.unique_steps;
    if(b4.codes!=1201917ull||b8.codes!=b4.codes||b16.codes!=b4.codes||b4.cooperative_exact!=b4.codes||b8.cooperative_exact!=b8.codes||b16.cooperative_exact!=b16.codes||b4.per_lane_steps!=104346ull||b8.per_lane_steps!=243417ull||b16.per_lane_steps!=521034ull||b4.unique_steps!=52183ull||b8.unique_steps!=60845ull||b16.unique_steps!=65159ull||l4!=653231ull||l8!=361431ull||l16!=215509ull||b4.max_groups!=4u||b8.max_groups!=8u||b16.max_groups!=16u||e4.warps!=37628ull||e8.warps!=37628ull||e16.warps!=37628ull||e4.ballots!=63478ull||e8.ballots!=72285ull||e16.ballots!=84715ull)return 2;
    auto emit=[](int B,const CoopStats&z,const EarlyStats&e,uint64_t loads){std::cout<<"block="<<B<<" codes="<<z.codes<<" blocks="<<z.blocks<<" per_lane_successor_loads="<<z.per_lane_steps<<" cooperative_successor_loads="<<z.unique_steps<<" cooperative_table_loads="<<loads<<" cooperative_loads_per_code="<<double(loads)/double(z.codes)<<" cooperative_exact="<<z.cooperative_exact<<" max_groups_per_block="<<z.max_groups<<" warps="<<e.warps<<" transition_rounds="<<e.transition_rounds<<" early_ballots="<<e.ballots<<" avg_early_ballots="<<double(e.ballots)/double(e.warps)<<" max_rounds="<<e.max_rounds<<" max_round_warps="<<e.max_round_warps<<'\n';};
    emit(4,b4,e4,l4);emit(8,b8,e8,l8);emit(16,b16,e16,l16);
    std::cout<<"gridfp-rankformula-nometa-coopgroup OK block8_old_warpshare_loads=544003 block8_cooperative_loads=361431 block8_successor_reduction_fraction="<<1.0-double(b8.unique_steps)/double(b8.per_lane_steps)<<" block8_early_ballots=72285 block8_avg_early_ballots="<<double(e8.ballots)/double(e8.warps)<<" block16_early_ballots=84715 block16_avg_early_ballots="<<double(e16.ballots)/double(e16.warps)<<" warp_uniform_early_exit=1 subgroup_prefix_safe=1 cooperative_exact_all=1\n";
    return 0;
}
