#define RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN
#include "ramstream32_cpu_low_domain_worker_locality_selftest.cu"
#undef RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN

#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_fixedpoint.hpp"

static void fp_make_parent(
    CpuLowSparsePersistentPool& d,CpuLowSparsePersistentPool& h,
    CpuLowSparsePersistentPool& out,const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,const StorageFactorHost& storage,
    const StorageLayout& layout,const CpuLowWorkerExactWorkspace& ws){
    auto ds=cpu_low_apply_domain_worker_unique_shared_coalesce(d,jobs,sparse,ws);
    cpu_low_apply_domain_worker_coalesce(h,jobs,sparse,storage,layout);
    auto hs=cpu_low_apply_domain_worker_unique_shared_coalesce(h,jobs,sparse,ws);
    CpuLowDomainWorkerUniqueCoalesceStats cd{},ch{};
    cd.unique_pages_2m_after=ds.unique_pages_2m_after;cd.unique_pages_4k_after=ds.unique_pages_4k_after;cd.owner_transitions_after=ds.owner_transitions_after;
    ch.unique_pages_2m_after=hs.unique_pages_2m_after;ch.unique_pages_4k_after=hs.unique_pages_4k_after;ch.owner_transitions_after=hs.owner_transitions_after;
    auto dec=cpu_low_choose_worker_multistart(d,cd,h,ch);
    cpu_low_copy_worker_static_schedule(out,dec.source==CPU_LOW_WORKER_MULTISTART_HYBRID?h:d);
}

static uint64_t fp_test_max(const CpuLowSparsePersistentPool&p){
    return p.sticky_worker_cells.empty()?0:*std::max_element(p.sticky_worker_cells.begin(),p.sticky_worker_cells.end());
}

int main(){
    constexpr Count mod=4294967291u; constexpr int W=TARGET_W;
    static_assert(W==LOW_LUT_K+HIGH_LUT_K+1);static_assert(W<=12,"fixedpoint selftest uses small W");
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);LowOrbitHost orbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuLowSparseHost sparse=build_cpu_low_sparse(storage,layout,lowdesc,orbit);WindowPlan low_wp=make_direct2d_window(false);auto jobs=make_cpu_low_jobs(W,low_wp);

    CpuLowSparsePersistentPool d(4,CPU_LOW_SCHEDULE_DOMAIN,2,true),h(4,CPU_LOW_SCHEDULE_DOMAIN,2,true),parent(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    for(auto*p:{&d,&h,&parent}){p->prepare_static_schedule(jobs,sparse);cpu_low_apply_domain_page_tiebreak(*p,jobs,sparse,storage,layout);cpu_low_apply_domain_worker_locality(*p,jobs,sparse);}
    CpuLowWorkerExactWorkspace ws=cpu_low_build_worker_exact_workspace(jobs,sparse,storage,layout);fp_make_parent(d,h,parent,jobs,sparse,storage,layout,ws);

    CpuLowSparsePersistentPool rs(4,CPU_LOW_SCHEDULE_DOMAIN,2,true),sr(4,CPU_LOW_SCHEDULE_DOMAIN,2,true),selected(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    rs.prepare_static_schedule(jobs,sparse);sr.prepare_static_schedule(jobs,sparse);selected.prepare_static_schedule(jobs,sparse);
    cpu_low_copy_worker_static_schedule(rs,parent);cpu_low_copy_worker_static_schedule(sr,parent);cpu_low_copy_worker_static_schedule(selected,parent);
    auto a=cpu_low_apply_worker_exact_fixedpoint(rs,jobs,sparse,ws,CPU_LOW_WORKER_FIXED_RUN_SWAP,4,4);
    auto b=cpu_low_apply_worker_exact_fixedpoint(sr,jobs,sparse,ws,CPU_LOW_WORKER_FIXED_SWAP_RUN,4,4);
    if(a.component_limit_hit||a.round_limit_hit||b.component_limit_hit||b.round_limit_hit)return 1;
    if(a.before.pages_2m!=b.before.pages_2m||a.before.pages_4k!=b.before.pages_4k||a.before.transitions!=b.before.transitions)return 2;
    if(cpu_low_worker_unique_score_less(a.before,a.after)||cpu_low_worker_unique_score_less(b.before,b.after))return 3;
    uint64_t pmax=fp_test_max(parent);if(fp_test_max(rs)>pmax||fp_test_max(sr)>pmax)return 4;
    bool bbetter=cpu_low_worker_unique_score_less(b.after,a.after);bool abetter=cpu_low_worker_unique_score_less(a.after,b.after);
    const CpuLowSparsePersistentPool*winner=&rs;
    if(bbetter||(!abetter&&fp_test_max(sr)<fp_test_max(rs)))winner=&sr;
    cpu_low_copy_worker_static_schedule(selected,*winner);if(fp_test_max(selected)>pmax)return 5;

    auto main_states=enum_states(W);auto block_states=enum_states(W-1);if(main_states.size()!=layout.main_size||block_states.size()!=layout.block_size)return 6;
    std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<main_states.size();++i)mi.emplace(main_states[i],i);for(size_t i=0;i<block_states.size();++i)di.emplace(block_states[i],i);
    std::vector<Count>init_main(main_states.size()),init_block(block_states.size());std::mt19937_64 rng(341618);for(auto&x:init_main)x=Count(rng()%mod);for(auto&x:init_block)x=Count(rng()%mod);
    auto one=reference_window(W,LOW_LUT_K,1,mod,main_states,block_states,mi,di,init_main,init_block);if(one.first.empty())return 7;
    auto two=reference_window(W,LOW_LUT_K,1,mod,main_states,block_states,mi,di,one.first,one.second);if(two.first.empty())return 8;
    RamCounts main_auth,block_auth;main_auth.alloc(layout.main_size,"fixedpoint main");block_auth.alloc(layout.block_size,"fixedpoint block");fill_factor(main_auth,block_auth,main_states,block_states,init_main,init_block,storage,layout);
    selected.run(jobs,main_auth,block_auth,storage,layout,sparse,mod);if(!compare_factor("fixedpoint-1",main_auth,block_auth,main_states,block_states,one.first,one.second,storage,layout))return 9;
    selected.run(jobs,main_auth,block_auth,storage,layout,sparse,mod);if(!compare_factor("fixedpoint-2",main_auth,block_auth,main_states,block_states,two.first,two.second,storage,layout))return 10;
    std::cout<<"cpu-low-worker-fixedpoint-selftest OK rs_rounds="<<a.rounds<<" sr_rounds="<<b.rounds<<" rs_moves="<<(a.run_accepted+a.swap_accepted)<<" sr_moves="<<(b.run_accepted+b.swap_accepted)<<" exact_generations=2\n";
    d.shutdown();h.shutdown();parent.shutdown();rs.shutdown();sr.shutdown();selected.shutdown();main_auth.release();block_auth.release();return 0;
}
