#define RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN
#include "ramstream32_cpu_low_domain_worker_locality_selftest.cu"
#undef RAMSTREAM_CPU_LOW_WORKER_LOCALITY_SELFTEST_NO_MAIN

#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_run_coalesce.hpp"

static void make_selected_parent(
    CpuLowSparsePersistentPool& direct,
    CpuLowSparsePersistentPool& hybrid,
    CpuLowSparsePersistentPool& out,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const StorageFactorHost& storage,
    const StorageLayout& layout,
    const CpuLowWorkerExactWorkspace& ws
) {
    CpuLowDomainWorkerUniqueDenseStats ds =
        cpu_low_apply_domain_worker_unique_shared_coalesce(direct, jobs, sparse, ws);
    cpu_low_apply_domain_worker_coalesce(hybrid, jobs, sparse, storage, layout);
    CpuLowDomainWorkerUniqueDenseStats hs =
        cpu_low_apply_domain_worker_unique_shared_coalesce(hybrid, jobs, sparse, ws);
    CpuLowDomainWorkerUniqueCoalesceStats cd{}, ch{};
    cd.unique_pages_2m_after=ds.unique_pages_2m_after;
    cd.unique_pages_4k_after=ds.unique_pages_4k_after;
    cd.owner_transitions_after=ds.owner_transitions_after;
    ch.unique_pages_2m_after=hs.unique_pages_2m_after;
    ch.unique_pages_4k_after=hs.unique_pages_4k_after;
    ch.owner_transitions_after=hs.owner_transitions_after;
    auto decision=cpu_low_choose_worker_multistart(direct,cd,hybrid,ch);
    cpu_low_copy_worker_static_schedule(
        out, decision.source==CPU_LOW_WORKER_MULTISTART_HYBRID ? hybrid : direct);
}

int main() {
    constexpr Count mod=4294967291u;
    constexpr int W=TARGET_W;
    static_assert(W==LOW_LUT_K+HIGH_LUT_K+1);
    static_assert(W<=12,"run selftest intentionally uses small W");

    build_full_dp();
    G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    LowOrbitHost orbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuLowSparseHost sparse=build_cpu_low_sparse(storage,layout,lowdesc,orbit);
    WindowPlan low_wp=make_direct2d_window(false);
    auto jobs=make_cpu_low_jobs(W,low_wp);

    CpuLowSparsePersistentPool d(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    CpuLowSparsePersistentPool h(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    CpuLowSparsePersistentPool parent(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    for (auto* p:{&d,&h,&parent}) {
        p->prepare_static_schedule(jobs,sparse);
        cpu_low_apply_domain_page_tiebreak(*p,jobs,sparse,storage,layout);
        cpu_low_apply_domain_worker_locality(*p,jobs,sparse);
    }
    CpuLowWorkerExactWorkspace ws=cpu_low_build_worker_exact_workspace(
        jobs,sparse,storage,layout);
    make_selected_parent(d,h,parent,jobs,sparse,storage,layout,ws);

    CpuLowSparsePersistentPool run1(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    CpuLowSparsePersistentPool run4(4,CPU_LOW_SCHEDULE_DOMAIN,2,true);
    run1.prepare_static_schedule(jobs,sparse);
    run4.prepare_static_schedule(jobs,sparse);
    cpu_low_copy_worker_static_schedule(run1,parent);
    cpu_low_copy_worker_static_schedule(run4,parent);

    auto s1=cpu_low_apply_domain_worker_unique_run_coalesce(
        run1,jobs,sparse,ws,1);
    auto s4=cpu_low_apply_domain_worker_unique_run_coalesce(
        run4,jobs,sparse,ws,4);
    CpuLowDomainWorkerUniqueScore b1{s1.unique_pages_2m_before,s1.unique_pages_4k_before,s1.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore a1{s1.unique_pages_2m_after,s1.unique_pages_4k_after,s1.owner_transitions_after};
    CpuLowDomainWorkerUniqueScore b4{s4.unique_pages_2m_before,s4.unique_pages_4k_before,s4.owner_transitions_before};
    CpuLowDomainWorkerUniqueScore a4{s4.unique_pages_2m_after,s4.unique_pages_4k_after,s4.owner_transitions_after};
    if (b1.pages_2m!=b4.pages_2m || b1.pages_4k!=b4.pages_4k || b1.transitions!=b4.transitions) return 1;
    if (cpu_low_worker_unique_score_less(b1,a1) || cpu_low_worker_unique_score_less(b4,a4)) return 2;
    if (s1.max_worker_cells_after>s1.max_worker_cells_before || s4.max_worker_cells_after>s4.max_worker_cells_before) return 3;
    if (s1.move_limit_hit || s4.move_limit_hit) return 4;

    auto main_states=enum_states(W);
    auto block_states=enum_states(W-1);
    if (main_states.size()!=layout.main_size || block_states.size()!=layout.block_size) return 5;
    std::unordered_map<MateID,size_t> mi,di;
    for(size_t i=0;i<main_states.size();++i) mi.emplace(main_states[i],i);
    for(size_t i=0;i<block_states.size();++i) di.emplace(block_states[i],i);
    std::vector<Count> init_main(main_states.size()),init_block(block_states.size());
    std::mt19937_64 rng(321618);
    for(auto&x:init_main)x=Count(rng()%mod);
    for(auto&x:init_block)x=Count(rng()%mod);
    auto one=reference_window(W,LOW_LUT_K,1,mod,main_states,block_states,mi,di,init_main,init_block);
    if(one.first.empty())return 6;
    auto two=reference_window(W,LOW_LUT_K,1,mod,main_states,block_states,mi,di,one.first,one.second);
    if(two.first.empty())return 7;

    RamCounts main_auth,block_auth;
    main_auth.alloc(layout.main_size,"run selftest main");
    block_auth.alloc(layout.block_size,"run selftest block");
    fill_factor(main_auth,block_auth,main_states,block_states,init_main,init_block,storage,layout);
    run4.run(jobs,main_auth,block_auth,storage,layout,sparse,mod);
    if(!compare_factor("bounded-run-1",main_auth,block_auth,main_states,block_states,one.first,one.second,storage,layout))return 8;
    run4.run(jobs,main_auth,block_auth,storage,layout,sparse,mod);
    if(!compare_factor("bounded-run-2",main_auth,block_auth,main_states,block_states,two.first,two.second,storage,layout))return 9;

    std::cout<<"cpu-low-worker-run-selftest OK"
             <<" run1_accepted="<<s1.accepted_runs
             <<" run4_accepted="<<s4.accepted_runs
             <<" run4_max_used="<<s4.max_run_used
             <<" exact_generations=2\n";
    d.shutdown();h.shutdown();parent.shutdown();run1.shutdown();run4.shutdown();
    main_auth.release();block_auth.release();
    return 0;
}
