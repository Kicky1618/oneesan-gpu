#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_domain_page.hpp"
#include "../ramstream32_cpu_low_domain_worker_locality.hpp"
#include "../ramstream32_cpu_low_domain_worker_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_multistart.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_run_coalesce.hpp"
#include "../ramstream32_cpu_low_domain_worker_unique_swap_coalesce.hpp"

static uint64_t swap_plan_max_cells(const CpuLowSparsePersistentPool& p) {
    return p.sticky_worker_cells.empty() ? 0
        : *std::max_element(p.sticky_worker_cells.begin(), p.sticky_worker_cells.end());
}

int main(int argc, char** argv) {
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    int workers=argc>2?std::max(1,std::atoi(argv[2])):64;
    int domain_size=argc>3?std::atoi(argv[3]):32;
    int max_swap=argc>4?std::atoi(argv[4]):4;
    int max_run=argc>5?std::atoi(argv[5]):4;
    if(n<2||n+1!=TARGET_W||n+1>MAXW||workers<=0||domain_size<=0
       ||domain_size>workers||max_swap<=0||max_swap>64||max_run<=0||max_run>64)return 1;
    if constexpr(LOW_LUT_K+HIGH_LUT_K!=TARGET_W-1)return 1;

    build_full_dp(); G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    LowOrbitHost orbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuLowSparseHost sparse=build_cpu_low_sparse(storage,layout,lowdesc,orbit);
    WindowPlan low_wp=make_direct2d_window(false);
    auto jobs=make_cpu_low_jobs(n+1,low_wp);

    CpuLowSparsePersistentPool direct(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    CpuLowSparsePersistentPool hybrid(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    CpuLowSparsePersistentPool selected(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    for(auto*p:{&direct,&hybrid,&selected}){
        p->prepare_static_schedule(jobs,sparse);
        cpu_low_apply_domain_page_tiebreak(*p,jobs,sparse,storage,layout);
        cpu_low_apply_domain_worker_locality(*p,jobs,sparse);
    }
    CpuLowWorkerExactWorkspace ws=cpu_low_build_worker_exact_workspace(jobs,sparse,storage,layout);
    auto ds=cpu_low_apply_domain_worker_unique_shared_coalesce(direct,jobs,sparse,ws);
    auto hl=cpu_low_apply_domain_worker_coalesce(hybrid,jobs,sparse,storage,layout);
    auto hs=cpu_low_apply_domain_worker_unique_shared_coalesce(hybrid,jobs,sparse,ws);
    CpuLowDomainWorkerUniqueCoalesceStats cd{},ch{};
    cd.unique_pages_2m_after=ds.unique_pages_2m_after; cd.unique_pages_4k_after=ds.unique_pages_4k_after; cd.owner_transitions_after=ds.owner_transitions_after;
    ch.unique_pages_2m_after=hs.unique_pages_2m_after; ch.unique_pages_4k_after=hs.unique_pages_4k_after; ch.owner_transitions_after=hs.owner_transitions_after;
    auto decision=cpu_low_choose_worker_multistart(direct,cd,hybrid,ch);
    cpu_low_copy_worker_static_schedule(selected,
        decision.source==CPU_LOW_WORKER_MULTISTART_HYBRID?hybrid:direct);

    auto run=cpu_low_apply_domain_worker_unique_run_coalesce(
        selected,jobs,sparse,ws,uint32_t(max_run));
    if(run.move_limit_hit)return 2;
    CpuLowDomainWorkerUniqueScore before{
        run.unique_pages_2m_after,run.unique_pages_4k_after,run.owner_transitions_after};
    uint64_t before_max=swap_plan_max_cells(selected);
    auto swap=cpu_low_apply_domain_worker_unique_swap_coalesce(
        selected,jobs,sparse,ws,uint32_t(max_swap));
    CpuLowDomainWorkerUniqueScore after{
        swap.unique_pages_2m_after,swap.unique_pages_4k_after,swap.owner_transitions_after};
    if(cpu_low_worker_unique_score_less(before,after))return 3;
    if(swap_plan_max_cells(selected)>before_max)return 4;

    std::cout<<std::setprecision(12)
             <<"cpu_low_worker_swap_plan OK"
             <<" objective=bounded-swap-global-unique-v5.33-plan"
             <<" n="<<n<<" workers="<<workers<<" domain_size="<<domain_size
             <<" max_run="<<max_run<<" max_swap="<<max_swap
             <<" selected_source="<<cpu_low_worker_multistart_source_name(decision.source)
             <<" before_pages_2m="<<before.pages_2m
             <<" before_pages_4k="<<before.pages_4k
             <<" before_transitions="<<before.transitions
             <<" after_pages_2m="<<after.pages_2m
             <<" after_pages_4k="<<after.pages_4k
             <<" after_transitions="<<after.transitions
             <<" pages_2m_delta="<<int64_t(after.pages_2m)-int64_t(before.pages_2m)
             <<" pages_4k_delta="<<int64_t(after.pages_4k)-int64_t(before.pages_4k)
             <<" transition_delta="<<int64_t(after.transitions)-int64_t(before.transitions)
             <<" max_worker_cells_before="<<before_max
             <<" max_worker_cells_after="<<swap_plan_max_cells(selected)
             <<" run_accepted="<<run.accepted_runs
             <<" run_max_used="<<run.max_run_used
             <<" swap_candidate_evaluations="<<swap.candidate_evaluations
             <<" swap_cap_rejections="<<swap.cap_rejections
             <<" accepted_swaps="<<swap.accepted_swaps
             <<" moved_jobs="<<swap.moved_jobs
             <<" max_left_used="<<swap.max_left_used
             <<" max_right_used="<<swap.max_right_used
             <<" page_improving_swaps="<<swap.page_improving_swaps
             <<" transition_only_swaps="<<swap.transition_only_swaps
             <<" move_limit_hit="<<(swap.move_limit_hit?1:0)
             <<" swap_build_s="<<swap.build_s
             <<" workspace_build_s="<<ws.build_s
             <<" hybrid_v526_build_s="<<hl.build_s
             <<'\n';
    return 0;
}
