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
#include "../ramstream32_cpu_low_domain_worker_unique_fixedpoint.hpp"

static uint64_t fp_max_cells(const CpuLowSparsePersistentPool& p) {
    return p.sticky_worker_cells.empty() ? 0
        : *std::max_element(p.sticky_worker_cells.begin(), p.sticky_worker_cells.end());
}

static bool fp_score_equal(
    const CpuLowDomainWorkerUniqueScore& a,
    const CpuLowDomainWorkerUniqueScore& b
) {
    return a.pages_2m == b.pages_2m
        && a.pages_4k == b.pages_4k
        && a.transitions == b.transitions;
}

int main(int argc, char** argv) {
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    int workers=argc>2?std::max(1,std::atoi(argv[2])):64;
    int domain_size=argc>3?std::atoi(argv[3]):32;
    int max_run=argc>4?std::atoi(argv[4]):4;
    int max_swap=argc>5?std::atoi(argv[5]):4;
    if(n<2||n+1!=TARGET_W||n+1>MAXW||workers<=0||domain_size<=0
       ||domain_size>workers||max_run<=0||max_run>64||max_swap<=0||max_swap>64)return 1;
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
    CpuLowSparsePersistentPool parent(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    for(auto*p:{&direct,&hybrid,&parent}){
        p->prepare_static_schedule(jobs,sparse);
        cpu_low_apply_domain_page_tiebreak(*p,jobs,sparse,storage,layout);
        cpu_low_apply_domain_worker_locality(*p,jobs,sparse);
    }
    CpuLowWorkerExactWorkspace ws=cpu_low_build_worker_exact_workspace(jobs,sparse,storage,layout);
    auto ds=cpu_low_apply_domain_worker_unique_shared_coalesce(direct,jobs,sparse,ws);
    cpu_low_apply_domain_worker_coalesce(hybrid,jobs,sparse,storage,layout);
    auto hs=cpu_low_apply_domain_worker_unique_shared_coalesce(hybrid,jobs,sparse,ws);
    CpuLowDomainWorkerUniqueCoalesceStats cd{},ch{};
    cd.unique_pages_2m_after=ds.unique_pages_2m_after; cd.unique_pages_4k_after=ds.unique_pages_4k_after; cd.owner_transitions_after=ds.owner_transitions_after;
    ch.unique_pages_2m_after=hs.unique_pages_2m_after; ch.unique_pages_4k_after=hs.unique_pages_4k_after; ch.owner_transitions_after=hs.owner_transitions_after;
    auto mdec=cpu_low_choose_worker_multistart(direct,cd,hybrid,ch);
    cpu_low_copy_worker_static_schedule(parent,
        mdec.source==CPU_LOW_WORKER_MULTISTART_HYBRID?hybrid:direct);

    CpuLowSparsePersistentPool rs(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    CpuLowSparsePersistentPool sr(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    CpuLowSparsePersistentPool selected(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    rs.prepare_static_schedule(jobs,sparse); sr.prepare_static_schedule(jobs,sparse); selected.prepare_static_schedule(jobs,sparse);
    cpu_low_copy_worker_static_schedule(rs,parent);
    cpu_low_copy_worker_static_schedule(sr,parent);
    cpu_low_copy_worker_static_schedule(selected,parent);
    uint64_t parent_max=fp_max_cells(parent);

    auto a=cpu_low_apply_worker_exact_fixedpoint(
        rs,jobs,sparse,ws,CPU_LOW_WORKER_FIXED_RUN_SWAP,uint32_t(max_run),uint32_t(max_swap));
    auto b=cpu_low_apply_worker_exact_fixedpoint(
        sr,jobs,sparse,ws,CPU_LOW_WORKER_FIXED_SWAP_RUN,uint32_t(max_run),uint32_t(max_swap));
    if(a.component_limit_hit||a.round_limit_hit||b.component_limit_hit||b.round_limit_hit)return 2;
    if(!fp_score_equal(a.before,b.before))return 3;

    bool b_better=cpu_low_worker_unique_score_less(b.after,a.after);
    bool a_better=cpu_low_worker_unique_score_less(a.after,b.after);
    const CpuLowSparsePersistentPool* winner=&rs;
    const char* winner_name="run-swap";
    CpuLowDomainWorkerUniqueScore best=a.after;
    if(b_better || (!a_better && fp_max_cells(sr)<fp_max_cells(rs))){
        winner=&sr; winner_name="swap-run"; best=b.after;
    }
    cpu_low_copy_worker_static_schedule(selected,*winner);
    if(cpu_low_worker_unique_score_less(a.before,best)
       ||cpu_low_worker_unique_score_less(a.after,best)
       ||cpu_low_worker_unique_score_less(b.after,best))return 4;
    if(fp_max_cells(selected)>parent_max)return 5;

    std::cout<<std::setprecision(12)
             <<"cpu_low_worker_fixedpoint_plan OK"
             <<" objective=alternating-run-swap-v5.34-plan"
             <<" n="<<n<<" workers="<<workers<<" domain_size="<<domain_size
             <<" max_run="<<max_run<<" max_swap="<<max_swap
             <<" multistart_source="<<cpu_low_worker_multistart_source_name(mdec.source)
             <<" parent_pages_2m="<<a.before.pages_2m
             <<" parent_pages_4k="<<a.before.pages_4k
             <<" parent_transitions="<<a.before.transitions
             <<" rs_pages_2m="<<a.after.pages_2m
             <<" rs_pages_4k="<<a.after.pages_4k
             <<" rs_transitions="<<a.after.transitions
             <<" rs_rounds="<<a.rounds
             <<" rs_run_accepted="<<a.run_accepted
             <<" rs_swap_accepted="<<a.swap_accepted
             <<" rs_build_s="<<a.build_s
             <<" sr_pages_2m="<<b.after.pages_2m
             <<" sr_pages_4k="<<b.after.pages_4k
             <<" sr_transitions="<<b.after.transitions
             <<" sr_rounds="<<b.rounds
             <<" sr_run_accepted="<<b.run_accepted
             <<" sr_swap_accepted="<<b.swap_accepted
             <<" sr_build_s="<<b.build_s
             <<" selected_order="<<winner_name
             <<" selected_pages_2m="<<best.pages_2m
             <<" selected_pages_4k="<<best.pages_4k
             <<" selected_transitions="<<best.transitions
             <<" parent_max_worker_cells="<<parent_max
             <<" selected_max_worker_cells="<<fp_max_cells(selected)
             <<" workspace_build_s="<<ws.build_s
             <<" workspace_mib="<<double(ws.bytes())/double(1<<20)
             <<" identical_parent_score=1 limits_clear=1\n";
    return 0;
}
