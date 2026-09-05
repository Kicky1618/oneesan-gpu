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
#include "../ramstream32_cpu_low_domain_worker_neutral_balance.hpp"

static uint64_t plateau_max_cells(const CpuLowSparsePersistentPool& p){
    return p.sticky_worker_cells.empty()?0:*std::max_element(p.sticky_worker_cells.begin(),p.sticky_worker_cells.end());
}
static bool plateau_score_eq(const CpuLowDomainWorkerUniqueScore&a,const CpuLowDomainWorkerUniqueScore&b){return a.pages_2m==b.pages_2m&&a.pages_4k==b.pages_4k&&a.transitions==b.transitions;}

struct PlateauFixedChoice{
    const CpuLowSparsePersistentPool* winner=nullptr;
    CpuLowDomainWorkerUniqueScore score{};
    const char* order="run-swap";
    CpuLowWorkerFixedPointStats rs{},sr{};
};
static PlateauFixedChoice plateau_run_both(
    CpuLowSparsePersistentPool&rs,CpuLowSparsePersistentPool&sr,
    const std::vector<CpuLowJob>&jobs,const CpuLowSparseHost&sparse,
    const CpuLowWorkerExactWorkspace&ws,uint32_t max_run,uint32_t max_swap){
    PlateauFixedChoice z;
    z.rs=cpu_low_apply_worker_exact_fixedpoint(rs,jobs,sparse,ws,CPU_LOW_WORKER_FIXED_RUN_SWAP,max_run,max_swap);
    z.sr=cpu_low_apply_worker_exact_fixedpoint(sr,jobs,sparse,ws,CPU_LOW_WORKER_FIXED_SWAP_RUN,max_run,max_swap);
    if(z.rs.component_limit_hit||z.rs.round_limit_hit||z.sr.component_limit_hit||z.sr.round_limit_hit){std::cerr<<"plateau fixedpoint limit hit\n";std::exit(299);}
    if(!plateau_score_eq(z.rs.before,z.sr.before)){std::cerr<<"plateau fixedpoint parent mismatch\n";std::exit(300);}
    bool sb=cpu_low_worker_unique_score_less(z.sr.after,z.rs.after);bool rb=cpu_low_worker_unique_score_less(z.rs.after,z.sr.after);
    z.winner=&rs;z.score=z.rs.after;
    if(sb||(!rb&&plateau_max_cells(sr)<plateau_max_cells(rs))){z.winner=&sr;z.score=z.sr.after;z.order="swap-run";}
    return z;
}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    int workers=argc>2?std::max(1,std::atoi(argv[2])):64;
    int domain_size=argc>3?std::atoi(argv[3]):32;
    int max_run=argc>4?std::atoi(argv[4]):4;
    int max_swap=argc>5?std::atoi(argv[5]):4;
    if(n<2||n+1!=TARGET_W||n+1>MAXW||workers<=0||domain_size<=0||domain_size>workers||max_run<=0||max_run>64||max_swap<=0||max_swap>64)return 1;
    if constexpr(LOW_LUT_K+HIGH_LUT_K!=TARGET_W-1)return 1;

    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);LowDescHost lowdesc=build_low_descriptors(storage,layout);LowOrbitHost orbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuLowSparseHost sparse=build_cpu_low_sparse(storage,layout,lowdesc,orbit);WindowPlan low_wp=make_direct2d_window(false);auto jobs=make_cpu_low_jobs(n+1,low_wp);

    CpuLowSparsePersistentPool d(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true),h(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true),parent(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    for(auto*p:{&d,&h,&parent}){p->prepare_static_schedule(jobs,sparse);cpu_low_apply_domain_page_tiebreak(*p,jobs,sparse,storage,layout);cpu_low_apply_domain_worker_locality(*p,jobs,sparse);}
    CpuLowWorkerExactWorkspace ws=cpu_low_build_worker_exact_workspace(jobs,sparse,storage,layout);
    auto ds=cpu_low_apply_domain_worker_unique_shared_coalesce(d,jobs,sparse,ws);cpu_low_apply_domain_worker_coalesce(h,jobs,sparse,storage,layout);auto hs=cpu_low_apply_domain_worker_unique_shared_coalesce(h,jobs,sparse,ws);
    CpuLowDomainWorkerUniqueCoalesceStats cd{},ch{};cd.unique_pages_2m_after=ds.unique_pages_2m_after;cd.unique_pages_4k_after=ds.unique_pages_4k_after;cd.owner_transitions_after=ds.owner_transitions_after;ch.unique_pages_2m_after=hs.unique_pages_2m_after;ch.unique_pages_4k_after=hs.unique_pages_4k_after;ch.owner_transitions_after=hs.owner_transitions_after;
    auto md=cpu_low_choose_worker_multistart(d,cd,h,ch);cpu_low_copy_worker_static_schedule(parent,md.source==CPU_LOW_WORKER_MULTISTART_HYBRID?h:d);

    CpuLowSparsePersistentPool brs(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true),bsr(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true),baseline(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    brs.prepare_static_schedule(jobs,sparse);bsr.prepare_static_schedule(jobs,sparse);baseline.prepare_static_schedule(jobs,sparse);cpu_low_copy_worker_static_schedule(brs,parent);cpu_low_copy_worker_static_schedule(bsr,parent);
    auto bc=plateau_run_both(brs,bsr,jobs,sparse,ws,uint32_t(max_run),uint32_t(max_swap));cpu_low_copy_worker_static_schedule(baseline,*bc.winner);
    auto baseline_profile=cpu_low_worker_sorted_load_profile(baseline.sticky_worker_cells);uint64_t baseline_max=plateau_max_cells(baseline);

    CpuLowSparsePersistentPool plateau(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);plateau.prepare_static_schedule(jobs,sparse);cpu_low_copy_worker_static_schedule(plateau,baseline);
    auto neutral=cpu_low_apply_worker_neutral_balance(plateau,jobs,sparse,ws);if(neutral.move_limit_hit)return 2;if(!plateau_score_eq(neutral.exact_before,bc.score)||!plateau_score_eq(neutral.exact_after,bc.score))return 3;

    CpuLowSparsePersistentPool prs(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true),psr(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true),selected(workers,CPU_LOW_SCHEDULE_DOMAIN,domain_size,true);
    prs.prepare_static_schedule(jobs,sparse);psr.prepare_static_schedule(jobs,sparse);selected.prepare_static_schedule(jobs,sparse);cpu_low_copy_worker_static_schedule(prs,plateau);cpu_low_copy_worker_static_schedule(psr,plateau);
    auto pc=plateau_run_both(prs,psr,jobs,sparse,ws,uint32_t(max_run),uint32_t(max_swap));cpu_low_copy_worker_static_schedule(selected,*pc.winner);
    if(cpu_low_worker_unique_score_less(bc.score,pc.score))return 4;if(plateau_max_cells(selected)>baseline_max)return 5;
    auto final_profile=cpu_low_worker_sorted_load_profile(selected.sticky_worker_cells);if(plateau_score_eq(pc.score,bc.score)&&baseline_profile<final_profile)return 6;

    std::cout<<std::setprecision(12)
             <<"cpu_low_worker_plateau_plan OK objective=neutral-load-bridge-v5.35-plan"
             <<" n="<<n<<" workers="<<workers<<" domain_size="<<domain_size
             <<" max_run="<<max_run<<" max_swap="<<max_swap
             <<" multistart_source="<<cpu_low_worker_multistart_source_name(md.source)
             <<" baseline_order="<<bc.order
             <<" baseline_pages_2m="<<bc.score.pages_2m<<" baseline_pages_4k="<<bc.score.pages_4k<<" baseline_transitions="<<bc.score.transitions
             <<" baseline_max_worker_cells="<<baseline_max
             <<" neutral_accepted_moves="<<neutral.accepted_moves
             <<" neutral_candidate_evaluations="<<neutral.candidate_evaluations
             <<" neutral_exact_rejections="<<neutral.exact_rejections
             <<" neutral_profile_rejections="<<neutral.profile_rejections
             <<" neutral_max_worker_cells_after="<<neutral.max_worker_cells_after
             <<" neutral_build_s="<<neutral.build_s
             <<" refixed_order="<<pc.order
             <<" final_pages_2m="<<pc.score.pages_2m<<" final_pages_4k="<<pc.score.pages_4k<<" final_transitions="<<pc.score.transitions
             <<" final_max_worker_cells="<<plateau_max_cells(selected)
             <<" final_2m_delta="<<int64_t(pc.score.pages_2m)-int64_t(bc.score.pages_2m)
             <<" final_4k_delta="<<int64_t(pc.score.pages_4k)-int64_t(bc.score.pages_4k)
             <<" final_transition_delta="<<int64_t(pc.score.transitions)-int64_t(bc.score.transitions)
             <<" refixed_rs_rounds="<<pc.rs.rounds<<" refixed_sr_rounds="<<pc.sr.rounds
             <<" workspace_build_s="<<ws.build_s<<" limits_clear=1\n";
    return 0;
}
