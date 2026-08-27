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
#include "../ramstream32_cpu_low_domain_worker_augmented_fixedpoint.hpp"

static uint64_t augmax(const CpuLowSparsePersistentPool&p){return p.sticky_worker_cells.empty()?0:*std::max_element(p.sticky_worker_cells.begin(),p.sticky_worker_cells.end());}

int main(int argc,char**argv){
 int n=argc>1?std::atoi(argv[1]):TARGET_W-1;int workers=argc>2?std::max(1,std::atoi(argv[2])):64;int domain=argc>3?std::atoi(argv[3]):32;int mr=argc>4?std::atoi(argv[4]):4;int ms=argc>5?std::atoi(argv[5]):4;
 if(n<2||n+1!=TARGET_W||n+1>MAXW||workers<=0||domain<=0||domain>workers||mr<=0||mr>64||ms<=0||ms>64)return 1;if constexpr(LOW_LUT_K+HIGH_LUT_K!=TARGET_W-1)return 1;
 build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);LowDescHost lowdesc=build_low_descriptors(storage,layout);LowOrbitHost orbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuLowSparseHost sparse=build_cpu_low_sparse(storage,layout,lowdesc,orbit);WindowPlan low_wp=make_direct2d_window(false);auto jobs=make_cpu_low_jobs(n+1,low_wp);
 CpuLowSparsePersistentPool d(workers,CPU_LOW_SCHEDULE_DOMAIN,domain,true),h(workers,CPU_LOW_SCHEDULE_DOMAIN,domain,true),parent(workers,CPU_LOW_SCHEDULE_DOMAIN,domain,true);for(auto*p:{&d,&h,&parent}){p->prepare_static_schedule(jobs,sparse);cpu_low_apply_domain_page_tiebreak(*p,jobs,sparse,storage,layout);cpu_low_apply_domain_worker_locality(*p,jobs,sparse);}CpuLowWorkerExactWorkspace ws=cpu_low_build_worker_exact_workspace(jobs,sparse,storage,layout);if(!ws.structural_audit_ok)return 6;
 auto ds=cpu_low_apply_domain_worker_unique_shared_coalesce(d,jobs,sparse,ws);cpu_low_apply_domain_worker_coalesce(h,jobs,sparse,storage,layout);auto hs=cpu_low_apply_domain_worker_unique_shared_coalesce(h,jobs,sparse,ws);CpuLowDomainWorkerUniqueCoalesceStats cd{},ch{};cd.unique_pages_2m_after=ds.unique_pages_2m_after;cd.unique_pages_4k_after=ds.unique_pages_4k_after;cd.owner_transitions_after=ds.owner_transitions_after;ch.unique_pages_2m_after=hs.unique_pages_2m_after;ch.unique_pages_4k_after=hs.unique_pages_4k_after;ch.owner_transitions_after=hs.owner_transitions_after;auto md=cpu_low_choose_worker_multistart(d,cd,h,ch);cpu_low_copy_worker_static_schedule(parent,md.source==CPU_LOW_WORKER_MULTISTART_HYBRID?h:d);
 CpuLowSparsePersistentPool baseline(workers,CPU_LOW_SCHEDULE_DOMAIN,domain,true),aug(workers,CPU_LOW_SCHEDULE_DOMAIN,domain,true);baseline.prepare_static_schedule(jobs,sparse);aug.prepare_static_schedule(jobs,sparse);cpu_low_copy_worker_static_schedule(baseline,parent);cpu_low_copy_worker_static_schedule(aug,parent);
 auto base=cpu_low_apply_best_exact_fixedpoint(baseline,jobs,sparse,ws,uint32_t(mr),uint32_t(ms));auto az=cpu_low_apply_worker_augmented_fixedpoint(aug,jobs,sparse,ws,uint32_t(mr),uint32_t(ms));if(az.neutral_limit_hit||az.round_limit_hit)return 2;if(cpu_low_worker_unique_score_less(base.after,az.after))return 3;if(augmax(aug)>augmax(baseline))return 4;
 auto bp=cpu_low_worker_sorted_load_profile(baseline.sticky_worker_cells),ap=cpu_low_worker_sorted_load_profile(aug.sticky_worker_cells);if(cpu_low_worker_score_equal_aug(base.after,az.after)&&bp<ap)return 5;
 std::cout<<std::setprecision(12)<<"cpu_low_worker_augmented_plan OK objective=exact-neutral-augmented-v5.36-plan n="<<n<<" workers="<<workers<<" domain_size="<<domain<<" max_run="<<mr<<" max_swap="<<ms<<" multistart_source="<<cpu_low_worker_multistart_source_name(md.source)<<" baseline_pages_2m="<<base.after.pages_2m<<" baseline_pages_4k="<<base.after.pages_4k<<" baseline_transitions="<<base.after.transitions<<" baseline_max_worker_cells="<<augmax(baseline)<<" augmented_pages_2m="<<az.after.pages_2m<<" augmented_pages_4k="<<az.after.pages_4k<<" augmented_transitions="<<az.after.transitions<<" augmented_max_worker_cells="<<augmax(aug)<<" pages_2m_delta="<<int64_t(az.after.pages_2m)-int64_t(base.after.pages_2m)<<" pages_4k_delta="<<int64_t(az.after.pages_4k)-int64_t(base.after.pages_4k)<<" transition_delta="<<int64_t(az.after.transitions)-int64_t(base.after.transitions)<<" augmented_rounds="<<az.rounds<<" exact_schedule_changes="<<az.exact_schedule_changes<<" exact_primary_improvements="<<az.exact_primary_improvements<<" exact_profile_improvements="<<az.exact_profile_improvements<<" neutral_moves="<<az.neutral_moves<<" neutral_candidates="<<az.neutral_candidates<<" augmented_build_s="<<az.build_s<<" workspace_audit_ok="<<int(ws.structural_audit_ok)<<" workspace_audited_jobs="<<ws.audited_jobs<<" workspace_audited_cells="<<ws.audited_cells<<" workspace_mib="<<double(ws.bytes())/double(1ULL<<20)<<" workspace_audit_s="<<ws.audit_s<<" workspace_build_s="<<ws.build_s<<" limits_clear=1\n";return 0;
}
