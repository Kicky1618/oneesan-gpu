#pragma once

#include "ramstream32_cpu_low_domain_worker_exact_workspace.hpp"
#include "ramstream32_cpu_low_domain_worker_unique_shared_coalesce.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Research-only v5.35 exact-neutral plateau descent.
// Primary exact tuple must remain identical. Among such neighbor moves, accept
// only a strict lexicographic improvement of the descending worker-load vector.
// The combined (exact tuple, load profile) potential strictly decreases, so the
// pass cannot cycle while it rearranges equal-page states to create future slack.

struct CpuLowWorkerNeutralBalanceStats {
    uint64_t candidate_evaluations = 0;
    uint64_t cap_rejections = 0;
    uint64_t exact_rejections = 0;
    uint64_t profile_rejections = 0;
    uint64_t accepted_moves = 0;
    uint64_t moved_cells = 0;
    bool move_limit_hit = false;
    CpuLowDomainWorkerUniqueScore exact_before{};
    CpuLowDomainWorkerUniqueScore exact_after{};
    uint64_t max_worker_cells_before = 0;
    uint64_t max_worker_cells_after = 0;
    double build_s = 0.0;
};

static std::vector<uint64_t> cpu_low_worker_sorted_load_profile(
    const std::vector<uint64_t>& loads
) {
    std::vector<uint64_t> z = loads;
    std::sort(z.begin(), z.end(), std::greater<uint64_t>());
    return z;
}

static CpuLowWorkerNeutralBalanceStats cpu_low_apply_worker_neutral_balance(
    CpuLowSparsePersistentPool& pool,
    const std::vector<CpuLowJob>& jobs,
    const CpuLowSparseHost& sparse,
    const CpuLowWorkerExactWorkspace& ws,
    uint64_t max_accepted = 4096
) {
    CpuLowWorkerNeutralBalanceStats stats;
    auto t0 = std::chrono::steady_clock::now();
    cpu_low_validate_worker_exact_workspace(ws, jobs, sparse);
    if (pool.schedule_mode != CPU_LOW_SCHEDULE_DOMAIN || pool.domain_size <= 0) {
        std::cerr << "cpu LOW neutral balance requires domain schedule\n";
        std::exit(287);
    }
    if (pool.sticky_source_jobs != &jobs || pool.sticky_source_sparse != &sparse) {
        std::cerr << "cpu LOW neutral balance requires prepared schedule\n";
        std::exit(288);
    }

    const auto& ordered = ws.ordered;
    const auto& dense = ws.dense;
    std::vector<int> owner(ordered.size(), -1), original_domain(ordered.size(), -1);
    for (int w = 0; w < pool.workers; ++w) {
        int d = w / pool.domain_size;
        for (size_t q : pool.sticky_worker_jobs[size_t(w)]) {
            if (q >= jobs.size() || ws.ordered_pos[q] == size_t(-1)) {
                std::cerr << "cpu LOW neutral balance bad job owner\n";
                std::exit(289);
            }
            size_t k = ws.ordered_pos[q];
            if (owner[k] >= 0) {
                std::cerr << "cpu LOW neutral balance duplicate owner\n";
                std::exit(290);
            }
            owner[k] = w;
            original_domain[k] = d;
        }
    }

    int domains = (pool.workers + pool.domain_size - 1) / pool.domain_size;
    std::vector<std::pair<size_t,size_t>> domain_segs(
        size_t(domains), {ordered.size(), ordered.size()});
    int last_domain = -1;
    for (size_t k = 0; k < ordered.size(); ++k) {
        int d = original_domain[k];
        if (owner[k] < 0 || d < 0 || d >= domains || d < last_domain) {
            std::cerr << "cpu LOW neutral balance lost domain ordering\n";
            std::exit(291);
        }
        last_domain = d;
        if (domain_segs[size_t(d)].first == ordered.size())
            domain_segs[size_t(d)].first = k;
        domain_segs[size_t(d)].second = k + 1;
    }

    std::vector<uint64_t> loads = pool.sticky_worker_cells;
    std::vector<uint64_t> domain_cap(size_t(domains), 0);
    for (int w = 0; w < pool.workers; ++w)
        domain_cap[size_t(w / pool.domain_size)] = std::max(
            domain_cap[size_t(w / pool.domain_size)], loads[size_t(w)]);
    stats.max_worker_cells_before = loads.empty() ? 0
        : *std::max_element(loads.begin(), loads.end());
    std::vector<uint64_t> profile = cpu_low_worker_sorted_load_profile(loads);

    std::vector<uint32_t> refs2m(dense.universe_2m.size(), 0), refs4k(dense.universe_4k.size(), 0);
    uint64_t transitions = 0;
    for (size_t k = 1; k < ordered.size(); ++k) {
        if (owner[k - 1] == owner[k]) continue;
        cpu_low_worker_dense_ref_add(refs2m, dense.boundary[k].pages_2m, +1);
        cpu_low_worker_dense_ref_add(refs4k, dense.boundary[k].pages_4k, +1);
        transitions += ws.transition_weight[k];
    }
    uint64_t unique2m = cpu_low_worker_dense_ref_unique(refs2m);
    uint64_t unique4k = cpu_low_worker_dense_ref_unique(refs4k);
    stats.exact_before = {unique2m, unique4k, transitions};

    struct Candidate {
        bool valid=false; size_t pos=0; int src=-1,dst=-1; uint64_t cells=0;
        std::vector<uint64_t> profile;
        CpuLowWorkerDenseDelta d2,d4; int64_t transition_delta=0;
    };

    for (;;) {
        Candidate best;
        best.profile = profile;
        for (int d = 0; d < domains; ++d) {
            auto seg=domain_segs[size_t(d)];
            if(seg.first>=seg.second)continue;
            for(size_t i=seg.first;i<seg.second;++i){
                int src=owner[i];uint64_t cells=ordered[i].cells;
                if(loads[size_t(src)]<cells){std::cerr<<"cpu LOW neutral source load underflow\n";std::exit(321);}
                int dsts[2]={-1,-1};int nd=0;
                if(i>seg.first&&owner[i-1]!=src)dsts[nd++]=owner[i-1];
                if(i+1<seg.second&&owner[i+1]!=src&&(nd==0||owner[i+1]!=dsts[0]))dsts[nd++]=owner[i+1];
                for(int q=0;q<nd;++q){
                    int dst=dsts[q];++stats.candidate_evaluations;
                    if(dst/pool.domain_size!=d){std::cerr<<"cpu LOW neutral balance crossed domain\n";std::exit(292);}
                    if(cells>domain_cap[size_t(d)]||loads[size_t(dst)]>domain_cap[size_t(d)]-cells){++stats.cap_rejections;continue;}
                    CpuLowWorkerDenseDelta d2,d4;int64_t td=0;
                    auto edge=[&](size_t b,int ol,int orr,int nl,int nr){
                        if(!b||b>=ordered.size())return;bool oa=ol!=orr,na=nl!=nr;if(oa==na)return;int s=na?1:-1;
                        cpu_low_worker_dense_delta_add(d2,dense.boundary[b].pages_2m,s);
                        cpu_low_worker_dense_delta_add(d4,dense.boundary[b].pages_4k,s);
                        td+=int64_t(s)*ws.transition_weight[b];
                    };
                    if(i>0)edge(i,owner[i-1],src,owner[i-1],dst);
                    if(i+1<ordered.size())edge(i+1,src,owner[i+1],dst,owner[i+1]);
                    cpu_low_worker_dense_delta_normalize(d2);cpu_low_worker_dense_delta_normalize(d4);
                    int64_t tr=int64_t(transitions)+td;if(tr<0){std::cerr<<"cpu LOW neutral transition underflow\n";std::exit(293);}
                    CpuLowDomainWorkerUniqueScore exact{
                        cpu_low_worker_dense_unique_after_delta(refs2m,unique2m,d2),
                        cpu_low_worker_dense_unique_after_delta(refs4k,unique4k,d4),uint64_t(tr)};
                    if(exact.pages_2m!=unique2m||exact.pages_4k!=unique4k||exact.transitions!=transitions){++stats.exact_rejections;continue;}
                    std::vector<uint64_t> cand_loads=loads;cand_loads[size_t(src)]-=cells;cand_loads[size_t(dst)]+=cells;
                    auto cand_profile=cpu_low_worker_sorted_load_profile(cand_loads);
                    if(!(cand_profile<profile)){++stats.profile_rejections;continue;}
                    bool take=!best.valid||cand_profile<best.profile||(cand_profile==best.profile&&(i<best.pos||(i==best.pos&&dst<best.dst)));
                    if(take){best.valid=true;best.pos=i;best.src=src;best.dst=dst;best.cells=cells;best.profile=std::move(cand_profile);best.d2=std::move(d2);best.d4=std::move(d4);best.transition_delta=td;}
                }
            }
        }
        if(!best.valid)break;
        cpu_low_worker_dense_apply_delta(refs2m,unique2m,best.d2);cpu_low_worker_dense_apply_delta(refs4k,unique4k,best.d4);
        transitions=uint64_t(int64_t(transitions)+best.transition_delta);
        if(loads[size_t(best.src)]<best.cells){std::cerr<<"cpu LOW neutral accepted source load underflow\n";std::exit(322);}
        loads[size_t(best.src)]-=best.cells;loads[size_t(best.dst)]+=best.cells;owner[best.pos]=best.dst;profile=std::move(best.profile);
        ++stats.accepted_moves;stats.moved_cells+=best.cells;
        if(stats.accepted_moves>=max_accepted){stats.move_limit_hit=true;break;}
    }

    uint64_t expected=0;for(const auto&x:ordered)expected+=x.cells;
    std::vector<std::vector<size_t>>next_jobs(size_t(pool.workers));std::vector<uint64_t>next_cells(size_t(pool.workers),0);
    for(size_t k=0;k<ordered.size();++k){int w=owner[k];if(w<0||w>=pool.workers||w/pool.domain_size!=original_domain[k]){std::cerr<<"cpu LOW neutral final owner mismatch\n";std::exit(294);}next_jobs[size_t(w)].push_back(ordered[k].index);next_cells[size_t(w)]+=ordered[k].cells;}
    uint64_t assigned=0;for(uint64_t x:next_cells)assigned+=x;if(assigned!=expected){std::cerr<<"cpu LOW neutral accounting mismatch\n";std::exit(295);}
    stats.max_worker_cells_after=next_cells.empty()?0:*std::max_element(next_cells.begin(),next_cells.end());if(stats.max_worker_cells_after>stats.max_worker_cells_before){std::cerr<<"cpu LOW neutral max-worker regression\n";std::exit(296);}
    std::vector<uint32_t>v2(refs2m.size(),0),v4(refs4k.size(),0);uint64_t vt=0;for(size_t k=1;k<ordered.size();++k){if(owner[k-1]==owner[k])continue;cpu_low_worker_dense_ref_add(v2,dense.boundary[k].pages_2m,+1);cpu_low_worker_dense_ref_add(v4,dense.boundary[k].pages_4k,+1);vt+=ws.transition_weight[k];}
    if(v2!=refs2m||v4!=refs4k||vt!=transitions){std::cerr<<"cpu LOW neutral exact accounting mismatch\n";std::exit(297);}
    stats.exact_after={cpu_low_worker_dense_ref_unique(v2),cpu_low_worker_dense_ref_unique(v4),vt};
    if(stats.exact_before.pages_2m!=stats.exact_after.pages_2m||stats.exact_before.pages_4k!=stats.exact_after.pages_4k||stats.exact_before.transitions!=stats.exact_after.transitions){std::cerr<<"cpu LOW neutral changed exact objective\n";std::exit(298);}
    pool.sticky_worker_jobs.swap(next_jobs);pool.sticky_worker_cells.swap(next_cells);stats.build_s=ram_seconds_since(t0);
    std::cerr<<"cpu_low_worker_neutral_balance objective=exact-neutral-load-descent-v5.35-plan accepted_moves="<<stats.accepted_moves<<" candidate_evaluations="<<stats.candidate_evaluations<<" exact_rejections="<<stats.exact_rejections<<" profile_rejections="<<stats.profile_rejections<<" move_limit_hit="<<(stats.move_limit_hit?1:0)<<" max_worker_cells_before="<<stats.max_worker_cells_before<<" max_worker_cells_after="<<stats.max_worker_cells_after<<" exact_pages_2m="<<stats.exact_after.pages_2m<<" exact_pages_4k="<<stats.exact_after.pages_4k<<" exact_transitions="<<stats.exact_after.transitions<<" build_s="<<stats.build_s<<'\n';
    return stats;
}
