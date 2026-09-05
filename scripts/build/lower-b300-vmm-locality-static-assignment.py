#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

PW_OLD='''struct PreparedWindow{
    WindowPlan wp;
    std::vector<PreparedGroup> groups;
    std::vector<std::vector<int>> by_gpu;
    std::vector<Code> assigned_work;
};'''
PW_NEW='''struct PreparedWindow{
    WindowPlan wp;
    std::vector<PreparedGroup> groups;
    std::vector<std::vector<int>> by_gpu;
    std::vector<Code> assigned_work;
    Code baseline_local_states=0,locality_local_states=0,total_authoritative_states=0,locality_work_cap=0;
};
static bool b300_group_choice_allowed(const GroupSpec&s,int p,int v){bool f=(s.fixed>>p)&1u,o=(s.occ>>p)&1u;if(!f)return true;if(!o)return v==0;return v!=0;}
static Code b300_group_fullrank_prefix(const GroupSpec&s,Code t){
    Code full=H_DP[s.width][1];if(t==0)return 0;if(t>=full)return s.size;Code ans=0,rem=t;int h=1;
    for(int p=s.width-1;p>=0;--p){
        Code fs=H_DP[p][h],gs=b300_group_choice_allowed(s,p,0)?s.dp[p][h]:0;
        if(rem<=fs){if(!b300_group_choice_allowed(s,p,0))return ans;if(rem==fs)return ans+gs;continue;}ans+=gs;rem-=fs;
        if(h){fs=H_DP[p][h-1];gs=b300_group_choice_allowed(s,p,1)?s.dp[p][h-1]:0;if(rem<=fs){if(!b300_group_choice_allowed(s,p,1))return ans;if(rem==fs)return ans+gs;--h;continue;}ans+=gs;rem-=fs;}
        fs=H_DP[p][h+1];gs=b300_group_choice_allowed(s,p,2)?s.dp[p][h+1]:0;if(rem<=fs){if(!b300_group_choice_allowed(s,p,2))return ans;if(rem==fs)return ans+gs;++h;continue;}ans+=gs;rem-=fs;std::abort();
    }
    return ans;
}
static Code b300_group_vmm_local_states(const GroupSpec&s,const b300_vmm::ContiguousStorage&store,int d){
    Code full=H_DP[s.width][1];Code lo=std::min<Code>(full,Code(store.offsets[size_t(d)]/sizeof(Count)));Code hi=std::min<Code>(full,Code(store.offsets[size_t(d+1)]/sizeof(Count)));return b300_group_fullrank_prefix(s,hi)-b300_group_fullrank_prefix(s,lo);
}'''

ASSIGN_OLD='''        pw.by_gpu.resize(ng);pw.assigned_work.assign(ng,0);
        for(int q=0;q<nj;++q){int d=0;for(int x=1;x<ng;++x)if(pw.assigned_work[x]<pw.assigned_work[d])d=x;pw.by_gpu[d].push_back(q);pw.assigned_work[d]+=pw.groups[q].work;}
        schedule.push_back(std::move(pw));hi=wp.p_lo-1;'''
ASSIGN_NEW='''        pw.by_gpu.resize(ng);pw.assigned_work.assign(ng,0);std::vector<Code> baseline_work(ng,0);Code total_window_work=0;for(auto const&pg:pw.groups)total_window_work+=pg.work;pw.locality_work_cap=(total_window_work*10001ull+Code(ng)*10000ull-1)/(Code(ng)*10000ull);
        for(int q=0;q<nj;++q){auto const&pg=pw.groups[q];std::array<Code,MAXGPU>local{};for(int d=0;d<ng;++d)local[d]=b300_group_vmm_local_states(pg.ms,main_store,d)+b300_group_vmm_local_states(pg.ds,block_store,d);
            int base_d=0;for(int d=1;d<ng;++d)if(baseline_work[d]<baseline_work[base_d])base_d=d;baseline_work[base_d]+=pg.work;pw.baseline_local_states+=local[base_d];
            int d=-1;for(int x=0;x<ng;++x)if(pw.assigned_work[x]+pg.work<=pw.locality_work_cap){if(d<0||local[x]>local[d]||(local[x]==local[d]&&(pw.assigned_work[x]<pw.assigned_work[d]||(pw.assigned_work[x]==pw.assigned_work[d]&&x<d))))d=x;}if(d<0){d=0;for(int x=1;x<ng;++x)if(pw.assigned_work[x]+pg.work<pw.assigned_work[d]+pg.work)d=x;}
            pw.by_gpu[d].push_back(q);pw.assigned_work[d]+=pg.work;pw.locality_local_states+=local[d];pw.total_authoritative_states+=pg.ms.size+pg.ds.size;}
        if(*std::max_element(pw.assigned_work.begin(),pw.assigned_work.end())>pw.locality_work_cap){std::cerr<<"VMM locality assignment exceeded work cap window="<<wp.p_hi<<":"<<wp.p_lo<<" max="<<*std::max_element(pw.assigned_work.begin(),pw.assigned_work.end())<<" cap="<<pw.locality_work_cap<<"\\n";return 19;}
        schedule.push_back(std::move(pw));hi=wp.p_lo-1;'''

STAGE_HEAD_OLD='''    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();'''
STAGE_HEAD_NEW='''    Code vmm_baseline_local_states=0,vmm_locality_local_states=0,vmm_locality_total_states=0;for(auto const&pw:schedule){vmm_baseline_local_states+=pw.baseline_local_states;vmm_locality_local_states+=pw.locality_local_states;vmm_locality_total_states+=pw.total_authoritative_states;}
    if(vmm_locality_local_states<vmm_baseline_local_states){std::cerr<<"VMM locality assignment regressed local states baseline="<<vmm_baseline_local_states<<" locality="<<vmm_locality_local_states<<"\\n";return 19;}
    double vmm_baseline_remote=double(vmm_locality_total_states-vmm_baseline_local_states),vmm_locality_remote=double(vmm_locality_total_states-vmm_locality_local_states);double vmm_remote_reduction=vmm_baseline_remote?100.0*(vmm_baseline_remote-vmm_locality_remote)/vmm_baseline_remote:0.0;
    std::cerr<<"VMM locality assignment: baseline_local="<<vmm_baseline_local_states<<" locality_local="<<vmm_locality_local_states<<" total_states="<<vmm_locality_total_states<<" baseline_local_pct="<<(100.0*double(vmm_baseline_local_states)/double(vmm_locality_total_states))<<" locality_local_pct="<<(100.0*double(vmm_locality_local_states)/double(vmm_locality_total_states))<<" remote_reduction_pct="<<vmm_remote_reduction<<" work_cap_over_ideal_pct=0.01\\n";
    size_t staged_group_count=0;for(auto const&pw:schedule)staged_group_count+=pw.groups.size();'''

GUARD_OLD='''    if(W==28&&ng==8&&effective_target_mib==16384&&max_window==14&&(staged_group_count!=16384||static_lpt_group_min!=2044||static_lpt_group_max!=2052||staged_group_meta_total_bytes!=228327424ull||staged_group_meta_max_bytes!=28596672ull)){std::cerr<<"static LPT default-plan proof mismatch: groups="<<staged_group_count<<" group_min="<<static_lpt_group_min<<" group_max="<<static_lpt_group_max<<" total_bytes="<<staged_group_meta_total_bytes<<" max_bytes="<<staged_group_meta_max_bytes<<"\\n";return 14;}'''
GUARD_NEW='''    if(W==28&&ng==8&&effective_target_mib==16384&&max_window==14&&(staged_group_count!=16384||staged_group_meta_total_bytes!=228327424ull||vmm_locality_total_states!=1041470024054ull)){std::cerr<<"VMM locality default-plan proof mismatch: groups="<<staged_group_count<<" total_bytes="<<staged_group_meta_total_bytes<<" total_states="<<vmm_locality_total_states<<"\\n";return 19;}'''

LOG_OLD='''copy_mode=H2D_once_local_then_D2D_per_group scheduler=static_lpt'''
LOG_NEW='''copy_mode=H2D_once_local_then_D2D_per_group scheduler=vmm_locality_lpt'''

def once(t,o,n,l):
    c=t.count(o)
    if c!=1:raise SystemExit(f'{l}: expected one static-LPT match, got {c}')
    return t.replace(o,n,1)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('src',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args();t=a.src.read_text();t=once(t,PW_OLD,PW_NEW,'locality helpers');t=once(t,ASSIGN_OLD,ASSIGN_NEW,'locality assignment');t=once(t,STAGE_HEAD_OLD,STAGE_HEAD_NEW,'locality runtime report');t=once(t,GUARD_OLD,GUARD_NEW,'locality default guard');t=once(t,LOG_OLD,LOG_NEW,'scheduler label')
    for stale in ('static_lpt_group_min!=2044','static_lpt_group_max!=2052','staged_group_meta_max_bytes!=28596672ull','scheduler=static_lpt'):
        if stale in t:raise SystemExit(f'fixed-LPT artifact remains: {stale}')
    for r in ('b300_group_fullrank_prefix','b300_group_vmm_local_states','pw.locality_work_cap','VMM locality assignment:','remote_reduction_pct=','scheduler=vmm_locality_lpt','vmm_locality_total_states!=1041470024054ull'):
        if r not in t:raise SystemExit(f'missing VMM locality artifact: {r}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(t);print(f'lowered {a.out} scheduler=vmm_locality_lpt exact_fullrank_segment_counts=1 work_cap_over_ideal_pct=0.01 runtime_layout_offsets=1 baseline_comparison=1')
if __name__=='__main__':main()
