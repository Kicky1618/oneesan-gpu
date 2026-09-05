#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Code=unsigned long long;
static constexpr int MAXW=28,NGPU=8;
static Code H_DP[MAXW+1][MAXW+2]{};

static void build_full_dp(){
    for(int h=0;h<=MAXW+1;++h)H_DP[0][h]=(h==0);
    for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){Code x=H_DP[w-1][h];if(h>0)x+=H_DP[w-1][h-1];if(h<MAXW+1)x+=H_DP[w-1][h+1];H_DP[w][h]=x;}
}
static Code spec_size(int width,std::uint32_t fixed,std::uint32_t occ){
    Code prev[MAXW+2]{},cur[MAXW+2]{};prev[0]=1;
    for(int w=1;w<=width;++w){
        std::fill(cur,cur+MAXW+2,Code(0));int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;
        for(int h=0;h<=MAXW;++h){Code x=0;if(!f||!o)x+=prev[h];if(!f||o){if(h>0)x+=prev[h-1];if(h<MAXW+1)x+=prev[h+1];}cur[h]=x;}
        std::copy(cur,cur+MAXW+2,prev);
    }
    return prev[1];
}
static std::vector<int> window_candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
static void window_masks(int W,int hi,int lo,const std::vector<int>&fp,std::uint32_t group,std::uint32_t&mf,std::uint32_t&mo,std::uint32_t&bf,std::uint32_t&bo){
    (void)W;(void)hi;mf=mo=bf=bo=0;
    for(size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=(q<lo-1)?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}
}
struct Plan{int k=0;std::uint64_t groups=0;std::size_t max_bytes=0;};
static Plan plan_window(int W,int hi,int lo,std::size_t target,int maxbits=20){
    auto cand=window_candidates(W,hi,lo);int klim=std::min<int>(cand.size(),maxbits);
    for(int k=0;k<=klim;++k){
        std::vector<int>fp(cand.begin(),cand.begin()+k);std::uint64_t ng=1ull<<k;std::size_t mx=0;
        for(std::uint64_t g=0;g<ng;++g){std::uint32_t mf,mo,bf,bo;window_masks(W,hi,lo,fp,std::uint32_t(g),mf,mo,bf,bo);Code ms=spec_size(W,mf,mo),ds=spec_size(W-1,bf,bo);std::size_t b=std::size_t(2*ms+2*ds)*sizeof(std::uint32_t);mx=std::max(mx,b);if(mx>target&&k<klim)break;}
        if(mx<=target||k==klim)return{k,ng,mx};
    }
    std::abort();
}
struct GroupWork{Code work=0,main=0,block=0;std::uint32_t g=0;};
static std::vector<GroupWork> group_works(int W,int hi,int lo,int k){
    auto cand=window_candidates(W,hi,lo);std::vector<int>fp(cand.begin(),cand.begin()+k);std::vector<GroupWork>w;w.reserve(size_t(1u)<<k);
    for(std::uint32_t g=0;g<(1u<<k);++g){std::uint32_t mf,mo,bf,bo;window_masks(W,hi,lo,fp,g,mf,mo,bf,bo);Code ms=spec_size(W,mf,mo),ds=spec_size(W-1,bf,bo);w.push_back({2*ms+ds,ms,ds,g});}
    std::sort(w.begin(),w.end(),[](auto const&a,auto const&b){if(a.work!=b.work)return a.work>b.work;if(a.main!=b.main)return a.main>b.main;if(a.block!=b.block)return a.block>b.block;return a.g<b.g;});return w;
}
struct LptStats{std::array<Code,NGPU>load{},main{},block{};std::array<std::uint32_t,NGPU>count{};};
static LptStats lpt_assign(const std::vector<GroupWork>& work){
    LptStats s;
    for(auto const&x:work){int d=0;for(int q=1;q<NGPU;++q)if(s.load[q]<s.load[d])d=q;s.load[d]+=x.work;s.main[d]+=x.main;s.block[d]+=x.block;++s.count[d];}
    return s;
}
template<class A>static Code sum_array(const A&a){Code s=0;for(auto x:a)s+=x;return s;}
template<class A>static Code spread_array(const A&a){auto mm=std::minmax_element(a.begin(),a.end());return *mm.second-*mm.first;}
int main(){
    build_full_dp();constexpr int W=28,max_window=14;constexpr std::size_t target=std::size_t(16384)<<20;constexpr std::size_t meta_bytes=13936;
    struct Win{int hi,lo,k;std::uint64_t groups;std::size_t max_bytes;};std::vector<Win>windows;
    for(int hi=W-1;hi>=1;){bool found=false;for(int lo=std::max(1,hi-max_window+1);lo<=hi;++lo){auto p=plan_window(W,hi,lo,target);if(p.max_bytes&&p.max_bytes<=target){windows.push_back({hi,lo,p.k,p.groups,p.max_bytes});hi=lo-1;found=true;break;}}if(!found){std::fprintf(stderr,"cannot fit hi=%d\n",hi);return 2;}}
    std::uint64_t groups=0;for(auto const&w:windows)groups+=w.groups;std::size_t replicated_per_gpu=std::size_t(groups)*meta_bytes;
    if(windows.size()!=2||windows[0].hi!=27||windows[0].lo!=14||windows[0].k!=13||windows[0].groups!=8192||windows[1].hi!=13||windows[1].lo!=1||windows[1].k!=13||windows[1].groups!=8192){std::fprintf(stderr,"unexpected default schedule\n");return 3;}
    if(groups!=16384||replicated_per_gpu!=228327424ull){std::fprintf(stderr,"unexpected staged footprint groups=%llu bytes=%zu\n",(unsigned long long)groups,replicated_per_gpu);return 4;}
    if(windows[0].max_bytes!=15859230032ull||windows[1].max_bytes!=15859230032ull){std::fprintf(stderr,"unexpected max group bytes\n");return 5;}

    constexpr std::array<std::uint32_t,NGPU> expected_count={1022,1023,1024,1024,1024,1024,1025,1026};
    constexpr std::array<Code,NGPU> expected_load={113306829941ull,113307078532ull,113306581843ull,113306710480ull,113306619065ull,113306619065ull,113307452970ull,113306626751ull};
    constexpr std::array<Code,NGPU> expected_main={48215315792ull,48215115745ull,48214903183ull,48214952011ull,48214803399ull,48214803399ull,48215070712ull,48214542379ull};
    constexpr std::array<Code,NGPU> expected_block={16876198357ull,16876847042ull,16876775477ull,16876806458ull,16877012267ull,16877012267ull,16877311546ull,16877541993ull};
    std::array<std::uint32_t,NGPU> combined_count{};std::array<Code,NGPU> combined_load{},combined_main{},combined_block{};Code max_window_spread=0;
    for(auto const&w:windows){auto work=group_works(W,w.hi,w.lo,w.k);auto st=lpt_assign(work);if(st.count!=expected_count||st.load!=expected_load||st.main!=expected_main||st.block!=expected_block){std::fprintf(stderr,"unexpected deterministic LPT assignment for %d:%d\n",w.hi,w.lo);return 6;}if(sum_array(st.load)!=906454518647ull||sum_array(st.main)!=385719506620ull||sum_array(st.block)!=135015505407ull){std::fprintf(stderr,"unexpected LPT component totals\n");return 7;}max_window_spread=std::max(max_window_spread,spread_array(st.load));for(int d=0;d<NGPU;++d){combined_count[d]+=st.count[d];combined_load[d]+=st.load[d];combined_main[d]+=st.main[d];combined_block[d]+=st.block[d];}}
    constexpr std::array<std::uint32_t,NGPU> expected_combined_count={2044,2046,2048,2048,2048,2048,2050,2052};
    constexpr std::array<Code,NGPU> expected_combined_load={226613659882ull,226614157064ull,226613163686ull,226613420960ull,226613238130ull,226613238130ull,226614905940ull,226613253502ull};
    constexpr std::array<Code,NGPU> expected_combined_main={96430631584ull,96430231490ull,96429806366ull,96429904022ull,96429606798ull,96429606798ull,96430141424ull,96429084758ull};
    constexpr std::array<Code,NGPU> expected_combined_block={33752396714ull,33753694084ull,33753550954ull,33753612916ull,33754024534ull,33754024534ull,33754623092ull,33755083986ull};
    if(combined_count!=expected_combined_count||combined_load!=expected_combined_load||combined_main!=expected_combined_main||combined_block!=expected_combined_block||max_window_spread!=871127ull){std::fprintf(stderr,"unexpected combined deterministic LPT assignment\n");return 8;}
    if(sum_array(combined_load)!=1812909037294ull||sum_array(combined_main)!=771439013240ull||sum_array(combined_block)!=270031010814ull){std::fprintf(stderr,"unexpected combined component totals\n");return 9;}
    if(spread_array(combined_load)!=1742254ull||spread_array(combined_main)!=1546826ull||spread_array(combined_block)!=2687272ull){std::fprintf(stderr,"unexpected combined component spreads\n");return 10;}

    std::size_t static_total_bytes=0,static_max_bytes=0;for(int d=0;d<NGPU;++d){std::size_t b=std::size_t(combined_count[d])*meta_bytes;static_total_bytes+=b;static_max_bytes=std::max(static_max_bytes,b);}
    if(static_total_bytes!=228327424ull||static_max_bytes!=28596672ull){std::fprintf(stderr,"unexpected static metadata footprint\n");return 11;}
    const double work_avg=double(sum_array(combined_load))/NGPU,main_avg=double(sum_array(combined_main))/NGPU,block_avg=double(sum_array(combined_block))/NGPU;
    const double max_ratio=double(*std::max_element(combined_load.begin(),combined_load.end()))/work_avg;
    const double work_spread_pct=100.0*double(spread_array(combined_load))/work_avg;
    const double main_spread_pct=100.0*double(spread_array(combined_main))/main_avg;
    const double block_spread_pct=100.0*double(spread_array(combined_block))/block_avg;
    const double any_linear_bound_pct=std::max(main_spread_pct,block_spread_pct);
    std::printf("b300-staged-group-meta-plan-proof OK W=28 target_mib=16384 max_window=14 windows=2 window0=27:14 window1=13:1 fixed_bits=13 groups_per_window=8192 total_groups=%llu meta_bytes=%zu replicated_staged_bytes_per_gpu=%zu replicated_staged_mib_per_gpu=217.75 replicated_total_h2d_gib=1.701171875 old_per_row_group_processings=16384 rows=28 total_group_processings=458752 old_meta_h2d_gib=5.9541015625 replicated_h2d_reduction=3.5x static_lpt=1 deterministic_tie_break=work_main_block_group lpt_count_min=2044 lpt_count_max=2052 lpt_max_load_ratio=%.9f lpt_work_spread_pct=%.12f lpt_main_total=%llu lpt_main_spread=%llu lpt_main_spread_pct=%.12f lpt_block_total=%llu lpt_block_spread=%llu lpt_block_spread_pct=%.12f lpt_any_nonnegative_linear_cost_spread_bound_pct=%.12f static_staged_total_h2d_gib=0.212646484375 static_staged_max_mib_per_gpu=27.271911621 static_vs_replicated_h2d_reduction=8x max_group_bytes=15859230032 exact=1\n",(unsigned long long)groups,meta_bytes,replicated_per_gpu,max_ratio,work_spread_pct,(unsigned long long)sum_array(combined_main),(unsigned long long)spread_array(combined_main),main_spread_pct,(unsigned long long)sum_array(combined_block),(unsigned long long)spread_array(combined_block),block_spread_pct,any_linear_bound_pct);
    return 0;
}
