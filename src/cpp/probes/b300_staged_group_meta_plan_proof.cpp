#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Code=unsigned long long;
static constexpr int MAXW=28;
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
int main(){
    build_full_dp();constexpr int W=28,max_window=14;constexpr std::size_t target=std::size_t(16384)<<20;constexpr std::size_t meta_bytes=13936;
    struct Win{int hi,lo,k;std::uint64_t groups;std::size_t max_bytes;};std::vector<Win>windows;
    for(int hi=W-1;hi>=1;){bool found=false;for(int lo=std::max(1,hi-max_window+1);lo<=hi;++lo){auto p=plan_window(W,hi,lo,target);if(p.max_bytes&&p.max_bytes<=target){windows.push_back({hi,lo,p.k,p.groups,p.max_bytes});hi=lo-1;found=true;break;}}if(!found){std::fprintf(stderr,"cannot fit hi=%d\n",hi);return 2;}}
    std::uint64_t groups=0;for(auto const&w:windows)groups+=w.groups;std::size_t per_gpu=std::size_t(groups)*meta_bytes;
    if(windows.size()!=2||windows[0].hi!=27||windows[0].lo!=14||windows[0].k!=13||windows[0].groups!=8192||windows[1].hi!=13||windows[1].lo!=1||windows[1].k!=13||windows[1].groups!=8192){std::fprintf(stderr,"unexpected default schedule\n");return 3;}
    if(groups!=16384||per_gpu!=228327424ull){std::fprintf(stderr,"unexpected staged footprint groups=%llu bytes=%zu\n",(unsigned long long)groups,per_gpu);return 4;}
    if(windows[0].max_bytes!=15859230032ull||windows[1].max_bytes!=15859230032ull){std::fprintf(stderr,"unexpected max group bytes\n");return 5;}
    std::printf("b300-staged-group-meta-plan-proof OK W=28 target_mib=16384 max_window=14 windows=2 window0=27:14 window1=13:1 fixed_bits=13 groups_per_window=8192 total_groups=%llu meta_bytes=%zu staged_bytes_per_gpu=%zu staged_mib_per_gpu=217.75 staged_total_h2d_gib=1.701171875 old_per_row_group_processings=16384 rows=28 total_group_processings=458752 old_meta_h2d_gib=5.9541015625 h2d_reduction=3.5x max_group_bytes=15859230032 exact=1\n",(unsigned long long)groups,meta_bytes,per_gpu);
    return 0;
}
