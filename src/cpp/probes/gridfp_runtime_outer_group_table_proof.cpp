#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
constexpr int MAX_W=28, ENTRIES=99;
using Rank64=std::uint64_t;
constexpr std::uint32_t GROUP[ENTRIES]={
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_outer_group_values.inc"
};
constexpr Rank64 PREFIX[ENTRIES]={
#include "../../cuda/gridfp/gridfp_reduced_production_runtime_outer_prefix_values.inc"
};
constexpr int ROW_BASE[11]={0,4,9,15,22,30,39,49,60,72,85};
static_assert(sizeof(GROUP)==396); static_assert(sizeof(PREFIX)==792);
Rank64 binom(int n,int k){if(n<0||k<0||k>n)return 0;Rank64 x=1;for(int i=1;i<=k;++i)x=x*Rank64(n-k+i)/Rank64(i);return x;}
std::array<std::array<Rank64,MAX_W+2>,MAX_W+1> primitive(){std::array<std::array<Rank64,MAX_W+2>,MAX_W+1> p{};p[0][0]=1;for(int r=1;r<=MAX_W;++r)for(int h=0;h<=MAX_W;++h)p[r][h]=p[r-1][h+1]+(h?p[r-1][h-1]:0);return p;}
Rank64 group_size(const std::array<std::array<Rank64,MAX_W+2>,MAX_W+1>&p,int L,int outer){Rank64 z=0;for(int l=0;l<=L;++l){int o=outer+l;if(!(o&1))continue;z+=(binom(L,l)+binom(L-2,l-1))*p[o][1];}return z;}
}
int main(){const auto p=primitive();int index=0;std::uint64_t configs=0;Rank64 max_group=0,max_prefix=0;for(int wi=0;wi<11;++wi){int W=8+2*wi,L=W/2+1,O=W-L;if(ROW_BASE[wi]!=index)return 2;Rank64 prefix=0;for(int r=0;r<=O;++r){Rank64 g=group_size(p,L,r);if(g>std::numeric_limits<std::uint32_t>::max())return 3;if(index>=ENTRIES||GROUP[index]!=g||PREFIX[index]!=prefix){std::cerr<<"outer group table mismatch W="<<W<<" r="<<r<<" index="<<index<<" group="<<GROUP[index]<<" expected_group="<<g<<" prefix="<<PREFIX[index]<<" expected_prefix="<<prefix<<'\n';return 4;}max_group=std::max(max_group,g);max_prefix=std::max(max_prefix,prefix);prefix+=binom(O,r)*g;++index;}++configs;}if(configs!=11||index!=ENTRIES||max_group!=1805186805ULL||max_prefix!=471591870896ULL)return 5;std::cout<<"gridfp-runtime-outer-group-table-proof OK W_configs="<<configs<<" entries="<<index<<" group_bytes="<<sizeof(GROUP)<<" prefix_bytes="<<sizeof(PREFIX)<<" total_bytes="<<sizeof(GROUP)+sizeof(PREFIX)<<" max_group="<<max_group<<" max_prefix="<<max_prefix<<" row_bases_exact=1 embedded_exact=1\n";return 0;}
