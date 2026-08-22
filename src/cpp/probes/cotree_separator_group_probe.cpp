#define main cotree_schedule_original_main
#include "cotree_schedule_zdd.cpp"
#undef main
#include <map>
#include <iomanip>

static std::uint64_t union_nodes(rapidd::ZddManager& m, std::vector<rapidd::Zdd> const& roots){
 std::unordered_set<std::uint32_t> seen;std::vector<rapidd::Zdd>q=roots;
 while(!q.empty()){auto z=q.back();q.pop_back();if(z.raw()<=1||!seen.insert(z.raw()).second)continue;auto[lo,hi]=m.split(z);q.push_back(lo);q.push_back(hi);}return seen.size();
}
static std::uint64_t occmask(std::string const&k,int na){std::uint64_t m=0;size_t p=1;for(int i=0;i<na;++i){auto d=(unsigned char)k[p++];p++;auto lab1=(unsigned char)k[p++];if(d||lab1)m|=1ull<<i;}return m;}
int main(int ac,char**av){int n=ac>1?std::stoi(av[1]):10;int st=ac>2?std::stoi(av[2]):-1;std::uint32_t cap=ac>3?std::stoul(av[3]):32000000u;B b(n,"formula",cap);auto root=b.build();if(st<0)st=b.Q/2;auto const&mm=b.memo[st];int na=b.active[st].size();
 std::map<std::uint64_t,std::vector<rapidd::Zdd>> g;for(auto const&kv:mm)g[occmask(kv.first,na)].push_back(kv.second);
 std::uint64_t totalu=0,maxu=0;size_t maxs=0;std::vector<std::pair<std::uint64_t,size_t>> rows;rows.reserve(g.size());int idx=0;
 for(auto&kv:g){auto u=union_nodes(b.m,kv.second);totalu+=u;maxu=std::max(maxu,u);maxs=std::max(maxs,kv.second.size());rows.push_back({u,kv.second.size()});if(++idx%100==0)std::cerr<<"group "<<idx<<"/"<<g.size()<<"\r";}
 std::sort(rows.rbegin(),rows.rend());auto global=union_nodes(b.m,[&](){std::vector<rapidd::Zdd>v;v.reserve(mm.size());for(auto const&kv:mm)v.push_back(kv.second);return v;}());
 std::cout<<"n="<<n<<" st="<<st<<" active="<<na<<" states="<<mm.size()<<" occ_groups="<<g.size()<<" global_union="<<global<<" sum_group_union="<<totalu<<" duplication="<<std::fixed<<std::setprecision(3)<<double(totalu)/global<<" max_group_union="<<maxu<<" max_group_states="<<maxs<<"\n";
 for(int i=0;i<std::min<int>(20,rows.size());++i)std::cout<<"  rank="<<i<<" nodes="<<rows[i].first<<" states="<<rows[i].second<<"\n";
}
