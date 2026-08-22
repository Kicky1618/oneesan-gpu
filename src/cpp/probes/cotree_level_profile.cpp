#define main cotree_schedule_original_main
#include "cotree_schedule_zdd.cpp"
#undef main
#include <map>
int main(int ac,char**av){int n=ac>1?std::stoi(av[1]):10;uint32_t cap=ac>2?std::stoul(av[2]):32000000u;B b(n,"formula",cap);auto root=b.build();std::vector<std::uint64_t> cnt(b.Q+1);std::unordered_set<uint32_t>seen;std::vector<rapidd::Zdd>q{root};while(!q.empty()){auto z=q.back();q.pop_back();if(z.raw()<=1||!seen.insert(z.raw()).second)continue;auto lv=b.m.top_level(z);++cnt[lv];auto[lo,hi]=b.m.split(z);q.push_back(lo);q.push_back(hi);}std::uint64_t mx=0,sum=0;int ml=0;for(int l=1;l<=b.Q;++l){sum+=cnt[l];if(cnt[l]>mx){mx=cnt[l];ml=l;}}std::cout<<"n="<<n<<" Q="<<b.Q<<" total="<<sum<<" peak_level_nodes="<<mx<<" peak_level="<<ml<<" ratio="<<double(mx)/sum<<"\n";for(int l=1;l<=b.Q;++l)if(cnt[l]){int st=b.Q-l;std::cout<<"level="<<l<<" st="<<st<<" nodes="<<cnt[l]<<" states="<<b.memo[st].size()<<"\n";}}
