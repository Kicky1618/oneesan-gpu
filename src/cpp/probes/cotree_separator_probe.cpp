#define main cotree_schedule_original_main
#include "cotree_schedule_zdd.cpp"
#undef main
#include <random>

static std::uint64_t subtree_nodes(rapidd::ZddManager& m, rapidd::Zdd root) {
    std::unordered_set<std::uint32_t> seen;
    std::vector<rapidd::Zdd> q{root};
    while(!q.empty()){
        auto z=q.back(); q.pop_back();
        if(z.raw()<=1 || !seen.insert(z.raw()).second) continue;
        auto [lo,hi]=m.split(z); q.push_back(lo);q.push_back(hi);
    }
    return seen.size();
}
static std::uint64_t union_nodes(rapidd::ZddManager& m, std::vector<rapidd::Zdd> const& roots) {
    std::unordered_set<std::uint32_t> seen;
    std::vector<rapidd::Zdd> q=roots;
    while(!q.empty()){
        auto z=q.back(); q.pop_back();
        if(z.raw()<=1 || !seen.insert(z.raw()).second) continue;
        auto [lo,hi]=m.split(z); q.push_back(lo);q.push_back(hi);
    }
    return seen.size();
}
int main(int ac,char**av){
    if(ac<2){std::cerr<<"usage: n [order=formula] [cap] [samples]\n";return 2;}
    int n=std::stoi(av[1]); std::string ord=ac>2?av[2]:"formula";
    std::uint32_t cap=ac>3?std::stoul(av[3]):32000000u; int samples=ac>4?std::stoi(av[4]):32;
    B b(n,ord,cap); auto root=b.build();
    std::cout<<"n="<<n<<" Q="<<b.Q<<" root_nodes="<<root.node_count()<<" allocated="<<b.m.allocated_nodes()<<"\n";
    std::mt19937_64 rng(1618+n);
    for(int st=0;st<b.Q;++st){
        bool boundary=(st==0||st==b.Q-1);
        if(st>0 && st<b.Q && b.vars[st-1].u/b.W != b.vars[st].u/b.W) boundary=true;
        if(!boundary && st%(std::max(1,n/2))!=0) continue;
        auto const& mm=b.memo[st];
        if(mm.empty()){std::cout<<"st="<<st<<" states=0\n";continue;}
        std::vector<rapidd::Zdd> roots; roots.reserve(std::min<int>(samples,mm.size()));
        std::vector<rapidd::Zdd> allroots; if(mm.size()<=50000)allroots.reserve(mm.size());
        int k=0; for(auto const& kv:mm){ if(mm.size()<=50000)allroots.push_back(kv.second); if((int)roots.size()<samples)roots.push_back(kv.second); ++k; }
        // Reservoir sample when large.
        if((int)mm.size()>samples){ roots.clear(); roots.reserve(samples); std::uint64_t i=0; for(auto const&kv:mm){++i;if((int)roots.size()<samples)roots.push_back(kv.second);else {auto j=rng()%i;if(j<(std::uint64_t)samples)roots[j]=kv.second;}} }
        std::uint64_t mx=0,sum=0; for(auto z:roots){auto q=subtree_nodes(b.m,z);mx=std::max(mx,q);sum+=q;}
        std::uint64_t un=allroots.empty()?0:union_nodes(b.m,allroots);
        int r=b.vars[st].u/b.W;
        std::cout<<"st="<<st<<" next_row="<<r<<" states="<<mm.size()<<" sample="<<roots.size()<<" suffix_avg="<<(double)sum/roots.size()<<" suffix_max_sample="<<mx;
        if(un)std::cout<<" suffix_union="<<un;
        std::cout<<"\n";
    }
}
