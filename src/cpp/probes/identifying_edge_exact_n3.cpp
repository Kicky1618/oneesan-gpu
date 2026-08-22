#define main identifying_main
#include "identifying_edge_search.cpp"
#undef main
#include <bit>
int main(){auto z=load("work/n3.zdd");std::vector<uint64_t>p;dfs(z,z.root,0,p);uint64_t lim=1ull<<24;uint64_t tested=0;for(uint64_t m=0;m<lim;++m)if(std::popcount(m)==8){++tested;if(injective(p,m)){std::cout<<"FOUND 8 mask="<<std::hex<<m<<"\n";return 0;}}std::cout<<"NO_8 tested="<<tested<<"\n";}
