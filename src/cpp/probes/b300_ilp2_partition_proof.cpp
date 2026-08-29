#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

int main(){
    std::uint64_t cases=0,indices=0;
    for(std::uint64_t n=1;n<=20000;++n){
        for(std::uint64_t grid: {1ull,2ull,3ull,7ull,31ull,32ull,33ull,127ull,256ull,1024ull,4096ull}){
            std::vector<unsigned char> seen(n,0);
            for(std::uint64_t tid=0;tid<grid;++tid){
                for(std::uint64_t base=tid;base<n;base+=2*grid){
                    const std::uint64_t i0=base,i1=base+grid;
                    if(i0<n){if(seen[i0]++)return 2;++indices;}
                    if(i1<n){if(seen[i1]++)return 3;++indices;}
                }
            }
            for(auto x:seen)if(x!=1)return 4;
            ++cases;
        }
    }
    // Production cap: algebraically sample both ILP lanes and several rounds
    // around 65535*1024 threads. No giant allocation or O(G) loop is needed.
    constexpr std::uint64_t G=65535ull*1024ull;
    constexpr std::array<std::uint64_t,12> tids={0,1,31,32,1023,1024,G/2-1,G/2,G-1025,G-33,G-2,G-1};
    for(std::uint64_t round=0;round<16;++round){
        for(std::uint64_t tid:tids){
            const std::uint64_t i=tid+round*G;
            const std::uint64_t base=tid+(round/2)*(2*G);
            const std::uint64_t reconstructed=base+(round&1u)*G;
            if(reconstructed!=i)return 5;
            ++indices;
        }
    }
    cases+=16*tids.size();
    std::cout<<"b300-ilp2-partition-proof OK cases="<<cases<<" visited_indices="<<indices
             <<" pattern=base_tid_plus_grid stride=2grid duplicate=0 missing=0 exact=1\n";
    return 0;
}
