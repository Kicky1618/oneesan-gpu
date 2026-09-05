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
                for(std::uint64_t base=tid;base<n;base+=4*grid){
                    for(unsigned lane=0;lane<4;++lane){
                        const std::uint64_t i=base+std::uint64_t(lane)*grid;
                        if(i<n){if(seen[i]++)return 2;++indices;}
                    }
                }
            }
            for(auto x:seen)if(x!=1)return 3;
            ++cases;
        }
    }
    constexpr std::uint64_t G=65535ull*1024ull;
    constexpr std::array<std::uint64_t,12> tids={0,1,31,32,1023,1024,G/2-1,G/2,G-1025,G-33,G-2,G-1};
    for(std::uint64_t round=0;round<32;++round){
        for(std::uint64_t tid:tids){
            const std::uint64_t i=tid+round*G;
            const std::uint64_t base=tid+(round/4)*(4*G);
            const std::uint64_t reconstructed=base+(round&3u)*G;
            if(reconstructed!=i)return 4;
            ++indices;
        }
    }
    cases+=32*tids.size();
    std::cout<<"b300-ilp4-partition-proof OK cases="<<cases<<" visited_indices="<<indices
             <<" pattern=base_tid_plus_lane_grid lanes=4 stride=4grid duplicate=0 missing=0 exact=1\n";
    return 0;
}
