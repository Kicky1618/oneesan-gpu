#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using Code = std::uint64_t;

int main(){
    std::uint64_t cases=0,covered=0;
    for(int threads: {32,64,128,256,512,1024}){
        for(Code n=1;n<=200000;n+=1+(n>>8)){
            const Code blocks=std::min<Code>(65535,std::max<Code>(1,(n+Code(threads)*8-1)/(Code(threads)*8)));
            const Code grid=blocks*Code(threads);
            std::vector<unsigned char> seen(n,0);
            for(Code tid=0;tid<grid;++tid){
                for(Code base=tid;base<n;base+=8*grid){
                    for(int k=0;k<8;++k){
                        const Code i=base+Code(k)*grid;
                        if(i<n){
                            if(seen[i]){std::cerr<<"duplicate n="<<n<<" threads="<<threads<<" i="<<i<<'\n';return 2;}
                            seen[i]=1;++covered;
                        }
                    }
                }
            }
            if(std::find(seen.begin(),seen.end(),0)!=seen.end()){
                std::cerr<<"hole n="<<n<<" threads="<<threads<<'\n';return 3;
            }
            ++cases;
        }
    }
    std::cout<<"b300-ilp8-partition-proof OK cases="<<cases
             <<" covered="<<covered
             <<" destinations_per_thread=8 exact_partition=1 launch_blocks=ceil_n_over_8threads_capped65535\n";
    return 0;
}
