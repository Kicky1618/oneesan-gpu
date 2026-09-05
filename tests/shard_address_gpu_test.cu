#include <cuda_runtime.h>
#include "../src/common/shard_address.hpp"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
struct Input { uint64_t g,chunk; int devices; };
static void check(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}
__global__ void address_test(const Input* inputs,oneesan::ShardAddress* outputs,size_t n){
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
        auto x=inputs[i];outputs[i]=oneesan::shard_address(x.g,x.chunk,x.devices);
    }
}
int main(){
    std::vector<Input> inputs;
    for(int devices=1;devices<=8;++devices)for(uint64_t chunk : {uint64_t(1),uint64_t(3),uint64_t(1)<<31,uint64_t(1)<<63,UINT64_MAX}){
        for(int o=0;o<devices;++o){
            __uint128_t base=__uint128_t(chunk)*o;
            if(base>UINT64_MAX)break;
            inputs.push_back({uint64_t(base),chunk,devices});
            if(base)inputs.push_back({uint64_t(base-1),chunk,devices});
            __uint128_t end=base+chunk-1;if(end>UINT64_MAX)end=UINT64_MAX;
            inputs.push_back({uint64_t(end),chunk,devices});
        }
    }
    std::mt19937_64 rng(20260905);
    for(int i=0;i<1000000;++i){
        int devices=1+rng()%8;uint64_t chunk=rng()|1,g=rng();
        __uint128_t size=__uint128_t(devices)*chunk;if(size<=UINT64_MAX)g%=uint64_t(size);
        inputs.push_back({g,chunk,devices});
    }
    Input* a;oneesan::ShardAddress* b;size_t n=inputs.size();
    check(cudaMalloc(&a,n*sizeof(Input)));check(cudaMalloc(&b,n*sizeof(*b)));
    check(cudaMemcpy(a,inputs.data(),n*sizeof(Input),cudaMemcpyHostToDevice));
    address_test<<<256,256>>>(a,b,n);check(cudaGetLastError());
    std::vector<oneesan::ShardAddress> result(n);
    check(cudaMemcpy(result.data(),b,n*sizeof(*b),cudaMemcpyDeviceToHost));
    for(size_t i=0;i<n;++i)if(result[i].owner!=inputs[i].g/inputs[i].chunk||result[i].offset!=inputs[i].g%inputs[i].chunk){
        std::fprintf(stderr,"GPU address mismatch at %zu\n",i);return 2;
    }
    check(cudaFree(a));check(cudaFree(b));std::printf("PASS %zu GPU shard addresses (1..8 simulated shards)\n",n);
}
