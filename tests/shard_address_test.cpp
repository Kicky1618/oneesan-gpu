#include "../src/common/shard_address.hpp"
#include <cstdlib>
#include <iostream>
#include <random>

static uint64_t checked=0;
static void check(uint64_t g,uint64_t chunk,int devices) {
    auto a=oneesan::shard_address(g,chunk,devices);
    if(a.owner!=g/chunk||a.offset!=g%chunk||a.owner>=unsigned(devices)) {
        std::cerr<<"shard mismatch g="<<g<<" chunk="<<chunk<<" devices="<<devices<<'\n';
        std::exit(1);
    }
    ++checked;
}
int main() {
    for(int devices=1;devices<=8;++devices) {
        for(uint64_t chunk=1;chunk<=128;++chunk)
            for(uint64_t g=0;g<chunk*devices;++g)check(g,chunk,devices);
        for(uint64_t chunk : {uint64_t(1),uint64_t(3),uint64_t(1)<<31,uint64_t(1)<<63,UINT64_MAX/4,UINT64_MAX/2,UINT64_MAX}) {
            for(int owner=0;owner<devices;++owner) {
                __uint128_t base=__uint128_t(chunk)*owner;
                if(base>UINT64_MAX)break;
                check(uint64_t(base),chunk,devices);
                if(base)check(uint64_t(base-1),chunk,devices);
                __uint128_t end=base+chunk-1;
                if(end>UINT64_MAX)end=UINT64_MAX;
                check(uint64_t(end),chunk,devices);
            }
        }
    }
    std::mt19937_64 rng(20260905);
    for(int i=0;i<1000000;++i) {
        int devices=1+rng()%8;uint64_t chunk=rng()|1;
        __uint128_t size=__uint128_t(devices)*chunk;
        uint64_t g=rng();if(size<=UINT64_MAX)g%=uint64_t(size);
        check(g,chunk,devices);
    }
    std::cout<<"PASS "<<checked<<" shard addresses, 1..8 devices\n";
}
