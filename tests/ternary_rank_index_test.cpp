#include "../src/common/ternary_rank_index.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(){
    std::array<uint16_t,16384> chunk7{};
    std::array<uint16_t,256> chunk4{};
    for(uint32_t i=0;i<chunk7.size();++i)chunk7[i]=oneesan::ternary_rank_index(i);
    for(uint32_t i=0;i<chunk4.size();++i)chunk4[i]=oneesan::ternary_rank_index(i);
    uint64_t checked=0;
    for(int width=0;width<=14;++width){
        uint32_t capacity=oneesan::ternary_rank_capacity(width);
        for(uint32_t rank=0;rank<capacity;++rank){
            uint32_t code=0,n=rank;
            for(int p=0;p<width;++p){code|=(n%3)<<(2*p);n/=3;}
            uint32_t r7=chunk7[code&16383u]+2187u*chunk7[code>>14];
            uint32_t r4=0,scale=1;
            for(int p=0;p<width;p+=4){r4+=scale*chunk4[(code>>(2*p))&255u];scale*=81;}
            if(oneesan::ternary_rank_index(code)!=rank||r7!=rank||r4!=rank)
                throw std::runtime_error("ternary rank indexing mismatch");
            ++checked;
        }
    }
    std::cout<<"PASS "<<checked<<" ternary words, widths 0..14, scalar and 4/7-symbol conversion\n";
}
