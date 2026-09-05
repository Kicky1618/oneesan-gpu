#pragma once
#include <cstdint>
namespace oneesan {
// A legal frontier segment has digits N=0,R=1,L=2. Reinterpret its two-bit
// digits in base 3; this is a bijection onto [0,3^width).
inline uint32_t ternary_rank_index(uint32_t code){
    uint32_t index=0,scale=1;
    while(code){index+=(code&3u)*scale;scale*=3;code>>=2;}
    return index;
}
inline uint32_t ternary_rank_capacity(int width){
    uint32_t n=1;while(width--)n*=3;return n;
}
}
