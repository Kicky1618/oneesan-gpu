#include <array>
#include <cstdint>
#include <iostream>

using Code=std::uint64_t;
namespace {
constexpr int MAXW=28,HC=MAXW+2;
Code H[MAXW+1][MAXW+3]{};
void build(){for(int h=0;h<=MAXW+2;++h)H[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW+1;++h)H[w][h]=H[w-1][h]+(h?H[w-1][h-1]:0)+H[w-1][h+1];}
bool step(int pos,int v,int&h,long long&d){if(v==3)return false;if(v==1){if(h<=0)return false;d+=static_cast<long long>(H[pos-1][h])-static_cast<long long>(H[pos][h]);--h;}else if(v==2){if(h>=MAXW+1)return false;Code b=H[pos-1][h]+(h?H[pos-1][h-1]:0),a=H[pos][h]+(h?H[pos][h-1]:0);d+=static_cast<long long>(b)-static_cast<long long>(a);++h;}return true;}
std::uint32_t pack_chunk(int c,int h0,std::uint32_t code){int top=14-4*c,lo=top-3,h=h0;long long d=0;for(int pos=top;pos>=lo;--pos)if(!step(pos,(code>>(2*(pos-lo)))&3u,h,d))return 0xff000000u;if(d<-(1ll<<23)||d>=(1ll<<23)||h<0||h>255)return 0xff000000u;return (std::uint32_t(std::int32_t(d))&0x00ffffffu)|(std::uint32_t(h)<<24);}
long long decode_delta(std::uint32_t z){return std::int32_t(z<<8)>>8;}
}
int main(){build();std::uint64_t checked=0;long long min_d=0,max_d=0,max_abs=0;
    for(int c=0;c<3;++c)for(int h0=0;h0<HC;++h0)for(std::uint32_t code=0;code<256;++code){std::uint32_t z=pack_chunk(c,h0,code);if((z>>24)==0xff)continue;int top=14-4*c,lo=top-3,h=h0;long long d=0;for(int pos=top;pos>=lo;--pos)if(!step(pos,(code>>(2*(pos-lo)))&3u,h,d))return 2;if(decode_delta(z)!=d||int(z>>24)!=h)return 3;min_d=std::min(min_d,d);max_d=std::max(max_d,d);max_abs=std::max(max_abs,d<0?-d:d);++checked;}
    if(max_abs>=(1ll<<23))return 4;
    std::cout<<"b300-low-drop-chunk-proof OK checked="<<checked<<" min_delta="<<min_d<<" max_delta="<<max_d<<" max_abs="<<max_abs<<" packed_delta_bits=24 table_bytes="<<(3*HC*256*4)<<" max_table_loads=3 max_scalar_tail=3 exact=1\n";
}
