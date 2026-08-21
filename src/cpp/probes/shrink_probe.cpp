#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
static MateID ours(MateID m,int k){MateID lo=k?(m&((1ULL<<(2*k))-1ULL)):0;MateID hi=m&~((1ULL<<(2*(k+1)))-1ULL);return lo|(hi>>2);}
int main(){msg=NONE;for(int width=3;width<=8;++width){MateCodec mc(width,(width+1)/2,1,0);for(int p=1;p<width;++p){int bad=0;for(Code bi=0;bi<mc.codeSizeL();++bi){auto const& b=mc.codeTable(bi);for(Code i=0;i<b.size;++i){Mate m=b.mateL|b.mateR[i];for(int k:{p,p-1}){auto a=m.shrink(k).id(),z=ours(m.id(),k);if(a!=z){if(bad++<8)std::cerr<<"w="<<width<<" p="<<p<<" k="<<k<<" m="<<m<<" orig="<<Mate(a)<<" ours="<<Mate(z)<<"\n";}}}}std::cout<<"w="<<width<<" p="<<p<<" bad="<<bad<<"\n";}}
}
