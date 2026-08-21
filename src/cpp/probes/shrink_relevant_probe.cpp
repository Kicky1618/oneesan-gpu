#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main
#include <iostream>
static Mate rem(Mate m,int k){MateID x=m.id(),lo=k?(x&((1ULL<<(2*k))-1)):0,hi=x&~((1ULL<<(2*(k+1)))-1);return Mate(lo|(hi>>2));}
int main(){msg=NONE;for(int width=3;width<=10;++width){MateCodec mc(width,(width+1)/2,1,0);for(int p=2;p<width;++p){int bad=0,total=0;for(Code bi=0;bi<mc.codeSizeL();++bi){auto const&b=mc.codeTable(bi);for(Code i=0;i<b.size;++i){Mate m=b.mateL|b.mateR[i], a,z;auto w=m.getPair(p);bool use=false;int k=-1;
 if(w==NR||w==NL){use=true;k=p;a=m.shrink(k);z=rem(m,k);} 
 else if(w==LL){m.setPair(p,NN);int q=p-1,s=1;while(s>0){--q;auto v=m.get(q);if(v==L)++s;else if(v==R)--s;}m.set(q,L);use=true;k=p-1;a=m.shrink(k);z=rem(m,k);} 
 else if(w==RR){m.setPair(p,NN);int q=p,s=1;while(s>0){++q;auto v=m.get(q);if(v==L)--s;else if(v==R)++s;}m.set(q,R);use=true;k=p-1;a=m.shrink(k);z=rem(m,k);} 
 else if(w==RL){m.setPair(p,NN);use=true;k=p-1;a=m.shrink(k);z=rem(m,k);} 
 if(use){++total;if(a.id()!=z.id()){if(bad++<10)std::cerr<<"BAD w="<<width<<" p="<<p<<" pair="<<w<<" orig="<<a<<" ours="<<z<<"\n";}}
 } } std::cout<<"w="<<width<<" p="<<p<<" total="<<total<<" bad="<<bad<<"\n";}}
}
